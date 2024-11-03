# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================
# detector_bayesian_torch.py
# Description: Implementation of Bayesian detector for SynthID algorithm in PyTorch
# ============================================

import abc
from collections.abc import Mapping, Sequence
import enum
import functools
import gc
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import model_selection
import tqdm


def pad_to_len(
    arr: torch.Tensor,
    target_len: int,
    *,
    left_pad: bool,
    eos_token: int,
    device: torch.device,
) -> torch.Tensor:
    """Pad or truncate array to given length."""
    if arr.shape[1] < target_len:
        shape_for_ones = list(arr.shape)
        shape_for_ones[1] = target_len - shape_for_ones[1]
        padded = torch.ones(
            shape_for_ones,
            device=device,
            dtype=torch.long,
        ) * eos_token
        
        if not left_pad:
            return torch.cat((arr, padded), dim=1)
        else:
            return torch.cat((padded, arr), dim=1)
    else:
        return arr[:, :target_len]


def filter_and_truncate(
    outputs: torch.Tensor,
    truncation_length,
    eos_token_mask: torch.Tensor, 
) -> torch.Tensor:
    """Filter and truncate outputs to given length."""
    if truncation_length:
        outputs = outputs[:, :truncation_length]
        truncation_mask = torch.sum(eos_token_mask, dim=1) >= truncation_length
        return outputs[truncation_mask, :]
    return outputs


def process_outputs_for_training(
    all_outputs: Sequence[torch.Tensor],
    logits_processor,
    tokenizer: Any,
    *,
    pos_truncation_length,
    neg_truncation_length, 
    max_length: int,
    is_cv: bool,
    is_pos: bool,
    torch_device: torch.device,
) -> tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
    """Process raw model outputs for training."""
    all_masks = []
    all_g_values = []
    
    for outputs in tqdm.tqdm(all_outputs):
        eos_token_mask = logits_processor.compute_eos_token_mask(
            input_ids=outputs,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        if is_pos or is_cv:
            outputs = filter_and_truncate(
                outputs, pos_truncation_length, eos_token_mask
            )
        elif not is_pos and not is_cv:
            outputs = filter_and_truncate(
                outputs, neg_truncation_length, eos_token_mask
            )
            
        if outputs.shape[0] == 0:
            continue
            
        outputs = pad_to_len(
            outputs,
            max_length,
            left_pad=False, 
            eos_token=tokenizer.eos_token_id,
            device=torch_device,
        )

        eos_token_mask = logits_processor.compute_eos_token_mask(
            input_ids=outputs,
            eos_token_id=tokenizer.eos_token_id,
        )

        context_repetition_mask = logits_processor.compute_context_repetition_mask(
            input_ids=outputs,
        )

        context_repetition_mask = pad_to_len(
            context_repetition_mask,
            max_length,
            left_pad=True,
            eos_token=0,
            device=torch_device,
        )
        
        combined_mask = context_repetition_mask * eos_token_mask

        g_values = logits_processor.compute_g_values(
            input_ids=outputs,
        )

        g_values = pad_to_len(
            g_values, 
            max_length,
            left_pad=True,
            eos_token=0,
            device=torch_device
        )

        all_masks.append(combined_mask)
        all_g_values.append(g_values)
        
    return all_masks, all_g_values


class LikelihoodModel(nn.Module, abc.ABC):
    """Base class for likelihood models."""

    @abc.abstractmethod
    def forward(self, g_values: torch.Tensor) -> torch.Tensor:
        """Compute likelihoods given g-values."""
        pass


class LikelihoodModelWatermarked(LikelihoodModel):
    """Model for P(g_values|watermarked)."""

    def __init__(self, watermarking_depth: int):
        super().__init__()
        self.watermarking_depth = watermarking_depth
        
        # Initialize parameters
        self.beta = nn.Parameter(
            -2.5 + 0.001 * torch.randn(1, 1, watermarking_depth)
        )
        self.delta = nn.Parameter(
            0.001 * torch.randn(1, 1, watermarking_depth, watermarking_depth)
        )

    def l2_loss(self) -> torch.Tensor:
        return torch.sum(self.delta ** 2)

    def _compute_latents(
        self, g_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute latent probabilities."""
        x = g_values.unsqueeze(-2).repeat(
            1, 1, self.watermarking_depth, 1
        )
        x = torch.tril(x, diagonal=-1)
        
        logits = torch.sum(self.delta * x, dim=-1) + self.beta
        
        p_two_unique_tokens = torch.sigmoid(logits)
        p_one_unique_token = 1 - p_two_unique_tokens
        
        return p_one_unique_token, p_two_unique_tokens

    def forward(self, g_values: torch.Tensor) -> torch.Tensor:
        """Compute P(g_values|watermarked)."""
        p_one_unique_token, p_two_unique_tokens = self._compute_latents(g_values)
        return 0.5 * ((g_values + 0.5) * p_two_unique_tokens + p_one_unique_token)


class LikelihoodModelUnwatermarked(LikelihoodModel):
    """Model for P(g_values|unwatermarked)."""

    def forward(self, g_values: torch.Tensor) -> torch.Tensor:
        """Compute P(g_values|unwatermarked)."""
        return 0.5 * torch.ones_like(g_values)


class BayesianDetectorModule(nn.Module):
    """Bayesian detector model."""

    def __init__(
        self,
        watermarking_depth: int,
        baserate: float = 0.5
    ):
        super().__init__()
        self.watermarking_depth = watermarking_depth
        self.baserate = baserate
        
        self.likelihood_model_watermarked = LikelihoodModelWatermarked(
            watermarking_depth=watermarking_depth
        )
        self.likelihood_model_unwatermarked = LikelihoodModelUnwatermarked()
        self.prior = nn.Parameter(torch.tensor([baserate]))

    def l2_loss(self) -> torch.Tensor:
        return self.likelihood_model_watermarked.l2_loss()

    def forward(
        self,
        g_values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute P(watermarked|g_values)."""
        likelihoods_watermarked = self.likelihood_model_watermarked(g_values)
        likelihoods_unwatermarked = self.likelihood_model_unwatermarked(g_values)
        
        mask = mask.unsqueeze(-1)
        prior = torch.clamp(self.prior, 1e-5, 1 - 1e-5)
        
        log_likelihoods_watermarked = torch.log(
            torch.clamp(likelihoods_watermarked, min=1e-30)
        )
        log_likelihoods_unwatermarked = torch.log(
            torch.clamp(likelihoods_unwatermarked, min=1e-30)
        )
        
        log_odds = log_likelihoods_watermarked - log_likelihoods_unwatermarked
        
        relative_surprisal_likelihood = torch.sum(
            log_odds * mask, dim=(1, 2)
        )
        
        relative_surprisal_prior = torch.log(prior) - torch.log(1 - prior)
        relative_surprisal = relative_surprisal_prior + relative_surprisal_likelihood
        
        return torch.sigmoid(relative_surprisal)


def train_model(
    detector_module: BayesianDetectorModule,
    g_values: torch.Tensor,
    mask: torch.Tensor,
    watermarked: torch.Tensor,
    epochs: int = 250,
    learning_rate: float = 1e-3,
    minibatch_size: int = 64,
    l2_weight: float = 0.0,
    g_values_val: torch.Tensor = None,
    mask_val: torch.Tensor = None,
    watermarked_val: torch.Tensor = None,
    verbose: bool = False,
) -> tuple[dict, float]:
    """Train the detector model."""
    
    optimizer = torch.optim.Adam(detector_module.parameters(), lr=learning_rate)
    
    history = {}
    min_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        detector_module.train()
        
        # Training
        train_losses = []
        for i in range(0, len(g_values), minibatch_size):
            batch_g = g_values[i:i+minibatch_size]
            batch_m = mask[i:i+minibatch_size] 
            batch_w = watermarked[i:i+minibatch_size]

            optimizer.zero_grad()
            
            pred = detector_module(batch_g, batch_m)
            loss = F.binary_cross_entropy(
                pred, batch_w.float()
            ) + l2_weight * detector_module.l2_loss()
            
            train_losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
        avg_train_loss = sum(train_losses) / len(train_losses)
            
        # Validation
        detector_module.eval()
        with torch.no_grad():
            if g_values_val is not None:
                val_pred = detector_module(g_values_val, mask_val)
                val_loss = F.binary_cross_entropy(
                    val_pred, watermarked_val.float()
                )
                
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    best_state = detector_module.state_dict()
                    
                if verbose:
                    print(f"Epoch {epoch}: train_loss {avg_train_loss:.4f}, val_loss {val_loss:.4f}")
                    
            history[epoch] = {
                'train_loss': avg_train_loss,
                'val_loss': val_loss.item() if g_values_val is not None else None
            }
            
    if best_state is not None:
        detector_module.load_state_dict(best_state)
    return history, min_val_loss


class RawBayesianDetector:
    """Bayesian detector for watermark detection."""

    def __init__(
        self,
        logits_processor,
        tokenizer: Any,
        detector_module: BayesianDetectorModule,
    ):
        self.detector_module = detector_module
        self.logits_processor = logits_processor
        self.tokenizer = tokenizer

    def score(self, outputs: torch.Tensor) -> torch.Tensor:
        """Score outputs for watermark detection."""
        eos_token_mask = self.logits_processor.compute_eos_token_mask(
            input_ids=outputs,
            eos_token_id=self.tokenizer.eos_token_id,
        )[:, self.logits_processor.ngram_len - 1:]

        context_repetition_mask = self.logits_processor.compute_context_repetition_mask(
            input_ids=outputs,
        )

        combined_mask = context_repetition_mask * eos_token_mask

        g_values = self.logits_processor.compute_g_values(
            input_ids=outputs,
        )

        with torch.no_grad():
            return self.detector_module(g_values, combined_mask)

    @classmethod
    def train_best_detector(
        cls,
        *,
        tokenized_wm_outputs,
        tokenized_uwm_outputs,
        logits_processor,
        tokenizer: Any,
        torch_device: torch.device,
        test_size: float = 0.3,
        pos_truncation_length: int = 200, 
        neg_truncation_length: int = 100,
        max_padded_length: int = 2300,
        n_epochs: int = 50,
        learning_rate: float = 2.1e-2,
        l2_weights: np.ndarray = np.logspace(-3, -2, num=4),
        verbose: bool = False,
    ) -> tuple["RawBayesianDetector", float]:
        """Train the best detector model."""
        
        if torch_device.type not in ("cuda", "tpu"):
            raise ValueError(
                "Training is unstable on CPUs. Please use GPU or TPU."
            )

        # Process outputs
        (
            train_g_values,
            train_masks,
            train_labels,
            cv_g_values,
            cv_masks, 
            cv_labels,
        ) = cls.process_raw_model_outputs(
            tokenized_wm_outputs=tokenized_wm_outputs,
            tokenized_uwm_outputs=tokenized_uwm_outputs,
            logits_processor=logits_processor,
            tokenizer=tokenizer,
            torch_device=torch_device,
            test_size=test_size,
            pos_truncation_length=pos_truncation_length,
            neg_truncation_length=neg_truncation_length,
            max_padded_length=max_padded_length,
        )

        # Convert numpy arrays to torch tensors
        train_g_values = torch.from_numpy(train_g_values).to(torch_device)
        train_masks = torch.from_numpy(train_masks).to(torch_device)
        train_labels = torch.from_numpy(train_labels).to(torch_device)
        cv_g_values = torch.from_numpy(cv_g_values).to(torch_device)
        cv_masks = torch.from_numpy(cv_masks).to(torch_device)
        cv_labels = torch.from_numpy(cv_labels).to(torch_device)

        # Train models with different L2 weights
        best_detector = None
        lowest_loss = float('inf')
        
        for l2_weight in l2_weights:
            detector_module = BayesianDetectorModule(
                watermarking_depth=len(logits_processor.keys)
            ).to(torch_device)
            
            _, min_val_loss = train_model(
                detector_module=detector_module,
                g_values=train_g_values,
                mask=train_masks,
                watermarked=train_labels,
                g_values_val=cv_g_values,
                mask_val=cv_masks,
                watermarked_val=cv_labels,
                learning_rate=learning_rate,
                l2_weight=l2_weight,
                epochs=n_epochs,
                verbose=verbose,
            )

            if min_val_loss < lowest_loss:
                lowest_loss = min_val_loss
                best_detector = detector_module

        return cls(logits_processor, tokenizer, best_detector), lowest_loss

    @classmethod
    def process_raw_model_outputs(
        cls,
        *,
        tokenized_wm_outputs,
        tokenized_uwm_outputs,
        logits_processor,
        tokenizer: Any,
        torch_device: torch.device,
        test_size: float = 0.3,
        pos_truncation_length: int = 200,
        neg_truncation_length: int = 100,
        max_padded_length: int = 2300,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process raw models outputs into inputs for training.

        Args:
            tokenized_wm_outputs: tokenized outputs of watermarked data
            tokenized_uwm_outputs: tokenized outputs of unwatermarked data
            logits_processor: logits processor used for watermarking
            tokenizer: tokenizer used for the model
            torch_device: torch device to use
            test_size: test size for train-test split
            pos_truncation_length: Length to truncate wm outputs
            neg_truncation_length: Length to truncate uwm outputs
            max_padded_length: Length to pad truncated outputs

        Returns:
            Tuple of (train_g_values, train_masks, train_labels, 
                     cv_g_values, cv_masks, cv_labels)
        """
        # Split data into train and CV
        train_wm_outputs, cv_wm_outputs = model_selection.train_test_split(
            tokenized_wm_outputs, test_size=test_size
        )

        train_uwm_outputs, cv_uwm_outputs = model_selection.train_test_split(
            tokenized_uwm_outputs, test_size=test_size
        )

        # Process both train and CV data
        wm_masks_train, wm_g_values_train = process_outputs_for_training(
            [torch.tensor(outputs, device=torch_device, dtype=torch.long)
             for outputs in train_wm_outputs],
            logits_processor=logits_processor,
            tokenizer=tokenizer,
            pos_truncation_length=pos_truncation_length,
            neg_truncation_length=neg_truncation_length,
            max_length=max_padded_length,
            is_pos=True,
            is_cv=False,
            torch_device=torch_device,
        )

        wm_masks_cv, wm_g_values_cv = process_outputs_for_training(
            [torch.tensor(outputs, device=torch_device, dtype=torch.long)
             for outputs in cv_wm_outputs],
            logits_processor=logits_processor,
            tokenizer=tokenizer,
            pos_truncation_length=pos_truncation_length,
            neg_truncation_length=neg_truncation_length,
            max_length=max_padded_length,
            is_pos=True,
            is_cv=True,
            torch_device=torch_device,
        )

        uwm_masks_train, uwm_g_values_train = process_outputs_for_training(
            [torch.tensor(outputs, device=torch_device, dtype=torch.long)
             for outputs in train_uwm_outputs],
            logits_processor=logits_processor,
            tokenizer=tokenizer,
            pos_truncation_length=pos_truncation_length,
            neg_truncation_length=neg_truncation_length,
            max_length=max_padded_length,
            is_pos=False,
            is_cv=False,
            torch_device=torch_device,
        )

        uwm_masks_cv, uwm_g_values_cv = process_outputs_for_training(
            [torch.tensor(outputs, device=torch_device, dtype=torch.long)
             for outputs in cv_uwm_outputs],
            logits_processor=logits_processor,
            tokenizer=tokenizer,
            pos_truncation_length=pos_truncation_length,
            neg_truncation_length=neg_truncation_length,
            max_length=max_padded_length,
            is_pos=False,
            is_cv=True,
            torch_device=torch_device,
        )

        # Concatenate data
        wm_masks_train = torch.cat(wm_masks_train, dim=0)
        wm_g_values_train = torch.cat(wm_g_values_train, dim=0)
        wm_labels_train = torch.ones((wm_masks_train.shape[0],), dtype=torch.bool)
        
        wm_masks_cv = torch.cat(wm_masks_cv, dim=0)
        wm_g_values_cv = torch.cat(wm_g_values_cv, dim=0)
        wm_labels_cv = torch.ones((wm_masks_cv.shape[0],), dtype=torch.bool)

        uwm_masks_train = torch.cat(uwm_masks_train, dim=0)
        uwm_g_values_train = torch.cat(uwm_g_values_train, dim=0)
        uwm_labels_train = torch.zeros((uwm_masks_train.shape[0],), dtype=torch.bool)
        
        uwm_masks_cv = torch.cat(uwm_masks_cv, dim=0)
        uwm_g_values_cv = torch.cat(uwm_g_values_cv, dim=0)
        uwm_labels_cv = torch.zeros((uwm_masks_cv.shape[0],), dtype=torch.bool)

        # Combine positive and negative examples
        train_g_values = torch.cat((wm_g_values_train, uwm_g_values_train), dim=0).cpu().numpy()
        train_masks = torch.cat((wm_masks_train, uwm_masks_train), dim=0).cpu().numpy()
        train_labels = torch.cat((wm_labels_train, uwm_labels_train), dim=0).cpu().numpy()

        cv_g_values = torch.cat((wm_g_values_cv, uwm_g_values_cv), dim=0).cpu().numpy()
        cv_masks = torch.cat((wm_masks_cv, uwm_masks_cv), dim=0).cpu().numpy()
        cv_labels = torch.cat((wm_labels_cv, uwm_labels_cv), dim=0).cpu().numpy()

        # Free GPU memory
        del (wm_g_values_train, wm_labels_train, wm_masks_train,
             wm_g_values_cv, wm_labels_cv, wm_masks_cv)
        gc.collect()
        torch.cuda.empty_cache()

        # Shuffle data
        train_g_values = np.squeeze(train_g_values)
        train_masks = np.squeeze(train_masks)
        train_labels = np.squeeze(train_labels)

        cv_g_values = np.squeeze(cv_g_values)
        cv_masks = np.squeeze(cv_masks)
        cv_labels = np.squeeze(cv_labels)

        # Shuffle training data
        train_idx = np.random.permutation(len(train_labels))
        train_g_values = train_g_values[train_idx]
        train_masks = train_masks[train_idx]
        train_labels = train_labels[train_idx]

        # Shuffle CV data
        cv_idx = np.random.permutation(len(cv_labels))
        cv_g_values = cv_g_values[cv_idx]
        cv_masks = cv_masks[cv_idx]
        cv_labels = cv_labels[cv_idx]

        return (train_g_values, train_masks, train_labels,
                cv_g_values, cv_masks, cv_labels)