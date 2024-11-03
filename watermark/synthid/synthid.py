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
# synthid.py
# Description: Implementation of SynthID algorithm
# ============================================

import torch
from math import sqrt
from functools import partial
from ..base import BaseWatermark
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from transformers import LogitsProcessor, LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization
from typing import Dict, Tuple, Union
import numpy as np
from .detector import get_detector


class SynthIDConfig:
    """Config class for Default Watermark algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str = None, transformers_config: TransformersConfig = None, *args, **kwargs) -> None:
        """
        Initialize the Default Watermark configuration.

        Parameters:
            algorithm_config (str): Path to the algorithm configuration file.
            transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/SynthID.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'SynthID':
            raise AlgorithmNameMismatchError('SynthID', config_dict['algorithm_name'])

        # SynthID specific parameters
        self.ngram_len = config_dict['ngram_len']
        self.keys = config_dict['keys']
        self.sampling_table_size = config_dict['sampling_table_size']
        self.sampling_table_seed = config_dict['sampling_table_seed']
        self.context_history_size = config_dict['context_history_size']
        self.detector_name = config_dict['detector_type']
        self.threshold = config_dict['threshold']
        
        # Model configuration
        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs
        self.top_k = getattr(transformers_config, 'top_k', -1)
        self.temperature = getattr(transformers_config, 'temperature', 0.7)
        


class SynthIDUtils:
    """Utility class for SynthID algorithm, contains helper functions."""

    def __init__(self, config: SynthIDConfig, *args, **kwargs) -> None:
        self.config = config
        self.rng = torch.Generator(device=self.config.device)
        self.rng.manual_seed(self.config.sampling_table_seed)
        
    def accumulate_hash(
        self,
        current_hash: torch.LongTensor,
        data: torch.LongTensor,
        multiplier: int = 6364136223846793005,
        increment: int = 1,
    ) -> torch.LongTensor:
        """Accumulate hash of data on current hash.

        Method uses adapted linear congruential generator with newlib/musl parameters.

        This function has following property -
        f(x, data[T]) = f(f(x, data[:T - 1]), data[T])

        This function expects current_hash.shape and data.shape[:-1] to
        match/broadcastable.

        Args:
            current_hash: (shape,)
            data: (shape, tensor_len)
            multiplier: (int) multiplier of linear congruential generator
            increment: (int) increment of linear congruential generator

        Returns:
            updated hash (shape,)
        """
        for i in range(data.shape[-1]):
            current_hash = torch.add(current_hash, data[..., i])
            current_hash = torch.mul(current_hash, multiplier)
            current_hash = torch.add(current_hash, increment)
        return current_hash

    def update_scores(
        self,
        scores: torch.FloatTensor,
        g_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Updates scores using the g values.

        We assume that the scores are in the log space.
        Args:
            scores: Scores (batch_size, vocab_size).
            g_values: G values (batch_size, vocab_size, depth).

        Returns:
            Updated scores (batch_size, vocab_size).
        """
        _, _, depth = g_values.shape
        device = scores.device

        probs = torch.softmax(scores, dim=1)

        for i in range(depth):
            g_values_at_depth = g_values[:, :, i]
            g_mass_at_depth = (g_values_at_depth * probs).sum(axis=1, keepdims=True)
            probs = probs * (1 + g_values_at_depth - g_mass_at_depth)

        log_probs = torch.log(probs)
        log_probs = torch.where(
            torch.isfinite(log_probs), log_probs, torch.tensor(-1e12, device=device)
        )
        return log_probs

    def update_scores_distortionary(
        self,
        scores: torch.FloatTensor,
        g_values: torch.FloatTensor,
        num_leaves: int,
    ) -> torch.FloatTensor:
        """Update scores using the g values for distortionary tournament watermarking.

        We assume that the scores are in the log space.
        Args:
            scores: Scores (batch_size, vocab_size).
            g_values: G values (batch_size, vocab_size, depth).
            num_leaves: Number of leaves per node in the tournament tree.

        Returns:
            Updated scores (batch_size, vocab_size).
        """
        _, _, depth = g_values.shape
        device = scores.device

        probs = torch.softmax(scores, dim=1)

        for i in range(depth):
            g_values_at_depth = g_values[:, :, i]
            g_mass_at_depth = (g_values_at_depth * probs).sum(axis=1, keepdims=True)
            coeff_not_in_g = (1 - g_mass_at_depth)**(num_leaves - 1)
            coeff_in_g = (1 - (1 - g_mass_at_depth)**(num_leaves)) / g_mass_at_depth
            coeffs = torch.where(
                torch.logical_and(g_values_at_depth == 1, probs > 0),
                coeff_in_g, coeff_not_in_g)
            probs = probs * coeffs

        log_probs = torch.log(probs)
        log_probs = torch.where(
            torch.isfinite(log_probs), log_probs, torch.tensor(-1e12, device=device)
        )
        return log_probs
    
    def mean_score_numpy(self,g_values, mask):
        """
        Args:
            g_values: shape [batch_size, seq_len, watermarking_depth]
            mask: shape [batch_size, seq_len]
        Returns:
            scores: shape [batch_size]
        """
        watermarking_depth = g_values.shape[-1]
        num_unmasked = np.sum(mask, axis=1)  # shape [batch_size]
        return np.sum(g_values * np.expand_dims(mask, 2), axis=(1, 2)) / (
                watermarking_depth * num_unmasked
        )
    
    def weighted_mean_score_numpy(
        self,
        g_values: np.ndarray,
        mask: np.ndarray,
        weights: np.ndarray = None,
    ) -> np.ndarray:
        """Computes the Weighted Mean score.

        Args:
            g_values: g-values of shape [batch_size, seq_len, watermarking_depth]
            mask: A binary array shape [batch_size, seq_len] indicating which g-values
                should be used. g-values with mask value 0 are discarded
            weights: array of non-negative floats, shape [watermarking_depth]. The
                weights to be applied to the g-values. If not supplied, defaults to
                linearly decreasing weights from 10 to 1

        Returns:
            Weighted Mean scores, of shape [batch_size]. This is the mean of the
            unmasked g-values, re-weighted using weights.
        """
        watermarking_depth = g_values.shape[-1]

        if weights is None:
            weights = np.linspace(start=10, stop=1, num=watermarking_depth)

        # Normalise weights so they sum to watermarking_depth
        weights *= watermarking_depth / np.sum(weights)

        # Apply weights to g-values
        g_values = g_values * np.expand_dims(weights, axis=(0, 1))

        num_unmasked = np.sum(mask, axis=1)  # shape [batch_size]
        return np.sum(g_values * np.expand_dims(mask, 2), axis=(1, 2)) / (
            watermarking_depth * num_unmasked
        )


class SynthIDState:
  """SynthID watermarking state."""

  def __init__(
      self,
      batch_size: int,
      ngram_len: int,
      context_history_size: int,
      device: torch.device,
  ):
    """Initializes the state.

    Args:
      batch_size: Batch size.
      ngram_len: Ngram length.
      context_history_size: Size of the tensor to keep track of seen contexts.
      device: Device to use.
    """
    self.context = torch.zeros(
        (batch_size, ngram_len - 1),
        dtype=torch.int64,
        device=device,
    )
    self.context_history = torch.zeros(
        (batch_size, context_history_size),
        dtype=torch.int64,
        device=device,
    )
    self.num_calls = 0


class SynthIDLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for SynthID algorithm, process logits to add watermark."""

    def __init__(self, config: SynthIDConfig, utils: SynthIDUtils, *args, **kwargs) -> None:
        self.config = config
        self.utils = utils
        self.state = None
        
        # Initialize parameters from config
        self.ngram_len = config.ngram_len
        self.keys = torch.tensor(config.keys, device=config.device)
        self.sampling_table_size = config.sampling_table_size
        self.context_history_size = config.context_history_size
        self.device = config.device
        
        # Initialize sampling table
        self.sampling_table = torch.randint(
            low=0,
            high=2,
            size=(self.sampling_table_size,),
            generator=self.utils.rng,
            device=self.device,
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        # Initialize state if needed
        scores_processed = scores / self.config.temperature
        batch_size, vocab_size = scores.shape

        if self.config.top_k > 0:
            top_k_result = torch.topk(scores_processed, k=self.config.top_k, dim=1)
            scores_top_k = top_k_result.values
            # scores_top_k shape [batch_size, top_k]
            top_k_indices = top_k_result.indices
            # top_k_indices shape [batch_size, top_k]
        else:
            scores_top_k = scores_processed
            top_k_indices = torch.stack([
                torch.arange(vocab_size, device=self.device)
                for _ in range(batch_size)
            ])
        
        
        if self.state is None:
            self.state = {
                "context": torch.zeros((batch_size, self.ngram_len - 1), dtype=torch.int64, device=self.device),
                "context_history": torch.zeros((batch_size, self.context_history_size), dtype=torch.int64, device=self.device),
                "num_calls": 0
            }
        
        # Update context with last input token
        if self.state["num_calls"] > 0:
            self.state["context"] = torch.cat((self.state["context"], input_ids[:, -1:]), dim=1)
            self.state["context"] = self.state["context"][:, 1:]
        
        self.state["num_calls"] += 1
        
        # Generate ngram keys and sample g values
        ngram_keys, hash_context = self._compute_keys(self.state["context"], top_k_indices)
        g_values = self.sample_g_values(ngram_keys)
        
        # Update scores based on g values
        updated_scores = self.utils.update_scores(scores_top_k, g_values)
        
        # Check for repeated context
        hash_context = hash_context[:, None]
        is_repeated = (self.state["context_history"] == hash_context).any(dim=1, keepdim=True)
        
        # Update context history
        self.state["context_history"] = torch.cat((hash_context, self.state["context_history"]), dim=1)[:, :-1]
        
        # Return original scores if context is repeated, otherwise return updated scores
        return torch.where(is_repeated, scores, updated_scores)

    def _compute_keys(self, context: torch.LongTensor, top_k_indices: torch.LongTensor) -> tuple[torch.LongTensor, torch.LongTensor]:
        """Compute ngram keys for given context and possible next tokens."""
        batch_size = context.shape[0]
        
        # Initial hash of context
        hash_result = torch.ones(batch_size, device=self.device, dtype=torch.long)
        hash_context = self.utils.accumulate_hash(hash_result, context)
        
        # Compute hash for each possible continuation
        hash_result = torch.vmap(self.utils.accumulate_hash, in_dims=(None, 1), out_dims=1)(
            hash_context, top_k_indices[:, :, None]
        )
        
        # Add watermarking keys
        keys = self.keys[None, None, :, None]
        hash_result = torch.vmap(self.utils.accumulate_hash, in_dims=(None, 2), out_dims=2)(
            hash_result, keys
        )
        
        return hash_result, hash_context

    def sample_g_values(self, ngram_keys: torch.LongTensor) -> torch.LongTensor:
        """Sample g values from pre-computed sampling table."""
        ngram_keys = ngram_keys % self.sampling_table_size
        sampling_table = self.sampling_table.reshape((1, 1, self.sampling_table_size))
        return torch.take_along_dim(sampling_table, indices=ngram_keys, dim=2)
    
    def compute_g_values(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Computes g values for each ngram from the given sequence of tokens.

        Args:
            input_ids: Input token ids (batch_size, input_len).

        Returns:
            G values (batch_size, input_len - (ngram_len - 1), depth).
        """ 
        ngrams = input_ids.unfold(dimension=1, size=self.ngram_len, step=1)
        ngram_keys = self.compute_ngram_keys(ngrams)
        return self.sample_g_values(ngram_keys)
    
    def compute_ngram_keys(
      self,
        ngrams: torch.LongTensor,
    ) -> torch.LongTensor:
        """Computes random keys for each ngram and depth.

        Args:
            ngrams: Ngrams (batch_size, num_ngrams, ngram_len).

        Returns:
            ngram keys (batch_size, num_ngrams, depth).
        """
        if len(ngrams.shape) != 3:
            raise ValueError(
                "Ngrams should be of shape (batch_size, num_ngrams, ngram_len), but"
                f" is {ngrams.shape}"
        )
        if ngrams.shape[2] != self.ngram_len:
            raise ValueError(
                "Ngrams should be of shape (batch_size, num_ngrams, ngram_len),"
                f" where ngram_len is {self.ngram_len}, but is {ngrams.shape}"
            )
        batch_size, _, _ = ngrams.shape

        hash_result = torch.ones(batch_size, device=self.device, dtype=torch.long)
        # hash_result shape [batch_size,]
        # ngrams shape [batch_size, num_ngrams, ngram_len]
        hash_result = torch.vmap(
            self.utils.accumulate_hash, in_dims=(None, 1), out_dims=1
        )(hash_result, ngrams)
        # hash_result shape [batch_size, num_ngrams]

        keys = self.keys[None, None, :, None]
        # hash_result shape [batch_size, num_ngrams]
        # keys shape [1, 1, depth, 1]
        hash_result = torch.vmap(
            self.utils.accumulate_hash, in_dims=(None, 2), out_dims=2
        )(hash_result, keys)
        # hash_result shape [batch_size, num_ngrams, depth]

        return hash_result

    def compute_eos_token_mask(
        self,
        input_ids: torch.LongTensor,
        eos_token_id: int,
    ) -> torch.LongTensor:
        """Computes repetitions mask.

        1 stands for ngrams that don't contain EOS tokens and vice versa.

        Args:
            input_ids: Input token ids (batch_size, input_len).
            eos_token_id: EOS token ID.

        Returns:
            EOS token mask (batch_size, input_len).
        """
        noneos_masks = []
        all_eos_equated = input_ids == eos_token_id
        for eos_equated in all_eos_equated:
            nonzero_idx = torch.nonzero(eos_equated)
            noneos_mask = torch.ones_like(eos_equated)
            if nonzero_idx.shape[0] != 0:
                noneos_mask[nonzero_idx[0][0]:] = 0
            noneos_masks.append(noneos_mask)
        return torch.stack(noneos_masks, dim=0)

    def compute_context_repetition_mask(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Computes repetition mask.

        0 and 1 stand for repeated and not repeated context n-1 grams respectively.

        Args:
            input_ids: Input token ids (batch_size, input_len).

        Returns:
            Repetitions mask (batch_size, input_len - (ngram_len - 1)).
        """
        batch_size, _ = input_ids.shape
        state = SynthIDState(
            batch_size=batch_size,
            ngram_len=self.ngram_len,
            context_history_size=self.context_history_size,
            device=self.device,
        )
        contexts = input_ids[:, :-1].unfold(
            dimension=1,
            size=self.ngram_len - 1,
            step=1,
        )
        _, num_contexts, _ = contexts.shape

        are_repeated_contexts = []
        for i in range(num_contexts):
            context = contexts[:, i, :]
            hash_result = torch.ones(batch_size, device=self.device, dtype=torch.long)
            context_hash = self.utils.accumulate_hash(hash_result, context)[
                :, None
            ]
            is_repeated_context = (state.context_history == context_hash).any(
                dim=1,
                keepdim=True,
            )
            are_repeated_contexts.append(is_repeated_context)
            state.context_history = torch.concat(
                (context_hash, state.context_history),
                dim=1,
            )[:, :-1]
        are_repeated_contexts = torch.concat(are_repeated_contexts, dim=1)

        return torch.logical_not(are_repeated_contexts)



class SynthID(BaseWatermark):
    """Top-level class for SynthID algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        self.config = SynthIDConfig(algorithm_config, transformers_config)
        self.utils = SynthIDUtils(self.config)
        self.logits_processor = SynthIDLogitsProcessor(self.config, self.utils)
        self.detector = get_detector(self.config.detector_name, self.logits_processor)
    
    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text."""
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]), 
            **self.config.gen_kwargs
        )
        
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the text.
        
        Args:
            text (str): Text to detect watermark in
            return_dict (bool): Whether to return results as dictionary
            
        Returns:
            Union[Dict[str, Union[bool, float]], Tuple[bool, float]]: Detection results
        """
        # Encode text to token ids
        encoded_text = self.config.generation_tokenizer(
            text, 
            return_tensors="pt", 
            add_special_tokens=False
        )["input_ids"].to(self.config.device)
        
        # Compute g-values for the text
        g_values = self.logits_processor.compute_g_values(encoded_text)
        
        # Create eos mask
        eos_mask = self.logits_processor.compute_eos_token_mask(
            input_ids=encoded_text,
            eos_token_id=self.config.generation_tokenizer.eos_token_id
        )[:, self.config.ngram_len - 1:]
        
        # Compute context repetition mask
        context_repetition_mask = self.logits_processor.compute_context_repetition_mask(
            input_ids=encoded_text
        )
        
        # Combine masks
        combined_mask = context_repetition_mask * eos_mask
        
        # Calculate mean score
        g_values_np = g_values.cpu().numpy()
        mask_np = combined_mask.cpu().numpy()
        score = self.detector.detect(g_values_np, mask_np)[0]  # Take first element as we have batch_size=1
    
        # Determine if text is watermarked based on score
        # A positive score indicates watermarking
        is_watermarked = score > self.config.threshold
        
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": float(score)}
        else:
            return (is_watermarked, float(score))
        
    def get_data_for_visualization(self, text: str, *args, **kwargs) -> DataForVisualization:
        """Get data for visualization."""
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        # Placeholder for visualization data generation
        decoded_tokens = []
        highlight_values = []
        
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)
            highlight_values.append(0)  # Placeholder values
        
        return DataForVisualization(decoded_tokens, highlight_values)