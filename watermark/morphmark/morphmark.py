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

# ==============================================
# morphmark.py
# Description: Implementation of MorphMark algorithm
# ==============================================

import os
import time
import numpy as np
import torch
from math import sqrt
from functools import partial
from ..base import BaseConfig, BaseWatermark
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from transformers import LogitsProcessor, LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization


class MorphMarkConfig(BaseConfig):
    """Config class for MorphMark algorithm, load config file and initialize parameters."""
    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""

        self.gamma = self.config_dict['gamma']  # Proportion of the green table

        self.type = self.config_dict['type']  # Type of MorphMark, three options: linear, exp, log
        self.k_linear = self.config_dict['k_linear']
        self.k_exp = self.config_dict['k_exp']
        self.k_log = self.config_dict['k_log']
        self.p_0 = self.config_dict['p_0']

        self.ewd = self.config_dict['ewd']  # Whether to use ewd as the watermark detection method
        self.delta = self.config_dict['delta_ewd']  # Parameter for ewd

        self.hash_key = self.config_dict['hash_key']
        self.z_threshold = self.config_dict['z_threshold']
        self.prefix_length = self.config_dict['prefix_length']
        self.f_scheme = self.config_dict['f_scheme']
        self.window_scheme = self.config_dict['window_scheme']

    @property
    def algorithm_name(self) -> str:
        """Return algorithm name."""
        return 'MorphMark'

class MorphMarkUtils:
    """Utility class for MorphMark algorithm, contains helper functions."""

    def __init__(self, config: MorphMarkConfig, *args, **kwargs) -> None:
        """
            Initialize the MorphMark utility class.

            Parameters:
                config (MorphMarkConfig): Configuration for the MorphMark algorithm.
        """
        self.config = config
        self.rng = torch.Generator(device=self.config.device)
        self.rng.manual_seed(self.config.hash_key)
        self.prf = torch.randperm(self.config.vocab_size, device=self.config.device, generator=self.rng)
        self.f_scheme_map = {"time": self._f_time, "additive": self._f_additive, "skip": self._f_skip, "min": self._f_min}
        self.window_scheme_map = {"left": self._get_greenlist_ids_left, "self": self._get_greenlist_ids_self}

        alpha = torch.exp(torch.tensor(self.config.delta)).item()
        self.z_value = ((1 - self.config.gamma) * (alpha - 1))/(1 - self.config.gamma + (alpha * self.config.gamma))
            
    def _f(self, input_ids: torch.LongTensor) -> int:
        """Get the previous token."""
        return int(self.f_scheme_map[self.config.f_scheme](input_ids))

    def _f_time(self, input_ids: torch.LongTensor):
        """Get the previous token time."""
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        return self.prf[time_result % self.config.vocab_size]

    def _f_additive(self, input_ids: torch.LongTensor):
        """Get the previous token additive."""
        additive_result = 0
        for i in range(0, self.config.prefix_length):
            additive_result += input_ids[-1 - i].item()
        return self.prf[additive_result % self.config.vocab_size]

    def _f_skip(self, input_ids: torch.LongTensor):
        """Get the previous token skip."""
        return self.prf[input_ids[- self.config.prefix_length].item()]

    def _f_min(self, input_ids: torch.LongTensor):
        """Get the previous token min."""
        return min(self.prf[input_ids[-1 - i].item()] for i in range(0, self.config.prefix_length))

    def get_greenlist_ids(self, input_ids: torch.LongTensor, is_detect=False) -> list[int]:
        """Get greenlist ids for the input_ids."""
        return self.window_scheme_map[self.config.window_scheme](input_ids, is_detect)

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed the random number generator with the last token in the input_ids."""
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        prev_token = time_result % self.config.vocab_size
        self.rng.manual_seed(self.config.hash_key * prev_token)
        return
        
    def _get_greenlist_ids_left(self, input_ids: torch.LongTensor, is_detect=False) -> list[int]:
        """Get greenlist ids for the input_ids via leftHash scheme."""
        if self.config.ewd:
            self._seed_rng(input_ids)
        else:
            self.rng.manual_seed((self.config.hash_key * self._f(input_ids)) % self.config.vocab_size)
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)
        greenlist_ids = vocab_permutation[:greenlist_size]

        return greenlist_ids

    def _get_greenlist_ids_self(self, input_ids: torch.LongTensor) -> list[int]:
        """Not Implement."""
        raise "Not Implemented 'def _get_greenlist_ids_self'"

    def calculate_entropy(self, model, tokenized_text) -> list[float]:
        """Calculate the entropy of the tokenized text using the model."""
        with torch.no_grad():
            output = model(torch.unsqueeze(tokenized_text, 0), return_dict=True)
            probs = torch.softmax(output.logits, dim=-1)
            denoms = 1+(self.z_value * probs)
            renormed_probs = probs / denoms
            sum_renormed_probs = renormed_probs.sum(dim=-1)
            entropy=sum_renormed_probs[0].cpu().tolist()
            entropy.insert(0, -10000.0)
            return entropy[:-1]
    
    def _get_weight_from_entropy(self, entropy_list: list[float]) -> list:
        """Compute the weights from the entropy list."""

        # Convert the entropy list to a tensor
        entropy_tensor = torch.tensor(entropy_list)

        # Compute the minimum entropy from the elements beyond the prefix length
        min_entropy = torch.min(entropy_tensor[self.config.prefix_length:])

        # Subtract this minimum entropy from all entropy values
        adjusted_entropies = entropy_tensor - min_entropy

        # Create a list where the prefix part is filled with -1, and the rest with adjusted entropies
        weights = [-1] * self.config.prefix_length + adjusted_entropies[self.config.prefix_length:].tolist()

        return weights

    def _compute_z_score(self, observed_count: int , T: int) -> float:
        """Compute z-score for the given observed count and total tokens."""
        expected_count = self.config.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _compute_z_score_ewd(self, observed_count: float, weight: list) -> float:
        """Compute the z-score for the given observed count and weight."""

        weight_tensor = torch.tensor(weight, dtype=torch.float)
        expected_count = self.config.gamma
        numer = observed_count - expected_count * torch.sum(weight_tensor)
        denom = torch.sqrt(torch.sum(torch.square(weight_tensor)) * expected_count * (1 - expected_count))
        z = numer / denom
        return z.item()

    def score_sequence(self, input_ids: torch.Tensor, entropy_list=None):
        """Score the input_ids and return z_score and green
        _token_flags."""
        num_tokens_scored = len(input_ids) - self.config.prefix_length
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                    f"the first min_prefix_len={self.config.prefix_length} tokens required by the seeding scheme."
                )
            )

        green_token_count = 0
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]

        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx]
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx], is_detect=True)
            if curr_token in greenlist_ids:
                green_token_count += 1
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)

        # compute z_score
        if self.config.ewd:
            # calculate weights
            weights = self._get_weight_from_entropy(entropy_list)
            # sum up weights where green_token_flags = 1 to get green_token_count
            green_token_count = sum(weights[i] for i in range(len(green_token_flags)) if green_token_flags[i] == 1)
            z_score = self._compute_z_score_ewd(green_token_count, weights[self.config.prefix_length:])
            return z_score, green_token_flags, weights
        else:
            z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            return z_score, green_token_flags


class MorphMarkLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for MorphMark algorithm, process logits to add watermark."""

    def __init__(self, config: MorphMarkConfig, utils: MorphMarkUtils, *args, **kwargs) -> None:
        """
            Initialize the MorphMark logits processor.

            Parameters:
                config (MorphMarkConfig): Configuration for the MorphMark algorithm.
                utils (MorphMarkUtils): Utility class for the MorphMark algorithm.
        """
        self.config = config
        self.utils = utils

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids: torch.LongTensor) -> torch.BoolTensor:
        """Calculate greenlist mask for the given scores and greenlist token ids."""
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor) -> torch.Tensor:
        # scores = scores.to(torch.float32)  # due to float16 can not equal to 10^50
        probs = torch.softmax(scores / 1.0, dim=-1)

        P_G = probs[greenlist_mask].sum().item()

        if P_G < self.config.p_0:
            r = 0.0
        else:
            if self.config.type == "linear":
                r = self.config.k_linear * P_G
            elif self.config.type == "exp":
                r = np.exp(self.config.k_exp * P_G) - 1
            elif self.config.type == "log":
                r = np.log(self.config.k_log * P_G + 1)
            else:
                raise ValueError(f"{self.config.type} is not defined.")

        beta = r * (1 - P_G)
        weights = probs[greenlist_mask]
        normalized_weights = weights / weights.sum()
        probs[greenlist_mask] = probs[greenlist_mask] + normalized_weights * beta
        weights = probs[~greenlist_mask]
        normalized_weights = weights / weights.sum()
        probs[~greenlist_mask] = probs[~greenlist_mask] - normalized_weights * beta
        probs = torch.nan_to_num(probs, nan=0)
        probs = torch.clamp(probs, min=0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        scores = torch.log(probs) * 1.0

        return scores


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx], is_detect=False)
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask)
        return scores
    

class MorphMark(BaseWatermark):
    """Top-level class for MorphMark algorithm."""

    def __init__(self, algorithm_config: str | MorphMarkConfig, transformers_config: TransformersConfig | None = None, *args, ** kwargs) -> None:
        """
            Initialize the MorphMark algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """

        if isinstance(algorithm_config, str):
            self.config = MorphMarkConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, MorphMarkConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be either a path string or a MorphMarkConfig instance")

        self.utils = MorphMarkUtils(self.config)
        self.logits_processor = MorphMarkLogitsProcessor(self.config, self.utils)
    
    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]), 
            **self.config.gen_kwargs
        )
        
        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        # Generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # Decode
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the text."""

        # Encode the text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)

        # calculate entropy
        if self.config.ewd:
            entropy_list = self.utils.calculate_entropy(self.config.generation_model, encoded_text)

        # Compute z_score using a utility method
        if self.config.ewd:
            z_score, _, _ = self.utils.score_sequence(encoded_text, entropy_list)
        else:
            z_score, _ = self.utils.score_sequence(encoded_text)

        # Determine if the z_score indicates a watermark
        is_watermarked = z_score > self.config.z_threshold

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)
        
    def get_data_for_visualization(self, text: str, *args, **kwargs) -> tuple[list[str], list[int]]:
        """Get data for visualization."""
        
        # Encode text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        # Compute z-score and highlight values
        z_score, highlight_values = self.utils.score_sequence(encoded_text)
        
        # decode single tokens

        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)
        
        return DataForVisualization(decoded_tokens, highlight_values)