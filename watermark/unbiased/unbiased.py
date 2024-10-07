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

# ===========================================================
# unbiased.py
# Description: Implementation of Unbiased Watermark algorithm
# ===========================================================

import torch
import hashlib
import random
from typing import Tuple, Union
import torch.nn.functional as F
from math import sqrt
from functools import partial
from ..base import BaseWatermark
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from transformers import LogitsProcessor, LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization


class UnbiasedConfig:
    """Config class for Unbiased watermark algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the Unbiased configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/Unbiased.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'Unbiased':
            raise AlgorithmNameMismatchError('Unbiased', config_dict['algorithm_name'])
        
        random.seed(config_dict['key'])
        hash_key = random.getrandbits(1024).to_bytes(128, "big")
        self.hash_key = hash_key

        self.gamma = config_dict['gamma']
        self.alpha = 0.5
        self.ignore_history = bool(config_dict['ignore_history'])
        self.z_threshold = config_dict['z_threshold']
        self.prefix_length = config_dict['prefix_length']

        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs

class UnbiasedUtils:
    """Utility class for Unbiased watermark algorithm, contains helper functions."""

    def __init__(self, config: UnbiasedConfig, *args, **kwargs) -> None:
        """
            Initialize the Unbiased utility class.

            Parameters:
                config (UnbiasedConfig): Configuration for the unbiased algorithm.
        """
        self.config = config
        self.rng = torch.Generator(device=self.config.device)
        self.cc_history = set()
        

    def _get_rng_seed(self, context_code: any) -> int:
        """Get the random seed from the given context code and private key."""
        if not self.config.ignore_history:
            self.cc_history.add(context_code)
            
        m = hashlib.sha256()
        m.update(context_code)
        m.update(self.config.hash_key)
        
        full_hash = m.digest()
        seed = int.from_bytes(full_hash, "big") % (2**32 - 1)
        return seed
    
    def _extract_context_code(self, context: torch.LongTensor) -> bytes:
        """Extract context code from the given context."""
        if self.config.prefix_length == 0:
            return context.detach().cpu().numpy().tobytes()
        else:
            return context[-self.config.prefix_length : ].detach().cpu().numpy().tobytes()
    
    def from_random(self, rng: Union[torch.Generator, list[torch.Generator]], vocab_size: int) -> torch.LongTensor:
        """Generate a permutation from the random number generator."""
        if isinstance(rng, list):
            batch_size = len(rng)
            shuffle = torch.stack(
                [
                    torch.randperm(vocab_size, generator=rng[i], device=rng[i].device)
                    for i in range(batch_size)
                ]
            )
        else:
            shuffle = torch.randperm(vocab_size, generator=rng, device=rng.device)
        return shuffle

    def reweight_logits(self, shuffle: torch.LongTensor, p_logits: torch.FloatTensor) -> torch.FloatTensor:
        """Reweight the logits using the shuffle and alpha."""
        unshuffle = torch.argsort(shuffle, dim=-1)
        
        s_p_logits = torch.gather(p_logits, -1, shuffle)
        s_log_cumsum = torch.logcumsumexp(s_p_logits, dim=-1)
        
        # normalize the log_cumsum to force the last element to be 0
        s_log_cumsum = s_log_cumsum - s_log_cumsum[..., -1:]
        s_cumsum = torch.exp(s_log_cumsum)
        s_p = F.softmax(s_p_logits, dim=-1)

        boundary_1 = torch.argmax((s_cumsum > self.config.alpha).to(torch.int), dim=-1, keepdim=True)
        p_boundary_1 = torch.gather(s_p, -1, boundary_1)
        portion_in_right_1 = (torch.gather(s_cumsum, -1, boundary_1) - self.config.alpha) / p_boundary_1
        portion_in_right_1 = torch.clamp(portion_in_right_1, 0, 1)
        s_all_portion_in_right_1 = (s_cumsum > self.config.alpha).type_as(p_logits)
        s_all_portion_in_right_1.scatter_(-1, boundary_1, portion_in_right_1)

        boundary_2 = torch.argmax((s_cumsum > (1-self.config.alpha)).to(torch.int), dim=-1, keepdim=True)
        p_boundary_2 = torch.gather(s_p, -1, boundary_2)
        portion_in_right_2 = (torch.gather(s_cumsum, -1, boundary_2) - (1-self.config.alpha)) / p_boundary_2
        portion_in_right_2 = torch.clamp(portion_in_right_2, 0, 1)
        s_all_portion_in_right_2 = (s_cumsum > (1-self.config.alpha)).type_as(p_logits)
        s_all_portion_in_right_2.scatter_(-1, boundary_2, portion_in_right_2)

        s_all_portion_in_right = s_all_portion_in_right_2/2 + s_all_portion_in_right_1/2
        s_shift_logits = torch.log(s_all_portion_in_right)
        shift_logits = torch.gather(s_shift_logits, -1, unshuffle)

        return p_logits + shift_logits
    
    def get_seed_for_cipher(self, input_ids: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Get the mask and seeds for the cipher."""
        batch_size = input_ids.size(0)
        context_codes = [
            self._extract_context_code(input_ids[i]) for i in range(batch_size)
        ]

        mask, seeds = zip(
            *[
                (context_code in self.cc_history, self._get_rng_seed(context_code))
                for context_code in context_codes 
            ]
        )
        
        return mask, seeds
    
    def _get_green_token_quantile(self, input_ids: torch.LongTensor, vocab_size, current_token):
        """Get the vocab quantile of current token"""
        mask, seeds = self.get_seed_for_cipher(input_ids.unsqueeze(0))
        
        rng = [
            torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds
        ]
        
        mask = torch.tensor(mask, device=input_ids.device)
        shuffle = self.from_random(
            rng, vocab_size
        )
        
        token_quantile = [(torch.where(shuffle[0] == current_token)[0] +1)/vocab_size]
        return token_quantile
    
    def _get_score(self, input_ids: torch.LongTensor, vocab_size):
        """Get the score of the input_ids"""
        scores = torch.zeros(input_ids.shape, device=input_ids.device)
        
        for i in range(input_ids.shape[-1] - 1):
            pre = input_ids[ : i+1]
            cur = input_ids[i+1]
            token_quantile = self._get_green_token_quantile(pre, vocab_size, cur)
            scores[i] = torch.stack(token_quantile).reshape(-1)
        
        return scores
    
    def score_sequence(self, input_ids: torch.LongTensor) -> tuple[float, list[int]]:
        """Score the input_ids and return z_score and green_token_flags."""
        score = self._get_score(input_ids, self.config.vocab_size)
        green_tokens = torch.sum(score >= self.config.gamma, dim=-1, keepdim=False)
        
        green_token_flags = torch.zeros_like(score)
        condition_indices = torch.nonzero(score >= self.config.gamma, as_tuple=False)
        green_token_flags[condition_indices] = 1
        green_token_flags[:self.config.prefix_length] = -1
        
        z_score = (green_tokens - (1-self.config.gamma) * input_ids.size(-1)) / sqrt(input_ids.size(-1))
        return z_score.item(), green_token_flags.tolist()

class UnbiasedLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for DiP algorithm, process logits to add watermark."""

    def __init__(self, config: UnbiasedConfig, utils: UnbiasedUtils, *args, **kwargs) -> None:
        """
            Initialize the Unbiased logits processor.

            Parameters:
                config (UnbiasedConfig): Configuration for the DiP algorithm.
                utils (UnbiasedUtils): Utility class for the DiP algorithm.
        """
        self.config = config
        self.utils = utils
    
    def _apply_watermark(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Apply watermark to the scores."""
        mask, seeds = self.utils.get_seed_for_cipher(input_ids)
        
        rng = [
            torch.Generator(device=scores.device).manual_seed(seed) for seed in seeds
        ]
        mask = torch.tensor(mask, device=scores.device)
        shuffle = self.utils.from_random(
            rng, scores.size(1)
        )

        reweighted_scores = self.utils.reweight_logits(shuffle, scores)
        
        return mask, reweighted_scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores
        
        mask, reweighted_scores = self._apply_watermark(input_ids, scores)
        
        if self.config.ignore_history:
            return reweighted_scores
        else:
            return torch.where(mask[:, None], scores, reweighted_scores)
    

class UnbiasedWatermark(BaseWatermark):
    """Top-level class for Unbiased algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the Unbiased algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = UnbiasedConfig(algorithm_config, transformers_config)
        self.utils = UnbiasedUtils(self.config)
        self.logits_processor = UnbiasedLogitsProcessor(self.config, self.utils)
    
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
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)

        # Compute z-score using a utility method
        z_score, _ = self.utils.score_sequence(encoded_text)
        
        # Determine if the z-score indicates a watermark
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
        
        # Compute the highlight values
        _, highlight_values = self.utils.score_sequence(encoded_text)
        
        # decode single tokens
        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)
        
        return DataForVisualization(decoded_tokens, highlight_values)

