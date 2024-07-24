# ================================================
# dip.py
# Description: Implementation of DiPmark algorithm
# ================================================

import torch
import time
import hashlib
import random
from typing import Tuple, Union
import torch.nn.functional as F
from math import sqrt, log, exp, ceil
from functools import partial
from ..base import BaseWatermark
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from transformers import LogitsProcessor, LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization


class DIPConfig:
    """Config class for DiP algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the DiP configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/DIP.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'DIP':
            raise AlgorithmNameMismatchError('DIP', config_dict['algorithm_name'])
        
        random.seed(config_dict['key'])
        hash_key = random.getrandbits(1024).to_bytes(128, "big")
        self.hash_key = hash_key

        self.gamma = config_dict['gamma']
        self.alpha = config_dict['alpha']
        self.ignore_history = bool(config_dict['ignore_history'])
        self.z_threshold = config_dict['z_threshold']
        self.p_threshold = config_dict['p_threshold']
        self.prefix_length = config_dict['prefix_length']

        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs

class DIPUtils:
    """Utility class for DiP algorithm, contains helper functions."""

    def __init__(self, config: DIPConfig, *args, **kwargs) -> None:
        """
            Initialize the DIP utility class.

            Parameters:
                config (DIPConfig): Configuration for the DiP algorithm.
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
    
    # def _f_alpha(self, shuffled_logits: torch.LongTensor, i: int, _alpha: float) -> float:
    #     """Calculate the f_alpha(i|theta) function."""
    #     return max(shuffled_logits[:i+1].sum().item() - _alpha, 0) / (1 - _alpha)
    
    # def _p_alpha_w(self, shuffled_logits: torch.LongTensor, i: int, _alpha: float) -> float:
    #     """Calculate the p_alpha(w_i|theta) function."""
    #     return self._f_alpha(shuffled_logits, i, _alpha) - self._f_alpha(shuffled_logits, i-1, _alpha)
    
    # def reweight_logits(self, shuffle: torch.Tensor, p_logits: torch.LongTensor) -> torch.LongTensor:
    #     """Reweight the logits using the shuffle and alpha."""
    #     unshuffle = torch.argsort(shuffle, dim=-1)
        
    #     shuffled_logits = torch.gather(p_logits, -1, shuffle)
    #     reweighted_logits = torch.zeros_like(p_logits)
    #     for i in range(1, len(p_logits) + 1):
    #         p_alpha_w = self._p_alpha_w(shuffled_logits, i, self.config.alpha)
    #         p_1_minus_alpha_w = self._p_alpha_w(shuffled_logits, i, 1 - self.config.alpha)
    #         reweighted_logits[i-1] = (1 - self.config.alpha) * p_alpha_w + self.config.alpha * p_1_minus_alpha_w
            
    #     # Restore the original order using unshuffle
    #     reweighted_logits = torch.gather(reweighted_logits, -1, unshuffle)
        
    #     return reweighted_logits


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
        
        # reweight_logits = p_logits + shift_logits
        # shuffled_reweight_logits = torch.gather(reweight_logits, -1, shuffle)
        # print(shuffled_reweight_logits[0][:boundary_1])

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
        # print(seeds[0])
        # time.sleep(0.1)
        
        return mask, seeds
    
    # def score_sequence(self, input_ids: torch.LongTensor) -> tuple[float, list[int]]:
    #     """Score the input_ids and return green_token_ratio and green_token_flags"""
        
    #     # print(self.config.generation_tokenizer.batch_decode(input_ids.unsqueeze(0), skip_special_tokens=True)[0])
    #     num_tokens_scored = len(input_ids) - self.config.prefix_length
    #     if num_tokens_scored < 1:
    #         raise ValueError(
    #             (
    #                 f"Must have at least {1} token to score after "
    #                 f"the first min_prefix_len={self.config.prefix_length} tokens required by the seeding scheme."
    #             )
    #         )
            
    #     green_token_count = 0
    #     green_token_flags = [-1 for _ in range(self.config.prefix_length)]
        
    #     for idx in range(self.config.prefix_length, len(input_ids)):
    #         curr_token = input_ids[idx]
            
    #         decoded_text = self.config.generation_tokenizer.batch_decode(input_ids[:idx].unsqueeze(0), skip_special_tokens=True)[0]

    #         # Generate a texture key s based on input_ids[i - prefix : i]
    #         mask, seeds = self.get_seed_for_cipher(input_ids[idx - self.config.prefix_length : idx].unsqueeze(0))
            
    #         # Generate the permutation of token set cipher
    #         rng = [
    #             torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds
    #         ]
    #         mask = torch.tensor(mask, device=input_ids.device)
    #         shuffle = self.from_random(
    #             rng, input_ids.size(0)
    #         )
            
    #         # print("detect shuffle: ", shuffle[:50])
    #         # Calculate the list of green tokens
    #         greenlist_ids = shuffle[int(self.config.gamma * len(shuffle)) : ]
    #         if curr_token in greenlist_ids:
    #             green_token_count += 1
    #             green_token_flags.append(1)
    #         else:
    #             green_token_flags.append(0)
            
    #     print(f"total length: {len(input_ids)}")
    #     print(f"green token count: {green_token_count}")
        
    #     # Calculate the score
    #     green_token_ratio = green_token_count / len(input_ids) - (1 - self.config.gamma)
    #     return green_token_ratio, green_token_flags
    
    def get_green_token_quantile(self, input_ids: torch.LongTensor, vocab_size, current_token):
        """Get the vocab quantile of current token"""
        mask, seeds = self.get_seed_for_cipher(input_ids.unsqueeze(0))
        # print(f"seed: f{seeds[0]}")
        
        rng = [
            torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds
        ]
        
        mask = torch.tensor(mask, device=input_ids.device)
        shuffle = self.from_random(
            rng, vocab_size
        )
        
        # （对各个batch）当前token的id 的位置下标组成的tensor
        # token_quantile是该token所处在词表vocab中的位置的百分比
        # 用于后续detect中的gamma阈值判断
        # print(shuffle[:3])
        # print(f"position: {torch.where(shuffle[0] == current_token)[0]}")
        
        token_quantile = [(torch.where(shuffle[0] == current_token)[0] +1)/vocab_size]
        
        return token_quantile
    
    def get_dip_score(self, input_ids: torch.LongTensor, vocab_size):
        """Get the DiP score of the input_ids"""
        scores = torch.zeros(input_ids.shape, device=input_ids.device)
        
        for i in range(input_ids.shape[-1] - 1):
            pre = input_ids[ : i+1]
            cur = input_ids[i+1]
            # print(f"cur token:{cur}")
            token_quantile = self.get_green_token_quantile(pre, vocab_size, cur)
            # print(f"token_quantile: {token_quantile}")
            scores[i] = torch.stack(token_quantile).reshape(-1)
        
        return scores
    
    def score_sequence(self, input_ids: torch.LongTensor) -> tuple[float, list[int]]:
        
        score = self.get_dip_score(input_ids, self.config.vocab_size)
        green_tokens = torch.sum(score >= self.config.gamma, dim=-1, keepdim=False)
        print(green_tokens.item())
        
        green_token_flags = torch.zeros_like(score)
        condition_indices = torch.nonzero(score >= self.config.gamma, as_tuple=False)
        green_token_flags[condition_indices] = 1
        green_token_flags[:self.config.prefix_length] = -1
        
        mid2 = (green_tokens - (1-self.config.gamma) * input_ids.size(-1)) / sqrt(input_ids.size(-1))
        
        return mid2.item(), green_token_flags.tolist()

class DIPLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for DiP algorithm, process logits to add watermark."""

    def __init__(self, config: DIPConfig, utils: DIPUtils, *args, **kwargs) -> None:
        """
            Initialize the DIP logits processor.

            Parameters:
                config (DIPConfig): Configuration for the DiP algorithm.
                utils (DIPUtils): Utility class for the DiP algorithm.
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
    

class DIP(BaseWatermark):
    """Top-level class for DIP algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the DIP algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = DIPConfig(algorithm_config, transformers_config)
        self.utils = DIPUtils(self.config)
        self.logits_processor = DIPLogitsProcessor(self.config, self.utils)
    
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

        # print("text:")
        # print(text)
        # Encode the text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)

        # decoded_text = self.config.generation_tokenizer.batch_decode(encoded_text.unsqueeze(0), skip_special_tokens=False)[0]
        
        # print('decode test:')
        # print(decoded_text)

        # Compute green token ratio using a utility method
        z_score, _ = self.utils.score_sequence(encoded_text)
        # green_token_ratio, _ = self.utils.score_sequence(encoded_text)
        
        # z_score = (green_token_ratio * sqrt(len(encoded_text)))
        
        # prob = exp(-2 * z_score * z_score)
        
        # print(prob)

        # Calculate the K-L divergence
        # gm_ratio = green_token_ratio + (1 - self.config.gamma)
        # kl_div = gm_ratio * log(gm_ratio / (1 - self.config.gamma)) + (1 - gm_ratio) * log((1 - gm_ratio) / self.config.gamma)
        # prob = exp(-kl_div * len(encoded_text))
        
        # Determine if the green token ratio indicates a watermark
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

