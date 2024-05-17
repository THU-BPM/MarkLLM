# ==============================================
# sweet.py
# Description: Implementation of SWEET algorithm
# ==============================================
import torch
from math import sqrt

from functools import partial
from ..base import BaseWatermark
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from transformers import LogitsProcessor, LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization


class SWEETConfig:
    """Config class for SWEET algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the SWEET configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/SWEET.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'SWEET':
            raise AlgorithmNameMismatchError('SWEET', config_dict['algorithm_name'])
        
        self.gamma = config_dict['gamma']
        self.delta = config_dict['delta']
        self.hash_key = config_dict['hash_key']
        self.z_threshold = config_dict['z_threshold']
        self.prefix_length = config_dict['prefix_length']
        self.entropy_threshold = config_dict['entropy_threshold']
        
        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs


class SWEETUtils:
    """Utility class for SWEET algorithm, contains helper functions."""

    def __init__(self, config: SWEETConfig, *args, **kwargs):
        self.config = config
        self.rng = torch.Generator(device=self.config.device)

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed the random number generator with the last prefix_length tokens of the input_ids."""
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        prev_token = time_result % self.config.vocab_size
        self.rng.manual_seed(self.config.hash_key * prev_token)
        return

    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids."""
        self._seed_rng(input_ids)
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)
        greenlist_ids = vocab_permutation[:greenlist_size] 
        return greenlist_ids
    
    def calculate_entropy(self, model, tokenized_text: torch.Tensor):
        """Calculate entropy for each token in the tokenized_text."""
        with torch.no_grad():
            output = model(torch.unsqueeze(tokenized_text, 0), return_dict=True)
            probs = torch.softmax(output.logits, dim=-1)
            entropy = -torch.where(probs > 0, probs * probs.log(), probs.new([0.0])).sum(dim=-1)
            entropy = entropy[0].cpu().tolist()
            entropy.insert(0, -10000.0)
            return entropy[:-1]

    def _compute_z_score(self, observed_count: int, T: int) -> float: 
        """Compute z-score for the observed count of green tokens."""
        expected_count = self.config.gamma
        numer = observed_count - expected_count * T 
        denom = sqrt(T * expected_count * (1 - expected_count))  
        z = numer / denom
        return z

    def score_sequence(self, input_ids: torch.Tensor, entropy_list: list[float]) -> tuple[float, list[int], list[int]]:
        """Score the input_ids based on the greenlist and entropy."""
        num_tokens_scored = (len(input_ids) - self.config.prefix_length - 
                             len([e for e in entropy_list[self.config.prefix_length:] if e <= self.config.entropy_threshold]))
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                )
            )

        green_token_flags = [-1 for _ in range(self.config.prefix_length)]
        weights = [-1 for _ in range(self.config.prefix_length)]

        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx]
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
            if curr_token in greenlist_ids:
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
            if entropy_list[idx] > self.config.entropy_threshold:
                weights.append(1)
            else:
                weights.append(0)

        # calculate number of green tokens where weight is 1
        green_token_count = sum([1 for i in range(len(green_token_flags)) if green_token_flags[i] == 1 and weights[i] == 1])
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        
        return z_score, green_token_flags, weights


class SWEETLogitsProcessor(LogitsProcessor):
    """Logits processor for SWEET algorithm, contains the logic to bias the logits."""

    def __init__(self, config: SWEETConfig, utils: SWEETUtils, *args, **kwargs) -> None:
        """
            Initialize the SWEET logits processor.

            Parameters:
                config (SWEETConfig): Configuration for the SWEET algorithm.
                utils (SWEETUtils): Utility class for the SWEET algorithm.
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

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        """Bias the scores for the greenlist tokens."""
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        # get entropy
        raw_probs = torch.softmax(scores, dim=-1)  
        ent = -torch.where(raw_probs > 0, raw_probs * raw_probs.log(), raw_probs.new([0.0])).sum(dim=-1)
        entropy_mask = (ent > self.config.entropy_threshold).view(-1, 1)
        
        green_tokens_mask = green_tokens_mask * entropy_mask

        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.config.delta)
        return scores


class SWEET(BaseWatermark):
    """Top-level class for SWEET algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the SWEET algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = SWEETConfig(algorithm_config, transformers_config)
        self.utils = SWEETUtils(self.config)
        self.logits_processor = SWEETLogitsProcessor(self.config, self.utils)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs):
        """Generate watermarked text."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]), 
            **self.config.gen_kwargs
        )
        
        # encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        # generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # decode
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the text."""

        # encode text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)

        # calculate entropy
        entropy_list = self.utils.calculate_entropy(self.config.generation_model, encoded_text)
        
        # compute z_score
        z_score, _, _ = self.utils.score_sequence(encoded_text, entropy_list)

        # Determine if the z_score indicates a watermark
        is_watermarked = z_score > self.config.z_threshold

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)

    def get_data_for_visualization(self, text: str, *args, **kwargs):
        """Get data for visualization."""
        
        # encode text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.generation_model.device)

        # calculate entropy
        entropy_list = self.utils.calculate_entropy(self.config.generation_model, encoded_text)
        
        # compute z-score, highlight_values, and weights
        z_score, highlight_values, weights = self.utils.score_sequence(encoded_text, entropy_list)
        
        # decode single tokens
        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)
        
        return DataForVisualization(decoded_tokens, highlight_values, weights)