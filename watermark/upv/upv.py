# ============================================
# upv.py
# Description: Implementation of UPV algorithm
# ============================================

import torch
from math import sqrt
from functools import partial
from ..base import BaseWatermark
from utils.utils import load_config_file
from .network_model import UPVGenerator, UPVDetector
from utils.transformers_config import TransformersConfig
from transformers import LogitsProcessor, LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization
from exceptions.exceptions import AlgorithmNameMismatchError, InvalidDetectModeError


class UPVConfig:
    """Config class for UPV algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig) -> None:
        """
            Initialize the UPV configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/UPV.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'UPV':
            raise AlgorithmNameMismatchError('UPV', config_dict['algorithm_name'])
        
        self.gamma = config_dict['gamma']
        self.delta = config_dict['delta']
        self.z_threshold = config_dict['z_threshold']
        self.prefix_length = config_dict['prefix_length']
        self.bit_number = config_dict['bit_number']
        self.sigma = config_dict['sigma']
        self.default_top_k = config_dict['default_top_k']
        self.generator_model_name = config_dict['generator_model_name']
        self.detector_model_name = config_dict['detector_model_name']
        self.detect_mode = config_dict['detect_mode']

        # Validate detect mode
        if self.detect_mode not in ['key', 'network']:
            raise InvalidDetectModeError(self.detect_mode)

        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs


class UPVUtils:
    """Utility class for UPV algorithm, contains helper functions."""

    def __init__(self, config: UPVConfig, *args, **kwargs) -> None:
        """
            Initialize the UPV utility class.

            Parameters:
                config (UPVConfig): Configuration for the UPV algorithm.
        """
        self.config = config
        self.generator_model = self._get_generator_model(self.config.bit_number, self.config.prefix_length + 1)
        self.detector_model = self._get_detector_model(self.config.bit_number)
        self.cache = {}
        self.top_k = self.config.gen_kwargs.get('top_k', self.config.default_top_k)
        self.num_beams = self.config.gen_kwargs.get('num_beams', None)  

    def _get_generator_model(self, input_dim: int, window_size: int) -> UPVGenerator:
        """Load the generator model from the specified file."""
        model = UPVGenerator(input_dim, window_size)
        model.load_state_dict(torch.load(self.config.generator_model_name))
        return model

    def _get_detector_model(self, bit_number: int) -> UPVDetector:
        """Load the detector model from the specified file."""
        model = UPVDetector(bit_number)
        model.load_state_dict(torch.load(self.config.detector_model_name))
        return model

    def _get_predictions_from_generator(self, input_x: torch.Tensor) -> bool:
        """Get predictions from the generator model."""
        output = self.generator_model(input_x)  
        output = (output > 0.5).bool().item()
        return output
    
    def int_to_bin_list(self, n: int, length=8) -> list[int]:
        """Convert an integer to a binary list of specified length."""
        bin_str = format(n, 'b').zfill(length)
        return [int(b) for b in bin_str]
    
    def _select_candidates(self, scores: torch.Tensor) -> torch.Tensor:
        """Select candidate tokens based on the scores."""
        if self.num_beams is not None:
            threshold_score = torch.topk(scores, self.num_beams, largest=True, sorted=False)[0][-1]
            return (scores >= (threshold_score - self.config.delta)).nonzero(as_tuple=True)[0]
        else:
            return torch.topk(scores, self.top_k, largest=True, sorted=False).indices
    
    def get_greenlist_ids(self, input_ids: torch.Tensor, scores: torch.Tensor) -> list[int]:
        """Get greenlist ids for the input_ids."""
        greenlist_ids = []
        candidate_tokens = self._select_candidates(scores)
        
        # Ensure input_ids is a list for concatenation
        input_ids_list = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids

        for v in candidate_tokens:
            # Now safely concatenate lists
            pair = input_ids_list[-self.config.prefix_length:] + [v.item()] if self.config.prefix_length > 0 else [v.item()]
            merged_tuple = tuple(pair)
            bin_list = [self.int_to_bin_list(num, self.config.bit_number) for num in pair]

            if merged_tuple in self.cache:
                result = self.cache[merged_tuple]
            else:
                result = self._get_predictions_from_generator(torch.FloatTensor(bin_list).unsqueeze(0))
                self.cache[merged_tuple] = result
            if result:
                greenlist_ids.append(int(v))

        return greenlist_ids
    
    def _judge_green(self, input_ids: torch.Tensor, current_number: int) -> bool:
        """Judge if the current token is green based on previous tokens."""

        # Get the last 'prefix_length' items from input_ids
        last_nums = input_ids[-self.config.prefix_length:] if self.config.prefix_length > 0 else []
        # Append the current number to the list
        pair = list(last_nums) + [current_number]
        merged_tuple = tuple(pair)
        bin_list = [self.int_to_bin_list(num, self.config.bit_number) for num in pair]

        # load & update cache
        if merged_tuple in self.cache:
            result = self.cache[merged_tuple]
        else:
            result = self._get_predictions_from_generator(torch.FloatTensor(bin_list).unsqueeze(0))
            self.cache[merged_tuple] = result

        return result
    
    def green_token_mask_and_stats(self, input_ids: torch.Tensor) -> tuple[list[bool], int, float]:
        """Get green token mask and statistics for the input_ids."""

        # Initialize a list with None for the prefix tokens which are not scored
        mask_list = [None] * self.config.prefix_length

        # Count of green tokens, initialized to zero
        green_token_count = 0

        # Iterate over each token in the input_ids starting from prefix_length
        for idx in range(self.config.prefix_length, len(input_ids)):
            # Get the current token
            curr_token = input_ids[idx]

            # Judge if the current token is green based on previous tokens
            if self._judge_green(input_ids[:idx], curr_token):
                mask_list.append(True)  # Mark this token as green
                green_token_count += 1  # Increment the green token counter
            else:
                mask_list.append(False)  # Mark this token as not green

        # Compute the number of tokens that were evaluated for green status
        num_tokens_scored = len(input_ids) - self.config.prefix_length

        # Calculate the z-score for the number of green tokens
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)

        # Return the mask list, count of green tokens, and the z-score
        return mask_list, green_token_count, z_score

    def _compute_z_score(self, observed_count: int, T: int) -> float:
        """Compute z-score for the given observed count and total tokens."""
        expected_count = self.config.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count) + self.config.sigma * self.config.sigma * T)
        z = numer / denom
        return z


class UPVLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for UPV algorithm, process logits to add watermark."""

    def __init__(self, config: UPVConfig, utils: UPVUtils, *args, **kwargs):
        """
            Initialize the UPV logits processor.

            Parameters:
                config (UPVConfig): Configuration for the UPV algorithm.
                utils (UPVUtils): Utility class for the UPV algorithm.
        """
        self.config = config
        self.utils = utils

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        """Bias the logits for the greenlist tokens."""
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process the logits to add watermark."""
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores
        
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx], scores=scores[b_idx])
            green_tokens_mask[b_idx][greenlist_ids] = 1 
        green_tokens_mask = green_tokens_mask.bool()

        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.config.delta)

        return scores


class UPV(BaseWatermark):
    """Top-level class for UPV algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the UPV algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = UPVConfig(algorithm_config, transformers_config)
        self.utils = UPVUtils(self.config)
        self.logits_processor = UPVLogitsProcessor(self.config, self.utils)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text based on the prompt."""

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

    def _detect_watermark_network_mode(self, encoded_text: torch.Tensor) -> tuple[bool, float]:
        """ Detect watermark using the network mode. """
        # Convert input IDs to binary sequence
        inputs_bin = [self.utils.int_to_bin_list(token_id, self.config.bit_number) for token_id in encoded_text]
        inputs_bin = torch.tensor(inputs_bin)

        # Run the model on the input binary sequence
        outputs = self.utils.detector_model(inputs_bin.unsqueeze(dim=0).float())
        outputs = outputs.reshape([-1])
        predicted = (outputs.data > 0.5).int()

        # Determine watermark presence based on predictions
        is_watermarked = (predicted == 1).sum().item() > 0

        # z_score is not applicable in network mode
        return is_watermarked, None  
    
    def detect_watermark(self, text: str, return_dict: bool = True) -> tuple[bool, float]:
        """Detect watermark in the given text."""
        
        # Encode the text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

        # Check the mode and perform detection accordingly
        if self.config.detect_mode == 'key':
            _, _, z_score = self.utils.green_token_mask_and_stats(encoded_text)
            # Determine if the z_score indicates a watermark
            is_watermarked = z_score > self.config.z_threshold
        else:
            is_watermarked, z_score = self._detect_watermark_network_mode(encoded_text)

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)

    def get_data_for_visualization(self, text: str) -> DataForVisualization:
        """Get data for visualization."""

        # Encode the text using the specified tokenizer configuration
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        
        # Get the mask indicating green tokens
        mask, _, _ = self.utils.green_token_mask_and_stats(encoded_text)
        
        # Create a list of flags from the mask where -1 = None, 1 = True (green), 0 = False (not green)
        highlight_values = [-1 if m is None else 1 if m else 0 for m in mask]
        
        # Decode each token ID to its corresponding token string
        decoded_tokens = [self.config.generation_tokenizer.decode(token_id.item()) for token_id in encoded_text]
        
        # Return the data for visualization
        return DataForVisualization(decoded_tokens, highlight_values)