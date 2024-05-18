# ============================================
# exp.py
# Description: Implementation of EXP algorithm
# ============================================

import torch
from math import log
from ..base import BaseWatermark
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from visualize.data_for_visualization import DataForVisualization


class EXPConfig:
    """Config class for EXP algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the EXP configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/EXP.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'EXP':
            raise AlgorithmNameMismatchError('KGW', config_dict['algorithm_name'])

        self.prefix_length = config_dict['prefix_length']
        self.hash_key = config_dict['hash_key']
        self.threshold = config_dict['threshold']
        self.sequence_length = config_dict['sequence_length']

        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs


class EXPUtils:
    """Utility class for EXP algorithm, contains helper functions."""

    def __init__(self, config: EXPConfig, *args, **kwargs) -> None:
        """
            Initialize the EXP utility class.

            Parameters:
                config (EXPConfig): Configuration for the EXP algorithm.
        """
        self.config = config
        self.rng = torch.Generator()

    def seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed the random number generator with the last `prefix_length` tokens of the input."""
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        prev_token = time_result % self.config.vocab_size
        self.rng.manual_seed(self.config.hash_key * prev_token)
        return
    
    def exp_sampling(self, probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Sample a token from the vocabulary using the exponential sampling method."""
        return torch.argmax(u ** (1 / probs), axis=1).unsqueeze(-1)
    
    def _value_transformation(self, value):
        """Transform the value to a range between 0 and 1."""
        return value/(value + 1)
    

class EXP(BaseWatermark):
    """Top-level class for the EXP algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the EXP algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = EXPConfig(algorithm_config, transformers_config)
        self.utils = EXPUtils(self.config)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text using the EXP algorithm."""

        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        
        # Initialize
        inputs = encoded_prompt
        attn = torch.ones_like(encoded_prompt)
        past = None

        # Generate tokens
        for i in range(self.config.sequence_length):
            with torch.no_grad():
                if past:
                    output = self.config.generation_model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
                else:
                    output = self.config.generation_model(inputs)
            
            # Get probabilities
            probs = torch.nn.functional.softmax(output.logits[:,-1, :self.config.vocab_size], dim=-1).cpu()
            
            # Generate r1, r2,..., rk
            self.utils.seed_rng(inputs[0])
            random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.rng)
            
            # Sample token to add watermark
            token = self.utils.exp_sampling(probs, random_numbers).to(self.config.device)

            # Update inputs
            inputs = torch.cat([inputs, token], dim=-1)

            # Update past
            past = output.past_key_values

            # Update attention mask
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
        
        watermarked_tokens = inputs[0].detach().cpu()
        watermarked_text = self.config.generation_tokenizer.decode(watermarked_tokens, skip_special_tokens=True)

        return watermarked_text    

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> dict:
        """Detect watermark in the text."""

        # Encode the text into tokens using the configured tokenizer
        encoded_text = self.config.generation_tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).numpy()[0]

        # Calculate the number of tokens to score, excluding the prefix
        num_scored = len(encoded_text) - self.config.prefix_length
        total_score = 0

        for i in range(self.config.prefix_length, len(encoded_text)):
            # Seed RNG with the prefix of the encoded text
            self.utils.seed_rng(encoded_text[:i])

            # Generate random numbers for each token in the vocabulary
            random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.rng)

            # Calculate score for the current token
            r = random_numbers[encoded_text[i]]
            total_score += log(1 / (1 - r))

        # Compute the average score across all scored tokens
        score = total_score / num_scored if num_scored > 0 else 0

        # Determine if the computed score exceeds the threshold for watermarking
        is_watermarked = score > self.config.threshold

        # Return results based on the `return_dict` flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": score}
        else:
            return (is_watermarked, score)
        
    def get_data_for_visualization(self, text: str, *args, **kwargs) -> DataForVisualization:
        """Get data for visualization."""

        # Encode the text into tokens using the configured tokenizer
        encoded_text = self.config.generation_tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).numpy()[0]

        # Initialize the list of values with None for the prefix length
        highlight_values = [None] * self.config.prefix_length

        # Calculate the value for each token beyond the prefix
        for i in range(self.config.prefix_length, len(encoded_text)):
            # Seed the random number generator using the prefix of the encoded text
            self.utils.seed_rng(encoded_text[:i])
            random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.rng)
            r = random_numbers[encoded_text[i]]
            v = log(1 / (1 - r))
            v = self.utils._value_transformation(v)
            highlight_values.append(v)

        # Decode each token id to its corresponding string token
        decoded_tokens = [self.config.generation_tokenizer.decode([token_id]) for token_id in encoded_text]
        
        return DataForVisualization(decoded_tokens, highlight_values)
        
    