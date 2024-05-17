# ================================================
# exp_edit.py
# Description: Implementation of EXPEdit algorithm
# ================================================

import torch
import numpy as np
from math import log
from ..base import BaseWatermark
from .mersenne import MersenneRNG
from utils.utils import load_config_file
from .cython_files.levenshtein import levenshtein
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from visualize.data_for_visualization import DataForVisualization


class EXPEditConfig:
    """Config class for EXPEdit algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the EXPEdit configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is not None:
            config_dict = load_config_file('config/EXPEdit.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'EXPEdit':
            raise AlgorithmNameMismatchError('EXPEdit', config_dict['algorithm_name'])

        self.pseudo_length = config_dict['pseudo_length']
        self.sequence_length = config_dict['sequence_length']
        self.n_runs = config_dict['n_runs']
        self.p_threshold = config_dict['p_threshold']
        self.key = config_dict['key']

        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs


class EXPEditUtils:
    """Utility class for EXPEdit algorithm, contains helper functions."""

    def __init__(self, config: EXPEditConfig, *args, **kwargs) -> None:
        """
            Initialize the EXPEdit utility class.

            Parameters:
                config (EXPEditConfig): Configuration for the EXPEdit algorithm.
        """
        self.config = config
        self.rng = MersenneRNG(self.config.key)
        self.xi = (torch.tensor([self.rng.rand() for _ in range(self.config.pseudo_length * self.config.vocab_size)])
                   .view(self.config.pseudo_length, self.config.vocab_size))

    def exp_sampling(self, probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Sample token using exponential distribution."""
        return torch.argmax(u ** (1 / probs), axis=1).unsqueeze(-1)
    
    def value_transformation(self, value: float) -> float:
        """Transform value to range [0, 1]."""
        return value/(value + 1)

    def one_run(self, tokens: np.ndarray, xi: np.ndarray) -> tuple:
        """Run one test."""
        k = len(tokens)
        n = len(xi)
        A = np.empty((1,n)) 
        for i in range(1):
            for j in range(n):
                A[i][j] = levenshtein(tokens[i:i+k],xi[(j+np.arange(k))%n],0.0)

        return np.min(A), np.argmin(A)


class EXPEdit(BaseWatermark):
    """Top-level class for the EXPEdit algorithm."""
    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        self.config = EXPEditConfig(algorithm_config, transformers_config)
        self.utils = EXPEditUtils(self.config)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs):
        """Generate watermarked text."""

        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)

        # Initialize
        shift = torch.randint(self.config.pseudo_length, (1,))
        inputs = encoded_prompt
        attn = torch.ones_like(inputs)
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
            
            # Sample token to add watermark
            token = self.utils.exp_sampling(probs, self.utils.xi[(shift + i) % self.config.pseudo_length,:]).to(self.config.device)
            
            # Update inputs
            inputs = torch.cat([inputs, token], dim=-1)

            # Update past
            past = output.past_key_values

            # Update attention mask
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
        
        watermarked_tokens = inputs[0].detach().cpu()
        watermarked_text = self.config.generation_tokenizer.decode(watermarked_tokens, skip_special_tokens=True)

        return watermarked_text

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the text."""

        encoded_text = self.config.generation_tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).numpy()[0]

        test_result, _ = self.utils.one_run(encoded_text, self.utils.xi.numpy())

        p_val = 0
        
        for i in range(self.config.n_runs):
            xi_alternative = np.random.rand(self.config.pseudo_length, self.config.vocab_size).astype(np.float32)
            null_result, _ = self.utils.one_run(encoded_text, xi_alternative)

            # assuming lower test values indicate presence of watermark
            p_val += null_result <= test_result
            print(f"round: {i + 1}, good: {null_result > test_result}")

        p_val = (p_val + 1.0) / (self.config.n_runs + 1.0)

        # Determine if the computed score exceeds the threshold for watermarking
        is_watermarked = p_val < self.config.p_threshold

        # Return results based on the `return_dict` flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": p_val}
        else:
            return (is_watermarked, p_val)

    def get_data_for_visualization(self, text: str, *args, **kwargs):
        """Get data for visualization."""

        # Encode text
        encoded_text = self.config.generation_tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).numpy()[0]

        # Find best match
        _, index = self.utils.one_run(encoded_text, self.utils.xi.numpy())
        random_numbers = self.utils.xi[(index + np.arange(len(encoded_text))) % len(self.utils.xi)]
        
        highlight_values = []

        # Compute highlight values
        for i in range(0, len(encoded_text)):
            r = random_numbers[i][encoded_text[i]]
            v = log(1/(1 - r))
            v = self.utils.value_transformation(v)
            highlight_values.append(v)

        # Decode each token id to its corresponding string token
        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)
        
        return DataForVisualization(decoded_tokens, highlight_values)