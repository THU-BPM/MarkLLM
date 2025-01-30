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
# exp.py
# Description: Implementation of EXP algorithm
# ============================================

import torch
import scipy
from math import log
from ..base import BaseWatermark, BaseConfig
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from visualize.data_for_visualization import DataForVisualization


class EXPConfig(BaseConfig):
    """Config class for EXP algorithm, load config file and initialize parameters."""

    
    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.prefix_length = self.config_dict['prefix_length']
        self.hash_key = self.config_dict['hash_key']
        self.threshold = self.config_dict['threshold']
        self.sequence_length = self.config_dict['sequence_length']
        self.top_k = getattr(self.transformers_config, 'top_k', -1)
        self.temperature = getattr(self.transformers_config, 'temperature', 0.7)
    
    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'EXP'


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
        
        # If top_k is not specified, use argmax
        if self.config.top_k <= 0:
            return torch.argmax(u ** (1 / probs), axis=1).unsqueeze(-1)
        
        # Ensure top_k is not greater than the vocabulary size
        top_k = min(self.config.top_k, probs.size(-1))
    
        # Get the top_k probabilities and their indices
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
    
        # Perform exponential sampling on the top_k probabilities
        sampled_indices = torch.argmax(u.gather(-1, top_indices) ** (1 / top_probs), dim=-1)
    
        # Map back the sampled indices to the original vocabulary indices
        return top_indices.gather(-1, sampled_indices.unsqueeze(-1))
    
    def _value_transformation(self, value):
        """Transform the value to a range between 0 and 1."""
        return value/(value + 1)
    

class EXP(BaseWatermark):
    """Top-level class for the EXP algorithm."""

    def __init__(self, algorithm_config: str | EXPConfig, transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        """
            Initialize the EXP algorithm.

            Parameters:
                algorithm_config (str | EXPConfig): Path to the algorithm configuration file or EXPConfig instance.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if isinstance(algorithm_config, str):
            self.config = EXPConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, EXPConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be either a path string or a EXPConfig instance")
        
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
            
            # Get probabilities with temperature
            probs = torch.nn.functional.softmax(output.logits[:,-1, :self.config.vocab_size] / self.config.temperature, dim=-1).cpu()
            
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

        # Calculate p_value
        p_value = scipy.stats.gamma.sf(total_score, num_scored, loc=0, scale=1)

        # Determine if the computed score exceeds the threshold for watermarking
        is_watermarked = p_value < self.config.threshold

        # Return results based on the `return_dict` flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": p_value}
        else:
            return (is_watermarked, p_value)
        
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
        
    