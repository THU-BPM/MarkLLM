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

# ==================================================
# exp_gumbel.py
# Description: Implementation of EXPGumbel algorithm
# ==================================================

import scipy
import torch
from math import log
from ..base import BaseWatermark
from utils.utils import load_config_file
from transformers import LogitsProcessor
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from visualize.data_for_visualization import DataForVisualization

class EXPGumbelConfig:
    """Config class for EXPGumbel algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the EXPGumbel configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/EXPGumbel.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'EXPGumbel':
            raise AlgorithmNameMismatchError('EXPGumbel', config_dict['algorithm_name'])

        self.prefix_length = config_dict['prefix_length']
        self.eps = config_dict['eps']
        self.threshold = config_dict['threshold']
        self.sequence_length = config_dict['sequence_length']
        self.temperature = config_dict['temperature']
        self.seed = config_dict['seed']

        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs


class EXPGumbelUtils:
    """Utility class for EXPGumbel algorithm, contains helper functions."""

    def __init__(self, config: EXPGumbelConfig, *args, **kwargs) -> None:
        """
            Initialize the EXPGumbel utility class.

            Parameters:
                config (EXPGumbelConfig): Configuration for the EXPGumbel algorithm.
        """
        self.config = config
        self.generator = torch.Generator().manual_seed(self.config.seed)
        self.uniform = torch.clamp(torch.rand((self.config.vocab_size * self.config.prefix_length, self.config.vocab_size), 
                                         generator=self.generator, dtype=torch.float32), min=self.config.eps)
        self.gumbel = (-torch.log(torch.clamp(-torch.log(self.uniform), min=self.config.eps))).to(self.config.device)
    
    def _value_transformation(self, value):
        """Transform the value to a range between 0 and 1."""
        return value/(value + 1)


class EXPGumbelLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for EXPGumbel algorithm, process logits to add watermark."""

    def __init__(self, config: EXPGumbelConfig, utils: EXPGumbelUtils, *args, **kwargs) -> None:
        """
            Initialize the EXPGumbel logits processor.

            Parameters:
                config (EXPGumbelConfig): Configuration for the KGW algorithm.
                utils (EXPGumbelUtils): Utility class for the KGW algorithm.
        """
        self.config = config
        self.utils = utils

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores
        prev_token = torch.sum(input_ids[:, -self.config.prefix_length:], dim=-1)  # (batch_size,)
        gumbel = self.utils.gumbel[prev_token]  # (batch_size, vocab_size)
        return scores[..., :gumbel.shape[-1]] / self.config.temperature + gumbel


class EXPGumbel(BaseWatermark):
    """Top-level class for the EXPGumbel algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the EXPGumbel algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = EXPGumbelConfig(algorithm_config, transformers_config)
        self.utils = EXPGumbelUtils(self.config)
        self.logits_processor = EXPGumbelLogitsProcessor(self.config, self.utils)
    
    def watermark_logits_argmax(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.LongTensor:
        """
        Applies watermarking to the last token's logits and returns the argmax for that token.
        Returns tensor of shape (batch,), where each element is the index of the selected token.
        """
        
        # Get the logits for the last token
        last_logits = logits[:, -1, :]  # (batch, vocab_size)
        
        # Get the argmax of the logits
        last_token = torch.argmax(last_logits, dim=-1).unsqueeze(-1)  # (batch,)
        return last_token

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text using the EXPGumbel algorithm."""

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
                    output_gumbel = self.logits_processor(input_ids=inputs, scores=output.logits)
                else:
                    output = self.config.generation_model(inputs)
                    output_gumbel = self.logits_processor(input_ids=inputs, scores=output.logits)

            # Sample token
            token = self.watermark_logits_argmax(inputs, output_gumbel)

            # Update past
            past = output.past_key_values

            # Update inputs
            inputs = torch.cat([inputs, token], dim=-1)

            # Update attention mask
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

        watermarked_tokens = inputs[0].detach().cpu()
        watermarked_text = self.config.generation_tokenizer.decode(watermarked_tokens, skip_special_tokens=True)

        return watermarked_text

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> dict:
        """Detect watermark in the text."""

        # Encode the text into tokens using the configured tokenizer
        encoded_text = self.config.generation_tokenizer.encode(text, return_tensors='pt', add_special_tokens=False)[0]

        seq_len = len(encoded_text)
        score = 0
        for i in range(self.config.prefix_length, seq_len):
            prev_tokens_sum = torch.sum(encoded_text[i - self.config.prefix_length:i], dim=-1)
            token = encoded_text[i]
            u = self.utils.uniform[prev_tokens_sum, token]
            score += log(1 / (1 - u))
        
        p_value = scipy.stats.gamma.sf(score, seq_len - self.config.prefix_length, loc=0, scale=1)
        
        # Determine if the computed score exceeds the threshold for watermarking
        is_watermarked = bool(p_value < self.config.threshold)

        # Return results based on the `return_dict` flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": p_value}
        else:
            return (is_watermarked, p_value)
    
    def get_data_for_visualization(self, text: str, *args, **kwargs) -> DataForVisualization:
        """Get data for visualization."""
        
        # Encode the text into tokens using the configured tokenizer
        encoded_text = self.config.generation_tokenizer.encode(text, return_tensors='pt', add_special_tokens=False)[0]

        # Initialize the list of values with None for the prefix length
        highlight_values = [None] * self.config.prefix_length

        # Calculate the value for each token beyond the prefix
        for i in range(self.config.prefix_length, len(encoded_text)):
            # Seed the random number generator using the prefix of the encoded text
            prev_tokens_sum = torch.sum(encoded_text[i - self.config.prefix_length:i], dim=-1)
            token = encoded_text[i]
            u = self.utils.uniform[prev_tokens_sum, token]
            score = log(1 / (1 - u))
            highlight_values.append(self.utils._value_transformation(score))

        # Decode each token id to its corresponding token
        decoded_tokens = [self.config.generation_tokenizer.decode([token_id]) for token_id in encoded_text]

        return DataForVisualization(decoded_tokens, highlight_values)
            