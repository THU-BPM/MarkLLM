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

# ===============================================================
# base.py
# Description: This is a generic watermark class that will be 
#              inherited by the watermark classes of the library.
# ===============================================================

from typing import Union
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from visualize.data_for_visualization import DataForVisualization


class BaseConfig:
    """Base configuration class for watermark algorithms."""

    def __init__(self, algorithm_config_path: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
        Initialize the base configuration.

        Parameters:
            algorithm_config (str): Path to the algorithm configuration file.
            transformers_config (TransformersConfig): Configuration for the transformers model.
            **kwargs: Additional parameters to override config values
        """
        # Load config file
        self.config_dict = load_config_file(f'config/{self.algorithm_name()}.json') if algorithm_config_path is None else load_config_file(algorithm_config_path)
            
        # Update config with kwargs
        if kwargs:
            self.config_dict.update(kwargs)
            
        # Load model-related configurations
        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs
        self.transformers_config = transformers_config
        
        # Initialize algorithm-specific parameters
        self.initialize_parameters()
        
    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters. Should be overridden by subclasses."""
        raise NotImplementedError
    
    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name. Should be overridden by subclasses."""
        raise NotImplementedError



class BaseWatermark:
    def __init__(self, algorithm_config: str | BaseConfig, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        pass

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str: 
        pass

    def generate_unwatermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate unwatermarked text."""
        
        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        # Generate unwatermarked text
        encoded_unwatermarked_text = self.config.generation_model.generate(**encoded_prompt, **self.config.gen_kwargs)
        # Decode
        unwatermarked_text = self.config.generation_tokenizer.batch_decode(encoded_unwatermarked_text, skip_special_tokens=True)[0]
        return unwatermarked_text

    def detect_watermark(self, text:str, return_dict: bool=True, *args, **kwargs) -> Union[tuple, dict]:
        pass

    def get_data_for_visualize(self, text, *args, **kwargs) -> DataForVisualization:
        pass



