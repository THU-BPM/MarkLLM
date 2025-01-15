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
from utils.transformers_config import TransformersConfig
from visualize.data_for_visualization import DataForVisualization


class BaseWatermark:
    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
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
    
    def generate_unwatermarked_code(self, prompt: str, *args, **kwargs) -> str:
        """Generate unwatermarked code."""
        
        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        # Generate unwatermarked code
        encoded_unwatermarked_code = self.config.generation_model.generate(**encoded_prompt, **self.config.gen_kwargs)
        # Decode
        unwatermarked_code = self.config.generation_tokenizer.batch_decode(encoded_unwatermarked_code, skip_special_tokens=True)[0]
        return unwatermarked_code

    def detect_watermark(self, text:str, return_dict: bool=True, *args, **kwargs) -> Union[tuple, dict]:
        pass

    def get_data_for_visualize(self, text, *args, **kwargs) -> DataForVisualization:
        pass


