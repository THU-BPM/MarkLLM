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

    def detect_watermark(self, text:str, return_dict: bool=True, *args, **kwargs) -> Union[tuple, dict]:
        pass

    def get_data_for_visualize(self, text, *args, **kwargs) -> DataForVisualization:
        pass


