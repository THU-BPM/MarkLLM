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

# =========================================================================
# AutoWatermark.py
# Description: This is a generic watermark class that will be instantiated
#              as one of the watermark classes of the library when created
#              with the [`AutoWatermark.load`] class method.
# =========================================================================

import torch
import importlib
from typing import List
from watermark.auto_config import AutoConfig

WATERMARK_MAPPING_NAMES={
    'KGW': 'watermark.kgw.KGW',
    'Unigram': 'watermark.unigram.Unigram',
    'SWEET': 'watermark.sweet.SWEET',
    'UPV': 'watermark.upv.UPV',
    'SIR': 'watermark.sir.SIR',
    'XSIR': 'watermark.xsir.XSIR',
    'Unbiased': 'watermark.unbiased.UnbiasedWatermark',
    'DIP': 'watermark.dip.DIP',
    'EWD': 'watermark.ewd.EWD',
    'EXP': 'watermark.exp.EXP',
    'EXPGumbel': 'watermark.exp_gumbel.EXPGumbel',
    'EXPEdit': 'watermark.exp_edit.EXPEdit',
    'ITSEdit': 'watermark.its_edit.ITSEdit',
    'SynthID': 'watermark.synthid.SynthID',
    'TS':'watermark.ts.TS',
    'PF':'watermark.pf.PF'
}

def watermark_name_from_alg_name(name):
    """Get the watermark class name from the algorithm name."""
    if name in WATERMARK_MAPPING_NAMES:
        return WATERMARK_MAPPING_NAMES[name]
    else:
        raise ValueError(f"Invalid algorithm name: {name}")

class AutoWatermark:
    """
        This is a generic watermark class that will be instantiated as one of the watermark classes of the library when
        created with the [`AutoWatermark.load`] class method.

        This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoWatermark is designed to be instantiated "
            "using the `AutoWatermark.load(algorithm_name, algorithm_config, transformers_config)` method."
        )

    def load(algorithm_name, algorithm_config=None, transformers_config=None, *args, **kwargs):
        """Load the watermark algorithm instance based on the algorithm name."""
        watermark_name = watermark_name_from_alg_name(algorithm_name)
        module_name, class_name = watermark_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        watermark_class = getattr(module, class_name)
        watermark_config = AutoConfig.load(algorithm_name, transformers_config, algorithm_config_path=algorithm_config, **kwargs)
        watermark_instance = watermark_class(watermark_config)
        return watermark_instance


vllm_supported_methods = ["UPV", "KGW", "Unigram"]
class AutoWatermarkForVLLM:
    def __init__(self, algorithm_name, algorithm_config, transformers_config):
        if not algorithm_name in vllm_supported_methods:
            raise NotImplementedError(f"vllm integrating currently supports {vllm_supported_methods}, but got {algorithm_name}")
        self.watermark = AutoWatermark.load(algorithm_name=algorithm_name, algorithm_config=algorithm_config, transformers_config=transformers_config)

    def __call__(self, prompt_tokens: List[int], generated_tokens: List[int], scores: torch.FloatTensor) -> torch.Tensor:
        if len(prompt_tokens) == 0:
            return scores
        
        # concencate prompt_tokens and generated_tokens
        input_ids = torch.LongTensor(prompt_tokens + generated_tokens).to(self.watermark.config.device)[None, :]
        scores = scores[None, :]
        assert len(input_ids.shape) == 2, input_ids.shape
        assert len(scores.shape) == 2, scores.shape
        
        scores = self.watermark.logits_processor(input_ids, scores)
        return scores[0, :]

    def get_data_for_visualization(self, text):
        data = self.watermark.get_data_for_visualization(text)
        return data

    def detect_watermark(self, text):
        if type(text) is list:
            return [self.watermark.detect_watermark(_) for _ in text]
        return self.watermark.detect_watermark(text)
