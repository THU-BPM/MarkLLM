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

import importlib

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
    'ITSEdit': 'watermark.its_edit.ITSEdit'
}

def watermark_name_from_alg_name(name):
    """Get the watermark class name from the algorithm name."""
    for algorithm_name, watermark_name in WATERMARK_MAPPING_NAMES.items():
        if name == algorithm_name:
            return watermark_name
    return None

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
        watermark_instance = watermark_class(algorithm_config, transformers_config)
        return watermark_instance

