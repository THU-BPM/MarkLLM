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
# auto_config.py
# Description: This is a generic configuration class that will be 
#              inherited by the configuration classes of the library.
# ===============================================================


import importlib
from typing import Dict, Optional, Any
from utils.transformers_config import TransformersConfig

CONFIG_MAPPING_NAMES = {
    'KGW': 'watermark.kgw.KGWConfig',
    'Unigram': 'watermark.unigram.UnigramConfig',
    'SWEET': 'watermark.sweet.SWEETConfig',
    'UPV': 'watermark.upv.UPVConfig',
    'SIR': 'watermark.sir.SIRConfig',
    'XSIR': 'watermark.xsir.XSIRConfig',
    'Unbiased': 'watermark.unbiased.UnbiasedConfig',
    'DIP': 'watermark.dip.DIPConfig',
    'EWD': 'watermark.ewd.EWDConfig',
    'EXP': 'watermark.exp.EXPConfig',
    'EXPGumbel': 'watermark.exp_gumbel.EXPGumbelConfig',
    'EXPEdit': 'watermark.exp_edit.EXPEditConfig',
    'ITSEdit': 'watermark.its_edit.ITSEditConfig',
    'SynthID': 'watermark.synthid.SynthIDConfig',
    'TS': 'watermark.ts.TSConfig',
    'PF': 'watermark.pf.PFConfig'
}

def config_name_from_alg_name(name: str) -> Optional[str]:
    """Get the config class name from the algorithm name."""
    if name in CONFIG_MAPPING_NAMES:
        return CONFIG_MAPPING_NAMES[name]
    else:
        raise ValueError(f"Invalid algorithm name: {name}")

class AutoConfig:
    """
    A generic configuration class that will be instantiated as one of the configuration classes
    of the library when created with the [`AutoConfig.load`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.load(algorithm_name, **kwargs)` method."
        )

    @classmethod
    def load(cls, algorithm_name: str, transformers_config: TransformersConfig, algorithm_config_path=None, **kwargs) -> Any:
        """
        Load the configuration class for the specified watermark algorithm.

        Args:
            algorithm_name (str): The name of the watermark algorithm
            transformers_config (TransformersConfig): Configuration for the transformers model
            algorithm_config_path (str): Path to the algorithm configuration file
            **kwargs: Additional keyword arguments to pass to the configuration class

        Returns:
            The instantiated configuration class for the specified algorithm
        """
        config_name = config_name_from_alg_name(algorithm_name)
        if config_name is None:
            raise ValueError(f"Unknown algorithm name: {algorithm_name}")
            
        module_name, class_name = config_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        config_class = getattr(module, class_name)
        if algorithm_config_path is None:
            algorithm_config_path = f'config/{algorithm_name}.json'
        config_instance = config_class(algorithm_config_path, transformers_config, **kwargs)
        return config_instance