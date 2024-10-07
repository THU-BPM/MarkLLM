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
# transformers_config.py
# Description: Configuration for transformers
# ============================================

class TransformersConfig:
    """Configuration class for transformers."""

    def __init__(self, model, tokenizer, vocab_size=None, device='cuda', *args, **kwargs):
        """
            Initialize the transformers configuration.

            Parameters:
                model (object): The model object.
                tokenizer (object): The tokenizer object.
                vocab_size (int): The vocabulary size.
                device (str): The device to use.
                kwargs: Additional keyword arguments.
        """
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer) if vocab_size is None else vocab_size
        self.gen_kwargs = {}
        self.gen_kwargs.update(kwargs)