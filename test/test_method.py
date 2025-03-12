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

# ==========================================
# test_method.py
# Description: Test the watermark algorithm
# ==========================================

import json
import torch
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


# Load data
with open('dataset/c4/processed_c4.json', 'r') as f:
    lines = f.readlines()
item = json.loads(lines[0])
prompt = item['prompt']
natural_text = item['natural_text']


def test_algorithm(algorithm_name):
    # Check algorithm name
    assert algorithm_name in ['KGW', 'Unigram', 'SWEET', 'EWD', 'SIR', 'XSIR', 'DIP', 'Unbiased', 
                              'UPV', 'TS', 'SynthID', 'EXP', 'EXPGumbel', 'EXPEdit', 'ITSEdit', 'PF']

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Transformers config
    transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to(device),
                                            tokenizer=AutoTokenizer.from_pretrained("facebook/opt-1.3b"),
                                            vocab_size=50272,
                                            device=device,
                                            max_new_tokens=200,
                                            min_length=230,
                                            do_sample=True,
                                            no_repeat_ngram_size=4)
                                            
        
    # Load watermark algorithm
    myWatermark = AutoWatermark.load(f'{algorithm_name}', 
                                     algorithm_config=f'config/{algorithm_name}.json',
                                     transformers_config=transformers_config, delta=1)

    watermarked_text = myWatermark.generate_watermarked_text(prompt)
    print(watermarked_text)
    unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)
    detect_result = myWatermark.detect_watermark(watermarked_text)
    print(detect_result)
    detect_result = myWatermark.detect_watermark(unwatermarked_text)
    print(detect_result)
    detect_result = myWatermark.detect_watermark(natural_text)
    print(detect_result)


if __name__ == '__main__':
    test_algorithm('KGW')


