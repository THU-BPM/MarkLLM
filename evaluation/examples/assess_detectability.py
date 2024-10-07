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

# =================================================================
# assess_detectability.py
# Description: Assess the detectability of a watermarking algorithm
# =================================================================

import torch
from evaluation.dataset import C4Dataset
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.tools.text_editor import TruncatePromptTextEditor
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from evaluation.pipelines.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def assess_detectability(algorithm_name, labels, rules, target_fpr):
    my_dataset = C4Dataset('dataset/c4/processed_c4.json')
    transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("/data2/shared_model/facebook/opt-1.3b/").to(device),
                                             tokenizer=AutoTokenizer.from_pretrained("/data2/shared_model/facebook/opt-1.3b/"),
                                             vocab_size=50272,
                                             device=device,
                                             max_new_tokens=200,
                                             min_length=230,
                                             do_sample=True,
                                             no_repeat_ngram_size=4)
    my_watermark = AutoWatermark.load(f'{algorithm_name}', 
                                    algorithm_config=f'config/{algorithm_name}.json',
                                    transformers_config=transformers_config)

    pipline1 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[TruncatePromptTextEditor()],
                                                show_progress=True, return_type=DetectionPipelineReturnType.SCORES) 

    pipline2 = UnWatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[],
                                                show_progress=True, return_type=DetectionPipelineReturnType.SCORES)

    calculator = DynamicThresholdSuccessRateCalculator(labels=labels, rule=rules, target_fpr=target_fpr)
    print(calculator.calculate(pipline1.evaluate(my_watermark), pipline2.evaluate(my_watermark)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='KGW')
    parser.add_argument('--labels', nargs='+', default=['TPR', 'F1'])
    parser.add_argument('--rules', type=str, default='best')
    parser.add_argument('--target_fpr', type=float, default=0.01)
    args = parser.parse_args()

    assess_detectability(args.algorithm, args.labels, args.rules, args.target_fpr)