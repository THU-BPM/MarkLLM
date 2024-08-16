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

# ================================================================
# assess_robustness.py
# Description: Assess the robustness of a watermarking algorithm
# ================================================================

import torch
from translate import Translator
from evaluation.dataset import C4Dataset
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForMaskedLM
from evaluation.pipelines.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType
from evaluation.tools.text_editor import TruncatePromptTextEditor, WordDeletion, SynonymSubstitution, ContextAwareSynonymSubstitution, GPTParaphraser, DipperParaphraser, BackTranslationTextEditor


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def assess_robustness(algorithm_name, attack_name):
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
    if attack_name == 'Word-D':
        attack = WordDeletion(ratio=0.3)
    elif attack_name == 'Word-S':
        attack = SynonymSubstitution(ratio=0.5)
    elif attack_name == 'Word-S(Context)':
        attack = ContextAwareSynonymSubstitution(ratio=0.5,
                                                 tokenizer=BertTokenizer.from_pretrained('/data2/shared_model/bert-large-uncased'),
                                                 model=BertForMaskedLM.from_pretrained('/data2/shared_model/bert-large-uncased').to(device))
    elif attack_name == 'Doc-P(GPT-3.5)':
        attack = GPTParaphraser(openai_model='gpt-3.5-turbo',
                                prompt='Please rewrite the following text: ')
    elif attack_name == 'Doc-P(Dipper)':
        attack = DipperParaphraser(tokenizer=T5Tokenizer.from_pretrained('/data2/shared_model/google/t5-v1_1-xxl/'),
                                   model=T5ForConditionalGeneration.from_pretrained('/data2/shared_model/kalpeshk2011/dipper-paraphraser-xxl/', device_map='auto'),
                                   lex_diversity=60, order_diversity=0, sent_interval=1, 
                                   max_new_tokens=100, do_sample=True, top_p=0.75, top_k=None)
    elif attack_name == 'Translation':
        attack = BackTranslationTextEditor(translate_to_intermediary = Translator(from_lang="en", to_lang="zh").translate,
                                           translate_to_source = Translator(from_lang="zh", to_lang="en").translate)

    pipline1 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[TruncatePromptTextEditor(), attack],
                                                show_progress=True, return_type=DetectionPipelineReturnType.SCORES) 

    pipline2 = UnWatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[],
                                                show_progress=True, return_type=DetectionPipelineReturnType.SCORES)

    calculator = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'F1'], rule='best')
    print(calculator.calculate(pipline1.evaluate(my_watermark), pipline2.evaluate(my_watermark)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='KGW')
    parser.add_argument('--attack', type=str, default='Word-D')
    args = parser.parse_args()

    assess_robustness(args.algorithm, args.attack)