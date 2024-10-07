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

# ==========================================================================
# assess_quality.py
# Description: Assess the impact on text quality of a watermarking algorithm
# ==========================================================================

import torch
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.dataset import C4Dataset, HumanEvalDataset, WMT16DE_ENDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, LlamaTokenizer
from evaluation.tools.text_editor import TruncatePromptTextEditor, TruncateTaskTextEditor, CodeGenerationTextEditor
from evaluation.tools.text_quality_analyzer import PPLCalculator, LogDiversityAnalyzer, BLEUCalculator, PassOrNotJudger, GPTTextDiscriminator
from evaluation.pipelines.quality_analysis import DirectTextQualityAnalysisPipeline, QualityPipelineReturnType, ReferencedTextQualityAnalysisPipeline, ExternalDiscriminatorTextQualityAnalysisPipeline


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def assess_quality(algorithm_name, metric):
    if metric == 'PPL':
        my_dataset = C4Dataset('dataset/c4/processed_c4.json')
        pipeline = DirectTextQualityAnalysisPipeline(dataset=my_dataset, 
                                                     watermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                     unwatermarked_text_editor_list=[],
                                                     analyzer=PPLCalculator(model=AutoModelForCausalLM.from_pretrained("/data2/shared_model/llama-7b/", device_map='auto'),
                                                                            tokenizer=LlamaTokenizer.from_pretrained("/data2/shared_model/llama-7b/"),
                                                                            device=device),
                                                    unwatermarked_text_source='natural', show_progress=True, 
                                                    return_type=QualityPipelineReturnType.MEAN_SCORES)
        transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("/data2/shared_model/facebook/opt-1.3b/").to(device),
                                                 tokenizer=AutoTokenizer.from_pretrained("/data2/shared_model/facebook/opt-1.3b/"),
                                                 vocab_size=50272,
                                                 device=device,
                                                 max_new_tokens=200,
                                                 min_length=230,
                                                 do_sample=True,
                                                 no_repeat_ngram_size=4)

    elif metric == 'Log Diversity':
        my_dataset = C4Dataset('dataset/c4/processed_c4.json')
        pipeline = DirectTextQualityAnalysisPipeline(dataset=my_dataset, 
                                                     watermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                     unwatermarked_text_editor_list=[],
                                                     analyzer=LogDiversityAnalyzer(),
                                                     unwatermarked_text_source='natural', show_progress=True, 
                                                     return_type=QualityPipelineReturnType.MEAN_SCORES)
        transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("/data2/shared_model/facebook/opt-1.3b/").to(device),
                                                 tokenizer=AutoTokenizer.from_pretrained("/data2/shared_model/facebook/opt-1.3b/"),
                                                 vocab_size=50272,
                                                 device=device,
                                                 max_new_tokens=200,
                                                 min_length=230,
                                                 do_sample=True,
                                                 no_repeat_ngram_size=4)
    elif metric == 'BLEU':
        my_dataset = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl')
        tokenizer = AutoTokenizer.from_pretrained("/data2/shared_model/facebook/nllb-200-distilled-600M/", src_lang="deu_Latn")
        transformers_config = TransformersConfig(model=AutoModelForSeq2SeqLM.from_pretrained("/data2/shared_model/facebook/nllb-200-distilled-600M/").to(device),
                                                 tokenizer=tokenizer,
                                                 device=device,
                                                 vocab_size=256206,
                                                 forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
        pipeline = ReferencedTextQualityAnalysisPipeline(dataset=my_dataset, 
                                                         watermarked_text_editor_list=[],
                                                         unwatermarked_text_editor_list=[],
                                                         analyzer=BLEUCalculator(),
                                                         unwatermarked_text_source='generated', show_progress=True, 
                                                         return_type=QualityPipelineReturnType.MEAN_SCORES)
    elif metric == 'pass@1':
        my_dataset = HumanEvalDataset('dataset/human_eval/test.jsonl')
        transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("/data2/shared_model/starcoder/", device_map='auto'),
                                                 tokenizer=AutoTokenizer.from_pretrained("/data2/shared_model/starcoder/"),
                                                 device=device,
                                                 min_length=200,
                                                 max_length=400)
        pipeline = ReferencedTextQualityAnalysisPipeline(dataset=my_dataset, 
                                                         watermarked_text_editor_list=[TruncateTaskTextEditor(),CodeGenerationTextEditor()],
                                                         unwatermarked_text_editor_list=[TruncateTaskTextEditor(), CodeGenerationTextEditor()],
                                                         analyzer=PassOrNotJudger(),
                                                         unwatermarked_text_source='generated', show_progress=True, 
                                                         return_type=QualityPipelineReturnType.MEAN_SCORES)
    elif metric == 'GPT-4 Judge':
        my_dataset = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl')
        tokenizer = AutoTokenizer.from_pretrained("/data2/shared_model/facebook/nllb-200-distilled-600M/", src_lang="deu_Latn")
        transformers_config = TransformersConfig(model=AutoModelForSeq2SeqLM.from_pretrained("/data2/shared_model/facebook/nllb-200-distilled-600M/").to(device),
                                                 tokenizer=tokenizer,
                                                 device=device,
                                                 vocab_size=256206,
                                                 forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
        pipeline = ExternalDiscriminatorTextQualityAnalysisPipeline(dataset=my_dataset, 
                                                                    watermarked_text_editor_list=[],
                                                                    unwatermarked_text_editor_list=[],
                                                                    analyzer=GPTTextDiscriminator(openai_model='gpt-4',
                                                                                                  task_description='Translate the following German text to English.'),
                                                                    unwatermarked_text_source='generated', show_progress=True, 
                                                                    return_type=QualityPipelineReturnType.MEAN_SCORES)
    else:
        raise ValueError('Invalid metric')
    
    
    my_watermark = AutoWatermark.load(f'{algorithm_name}', 
                                    algorithm_config=f'config/{algorithm_name}.json',
                                    transformers_config=transformers_config)
    print(pipeline.evaluate(my_watermark))

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='KGW')
    parser.add_argument('--metric', type=str, default='PPL')
    args = parser.parse_args()

    assess_quality(args.algorithm, args.metric)