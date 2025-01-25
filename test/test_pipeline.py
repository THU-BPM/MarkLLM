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

# ============================================================================
# test_pipeline.py
# Description: This file contains the test cases for the evaluation pipelines.
# ============================================================================

import torch
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.dataset import C4Dataset, WMT16DE_ENDataset, HumanEvalDataset, CNN_DailyMailDataset
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, LlamaTokenizer
from evaluation.tools.text_editor import TruncatePromptTextEditor, TruncateTaskTextEditor ,CodeGenerationTextEditor, WordDeletion
from evaluation.tools.text_quality_analyzer import (PPLCalculator, LogDiversityAnalyzer, BLEUCalculator, 
                                                    PassOrNotJudger, GPTTextDiscriminator, ROUGE1Calculator, 
                                                    ROUGE2Calculator, ROUGELCalculator, BERTScoreCalculator)
from evaluation.pipelines.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType
from evaluation.pipelines.quality_analysis import (DirectTextQualityAnalysisPipeline, ReferencedTextQualityAnalysisPipeline, ExternalDiscriminatorTextQualityAnalysisPipeline, 
                                                   QualityPipelineReturnType)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_detection_pipeline():
    my_dataset = C4Dataset('dataset/c4/processed_c4.json')
    transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("/data2/shared_model/facebook/opt-1.3b/").to(device),
                                             tokenizer=AutoTokenizer.from_pretrained("/data2/shared_model/facebook/opt-1.3b/"),
                                             vocab_size=50272,
                                             device=device,
                                             max_new_tokens=200,
                                             do_sample=True,
                                             min_length=230,
                                             no_repeat_ngram_size=4)
    my_watermark = AutoWatermark.load('KGW', 
                                    algorithm_config='config/KGW.json',
                                    transformers_config=transformers_config)

    pipeline1 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[TruncatePromptTextEditor(), WordDeletion(ratio=0.3)],
                                                show_progress=True, return_type=DetectionPipelineReturnType.SCORES) 

    pipeline2 = UnWatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[],
                                                show_progress=True, return_type=DetectionPipelineReturnType.SCORES)

    calculator = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'F1'], rule='best')
    print(calculator.calculate(pipeline1.evaluate(my_watermark), pipeline2.evaluate(my_watermark)))


def test_direct_quality_analysis_pipeline_1():
    my_dataset = C4Dataset('dataset/c4/processed_c4.json', max_samples=10)
    transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("/workspace/intern_ckpt/panleyi/opt-1.3b").to(device),
                                                tokenizer=AutoTokenizer.from_pretrained("/workspace/intern_ckpt/panleyi/opt-1.3b"),
                                                vocab_size=50272,
                                                device=device,
                                                max_new_tokens=200,
                                                min_length=230,
                                                do_sample=True,
                                                no_repeat_ngram_size=4)
    my_watermark = AutoWatermark.load('KGW', 
                                    algorithm_config='config/KGW.json',
                                    transformers_config=transformers_config)
    quality_pipeline = DirectTextQualityAnalysisPipeline(dataset=my_dataset, 
                                                         watermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                         unwatermarked_text_editor_list=[],
                                                         analyzers=[PPLCalculator(model=AutoModelForCausalLM.from_pretrained("/workspace/intern_ckpt/panleyi/Llama-7b/", device_map='auto'),
                                                                                tokenizer=LlamaTokenizer.from_pretrained("/workspace/intern_ckpt/panleyi/Llama-7b/"),
                                                                                device=device)],
                                                         unwatermarked_text_source='natural', show_progress=True, 
                                                         return_type=QualityPipelineReturnType.MEAN_SCORES)
    print(quality_pipeline.evaluate(my_watermark))


def test_direct_quality_analysis_pipeline_2():
    my_dataset = C4Dataset('dataset/c4/processed_c4.json')
    transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("/data2/shared_model/facebook/opt-1.3b/").to(device),
                                                tokenizer=AutoTokenizer.from_pretrained("/data2/shared_model/facebook/opt-1.3b/"),
                                                vocab_size=50272,
                                                device=device,
                                                max_new_tokens=200,
                                                min_length=230,
                                                do_sample=True,
                                                no_repeat_ngram_size=4)
    my_watermark = AutoWatermark.load('KGW', 
                                    algorithm_config='config/KGW.json',
                                    transformers_config=transformers_config)
    quality_pipeline = DirectTextQualityAnalysisPipeline(dataset=my_dataset, 
                                                         watermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                         unwatermarked_text_editor_list=[],
                                                         analyzers=[LogDiversityAnalyzer()],
                                                         unwatermarked_text_source='natural', show_progress=True, 
                                                         return_type=QualityPipelineReturnType.MEAN_SCORES)
    print(quality_pipeline.evaluate(my_watermark))


# Machine Translation
def test_referenced_quality_analysis_pipeline_1():
    my_dataset = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl')
    tokenizer= AutoTokenizer.from_pretrained("/data2/shared_model/facebook/nllb-200-distilled-600M/", src_lang="deu_Latn")
    transformers_config = TransformersConfig(model=AutoModelForSeq2SeqLM.from_pretrained("/data2/shared_model/facebook/nllb-200-distilled-600M/").to(device),
                                                tokenizer=tokenizer,
                                                device=device,
                                                forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
    my_watermark = AutoWatermark.load('KGW', 
                                    algorithm_config='config/KGW.json',
                                    transformers_config=transformers_config)
    quality_pipeline = ReferencedTextQualityAnalysisPipeline(dataset=my_dataset, 
                                                             watermarked_text_editor_list=[],
                                                             unwatermarked_text_editor_list=[],
                                                             analyzers=[BLEUCalculator()],
                                                             unwatermarked_text_source='generated', show_progress=True, 
                                                             return_type=QualityPipelineReturnType.MEAN_SCORES)
    print(quality_pipeline.evaluate(my_watermark))


# Code Generation
def test_referenced_quality_analysis_pipeline_2():
    my_dataset = HumanEvalDataset('dataset/human_eval/test.jsonl')
    transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("/data2/shared_model/starcoder/", device_map='auto'),
                                             tokenizer=AutoTokenizer.from_pretrained("/data2/shared_model/starcoder/"),
                                             device=device,
                                             min_length=200,
                                             max_length=300)
    my_watermark = AutoWatermark.load('KGW', 
                                      algorithm_config='config/KGW.json',
                                      transformers_config=transformers_config)
    quality_pipeline = ReferencedTextQualityAnalysisPipeline(dataset=my_dataset, 
                                                             watermarked_text_editor_list=[TruncateTaskTextEditor(),CodeGenerationTextEditor()],
                                                             unwatermarked_text_editor_list=[TruncateTaskTextEditor(), CodeGenerationTextEditor()],
                                                             analyzers=[PassOrNotJudger()],
                                                             unwatermarked_text_source='generated', show_progress=True, 
                                                             return_type=QualityPipelineReturnType.MEAN_SCORES)
    print(quality_pipeline.evaluate(my_watermark))


# Text Summarization
def test_referenced_quality_analysis_pipeline_3():
    my_dataset = CNN_DailyMailDataset('dataset/cnn_dailymail/test-00000-of-00001.jsonl', 
                                      max_samples=10,
                                      global_prompt="Please summarize the following article: ")
    transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("/workspace/intern_ckpt/panleyi/glm-4-9b-chat/", trust_remote_code=True, device_map='auto'),
                                             tokenizer=AutoTokenizer.from_pretrained("/workspace/intern_ckpt/panleyi/glm-4-9b-chat/", trust_remote_code=True),
                                             device=device,
                                             max_new_tokens=100,
                                             top_p=0.9,
                                             temperature=0.7)
    my_watermark = AutoWatermark.load('KGW',
                                       algorithm_config='config/KGW.json',
                                       transformers_config=transformers_config)
    
    quality_pipeline = ReferencedTextQualityAnalysisPipeline(dataset=my_dataset,
                                                             watermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                             unwatermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                             analyzers=[ROUGE1Calculator(), ROUGE2Calculator(), ROUGELCalculator(), 
                                                                        BERTScoreCalculator(model_path="/workspace/intern_ckpt/panleyi/bert-base-uncased")],
                                                             unwatermarked_text_source='generated', show_progress=True,
                                                             return_type=QualityPipelineReturnType.MEAN_SCORES)
    print(quality_pipeline.evaluate(my_watermark))


def test_discriminator_quality_analysis_pipeline():
    my_dataset = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl')
    tokenizer= AutoTokenizer.from_pretrained("/data2/shared_model/facebook/nllb-200-distilled-600M/", src_lang="deu_Latn")
    transformers_config = TransformersConfig(model=AutoModelForSeq2SeqLM.from_pretrained("/data2/shared_model/facebook/nllb-200-distilled-600M/").to(device),
                                                tokenizer=tokenizer,
                                                device=device,
                                                forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
    my_watermark = AutoWatermark.load('KGW', 
                                    algorithm_config='config/KGW.json',
                                    transformers_config=transformers_config)
    quality_pipeline = ExternalDiscriminatorTextQualityAnalysisPipeline(dataset=my_dataset, 
                                                             watermarked_text_editor_list=[],
                                                             unwatermarked_text_editor_list=[],
                                                             analyzers=[GPTTextDiscriminator(openai_model='gpt-4',
                                                                                           task_description='Translate the following German text to English')],
                                                             unwatermarked_text_source='generated', show_progress=True, 
                                                             return_type=QualityPipelineReturnType.MEAN_SCORES)
    print(quality_pipeline.evaluate(my_watermark))


if __name__ == '__main__':
    # test_detection_pipeline()
    # test_direct_quality_analysis_pipeline_1()
    # test_direct_quality_analysis_pipeline_2()
    # test_referenced_quality_analysis_pipeline_1()
    # test_referenced_quality_analysis_pipeline_2()
    test_referenced_quality_analysis_pipeline_3()
    # test_discriminator_quality_analysis_pipeline()