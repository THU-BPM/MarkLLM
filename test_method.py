# ==========================================
# test_method.py
# Description: Test the watermark algorithm
# ==========================================

import json
import torch
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from visualize.font_settings import FontSettings

from visualize.visualizer import DiscreteVisualizer
from visualize.legend_settings import DiscreteLegendSettings
from visualize.page_layout_settings import PageLayoutSettings
from visualize.color_scheme import ColorSchemeForDiscreteVisualization


# Load data
with open('dataset/c4/processed_c4.json', 'r') as f:
    lines = f.readlines()
items = []
prompts = []
natural_texts = []

for line in lines:
    item = json.loads(line)
    items.append(item)
    prompts.append(item['prompt'])
    natural_texts.append(item['natural_text'])
# prompt = item['prompt']
# natural_text = item['natural_text']


def test_algorithm(algorithm_name):
    # Check algorithm name
    assert algorithm_name in ['KGW', 'Unigram', 'SWEET', 'EWD', 'SIR', 'XSIR', 'DIP', 'UPV', 'EXP', 'EXPEdit','ITSEdit']
    print("done")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transformers config
    transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained('facebook/opt-1.3b').to(device),
                                            tokenizer=AutoTokenizer.from_pretrained('facebook/opt-1.3b'),
                                            vocab_size=50272,
                                            device=device,
                                            max_new_tokens=600,
                                            min_length=230,
                                            do_sample=True,
                                            no_repeat_ngram_size=4)
    print("done")
    visualizer = DiscreteVisualizer(color_scheme=ColorSchemeForDiscreteVisualization(),
                                font_settings=FontSettings(), 
                                page_layout_settings=PageLayoutSettings(),
                                legend_settings=DiscreteLegendSettings())
    
    # Load watermark algorithm
    myWatermark = AutoWatermark.load(f'{algorithm_name}', 
                                     algorithm_config=f'config/{algorithm_name}.json',
                                     transformers_config=transformers_config)

    i = 1
    print("done")
    for prompt, natural_text in zip(prompts, natural_texts):
    
        watermarked_text = myWatermark.generate_watermarked_text(prompt)
        unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)
        detect_result = myWatermark.detect_watermark(watermarked_text)
        watermarked_data = myWatermark.get_data_for_visualization(watermarked_text)
        watermarked_img = visualizer.visualize(data=watermarked_data, 
                                       show_text=True, 
                                       visualize_weight=True, 
                                       display_legend=True)
        watermarked_img.save(f"DIP_watermarked{i}.png")
        
        print(detect_result)
        detect_result = myWatermark.detect_watermark(unwatermarked_text)
        unwatermarked_data = myWatermark.get_data_for_visualization(unwatermarked_text)
        unwatermarked_img = visualizer.visualize(data=unwatermarked_data,
                                         show_text=True, 
                                         visualize_weight=True, 
                                         display_legend=True)
        unwatermarked_img.save(f"DIP_unwatermarked{i}.png")
        
        print(detect_result)
        detect_result = myWatermark.detect_watermark(natural_text)
        print(detect_result)
        
        i = i + 1
        print()


if __name__ == '__main__':
    test_algorithm('DIP')


