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
# test_visualize.py
# Description: This file contains the test cases for the visualization module.
# ============================================================================

import json
import torch
from visualize.font_settings import FontSettings
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from visualize.page_layout_settings import PageLayoutSettings
from evaluation.tools.text_editor import TruncatePromptTextEditor
from visualize.data_for_visualization import DataForVisualization
from visualize.visualizer import DiscreteVisualizer, ContinuousVisualizer
from visualize.legend_settings import DiscreteLegendSettings, ContinuousLegendSettings
from visualize.color_scheme import ColorSchemeForDiscreteVisualization, ColorSchemeForContinuousVisualization


def test_discreet_visualization():
    tokens = ["PREFIX", "PREFIX", "Hello", "world", "this", "is", "a", "test", "Hello", "world", "this", "is", "a", "test", "Hello", "world", "this", "is", "a", "test", "Hello", "world", "this", "is", "a", "test", "Hello", "world", "this", "is", "a", "test"]
    flags = [-1, -1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0]
    weights = [0, 0, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4]

    discreet_visualizer = DiscreteVisualizer(color_scheme=ColorSchemeForDiscreteVisualization(),
                                            font_settings=FontSettings(), 
                                            page_layout_settings=PageLayoutSettings(),
                                            legend_settings=DiscreteLegendSettings())
    img = discreet_visualizer.visualize(data=DataForVisualization(tokens, flags, weights), 
                                        show_text=True, visualize_weight=True, display_legend=True)
    img.save("test1.png")


def test_continuous_visualization():
    tokens = ["PREFIX", "PREFIX", "Hello", "world", "this", "is", "a", "test", "Hello", "world", "this", "is", "a", "test", "Hello", "world", "this", "is", "a", "test", "Hello", "world", "this", "is", "a", "test", "Hello", "world", "this", "is", "a", "test"]
    values = [None, None, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4]
    weights = [0, 0, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4]

    continuous_visualizer = ContinuousVisualizer(color_scheme=ColorSchemeForContinuousVisualization(),
                                                    font_settings=FontSettings(), 
                                                    page_layout_settings=PageLayoutSettings(),
                                                    legend_settings=ContinuousLegendSettings())
    img = continuous_visualizer.visualize(data=DataForVisualization(tokens, values, weights), 
                                        show_text=False, visualize_weight=False, display_legend=True)
    img.save("test2.png")


def get_data(algorithm_name):
    # Load data
    with open('dataset/c4/processed_c4.json', 'r') as f:
        lines = f.readlines()
    item = json.loads(lines[0])
    prompt = item['prompt']

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transformers config
    transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("/data2/shared_model/facebook/opt-1.3b/").to(device),
                                            tokenizer=AutoTokenizer.from_pretrained("/data2/shared_model/facebook/opt-1.3b/"),
                                            vocab_size=50272,
                                            device=device,
                                            max_new_tokens=40,
                                            min_length=70,
                                            do_sample=True,
                                            no_repeat_ngram_size=4)
    
    # Load watermark algorithm
    myWatermark = AutoWatermark.load(f'{algorithm_name}', 
                                     algorithm_config=f'config/{algorithm_name}.json',
                                     transformers_config=transformers_config)
    
    # Generate watermarked and unwatermarked text
    watermarked_text = myWatermark.generate_watermarked_text(prompt)
    unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)

    # Truncate prompt
    editor = TruncatePromptTextEditor()
    watermarked_text = editor.edit(watermarked_text, prompt)
    unwatermarked_text = editor.edit(unwatermarked_text, prompt)

    # Get data for visualization
    watermarked_data = myWatermark.get_data_for_visualization(watermarked_text)
    unwatermarked_data = myWatermark.get_data_for_visualization(unwatermarked_text)
    
    return watermarked_data, unwatermarked_data


def test_visualization_without_weight(algorithm_name, visualize_type='discrete'):
    # Validate input
    assert visualize_type in ['discrete', 'continuous']
    if visualize_type == 'discrete':
        assert algorithm_name in ['KGW', 'Unigram', 'SWEET', 'UPV', 'SIR', 'XSIR', 'EWD', 'DIP', 'Unbiased']
    else:
        assert algorithm_name in ['EXP', 'EXPEdit', 'ITSEdit', 'EXPGumbel']

    # Get data for visualization
    watermarked_data, unwatermarked_data = get_data(algorithm_name)

    # Init visualizer
    if visualize_type == 'discrete':
        visualizer = DiscreteVisualizer(color_scheme=ColorSchemeForDiscreteVisualization(),
                                        font_settings=FontSettings(), 
                                        page_layout_settings=PageLayoutSettings(),
                                        legend_settings=DiscreteLegendSettings())
    else:
        visualizer = ContinuousVisualizer(color_scheme=ColorSchemeForContinuousVisualization(),
                                        font_settings=FontSettings(), 
                                        page_layout_settings=PageLayoutSettings(),
                                        legend_settings=ContinuousLegendSettings())
    
    # Visualize
    watermarked_img = visualizer.visualize(data=watermarked_data, 
                                           show_text=False, 
                                           visualize_weight=False, 
                                           display_legend=True)
    
    unwatermarked_img = visualizer.visualize(data=unwatermarked_data,
                                             show_text=False, 
                                             visualize_weight=False, 
                                             display_legend=True)
    
    # Save
    watermarked_img.save(f"{algorithm_name}_watermarked.png")
    unwatermarked_img.save(f"{algorithm_name}_unwatermarked.png")


def test_visualization_with_weight(algorithm_name):
    # Validate input
    assert algorithm_name in ['SWEET', 'EWD']

    # Get data for visualization
    watermarked_data, unwatermarked_data = get_data(algorithm_name)

    # Init visualizer
    visualizer = DiscreteVisualizer(color_scheme=ColorSchemeForDiscreteVisualization(),
                                    font_settings=FontSettings(), 
                                    page_layout_settings=PageLayoutSettings(),
                                    legend_settings=DiscreteLegendSettings())
    
    # Visualize
    watermarked_img = visualizer.visualize(data=watermarked_data, 
                                           show_text=True, 
                                           visualize_weight=True, 
                                           display_legend=True)
    
    unwatermarked_img = visualizer.visualize(data=unwatermarked_data,
                                             show_text=True, 
                                             visualize_weight=True, 
                                             display_legend=True)
    
    # Save
    watermarked_img.save(f"{algorithm_name}_watermarked.png")
    unwatermarked_img.save(f"{algorithm_name}_unwatermarked.png")


if __name__ == '__main__':
    test_discreet_visualization()
    test_continuous_visualization()
    test_visualization_without_weight('KGW', 'discrete')
    test_visualization_without_weight('EXP', 'continuous')
    test_visualization_with_weight('SWEET')
    test_visualization_with_weight('EWD')
