import os.path
import numpy as np

from visualize.color_scheme import ColorSchemeForDiscreteVisualization
from visualize.font_settings import FontSettings
from visualize.legend_settings import DiscreteLegendSettings
from visualize.page_layout_settings import PageLayoutSettings
from visualize.visualizer import DiscreteVisualizer
from vllm import LLM, SamplingParams

import gc
import sys
import json
import torch
from watermark.auto_watermark import AutoWatermarkForVLLM
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Clean gpu memory
assert torch.cuda.is_available()
gc.collect()
torch.cuda.empty_cache()
with torch.no_grad():
    torch.cuda.empty_cache()

# Load data
with open('dataset/c4/processed_c4.json', 'r') as f:
    lines = f.readlines()
    lines = [json.loads(line) for line in lines]


def main(algorithm_name, model_path):
    model = LLM(
        model=model_path, trust_remote_code=True,
        max_model_len=256,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
        dtype="auto",
        disable_custom_all_reduce=False,
        disable_log_stats=False,
        swap_space=32,
        seed=42
    )
    config = AutoConfig.from_pretrained(model_path)
    transformers_config = TransformersConfig(
        model=AutoModelForCausalLM.from_pretrained(model_path),
        tokenizer=AutoTokenizer.from_pretrained(model_path),
        vocab_size=config.vocab_size,
        device="cuda",
        max_new_tokens=256,
        max_length=256,
        do_sample=True,
        no_repeat_ngram_size=4
    )
    watermark = AutoWatermarkForVLLM(algorithm_name=algorithm_name, algorithm_config=f'config/{algorithm_name}.json', transformers_config=transformers_config)
    visualizer = DiscreteVisualizer(color_scheme=ColorSchemeForDiscreteVisualization(),
                                    font_settings=FontSettings(),
                                    page_layout_settings=PageLayoutSettings(),
                                    legend_settings=DiscreteLegendSettings())

    prompts = [line['prompt'] for line in lines]
    references = [line['natural_text'] for line in lines]

    # without watermark
    outputs = model.generate(
        prompts=prompts,
        sampling_params=SamplingParams(
            n=1, temperature=1.0, seed=42,
            max_tokens=256, min_tokens=16,
            logits_processors=[]
        ),
        use_tqdm=True,
    )
    nowatermark_text = [output.outputs[0].text for output in outputs]
    nowatermark_ppl = np.mean([-output.outputs[0].cumulative_logprob/len(output.outputs[0].token_ids) for output in outputs])
    nowatermark_detect_results = np.mean([r['is_watermarked'] for r in watermark.detect_watermark(nowatermark_text)])
    print(f"nowatermark_ppl: {nowatermark_ppl:.3f}")
    print(f"nowatermark_detect_results: {nowatermark_detect_results:.3f}")

    # with watermark
    outputs = model.generate(
        prompts=prompts,
        sampling_params=SamplingParams(
            n=1, temperature=1.0, seed=42,
            max_tokens=256, min_tokens=16,
            logits_processors=[watermark]
        ),
        use_tqdm=True,
    )
    watermark_text = [output.outputs[0].text for output in outputs]
    watermark_ppl = np.mean([-output.outputs[0].cumulative_logprob/len(output.outputs[0].token_ids) for output in outputs])
    watermark_detect_results = np.mean([r['is_watermarked'] for r in watermark.detect_watermark(watermark_text)])
    print(f"watermark_ppl: {watermark_ppl:.3f}")
    print(f"watermark_detect_results: {watermark_detect_results:.3f}")

    # visualize
    nowatermarked_img = visualizer.visualize(
        data=watermark.get_data_for_visualization(text=nowatermark_text[0]),
        show_text=True, visualize_weight=True, display_legend=True
    )
    nowatermarked_img.save(os.path.join(model_path, f"{algorithm_name}-nowatermark-vllm.png"))
    watermarked_img = visualizer.visualize(
        data=watermark.get_data_for_visualization(text=watermark_text[0]),
        show_text=True, visualize_weight=True, display_legend=True
    )
    watermarked_img.save(os.path.join(model_path, f"{algorithm_name}-watermark-vllm.png"))


if __name__ == "__main__":
    model_path = sys.argv[-2] # "meta-llama/Meta-Llama-3-8B-Instruct"
    method = sys.argv[-1] # "UPV" "KGW" "Unigram"
    main(model_path=model_path, algorithm_name=method)
    """
    --------------------------------------------------------------
    llama3-8b-instruct (vLLM)
                           KGW               UPV           Unigram
    PPL         1.191 -> 1.346    1.191 -> 0.926    1.191 -> 1.344
    detect      0.001 -> 0.929    0.001 -> 0.430    0.001 -> 0.508
    time (h)      0.19 -> 0.52      0.18 -> 2.02      0.18 -> 0.45
    --------------------------------------------------------------
    llama3-8b-instruct (huggingface)
                           KGW               UPV           Unigram
    detect      0.001 -> 0.934    0.001 -> 0.358    0.001 -> 0.505
    time (h)    20.00 -> 20.75    19.50 -> 21.50    20.50 -> 20.50
    --------------------------------------------------------------
    """
