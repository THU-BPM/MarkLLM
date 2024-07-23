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
    
    # Load watermark algorithm
    myWatermark = AutoWatermark.load(f'{algorithm_name}', 
                                     algorithm_config=f'config/{algorithm_name}.json',
                                     transformers_config=transformers_config)

    #for prompt, natural_text in zip(prompts, natural_texts):
    with open('output.txt', 'a') as file:
            file.write('gen started!' + '\n\n')  # 写入解码后的文本并换行
            
            
    #prompts[0] = 'Let\'s generate a random texts with over 500 words.'
    watermarked_text = myWatermark.generate_watermarked_text(prompts[0])
    print(watermarked_text[len(prompts[0]):])
    # unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)
    with open('output.txt', 'a') as file:
            file.write('\n\ngen end!' + '\n\n')  # 写入解码后的文本并换行
            file.write('detect started!\n\n')
    detect_result = myWatermark.detect_watermark(watermarked_text[len(prompts[0]):])
    
    print(detect_result)
    # detect_result = myWatermark.detect_watermark(unwatermarked_text)
    # print(detect_result)
    detect_result = myWatermark.detect_watermark(natural_texts[0])
    print(detect_result)
    
    print()


if __name__ == '__main__':
    test_algorithm('DIP')


