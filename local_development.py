import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark

# Load watermark algorithm
device = "cuda" if torch.cuda.is_available() else "cpu"
transformers_config = TransformersConfig(
    model=AutoModelForCausalLM.from_pretrained('facebook/opt-1.3b').to(device),
    tokenizer=AutoTokenizer.from_pretrained('facebook/opt-1.3b'),
    vocab_size=50272,
    device=device,
    max_new_tokens=200,
    min_length=230,
    do_sample=True,
    no_repeat_ngram_size=4)
myWatermark = AutoWatermark.load('KGW',
                                 algorithm_config='config/KGW.json',
                                 transformers_config=transformers_config)

# Prompt
prompt = 'Good Morning.'

# Generate text
watermarked_text = myWatermark.generate_watermarked_text(prompt)
unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)

# Detect
detect_result1 = myWatermark.detect_watermark(watermarked_text)
detect_result2 = myWatermark.detect_watermark(unwatermarked_text)
detect_result3 = myWatermark.detect_watermark(prompt)

print("LLM-generated watermarked text:")
print(watermarked_text)
print('\n')
print(detect_result1)
print('\n')

print("LLM-generated unwatermarked text:")
print(unwatermarked_text)
print('\n')
print(detect_result2)
print('\n')

print("Natural text:")
print(prompt)
print('\n')
print(detect_result3)
