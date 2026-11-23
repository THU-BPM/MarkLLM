# TextSeal2 integration

This module adds [TextSeal2](https://arxiv.org/abs/2610.40613), a distortion-free two-key watermark from Meta's [TextSeal](https://github.com/facebookresearch/textseal) toolkit, to MarkLLM's `BaseWatermark` interface.
It also adds watermarked speculative decoding, which keeps the output law of the target model exactly.

## Method

**Generation.** For the window `w` of the last `ngram` tokens and each candidate token `v`, each key gives a uniform `u_k(w, v)` (53-bit PRF) and a Gaussian `G_k = Phi^-1(u_k)`.
The two keys are merged into a virtual key `G* = (G_A + G_B) / sqrt(2)`, blended with fresh noise, `z = rho G* + sqrt(1 - rho^2) eta`, and turned into an exponential clock `E_v = -log Phi(z)`.
The next token is the winner of the race `argmin_v E_v / p_v`, where `p` is the temperature and top-p law of the model.
The clocks are i.i.d. Exp(1) over keys and noise, so each token follows `p` exactly; `rho` trades diversity for detection strength.
With `repeated_windows: "skip"`, a window that already produced a keyed token in the answer is sampled with fresh clocks.

**Detection.** Each (window, token) pair is scored once with `s = -log Phi(-G*)`, which is Exp(1) without watermark, and the sum over `n` pairs is tested against Gamma(n, 1).
MarkLLM's `score` is `-log10(p_value)`, so larger values mean stronger evidence.

**Speculative decoding.** A drafter proposes `num_speculative_tokens` tokens per target call, each drawn by a race on the drafter's law.
The target verifies them with block verification (Sun et al., 2025): the number of kept drafts and the last token are drawn jointly by a race on the leftovers of the draft races (the residual race).
The output follows the target law exactly, the acceptance rate is exactly block verification's, and the emitted tokens stay keyed, so the same detector works.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype="bfloat16", device_map="cuda")
transformers_config = TransformersConfig(model=target, tokenizer=tokenizer, device="cuda",
                                         max_new_tokens=200, do_sample=True, temperature=1.0, top_p=0.95)

# Standard generation (Hugging Face generate with a logits processor).
watermark = AutoWatermark.load("TextSeal2", algorithm_config="config/TextSeal2.json",
                               transformers_config=transformers_config)

# Speculative decoding: pass a drafter that shares the tokenizer.
draft = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype="bfloat16", device_map="cuda")
watermark = AutoWatermark.load("TextSeal2", algorithm_config="config/TextSeal2.json",
                               transformers_config=transformers_config, draft_model=draft)

text = watermark.generate_watermarked_text(prompt)
watermark.detect_watermark(text)  # {"is_watermarked": ..., "score": -log10(p_value)}
```

`config/TextSeal2.json`:

| Field | Meaning |
|---|---|
| `ngram` | Window length (1 to 3; the paper uses 3) |
| `key_a`, `key_b` | The two secret keys |
| `rho` | Weight of the keyed Gaussian against fresh noise (paper: 0.7 to 0.99) |
| `repeated_windows` | `skip` (fresh sampling on a repeated window) or `reuse` (plain generation only) |
| `p_threshold` | Detection threshold on the p-value |
| `num_speculative_tokens` | Drafts per target call (speculative decoding only) |
| `compile` | Speculative decoding: run the model forwards through CUDA graphs (`torch.compile`); the first answers compile for about a minute |
| `default_temperature`, `default_top_p` | Used when `TransformersConfig` does not set them |

Generation, detection and visualization reuse MarkLLM's `TextSeal` class; detection scores each (window, token) pair once.
The speculative path uses `max_new_tokens`, `temperature` and `top_p` from `TransformersConfig` and ignores the other generation arguments.
It runs one prompt at a time with a static KV cache for both models (attention-only models such as Llama or Qwen3, transformers 5).
Hugging Face's eager forwards are bound by Python overhead, so a drafter forward costs almost as much as a target forward and speculative decoding only pays off with `compile`.
