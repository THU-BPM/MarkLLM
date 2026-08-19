# TextSeal integration

This module adapts Meta's
[TextSeal](https://github.com/facebookresearch/textseal) dual-key
generation-time watermark to MarkLLM's `BaseWatermark` interface.

At each generation step, it applies the official dual-key PRF and Gumbel-max
selection rule, then forces the selected token through Hugging Face generation.
Detection fuses evidence from both keys and applies the paper's moment-matched
Gamma test. MarkLLM's returned `score` is `-log10(p_value)`, so larger values
consistently mean stronger watermark evidence.

`config/TextSeal.json` controls:

- `ngram`, `key_a`, `key_b`, and `mixing_alpha` for generation;
- `p_threshold` and `scoring_method` for detection;
- optional entropy weighting with the generation model;
- fallback temperature and top-p values when they are not supplied through
  `TransformersConfig`.

This focused integration provides standard generation, global detection, and
token visualization. TextSeal's speculative decoding and localized multi-region
API remain available in the upstream toolkit but are outside MarkLLM's current
single-text `BaseWatermark` contract.
