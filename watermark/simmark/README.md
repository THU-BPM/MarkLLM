# SimMark integration

This module adapts the cosine-similarity variant of
[SimMark](https://github.com/DabiriAghdam/SimMark) to MarkLLM.

Generation proceeds one sentence at a time. Candidate sentences are generated
with the configured causal language model and accepted when their embedding
similarity with the previous sentence falls in the configured interval. The
detector applies SimMark's soft counting function and returns its z-score.

The default `config/SimMark.json` follows the official RealNews/C4 cosine
setting (`hkunlp/instructor-large`, interval `[0.68, 0.76]`, softness `K=250`).
`expected_valid_fraction` is the empirical probability that a human sentence
pair falls inside the interval. The upstream evaluation estimates it on a
domain-matched human calibration set; users should likewise recalibrate this
field when changing the domain, embedder, or interval.

The cosine variant does not require the optional PCA checkpoint used by the
paper's Euclidean variant.
