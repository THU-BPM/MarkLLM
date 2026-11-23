# Copyright 2026 THU-BPM MarkLLM.
# Licensed under the Apache License, Version 2.0.

import itertools
import json
from collections import Counter

import torch
from scipy import stats
from transformers import LlamaConfig, LlamaForCausalLM

from test.test_textseal import _Tokenizer as _TextSealTokenizer
from utils.transformers_config import TransformersConfig
from watermark.auto_config import config_name_from_alg_name
from watermark.auto_watermark import AutoWatermark, watermark_name_from_alg_name
from watermark.textseal2.speculative import CachedCausalLM, TextSeal2SpeculativeGenerator, block_law, residual_race
from watermark.textseal2.textseal2 import TextSeal2LogitsProcessor, copula_clocks, detection_scores, keyed_uniform, race, top_p_law


class _Tokenizer(_TextSealTokenizer):
    eos_token_id = None


def _tiny_llama(seed: int, vocab_size: int = 16) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, initializer_range=0.5, bos_token_id=None, eos_token_id=None, pad_token_id=None)
    model = LlamaForCausalLM(config).eval()
    model.generation_config.eos_token_id, model.generation_config.pad_token_id = None, 0
    return model


def _load_watermark(tmp_path, model=None, draft_model=None, **overrides):
    config = {"algorithm_name": "TextSeal2", "ngram": 3, "key_a": 42, "key_b": 12387, "rho": 0.9, "repeated_windows": "skip",
              "p_threshold": 0.001, "num_speculative_tokens": 4, "compile": False, **overrides}
    (tmp_path / "TextSeal2.json").write_text(json.dumps(config))
    transformers_config = TransformersConfig(model=model or _tiny_llama(0), tokenizer=_Tokenizer(), vocab_size=16, device="cpu",
                                             max_new_tokens=40, do_sample=True, temperature=1.0, top_p=0.95, pad_token_id=0)
    return AutoWatermark.load("TextSeal2", algorithm_config=str(tmp_path / "TextSeal2.json"), transformers_config=transformers_config,
                              draft_model=draft_model)


def _chi_square_pvalue(observed: Counter, expected: dict) -> float:
    """Chi-squared test of counts against expected counts, pooling the cells expected below 5."""
    assert set(observed) <= set(expected), "impossible outcome"
    small = [key for key in expected if expected[key] < 5]
    pairs = [(observed.get(key, 0), expected[key]) for key in expected if expected[key] >= 5]
    pairs.append((sum(observed.get(key, 0) for key in small), sum(expected[key] for key in small)))
    statistic = sum((obs - exp) ** 2 / exp for obs, exp in pairs if exp > 0)
    return float(stats.chi2.sf(statistic, len(pairs) - 1))


def test_registration_and_prf_reference_values():
    assert config_name_from_alg_name("TextSeal2") == "watermark.textseal2.TextSeal2Config"
    assert watermark_name_from_alg_name("TextSeal2") == "watermark.textseal2.TextSeal2"
    windows, token_ids = torch.tensor([[1, 2, 3], [3, 2, 1], [151000, 7, 99]]), torch.tensor([5, 5, 151935])
    # Values of the TextSeal toolkit's 53-bit PRF (textseal.watermarking.core.prf_hires).
    assert keyed_uniform(windows, token_ids, 42).tolist() == [0.7209667037923795, 0.6275936084811751, 0.7611621051813764]
    assert keyed_uniform(windows, token_ids, 12387).tolist() == [0.14869118674528664, 0.07463465706171274, 0.15038475637104848]


def test_clocks_scores_and_race_are_unbiased():
    torch.manual_seed(0)
    windows, token_ids = torch.randint(0, 10**5, (4000, 3)), torch.randint(0, 10**5, (4000, 1))
    assert stats.kstest(keyed_uniform(windows, token_ids[:, 0], 7).numpy(), "uniform").pvalue > 1e-3
    for rho in (0.0, 0.9, 1.0):
        assert stats.kstest(copula_clocks(windows, token_ids, [42, 12387], rho)[:, 0].numpy(), "expon").pvalue > 1e-3
    assert stats.kstest(detection_scores(windows, token_ids[:, 0], [42, 12387]).numpy(), "expon").pvalue > 1e-3
    probabilities = torch.tensor([0.4, 0.25, 0.15, 0.1, 0.07, 0.03, 0.0, 0.0])
    clocks = copula_clocks(torch.randint(0, 10**6, (20000, 3)), torch.arange(8).expand(20000, -1), [42, 12387], 0.9)
    winners = Counter(race(clocks, probabilities.expand(20000, -1)).tolist())
    assert _chi_square_pvalue(winners, {tok: 20000 * float(p) for tok, p in enumerate(probabilities) if p > 0}) > 1e-3


def test_residual_race_follows_block_verification_exactly():
    # Law of (drafts, kept drafts, last token) against Q(drafts) W_L R_L(X), with fresh Exp(1) clocks per call.
    torch.manual_seed(2)
    vocab_size, n_drafts, n_calls = 4, 2, 20000
    target = torch.softmax(torch.randn(n_drafts + 1, vocab_size, dtype=torch.float64) * 1.3, 1)
    draft = torch.softmax(torch.log(target[:n_drafts]) + torch.randn(n_drafts, vocab_size, dtype=torch.float64), 1)
    target[1, 3], draft[0, 0] = 0.0, 0.0  # Top-p style zeros.
    target, draft = target / target.sum(1, keepdim=True), draft / draft.sum(1, keepdim=True)
    expected = {}
    for drafts in itertools.product(range(vocab_size), repeat=n_drafts):
        draft_prob = float(torch.prod(draft[torch.arange(n_drafts), torch.tensor(drafts)]))
        weights, laws = block_law(target, draft, torch.tensor(drafts))
        for kept, token in itertools.product(range(n_drafts + 1), range(vocab_size)):
            if draft_prob * float(weights[kept] * laws[kept, token]) > 0:
                expected[(drafts, kept, token)] = n_calls * draft_prob * float(weights[kept] * laws[kept, token])
    observed = Counter()
    for _ in range(n_calls):
        clocks = torch.empty(n_drafts + 1, vocab_size, dtype=torch.float64).exponential_()
        drafts = race(clocks[:n_drafts], draft)
        observed[(tuple(drafts.tolist()), *residual_race(target, draft, drafts, clocks))] += 1
    assert _chi_square_pvalue(observed, expected) > 1e-3


def test_detection_scores_each_pair_once_and_is_calibrated(tmp_path):
    watermark = _load_watermark(tmp_path)
    assert watermark.utils.score_sequence(torch.tensor([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 5])).token_positions == [3, 4, 5, 6, 11]
    assert watermark.utils.score_sequence(torch.tensor([1, 2, 3])).p_value == 1.0
    torch.manual_seed(4)
    null_p_values = [watermark.utils.score_sequence(torch.randint(0, 10**5, (60,))).p_value for _ in range(500)]
    assert stats.kstest(null_p_values, "uniform").pvalue > 1e-3


def test_logits_processor_forces_one_token_and_skips_repeated_windows(tmp_path):
    processor = TextSeal2LogitsProcessor(_load_watermark(tmp_path, rho=1.0).config)
    input_ids, scores = torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.linspace(-1, 1, 16).repeat(2, 1)
    processed = processor(input_ids, scores)
    assert torch.isfinite(processed).sum(dim=1).tolist() == [1, 1]
    assert processor.used_windows == [{(1, 2, 3)}, {(4, 5, 6)}]
    # With rho = 1 the keyed choice is deterministic; a repeated window samples with fresh clocks instead.
    assert torch.equal(processed.argmax(1), TextSeal2LogitsProcessor(processor.config)(input_ids, scores).argmax(1))
    assert len(Counter(processor(input_ids[:1], scores[:1]).argmax(1).item() for _ in range(200))) > 1


def test_cached_lm_rolls_back_to_the_full_forward():
    model = _tiny_llama(5)
    cached = CachedCausalLM(model, compile=False)
    cached.sync([1, 2, 3, 4], max_length=32)
    cached.feed([5, 6, 7])
    cached.sync([1, 2, 3, 4, 5, 9], max_length=32)  # Back to [1..5], then 9.
    with torch.no_grad():
        reference = model(torch.tensor([[1, 2, 3, 4, 5, 9, 10]])).logits[0, -1]
    assert torch.allclose(cached.feed([10])[-1], reference, atol=1e-4)
    assert cached.tokens == [1, 2, 3, 4, 5, 9, 10]


def test_plain_and_speculative_generation_are_detected(tmp_path):
    torch.manual_seed(6)
    plain = _load_watermark(tmp_path)
    speculative = _load_watermark(tmp_path, draft_model=_tiny_llama(1))
    unwatermarked_score = plain.detect_watermark(plain.generate_unwatermarked_text("1 2 3"))["score"]
    for watermark in (plain, speculative):
        generated = watermark.generate_watermarked_text("1 2 3")
        assert len(generated.split()) == 44  # Special token 0, prompt, 40 new tokens.
        assert watermark.detect_watermark(generated)["score"] > unwatermarked_score
        visualization = watermark.get_data_for_visualization(generated)
        assert visualization.highlight_values[:3] == [None] * 3  # No full window yet.
        assert all(value is None or 0 <= value <= 1 for value in visualization.highlight_values)
    run = speculative.speculative_generator.last_stats
    assert run["n_tokens"] == 40 and run["n_calls"] < 40 and 0 < run["accepted"] <= run["proposed"]


def test_speculative_generation_follows_the_target_law(tmp_path):
    # Joint law of the first two generated tokens, one key pair per answer, against the target model's exact law.
    vocab_size, prompt, n_answers = 6, [1, 2, 3], 1500
    target_model = _tiny_llama(10, vocab_size)
    watermark = _load_watermark(tmp_path, model=target_model, num_speculative_tokens=2)
    generator = TextSeal2SpeculativeGenerator(_tiny_llama(11, vocab_size), watermark.config)
    with torch.no_grad():
        first = top_p_law(target_model(torch.tensor([prompt])).logits[0, -1:], 1.0, 0.95)[0]
        second = top_p_law(target_model(torch.tensor([prompt + [x] for x in range(vocab_size)])).logits[:, -1], 1.0, 0.95)
    expected = {(a, b): n_answers * float(first[a] * second[a, b]) for a in range(vocab_size) for b in range(vocab_size) if first[a] * second[a, b] > 0}
    observed = Counter()
    torch.manual_seed(12)
    for answer in range(n_answers):
        watermark.config.key_a, watermark.config.key_b = 1000 + 2 * answer, 1001 + 2 * answer
        observed[tuple(generator.generate(prompt, 2))] += 1
    assert _chi_square_pvalue(observed, expected) > 1e-3
