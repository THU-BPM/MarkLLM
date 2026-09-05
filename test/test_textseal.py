# Copyright 2026 THU-BPM MarkLLM.
# Licensed under the Apache License, Version 2.0.

import json
import math

import torch

from utils.transformers_config import TransformersConfig
from watermark.auto_config import config_name_from_alg_name
from watermark.auto_watermark import AutoWatermark, watermark_name_from_alg_name
from watermark.textseal import TextSeal
from watermark.textseal.textseal import textseal_prf, textseal_prf_dual


class _Encoding(dict):
    def to(self, device):
        return _Encoding({key: value.to(device) for key, value in self.items()})


class _Tokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __len__(self):
        return 16

    def get_vocab(self):
        return {str(index): index for index in range(16)}

    def encode(self, text, add_special_tokens=False):
        result = [int(token) % 16 for token in text.split() if token.isdigit()]
        if add_special_tokens:
            result.insert(0, 0)
        return result

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        token_ids = self.encode(text, add_special_tokens)
        if return_tensors == "pt":
            return _Encoding(input_ids=torch.tensor([token_ids], dtype=torch.long))
        return {"input_ids": token_ids}

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        elif isinstance(token_ids, torch.Tensor) and token_ids.ndim == 0:
            token_ids = [int(token_ids)]
        return " ".join(str(int(token_id)) for token_id in token_ids)

    def batch_decode(self, sequences, skip_special_tokens=False):
        return [self.decode(sequence, skip_special_tokens) for sequence in sequences]


class _Output:
    def __init__(self, logits):
        self.logits = logits


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        vocabulary_logits = torch.linspace(-1, 1, 16, device=input_ids.device)
        logits = vocabulary_logits.expand(input_ids.shape[0], input_ids.shape[1], -1)
        return _Output(logits)

    def generate(self, input_ids, logits_processor=None, max_new_tokens=4, **kwargs):
        output = input_ids.clone()
        vocabulary_logits = torch.linspace(-1, 1, 16).expand(output.shape[0], -1)
        for _ in range(max_new_tokens):
            scores = vocabulary_logits.clone()
            if logits_processor is not None:
                scores = logits_processor(output, scores)
            output = torch.cat([output, scores.argmax(dim=-1, keepdim=True)], dim=-1)
        return output


def _load_watermark(tmp_path):
    config_path = tmp_path / "TextSeal.json"
    config_path.write_text(
        json.dumps(
            {
                "algorithm_name": "TextSeal",
                "ngram": 1,
                "key_a": 42,
                "key_b": 12387,
                "mixing_alpha": 0.5,
                "p_threshold": 0.01,
                "scoring_method": "v2",
                "use_entropy_weighting": False,
                "default_temperature": 0.8,
                "default_top_p": 0.95,
                "min_p_value": 1e-300,
            }
        )
    )
    tokenizer = _Tokenizer()
    transformers_config = TransformersConfig(
        model=_Model(),
        tokenizer=tokenizer,
        vocab_size=16,
        device="cpu",
        max_new_tokens=4,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
    )
    return AutoWatermark.load(
        "TextSeal",
        algorithm_config=str(config_path),
        transformers_config=transformers_config,
    )


def test_textseal_is_registered_with_auto_classes():
    assert config_name_from_alg_name("TextSeal") == (
        "watermark.textseal.TextSealConfig"
    )
    assert watermark_name_from_alg_name("TextSeal") == ("watermark.textseal.TextSeal")


def test_prf_is_deterministic_order_sensitive_and_keyed():
    contexts = torch.tensor([[1, 2], [2, 1]])
    targets = torch.tensor([3, 3])

    first = textseal_prf(contexts, targets, 42)
    second = textseal_prf(contexts, targets, 42)
    key_a, key_b = textseal_prf_dual(contexts, targets, 42, 12387)

    assert torch.equal(first, second)
    assert first[0] != first[1]
    assert torch.equal(first, key_a)
    assert not torch.equal(key_a, key_b)
    assert torch.all((0 <= first) & (first < 1))


def test_logits_processor_forces_one_token_per_batch_row(tmp_path):
    torch.manual_seed(4)
    watermark = _load_watermark(tmp_path)
    input_ids = torch.tensor([[1, 2], [2, 1]])
    scores = torch.linspace(-1, 1, 16).repeat(2, 1)

    processed = watermark.logits_processor(input_ids, scores)

    assert processed.shape == scores.shape
    assert torch.isfinite(processed).sum(dim=1).tolist() == [1, 1]
    assert processed.argmax(dim=1).tolist()[0] != processed.argmax(dim=1).tolist()[1]


def test_v2_detection_deduplicates_context_target_pairs(tmp_path):
    watermark = _load_watermark(tmp_path)
    input_ids = torch.tensor([1, 2, 3, 2, 3])

    result = watermark.utils.score_sequence(input_ids)

    assert result.token_positions == [2, 3]
    assert len(result.token_scores) == 2
    assert 0 < result.p_value <= 1


def test_generation_detection_and_visualization_use_markllm_interfaces(tmp_path):
    torch.manual_seed(8)
    watermark = _load_watermark(tmp_path)

    generated = watermark.generate_watermarked_text("1 2 3")
    detection = watermark.detect_watermark(generated)
    visualization = watermark.get_data_for_visualization("1 2 3 2 3")

    assert isinstance(watermark, TextSeal)
    assert len(generated.split()) == 8
    assert set(detection) == {"is_watermarked", "score"}
    assert isinstance(detection["is_watermarked"], bool)
    assert math.isfinite(detection["score"])
    assert visualization.highlight_values[:2] == [None, None]
    assert visualization.highlight_values[4] is None
    assert all(
        value is None or 0 <= value <= 1 for value in visualization.highlight_values
    )
