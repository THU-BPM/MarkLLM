# Copyright 2026 THU-BPM MarkLLM.
# Licensed under the Apache License, Version 2.0.

import json
import math

import torch

from utils.transformers_config import TransformersConfig
from watermark.auto_config import config_name_from_alg_name
from watermark.auto_watermark import AutoWatermark, watermark_name_from_alg_name
from watermark.simmark import SimMark
from watermark.simmark.simmark import split_sentences


class _Encoding(dict):
    def to(self, device):
        return _Encoding({key: value.to(device) for key, value in self.items()})


class _Tokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __len__(self):
        return 16

    def encode(self, text, add_special_tokens=False):
        pieces = text.replace(".", " . ").split()
        token_ids = [15 if piece == "." else int(piece) % 15 for piece in pieces]
        if add_special_tokens:
            token_ids.insert(0, 0)
        return token_ids

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        token_ids = self.encode(text, add_special_tokens)
        if return_tensors == "pt":
            return _Encoding(input_ids=torch.tensor([token_ids], dtype=torch.long))
        return {"input_ids": token_ids}

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        pieces = []
        for token_id in token_ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id == 0:
                continue
            pieces.append("." if token_id == 15 else str(token_id))
        return " ".join(pieces)

    def batch_decode(self, sequences, skip_special_tokens=False):
        return [self.decode(sequence, skip_special_tokens) for sequence in sequences]


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def generate(self, input_ids, **kwargs):
        self.calls += 1
        candidate = torch.tensor([[3, 4, 15]], device=input_ids.device)
        return torch.cat([input_ids, candidate], dim=1)


class _Embedder:
    def __init__(self):
        self.calls = []

    def encode(
        self,
        sentences,
        prompt=None,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        self.calls.append(sentences)
        return torch.tensor([[1.0, 0.0] for _ in sentences])


def _load_watermark(tmp_path):
    config_path = tmp_path / "SimMark.json"
    config_path.write_text(
        json.dumps(
            {
                "algorithm_name": "SimMark",
                "embedding_model_name": "unused-in-tests",
                "embedding_prompt": "Represent the sentence:",
                "interval_low": 0.9,
                "interval_high": 1.0,
                "expected_valid_fraction": 0.25,
                "softness": 10.0,
                "z_threshold": 2.0,
                "max_trials": 3,
                "candidate_max_new_tokens": 4,
                "ensure_sentence_punctuation": True,
            }
        )
    )
    model = _Model()
    tokenizer = _Tokenizer()
    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=16,
        device="cpu",
        max_new_tokens=6,
        do_sample=True,
    )
    watermark = AutoWatermark.load(
        "SimMark",
        algorithm_config=str(config_path),
        transformers_config=transformers_config,
        embedding_model=_Embedder(),
    )
    return watermark, model


def test_simmark_is_registered_with_auto_classes():
    assert config_name_from_alg_name("SimMark") == ("watermark.simmark.SimMarkConfig")
    assert watermark_name_from_alg_name("SimMark") == ("watermark.simmark.SimMark")


def test_sentence_splitter_and_soft_interval_score(tmp_path):
    watermark, _ = _load_watermark(tmp_path)

    assert split_sentences("First sentence. Second sentence!") == [
        "First sentence.",
        "Second sentence!",
    ]
    assert watermark.utils.soft_score(0.95) == 1.0
    assert math.isclose(watermark.utils.soft_score(0.8), math.exp(-1.0))


def test_soft_z_detection_scores_sentence_pairs(tmp_path):
    watermark, _ = _load_watermark(tmp_path)

    result = watermark.utils.score_text(
        "First sentence. Second sentence. Third sentence."
    )

    expected = (2 - 0.25 * 2) / math.sqrt(2 * 0.25 * 0.75 + 1e-12)
    assert result.similarities == [1.0, 1.0]
    assert result.soft_scores == [1.0, 1.0]
    assert math.isclose(result.z_score, expected)
    assert watermark.utils.embedder.calls[-1] == [
        "First sentence.",
        " Second sentence.",
        " Third sentence.",
    ]


def test_generation_detection_and_visualization_use_markllm_interfaces(tmp_path):
    watermark, model = _load_watermark(tmp_path)

    generated = watermark.generate_watermarked_text("1 2 .")
    detection = watermark.detect_watermark(generated)
    visualization = watermark.get_data_for_visualization(generated)

    assert isinstance(watermark, SimMark)
    assert model.calls == 2
    assert len(split_sentences(generated)) == 3
    assert set(detection) == {"is_watermarked", "score"}
    assert detection["is_watermarked"] is True
    assert detection["score"] > 2.0
    assert visualization.highlight_values == [
        None,
        None,
        None,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert visualization.decoded_tokens == [
        "1 ",
        "2 ",
        ".",
        "3 ",
        "4 ",
        ".",
        "3 ",
        "4 ",
        ".",
    ]
