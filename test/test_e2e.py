# Copyright 2026 THU-BPM MarkLLM.
# Licensed under the Apache License, Version 2.0.

import json
import math

import torch

from utils.transformers_config import TransformersConfig
from watermark.auto_config import config_name_from_alg_name
from watermark.auto_watermark import AutoWatermark, watermark_name_from_alg_name
from watermark.e2e import E2E
from watermark.e2e.model import E2EDetector, E2EEncoder


class _Encoding(dict):
    def to(self, device):
        return _Encoding({key: value.to(device) for key, value in self.items()})


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self, vocabulary_prefix="token"):
        self.vocabulary_prefix = vocabulary_prefix

    def __len__(self):
        return 8

    def get_vocab(self):
        return {f"{self.vocabulary_prefix}-{index}": index for index in range(8)}

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        token_ids = [int(token) % 8 for token in text.split() if token.isdigit()]
        if add_special_tokens:
            token_ids.insert(0, self.eos_token_id)
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


class _Model(torch.nn.Module):
    def __init__(self, embedding_dim=4):
        super().__init__()
        self.embeddings = torch.nn.Embedding(8, embedding_dim)

    def get_input_embeddings(self):
        return self.embeddings

    def generate(self, input_ids, logits_processor=None, max_new_tokens=2, **kwargs):
        output = input_ids.clone()
        base_scores = torch.arange(8, dtype=torch.float).repeat(output.shape[0], 1)
        for _ in range(max_new_tokens):
            scores = base_scores.clone()
            if logits_processor is not None:
                scores = logits_processor(output, scores)
            next_token = scores.argmax(dim=-1, keepdim=True)
            output = torch.cat([output, next_token], dim=-1)
        return output


def _write_checkpoint(path):
    torch.manual_seed(7)
    encoder = E2EEncoder(input_dim=4, mapper_layers=2, window_size=3, hidden_dim=4)
    detector = E2EDetector(input_dim=4, hidden_dim=4, num_layers=1)
    torch.save({"enc": encoder.state_dict(), "dec": detector.state_dict()}, path)


def _write_config(path, checkpoint_path, tokenizer_conversion_multiplier=2):
    path.write_text(
        json.dumps(
            {
                "algorithm_name": "E2E",
                "delta": 1.25,
                "top_k": 4,
                "window_size": 3,
                "detection_threshold": 0.5,
                "checkpoint_path": str(checkpoint_path),
                "reference_model_name": "unused-in-tests",
                "reference_tokenizer_name": "unused-in-tests",
                "tokenizer_conversion_multiplier": tokenizer_conversion_multiplier,
                "mapper_layers": 2,
                "hidden_dim": 4,
                "detector_layers": 1,
            }
        )
    )


def _load_watermark(tmp_path, generation_tokenizer=None, reference_tokenizer=None):
    checkpoint_path = tmp_path / "checkpoint.pth"
    config_path = tmp_path / "E2E.json"
    _write_checkpoint(checkpoint_path)
    _write_config(config_path, checkpoint_path)
    generation_tokenizer = generation_tokenizer or _Tokenizer()
    reference_tokenizer = reference_tokenizer or generation_tokenizer
    model = _Model()
    transformers_config = TransformersConfig(
        model=model,
        tokenizer=generation_tokenizer,
        vocab_size=8,
        device="cpu",
        max_new_tokens=2,
    )
    return AutoWatermark.load(
        "E2E",
        algorithm_config=str(config_path),
        transformers_config=transformers_config,
        reference_model=model,
        reference_tokenizer=reference_tokenizer,
    )


def test_e2e_is_registered_with_auto_classes():
    assert config_name_from_alg_name("E2E") == "watermark.e2e.E2EConfig"
    assert watermark_name_from_alg_name("E2E") == "watermark.e2e.E2E"


def test_logits_processor_handles_each_batch_row(tmp_path):
    watermark = _load_watermark(tmp_path)
    input_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
    scores = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]] * 2)

    processed = watermark.logits_processor(input_ids, scores)
    differences = processed - scores

    assert processed.shape == scores.shape
    assert torch.count_nonzero(differences[0, :4]) == 0
    assert torch.count_nonzero(differences[0, 4:]) > 0
    assert not torch.equal(differences[0], differences[1])


def test_generation_detection_and_visualization_use_markllm_interfaces(tmp_path):
    watermark = _load_watermark(tmp_path)

    generated = watermark.generate_watermarked_text("1 2 3")
    result = watermark.detect_watermark(generated)
    visualization = watermark.get_data_for_visualization(generated)

    assert isinstance(watermark, E2E)
    assert len(generated.split()) == 6
    assert set(result) == {"is_watermarked", "score"}
    assert isinstance(result["is_watermarked"], bool)
    assert math.isfinite(result["score"])
    assert 0 <= result["score"] <= 1
    assert len(visualization.decoded_tokens) == 6
    assert len(visualization.highlight_values) == 6
    assert all(0 <= value <= 1 for value in visualization.highlight_values)


def test_different_tokenizers_use_reference_conversion(tmp_path):
    watermark = _load_watermark(
        tmp_path,
        generation_tokenizer=_Tokenizer("generation"),
        reference_tokenizer=_Tokenizer("reference"),
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    scores = torch.arange(8, dtype=torch.float).unsqueeze(0)

    processed = watermark.logits_processor(input_ids, scores)

    assert watermark.utils.same_tokenizer is False
    assert watermark.utils.prefix_length == 6
    assert processed.shape == scores.shape
    assert not torch.equal(processed, scores)
