# Copyright 2026 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from types import SimpleNamespace

import torch

from watermark.auto_config import config_name_from_alg_name
from watermark.auto_watermark import AutoWatermark, watermark_name_from_alg_name
from watermark.ssg.ssg import SSG, SSGConfig, SSGLogitsProcessor, SSGUtils
from utils.transformers_config import TransformersConfig


def _config(detection_method="KGW"):
    config = SSGConfig.__new__(SSGConfig)
    config.gamma = 0.5
    config.delta = 2.0
    config.hash_key = 15485863
    config.prefix_length = 1
    config.topk = 4
    config.vocab_size = 8
    config.device = "cpu"
    config.detection_method = detection_method
    config.detection_entropy_threshold = 0.9
    return config


def test_ssg_is_registered_with_auto_classes():
    assert config_name_from_alg_name("SSG") == "watermark.ssg.SSGConfig"
    assert watermark_name_from_alg_name("SSG") == "watermark.ssg.SSG"


def test_greenlist_is_balanced_exact_and_deterministic():
    utils = SSGUtils(_config())
    input_ids = torch.tensor([3, 5])
    scores = torch.tensor([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

    first_greenlist = utils.get_greenlist_ids(input_ids, scores)
    second_greenlist = utils.get_greenlist_ids(input_ids, scores)

    assert torch.equal(first_greenlist, second_greenlist)
    assert len(first_greenlist) == 4
    assert len(torch.unique(first_greenlist)) == 4
    assert sum(token in first_greenlist for token in (0, 1)) == 1
    assert sum(token in first_greenlist for token in (2, 3)) == 1


def test_logits_processor_partitions_each_batch_row_independently():
    config = _config()
    utils = SSGUtils(config)
    processor = SSGLogitsProcessor(config, utils)
    input_ids = torch.tensor([[1, 2], [6, 7]])
    scores = torch.tensor(
        [
            [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ]
    )
    original_scores = scores.clone()
    expected_greenlists = [
        utils.get_greenlist_ids(input_ids[index], original_scores[index])
        for index in range(2)
    ]

    processed_scores = processor(input_ids, scores)

    for index, greenlist in enumerate(expected_greenlists):
        expected_delta = torch.zeros(config.vocab_size)
        expected_delta[greenlist] = config.delta
        assert torch.equal(
            processed_scores[index] - original_scores[index], expected_delta
        )


def test_score_sequence_returns_aligned_finite_kgw_data():
    config = _config("KGW")
    utils = SSGUtils(config)
    input_ids = torch.tensor([1, 3, 4, 2])
    logits = torch.tensor(
        [
            [
                [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                [1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0],
                [5.0, 4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0],
                [8.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0],
            ]
        ]
    )
    entropy = [-10000.0, 0.5, 0.8, 0.7]

    score, flags, weights = utils.score_sequence(input_ids, entropy, logits)

    assert math.isfinite(score)
    assert len(flags) == len(input_ids)
    assert flags[0] == -1
    assert set(flags[1:]) <= {0, 1}
    assert weights == [-1.0, 1.0, 1.0, 1.0]


def test_zero_ewd_variance_produces_neutral_score():
    utils = SSGUtils(_config("EWD"))

    assert utils._compute_z_score(0.0, [0.0, 0.0]) == 0.0


class _Encoding(dict):
    def to(self, device):
        return _Encoding({key: value.to(device) for key, value in self.items()})


class _Tokenizer:
    def __call__(self, text, return_tensors, add_special_tokens):
        token_ids = [int(token) for token in text.split()]
        if add_special_tokens:
            token_ids.insert(0, 0)
        return _Encoding(input_ids=torch.tensor([token_ids]))

    def decode(self, token_id):
        return str(token_id)


class _Model:
    def __call__(self, input_ids, return_dict):
        base_logits = torch.arange(8, dtype=torch.float)
        logits = torch.stack(
            [base_logits * (index + 1) for index in range(input_ids.shape[1])]
        ).unsqueeze(0)
        return SimpleNamespace(logits=logits)


def test_detection_and_visualization_use_the_markllm_interfaces():
    transformers_config = TransformersConfig(
        model=_Model(), tokenizer=_Tokenizer(), vocab_size=8, device="cpu"
    )
    watermark = AutoWatermark.load(
        "SSG",
        algorithm_config="config/SSG.json",
        transformers_config=transformers_config,
    )

    result = watermark.detect_watermark("1 3 4")
    visualization = watermark.get_data_for_visualization("1 3 4")

    assert set(result) == {"is_watermarked", "score"}
    assert isinstance(watermark, SSG)
    assert isinstance(result["is_watermarked"], bool)
    assert math.isfinite(result["score"])
    assert visualization.decoded_tokens == ["0", "1", "3", "4"]
    assert len(visualization.highlight_values) == 4
    assert len(visualization.weights) == 4
