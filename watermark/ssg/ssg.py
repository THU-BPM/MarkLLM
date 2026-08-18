# Copyright 2026 Chenxi Gu, Xiaoning Du, and John Grundy.
# Copyright 2024 THU-BPM MarkLLM.
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

"""SSG: Logit-Balanced Vocabulary Partitioning for LLM Watermarking.

Adapted from the authors' implementation at https://github.com/Leileileisa/SSG.
"""

import math
from functools import partial

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from utils.transformers_config import TransformersConfig
from visualize.data_for_visualization import DataForVisualization
from watermark.base import BaseConfig, BaseWatermark


class SSGConfig(BaseConfig):
    """Configuration for the SSG algorithm."""

    def initialize_parameters(self) -> None:
        self.gamma = self.config_dict["gamma"]
        self.delta = self.config_dict["delta"]
        self.hash_key = self.config_dict["hash_key"]
        self.z_threshold = self.config_dict["z_threshold"]
        self.prefix_length = self.config_dict["prefix_length"]
        self.topk = self.config_dict["topk"]
        self.detection_method = self.config_dict["detection_method"].upper()
        self.detection_entropy_threshold = self.config_dict[
            "detection_entropy_threshold"
        ]

        if not 0 < self.gamma < 1:
            raise ValueError("gamma must be between 0 and 1")
        if self.prefix_length < 1:
            raise ValueError("prefix_length must be at least 1")
        if self.topk < 2 or self.topk > self.vocab_size or self.topk % 2:
            raise ValueError("topk must be an even integer between 2 and vocab_size")
        if self.detection_method not in {"EWD", "KGW", "SWEET"}:
            raise ValueError("detection_method must be EWD, KGW, or SWEET")

        target_green = int(self.vocab_size * self.gamma)
        paired_green = self.topk // 2
        tail_size = self.vocab_size - self.topk
        if not paired_green <= target_green <= paired_green + tail_size:
            raise ValueError(
                "gamma and topk cannot produce the requested green-list size"
            )

    @property
    def algorithm_name(self) -> str:
        return "SSG"


class SSGUtils:
    """Vocabulary partitioning and detection utilities for SSG."""

    def __init__(self, config: SSGConfig, *args, **kwargs) -> None:
        self.config = config
        self.rng = torch.Generator(device=self.config.device)
        alpha = math.exp(self.config.delta)
        self.z_value = ((1 - self.config.gamma) * (alpha - 1)) / (
            1 - self.config.gamma + alpha * self.config.gamma
        )

    def _seed_rng(
        self, input_ids: torch.LongTensor, first_token: int, second_token: int
    ) -> None:
        """Seed the RNG from the secret key, context, and a token pair."""
        context_hash = 1
        for token in input_ids[-self.config.prefix_length :]:
            context_hash *= token.item()
        context_hash %= self.config.vocab_size
        pair_hash = (first_token + second_token) % self.config.vocab_size
        self.rng.manual_seed(self.config.hash_key * context_hash + pair_hash)

    def get_greenlist_ids(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.LongTensor:
        """Create the logit-balanced green list for one generation step."""
        scores = scores.reshape(-1)
        if scores.numel() != self.config.vocab_size:
            raise ValueError(
                f"Expected {self.config.vocab_size} scores, got {scores.numel()}"
            )

        top_ids = torch.topk(
            scores.float(), k=self.config.topk, largest=True, sorted=True
        ).indices
        green_ids = []

        for index in range(0, self.config.topk, 2):
            first = int(top_ids[index].item())
            second = int(top_ids[index + 1].item())
            self._seed_rng(input_ids, first, second)
            choice = int(
                torch.randint(
                    0,
                    2,
                    (1,),
                    device=scores.device,
                    generator=self.rng,
                ).item()
            )
            green_ids.append(min(first, second) if choice == 0 else max(first, second))

        vocab_ids = torch.arange(
            self.config.vocab_size, device=scores.device, dtype=torch.long
        )
        tail_mask = torch.ones(
            self.config.vocab_size, device=scores.device, dtype=torch.bool
        )
        tail_mask[top_ids] = False
        tail_ids = vocab_ids[tail_mask]

        remaining_green = int(self.config.vocab_size * self.config.gamma) - len(
            green_ids
        )
        if remaining_green:
            self._seed_rng(input_ids, 0, 0)
            permutation = torch.randperm(
                tail_ids.numel(), device=scores.device, generator=self.rng
            )
            green_ids.extend(tail_ids[permutation[:remaining_green]].tolist())

        return torch.tensor(green_ids, device=scores.device, dtype=torch.long)

    def calculate_entropy(
        self, model, tokenized_text: torch.LongTensor
    ) -> tuple[list[float], torch.FloatTensor]:
        """Calculate EWD's entropy-derived weights and return the model logits."""
        with torch.no_grad():
            output = model(torch.unsqueeze(tokenized_text, 0), return_dict=True)
            probabilities = torch.softmax(output.logits, dim=-1)
            renormalized = probabilities / (1 + self.z_value * probabilities)
            entropy = renormalized.sum(dim=-1)[0].cpu().tolist()
            entropy.insert(0, -10000.0)
            return entropy[:-1], output.logits

    def _get_weights(self, entropy_list: list[float]) -> list[float]:
        entropy = torch.tensor(entropy_list)
        scored_entropy = entropy[self.config.prefix_length :]

        if self.config.detection_method == "EWD":
            scored_weights = scored_entropy - torch.min(scored_entropy)
        elif self.config.detection_method == "KGW":
            scored_weights = torch.ones_like(scored_entropy)
        else:
            scored_weights = torch.where(
                scored_entropy > self.config.detection_entropy_threshold,
                1.0,
                0.01,
            )

        return [-1.0] * self.config.prefix_length + scored_weights.tolist()

    def _compute_z_score(self, observed_count: float, weights: list[float]) -> float:
        weight_tensor = torch.tensor(weights, dtype=torch.float)
        variance = (
            torch.square(weight_tensor).sum()
            * self.config.gamma
            * (1 - self.config.gamma)
        )
        if variance <= 0:
            return 0.0
        expected_count = self.config.gamma * weight_tensor.sum()
        return ((observed_count - expected_count) / torch.sqrt(variance)).item()

    def score_sequence(
        self,
        input_ids: torch.LongTensor,
        entropy_list: list[float],
        scores: torch.FloatTensor,
    ) -> tuple[float, list[int], list[float]]:
        """Score a token sequence by reconstructing each SSG green list."""
        num_tokens_scored = len(input_ids) - self.config.prefix_length
        if num_tokens_scored < 1:
            raise ValueError(
                "Must have at least one token after the prefix required by SSG"
            )

        green_token_flags = [-1] * self.config.prefix_length
        for index in range(self.config.prefix_length, len(input_ids)):
            green_ids = self.get_greenlist_ids(
                input_ids[:index], scores[0, index - 1, :]
            )
            is_green = torch.any(green_ids == input_ids[index]).item()
            green_token_flags.append(int(is_green))

        weights = self._get_weights(entropy_list)
        observed_count = sum(
            weights[index] for index, flag in enumerate(green_token_flags) if flag == 1
        )
        z_score = self._compute_z_score(
            observed_count, weights[self.config.prefix_length :]
        )
        return z_score, green_token_flags, weights


class SSGLogitsProcessor(LogitsProcessor):
    """Apply SSG's logit-balanced green-list bias."""

    def __init__(self, config: SSGConfig, utils: SSGUtils, *args, **kwargs) -> None:
        self.config = config
        self.utils = utils

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        green_mask = torch.zeros_like(scores, dtype=torch.bool)
        for batch_index in range(input_ids.shape[0]):
            green_ids = self.utils.get_greenlist_ids(
                input_ids[batch_index], scores[batch_index]
            )
            green_mask[batch_index, green_ids] = True

        scores[green_mask] += self.config.delta
        return scores


class SSG(BaseWatermark):
    """Top-level SSG watermark implementation."""

    def __init__(
        self,
        algorithm_config: str | SSGConfig,
        transformers_config: TransformersConfig | None = None,
        *args,
        **kwargs,
    ) -> None:
        if isinstance(algorithm_config, str):
            self.config = SSGConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, SSGConfig):
            self.config = algorithm_config
        else:
            raise TypeError(
                "algorithm_config must be either a path string or an SSGConfig instance"
            )

        self.utils = SSGUtils(self.config)
        self.logits_processor = SSGLogitsProcessor(self.config, self.utils)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs,
        )
        encoded_prompt = self.config.generation_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.config.device)
        encoded_text = generate_with_watermark(**encoded_prompt)
        return self.config.generation_tokenizer.batch_decode(
            encoded_text, skip_special_tokens=True
        )[0]

    def detect_watermark(
        self, text: str, return_dict: bool = True, *args, **kwargs
    ) -> dict | tuple:
        encoded_text = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=True
        )["input_ids"][0].to(self.config.device)
        entropy_list, scores = self.utils.calculate_entropy(
            self.config.generation_model, encoded_text
        )
        z_score, _, _ = self.utils.score_sequence(encoded_text, entropy_list, scores)
        is_watermarked = z_score > self.config.z_threshold
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        return is_watermarked, z_score

    def get_data_for_visualization(
        self, text: str, *args, **kwargs
    ) -> DataForVisualization:
        encoded_text = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=True
        )["input_ids"][0].to(self.config.device)
        entropy_list, scores = self.utils.calculate_entropy(
            self.config.generation_model, encoded_text
        )
        _, highlight_values, weights = self.utils.score_sequence(
            encoded_text, entropy_list, scores
        )
        decoded_tokens = [
            self.config.generation_tokenizer.decode(token_id.item())
            for token_id in encoded_text
        ]
        return DataForVisualization(decoded_tokens, highlight_values, weights)
