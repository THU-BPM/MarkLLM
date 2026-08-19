# Copyright 2025 Meta Platforms, Inc. and affiliates.
# Copyright 2026 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0. This implementation adapts
# the TextSeal dual-key generation and global detection method. See NOTICE.

"""MarkLLM integration of TextSeal dual-key Gumbel-max watermarking."""

from dataclasses import dataclass
from functools import partial
import math

import numpy as np
from scipy import special
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from exceptions.exceptions import AlgorithmNameMismatchError
from utils.transformers_config import TransformersConfig
from visualize.data_for_visualization import DataForVisualization
from watermark.base import BaseConfig, BaseWatermark


_PRIMES = [
    10000019,
    10000247,
    10000439,
    10000643,
    10000747,
    10000867,
    10000993,
    10001213,
    10001357,
    10001501,
]
_P2 = 100000007
_P3 = 500001713
_P4 = 15485863
_M = 2**13 - 1
_MIXING_PRIME = 40499
_MIXING_SHIFT = 13


class TextSealConfig(BaseConfig):
    """Load TextSeal algorithm and detection parameters."""

    def initialize_parameters(self) -> None:
        if self.config_dict["algorithm_name"] != self.algorithm_name:
            raise AlgorithmNameMismatchError(
                self.algorithm_name, self.config_dict["algorithm_name"]
            )

        self.ngram = int(self.config_dict["ngram"])
        self.key_a = int(self.config_dict["key_a"])
        self.key_b = int(self.config_dict["key_b"])
        self.mixing_alpha = float(self.config_dict["mixing_alpha"])
        self.p_threshold = float(self.config_dict["p_threshold"])
        self.scoring_method = self.config_dict.get("scoring_method", "v2")
        self.use_entropy_weighting = bool(
            self.config_dict.get("use_entropy_weighting", True)
        )
        self.default_temperature = float(
            self.config_dict.get("default_temperature", 0.8)
        )
        self.default_top_p = float(self.config_dict.get("default_top_p", 0.95))
        self.min_p_value = float(self.config_dict.get("min_p_value", 1e-300))
        self.temperature = float(
            self.gen_kwargs.get("temperature", self.default_temperature)
        )
        self.top_p = float(self.gen_kwargs.get("top_p", self.default_top_p))

        if not 1 <= self.ngram <= len(_PRIMES):
            raise ValueError(f"TextSeal ngram must be in [1, {len(_PRIMES)}]")
        if not 0 < self.mixing_alpha < 1:
            raise ValueError("TextSeal mixing_alpha must be in (0, 1)")
        if self.key_a == self.key_b:
            raise ValueError("TextSeal key_a and key_b must be different")
        if not 0 < self.p_threshold < 1:
            raise ValueError("TextSeal p_threshold must be in (0, 1)")
        if self.scoring_method not in {"none", "v1", "v2"}:
            raise ValueError("TextSeal scoring_method must be none, v1, or v2")
        if self.temperature <= 0:
            raise ValueError("TextSeal requires a positive sampling temperature")
        if not 0 < self.top_p <= 1:
            raise ValueError("TextSeal top_p must be in (0, 1]")
        if not 0 < self.min_p_value <= 1:
            raise ValueError("TextSeal min_p_value must be in (0, 1]")

    @property
    def algorithm_name(self) -> str:
        return "TextSeal"


def _weighted_context(context: torch.Tensor) -> torch.Tensor:
    primes = torch.tensor(
        _PRIMES[: context.shape[-1]], dtype=torch.long, device=context.device
    )
    return (context.long() * primes).sum(dim=-1)


def textseal_prf(
    context: torch.Tensor, token_ids: torch.Tensor, secret_key: int
) -> torch.Tensor:
    """Return TextSeal's uniform PRF value for context/token pairs."""
    weighted = _weighted_context(context)
    while weighted.ndim < token_ids.ndim:
        weighted = weighted.unsqueeze(-1)
    hashed = (weighted + _P2 * token_ids.long() + _P3 * int(secret_key)) * _P4
    hashed = hashed * _MIXING_PRIME
    hashed = hashed ^ (hashed >> _MIXING_SHIFT)
    return (hashed % _M).float() / _M


def textseal_prf_dual(
    context: torch.Tensor,
    token_ids: torch.Tensor,
    key_a: int,
    key_b: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PRF values for both TextSeal keys."""
    return (
        textseal_prf(context, token_ids, key_a),
        textseal_prf(context, token_ids, key_b),
    )


@dataclass
class TextSealSequenceScore:
    """Document and token-level TextSeal detection evidence."""

    p_value: float
    p_value_unweighted: float
    p_value_weighted: float | None
    token_scores: list[float]
    token_positions: list[int]
    entropy_weights: list[float] | None


class TextSealUtils:
    """Hashing and global Gamma-test detection helpers."""

    def __init__(self, config: TextSealConfig) -> None:
        self.config = config

    @property
    def base_variance(self) -> float:
        alpha = self.config.mixing_alpha
        return alpha**2 + (1 - alpha) ** 2

    def _entropy_values(self, input_ids: torch.LongTensor) -> list[float] | None:
        if not self.config.use_entropy_weighting or input_ids.numel() <= 1:
            return None
        with torch.inference_mode():
            logits = self.config.generation_model(input_ids.unsqueeze(0)).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            probabilities = log_probs.exp()
            entropies = -(probabilities * log_probs).sum(dim=-1)
        return entropies.squeeze(0)[:-1].cpu().tolist()

    @staticmethod
    def _entropy_weights(entropies: list[float]) -> list[float]:
        entropy_min = min(entropies)
        entropy_max = max(entropies)
        if entropy_max - entropy_min < 1e-6:
            entropy_min, entropy_max = 0.0, 5.0
        denominator = entropy_max - entropy_min
        return [
            0.1 + 0.9 * max(0.0, min(1.0, (value - entropy_min) / denominator))
            for value in entropies
        ]

    def score_sequence(self, input_ids: torch.LongTensor) -> TextSealSequenceScore:
        """Compute fused dual-key scores and the document p-value."""
        start_position = self.config.ngram + 1
        if input_ids.numel() <= start_position:
            return TextSealSequenceScore(1.0, 1.0, None, [], [], None)

        entropies = self._entropy_values(input_ids)
        contexts = []
        targets = []
        token_positions = []
        scored_entropies = []
        seen = set()

        for position in range(start_position, input_ids.numel()):
            context = input_ids[position - self.config.ngram : position]
            target = input_ids[position]
            if self.config.scoring_method == "v1":
                dedup_key = tuple(context.tolist())
            elif self.config.scoring_method == "v2":
                dedup_key = tuple(context.tolist()) + (int(target),)
            else:
                dedup_key = None
            if dedup_key is not None:
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

            contexts.append(context)
            targets.append(target)
            token_positions.append(position)
            if entropies is not None and position - 1 < len(entropies):
                scored_entropies.append(entropies[position - 1])

        if not targets:
            return TextSealSequenceScore(1.0, 1.0, None, [], [], None)

        context_tensor = torch.stack(contexts)
        target_tensor = torch.stack(targets)
        r_a, r_b = textseal_prf_dual(
            context_tensor,
            target_tensor,
            self.config.key_a,
            self.config.key_b,
        )
        score_a = -torch.log1p(-r_a.clamp(max=1 - 1e-10))
        score_b = -torch.log1p(-r_b.clamp(max=1 - 1e-10))
        fused = (
            self.config.mixing_alpha * score_a
            + (1 - self.config.mixing_alpha) * score_b
        )
        token_scores = fused.cpu().tolist()
        token_count = len(token_scores)
        score_sum = float(fused.sum().item())
        p_unweighted = float(
            special.gammaincc(
                token_count / self.base_variance,
                score_sum / self.base_variance,
            )
        )

        weights = None
        p_weighted = None
        if len(scored_entropies) == token_count:
            weights = self._entropy_weights(scored_entropies)
            weights_array = np.asarray(weights, dtype=np.float64)
            scores_array = np.asarray(token_scores, dtype=np.float64)
            weighted_sum = float(np.sum(weights_array * scores_array))
            mean = float(np.sum(weights_array))
            variance = float(np.sum(weights_array**2) * self.base_variance)
            if variance > 1e-10 and mean > 0:
                shape = mean**2 / variance
                scale = variance / mean
                p_weighted = float(special.gammaincc(shape, weighted_sum / scale))

        candidates = [p_unweighted]
        if p_weighted is not None:
            candidates.append(p_weighted)
        p_value = max(min(candidates), self.config.min_p_value)
        return TextSealSequenceScore(
            p_value,
            p_unweighted,
            p_weighted,
            token_scores,
            token_positions,
            weights,
        )

    def token_percentiles(self, token_scores: list[float]) -> list[float]:
        """Map raw evidence to a [0, 1] null percentile for visualization."""
        shape = 1 / self.base_variance
        return [
            float(special.gammainc(shape, score / self.base_variance))
            for score in token_scores
        ]


class TextSealLogitsProcessor(LogitsProcessor):
    """Perform dual-key Gumbel-max sampling and force the selected token."""

    def __init__(self, config: TextSealConfig) -> None:
        self.config = config

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.config.ngram:
            return scores

        probabilities = torch.softmax(scores / self.config.temperature, dim=-1)
        sorted_probabilities, sorted_ids = torch.sort(
            probabilities, dim=-1, descending=True
        )
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove_mask = cumulative - sorted_probabilities > self.config.top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove_mask, 0.0)
        sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(
            dim=-1, keepdim=True
        )

        context = input_ids[:, -self.config.ngram :]
        r_a, r_b = textseal_prf_dual(
            context,
            sorted_ids,
            self.config.key_a,
            self.config.key_b,
        )
        use_key_a = (
            torch.rand(scores.shape[0], device=scores.device) < self.config.mixing_alpha
        )
        random_values = torch.where(use_key_a.unsqueeze(1), r_a, r_b)
        selection_scores = torch.log(random_values + 1e-30) / (
            sorted_probabilities + 1e-30
        )
        selected_rank = torch.argmax(selection_scores, dim=-1, keepdim=True)
        selected_ids = torch.gather(sorted_ids, 1, selected_rank)

        forced_scores = torch.full_like(scores, float("-inf"))
        forced_scores.scatter_(1, selected_ids, 0.0)
        return forced_scores


class TextSeal(BaseWatermark):
    """TextSeal dual-key generation-time watermark."""

    def __init__(
        self,
        algorithm_config: str | TextSealConfig,
        transformers_config: TransformersConfig | None = None,
        *args,
        **kwargs,
    ) -> None:
        if isinstance(algorithm_config, str):
            self.config = TextSealConfig(
                algorithm_config, transformers_config, **kwargs
            )
        elif isinstance(algorithm_config, TextSealConfig):
            self.config = algorithm_config
        else:
            raise TypeError(
                "algorithm_config must be a path string or TextSealConfig instance"
            )
        self.utils = TextSealUtils(self.config)
        self.logits_processor = TextSealLogitsProcessor(self.config)

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

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        input_ids = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(self.config.device)
        result = self.utils.score_sequence(input_ids)
        is_watermarked = bool(result.p_value < self.config.p_threshold)
        score = float(-math.log10(result.p_value))
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": score}
        return is_watermarked, score

    def get_data_for_visualization(
        self, text: str, *args, **kwargs
    ) -> DataForVisualization:
        input_ids = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(self.config.device)
        result = self.utils.score_sequence(input_ids)
        values = [None] * input_ids.numel()
        for position, value in zip(
            result.token_positions,
            self.utils.token_percentiles(result.token_scores),
        ):
            values[position] = value
        decoded_tokens = [
            self.config.generation_tokenizer.decode(int(token_id))
            for token_id in input_ids
        ]
        return DataForVisualization(decoded_tokens, values)
