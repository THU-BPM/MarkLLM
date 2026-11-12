# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2026 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0. This implementation adapts
# the TextSeal2 generation, detection and speculative decoding method. See NOTICE.

"""MarkLLM integration of TextSeal2, a distortion-free two-key copula watermark.

Each key k gives a uniform u_k(w, v) for the window w of the last `ngram` tokens and the candidate token v. The virtual
key G* = (Phi^-1(u_A) + Phi^-1(u_B)) / sqrt(2) is blended with fresh noise, z = rho G* + sqrt(1 - rho^2) eta, and turned
into an Exp(1) clock E_v = -log Phi(z). The next token wins the race argmin_v E_v / p_v, so it follows p exactly.
Detection scores each (window, token) pair once with -log Phi(-G*), Exp(1) without watermark (Gamma(n, 1) test).
Generation, detection and visualization reuse MarkLLM's TextSeal integration.
"""

import math

import torch
from scipy import special
from transformers import LogitsProcessor

from exceptions.exceptions import AlgorithmNameMismatchError
from utils.transformers_config import TransformersConfig
from watermark.base import BaseConfig
from watermark.textseal.textseal import TextSeal, TextSealSequenceScore


# 53-bit PRF of the TextSeal2 paper: splitmix64 of (window, token, key).
_WINDOW_MULTIPLIERS = [1000003, 998244353, 1000000007]
_HIRES_C1, _HIRES_C2 = -4658895280553007687, -7723592293110705685


class TextSeal2Config(BaseConfig):
    """Load TextSeal2 parameters. An optional `draft_model` keyword argument enables speculative decoding."""

    def initialize_parameters(self) -> None:
        if self.config_dict["algorithm_name"] != self.algorithm_name:
            raise AlgorithmNameMismatchError(self.algorithm_name, self.config_dict["algorithm_name"])
        self.draft_model = self.config_dict.pop("draft_model", None)
        config = self.config_dict
        self.ngram, self.key_a, self.key_b = int(config["ngram"]), int(config["key_a"]), int(config["key_b"])
        self.rho = float(config["rho"])
        self.repeated_windows = config.get("repeated_windows", "skip")
        self.p_threshold = float(config["p_threshold"])
        self.num_speculative_tokens = int(config.get("num_speculative_tokens", 4))
        self.compile = bool(config.get("compile", False))  # Speculative decoding: CUDA graphs through torch.compile.
        self.temperature = float(self.gen_kwargs.get("temperature", config.get("default_temperature", 1.0)))
        self.top_p = float(self.gen_kwargs.get("top_p", config.get("default_top_p", 0.95)))
        valid = (
            1 <= self.ngram <= len(_WINDOW_MULTIPLIERS) and self.key_a != self.key_b and 0 <= self.rho <= 1
            and self.repeated_windows in {"skip", "reuse"} and 0 < self.p_threshold < 1
            and self.num_speculative_tokens >= 1 and self.temperature > 0 and 0 < self.top_p <= 1
        )
        if not valid:
            raise ValueError(f"Invalid TextSeal2 configuration: {config}")

    @property
    def algorithm_name(self) -> str:
        return "TextSeal2"


def _shift_right(values: torch.Tensor, shift: int) -> torch.Tensor:
    return (values >> shift) & ((1 << (64 - shift)) - 1)


def keyed_uniform(windows: torch.Tensor, token_ids: torch.Tensor, key: int) -> torch.Tensor:
    """Keyed uniforms in (0, 1), float64. windows (..., ngram), token_ids (...) or (..., n) for n tokens per window."""
    windows, token_ids = windows.long(), token_ids.to(windows.device).long()
    hashed = (windows * torch.tensor(_WINDOW_MULTIPLIERS[: windows.shape[-1]], device=windows.device)).sum(-1)
    if hashed.dim() < token_ids.dim():
        hashed = hashed.unsqueeze(-1)
    hashed = hashed * 0x9E3779B9 + token_ids * 2654435761 + int(key) * 40503
    for _ in range(2):
        hashed = hashed + 7046029254386353131
        hashed = (hashed ^ _shift_right(hashed, 30)) * _HIRES_C1
        hashed = (hashed ^ _shift_right(hashed, 27)) * _HIRES_C2
        hashed = hashed ^ _shift_right(hashed, 31)
    return (_shift_right(hashed, 11).double() + 0.5) / float(1 << 53)


def virtual_key(windows: torch.Tensor, token_ids: torch.Tensor, keys: list[int], eps: float = 1e-15) -> torch.Tensor:
    """G* = sum_k Phi^-1(u_k) / sqrt(K), standard normal for each (window, token) pair."""
    return sum(torch.special.ndtri(keyed_uniform(windows, token_ids, key).clamp(eps, 1 - eps)) for key in keys) / math.sqrt(len(keys))


def copula_clocks(windows: torch.Tensor, token_ids: torch.Tensor, keys: list[int], rho: float) -> torch.Tensor:
    """Exp(1) clocks -log Phi(rho G* + sqrt(1 - rho^2) eta) with fresh eta. windows (b, ngram), token_ids (b, n) -> (b, n)."""
    z = virtual_key(windows, token_ids, keys)
    if rho < 1:
        z = rho * z + math.sqrt(1 - rho**2) * torch.randn_like(z)
    return -torch.special.log_ndtr(z)


def detection_scores(windows: torch.Tensor, token_ids: torch.Tensor, keys: list[int]) -> torch.Tensor:
    """Scores -log Phi(-G*), Exp(1) without watermark. windows (n, ngram), token_ids (n,) -> (n,)."""
    return -torch.special.log_ndtr(-virtual_key(windows, token_ids, keys, eps=1e-12))


def race(clocks: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """argmin clocks / probabilities over the support of probabilities, along the last dimension."""
    ratio = clocks / probabilities.double()
    return torch.argmin(torch.where(probabilities > 0, ratio, torch.full_like(ratio, float("inf"))), dim=-1)


def top_p_law(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Temperature and top-p law over the full vocabulary, as TextSeal's logits processor. (b, V) -> (b, V)."""
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_probabilities, sorted_ids = torch.sort(probabilities, dim=-1, descending=True)
    sorted_probabilities[torch.cumsum(sorted_probabilities, dim=-1) - sorted_probabilities > top_p] = 0.0
    sorted_probabilities /= sorted_probabilities.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probabilities).scatter_(-1, sorted_ids, sorted_probabilities)


class TextSeal2Utils:
    """Detection: each (window, token) pair is scored once, and the sum is tested against Gamma(n, 1)."""

    def __init__(self, config: TextSeal2Config) -> None:
        self.config = config

    def score_sequence(self, input_ids: torch.LongTensor) -> TextSealSequenceScore:
        ngram, token_list = self.config.ngram, input_ids.tolist()
        pairs = {tuple(token_list[pos - ngram : pos + 1]): pos for pos in reversed(range(ngram, len(token_list)))}
        positions = sorted(pairs.values())
        if not positions:
            return TextSealSequenceScore(1.0, 1.0, None, [], [], None)
        index = torch.tensor(positions, device=input_ids.device)
        windows = input_ids[index[:, None] + torch.arange(-ngram, 0, device=input_ids.device)]  # n ngram
        scores = detection_scores(windows, input_ids[index], [self.config.key_a, self.config.key_b])
        p_value = max(float(special.gammaincc(len(positions), float(scores.sum()))), 1e-300)
        return TextSealSequenceScore(p_value, p_value, None, scores.tolist(), positions, None)

    @staticmethod
    def token_percentiles(token_scores: list[float]) -> list[float]:
        return [1.0 - math.exp(-score) for score in token_scores]


class TextSeal2LogitsProcessor(LogitsProcessor):
    """Run the race on the temperature and top-p law and force the winner.

    With repeated_windows = "skip", a window that already produced a keyed token in the answer gets fresh Exp(1)
    clocks. Set `used_windows = None` before each generation.
    """

    def __init__(self, config: TextSeal2Config) -> None:
        self.config = config
        self.used_windows = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.config.ngram:
            return scores
        if self.used_windows is None:
            self.used_windows = [set() for _ in range(scores.shape[0])]
        sorted_probabilities, sorted_ids = torch.sort(top_p_law(scores, self.config.temperature, self.config.top_p), dim=-1, descending=True)
        support = int((sorted_probabilities > 0).sum(dim=-1).max())
        sorted_probabilities, sorted_ids = sorted_probabilities[:, :support], sorted_ids[:, :support]
        windows = input_ids[:, -self.config.ngram :]
        clocks = copula_clocks(windows, sorted_ids, [self.config.key_a, self.config.key_b], self.config.rho)
        for row, window in enumerate(map(tuple, windows.tolist())):
            if self.config.repeated_windows == "skip" and window in self.used_windows[row]:
                clocks[row] = torch.empty_like(clocks[row]).exponential_()
            self.used_windows[row].add(window)
        selected_ids = torch.gather(sorted_ids, 1, race(clocks, sorted_probabilities)[:, None])
        return torch.full_like(scores, float("-inf")).scatter_(1, selected_ids, 0.0)


class TextSeal2(TextSeal):
    """TextSeal2 watermark. With `draft_model=...` passed to `AutoWatermark.load`, generation uses watermarked
    speculative decoding (block verification with the residual race), which keeps the target law exactly."""

    def __init__(self, algorithm_config: str | TextSeal2Config, transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        if isinstance(algorithm_config, TextSeal2Config):
            self.config = algorithm_config
        else:
            self.config = TextSeal2Config(algorithm_config, transformers_config, **kwargs)
        self.utils = TextSeal2Utils(self.config)
        self.logits_processor = TextSeal2LogitsProcessor(self.config)
        self.speculative_generator = None
        if self.config.draft_model is not None:
            from watermark.textseal2.speculative import TextSeal2SpeculativeGenerator

            self.speculative_generator = TextSeal2SpeculativeGenerator(self.config.draft_model, self.config)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        if self.speculative_generator is None:
            self.logits_processor.used_windows = None
            return super().generate_watermarked_text(prompt, *args, **kwargs)
        tokenizer = self.config.generation_tokenizer
        prompt_ids = list(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        generated_ids = self.speculative_generator.generate(prompt_ids, int(self.config.gen_kwargs.get("max_new_tokens", 200)))
        return tokenizer.decode(prompt_ids + generated_ids, skip_special_tokens=True)
