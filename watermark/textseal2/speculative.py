# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2026 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0. See NOTICE.

"""Watermarked speculative decoding for TextSeal2: block verification with the residual race.

A call has rows 0..N: row i < N drew the draft d_i from the drafter law q_i, row N is the bonus row, and p_i is the
target law at row i. Each row has Exp(1) clocks E_i (keyed, or fresh when the row's window was already used in the
answer), and d_i = argmin E_i / q_i with time tau_i. Block verification (Sun et al., 2025) emits (L kept drafts, last
token X) with probability W_L R_L(X):
    r_0 = 1, r_i = min(1, r_{i-1} p_{i-1}(d_{i-1}) / q_{i-1}(d_{i-1})),  R_l = (r_l p_l - q_l)_+ / z_l (l < N), R_N = p_N,
    h_i = z_i / (z_i + 1 - r_i) (1 <= i < N), h_N = r_N,  W_l = h_l prod_{j > l} (1 - h_j).
The residual race draws (L, X) = argmin over (l, x) of E'_l(x) / (W_l R_l(x)) on the leftovers E'_i = E_i - tau_i q_i of
the draft races (E'_N = E_N): the output law and the acceptance are exactly block verification's, and the emitted
tokens stay keyed.
"""

import torch
from transformers import StaticCache

from watermark.textseal2.textseal2 import TextSeal2Config, copula_clocks, race, top_p_law


def block_law(target_probs: torch.Tensor, draft_probs: torch.Tensor, drafts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """target_probs (N+1, V), draft_probs (N, V), drafts (N,) -> W (N+1,), R (N+1, V), float64."""
    n_drafts = draft_probs.shape[0]
    rows = torch.arange(n_drafts, device=drafts.device)
    acceptance = [1.0]
    for ratio in (target_probs[rows, drafts] / draft_probs[rows, drafts]).tolist():
        acceptance.append(min(1.0, acceptance[-1] * ratio))
    acceptance = torch.tensor(acceptance, dtype=torch.float64, device=target_probs.device)  # N+1
    residuals = (acceptance[:n_drafts, None] * target_probs[:n_drafts] - draft_probs).clamp(min=0)  # N V
    mass = residuals.sum(1)
    laws = torch.cat([residuals / torch.where(mass > 0, mass, 1.0)[:, None], target_probs[n_drafts:]])  # N+1 V
    denominator = mass[1:] + (1 - acceptance[1:n_drafts])
    stop = torch.cat([torch.where(denominator > 0, mass[1:] / denominator, 1.0), acceptance[n_drafts:]])  # h_1..h_N
    survival = torch.flip(torch.cumprod(torch.flip(1 - stop, [0]), 0), [0])  # prod_{j >= i} (1 - h_j), i = 1..N
    weights = torch.cat([survival[:1], torch.cat([survival[1:], survival.new_ones(1)]) * stop])
    return weights, laws


def residual_race(target_probs: torch.Tensor, draft_probs: torch.Tensor, drafts: torch.Tensor, clocks: torch.Tensor) -> tuple[int, int]:
    """clocks (N+1, V): the row clocks the drafts raced on -> (kept drafts L, last token X)."""
    n_drafts = draft_probs.shape[0]
    weights, laws = block_law(target_probs, draft_probs, drafts)
    rows = torch.arange(n_drafts, device=drafts.device)
    leftovers = clocks.clone()
    leftovers[:n_drafts] -= (clocks[rows, drafts] / draft_probs[rows, drafts])[:, None] * draft_probs
    flat_index = int(race(leftovers.flatten(), (weights[:, None] * laws).flatten()))
    return flat_index // laws.shape[1], flat_index % laws.shape[1]


class CachedCausalLM:
    """A causal LM with a static KV cache. Rolling back only moves the write position: later writes overwrite the
    stale slots, which the causal mask hides until then. With `compile`, decoding runs through CUDA graphs."""

    def __init__(self, model, compile: bool) -> None:
        self.model, self.device = model, model.device
        self.cache, self.tokens = None, []
        self.decode = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True) if compile else model.forward

    @torch.no_grad()
    def feed(self, token_ids: list[int], eager: bool = False) -> torch.Tensor:
        """Append token_ids and return the logits after each of them, (len(token_ids), V)."""
        forward = self.model.forward if eager or not self.tokens else self.decode
        if forward is not self.model.forward:
            torch.compiler.cudagraph_mark_step_begin()
        # An all-ones padding mask over the whole static cache keeps the mask code free of data-dependent branches.
        logits = forward(input_ids=torch.tensor([token_ids], device=self.device), attention_mask=self.attention_mask,
                         past_key_values=self.cache, use_cache=True).logits[0]
        self.tokens = self.tokens + token_ids
        return logits

    @torch.no_grad()
    def sync(self, token_ids: list[int], max_length: int) -> None:
        """Make the cache hold exactly token_ids (crop to the common prefix, feed the rest), with room for max_length."""
        if self.cache is None or self.cache.layers[0].max_cache_len < max_length:
            self.cache, self.tokens = StaticCache(config=self.model.config, max_cache_len=max(max_length, 2048)), []
            self.attention_mask = torch.ones(1, self.cache.layers[0].max_cache_len, dtype=torch.long, device=self.device)
        common = next((ii for ii, (a, b) in enumerate(zip(self.tokens, token_ids)) if a != b), min(len(self.tokens), len(token_ids)))
        self.tokens = self.tokens[:common]
        for layer in self.cache.layers:
            layer.cumulative_length.fill_(common)
        if common < len(token_ids):
            self.feed(token_ids[common:], eager=len(token_ids) - common > 1)  # Prefills run eagerly.


class TextSeal2SpeculativeGenerator:
    """TextSeal2 speculative decoding, one prompt at a time. Every keyed row a call reads (drafts and bonus) marks its
    window as used, and later rows on a used window get fresh Exp(1) clocks."""

    def __init__(self, draft_model, config: TextSeal2Config) -> None:
        self.config = config
        self.draft = CachedCausalLM(draft_model, config.compile)
        self.target = CachedCausalLM(config.generation_model, config.compile)
        self.device = self.target.device
        eos = [getattr(config.generation_tokenizer, "eos_token_id", None), config.generation_model.generation_config.eos_token_id]
        self.eos_token_ids = {t for e in eos if e is not None for t in (e if isinstance(e, (list, tuple)) else [e])}
        self.last_stats: dict = {}

    def _law(self, logits: torch.Tensor) -> torch.Tensor:
        return top_p_law(logits, self.config.temperature, self.config.top_p).double()

    def _clocks(self, window: tuple, keyed: bool, vocab_size: int) -> torch.Tensor:
        if not keyed:
            return torch.empty(vocab_size, dtype=torch.float64, device=self.device).exponential_()
        token_ids = torch.arange(vocab_size, device=self.device)[None]
        return copula_clocks(torch.tensor([window], device=self.device), token_ids, [self.config.key_a, self.config.key_b], self.config.rho)[0]

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
        """Watermarked continuation of prompt_ids (EOS included if generated), at most max_new_tokens tokens."""
        ngram, n_max = self.config.ngram, self.config.num_speculative_tokens
        if len(prompt_ids) < ngram:
            raise ValueError(f"TextSeal2 speculative decoding needs a prompt of at least {ngram} tokens")
        committed, used_windows, stats = list(prompt_ids), set(), {"n_calls": 0, "proposed": 0, "accepted": 0}
        max_length = len(prompt_ids) + max_new_tokens + n_max + 1
        while len(committed) - len(prompt_ids) < max_new_tokens and not self.eos_token_ids & set(committed[len(prompt_ids):]):
            self.draft.sync(committed[:-1], max_length)
            self.target.sync(committed[:-1], max_length)
            drafts, draft_laws, clocks, windows, keyed = [], [], [], [], []

            def add_row(vocab_size: int) -> None:
                windows.append(tuple((committed + drafts)[-ngram:]))
                keyed.append(windows[-1] not in used_windows and windows[-1] not in windows[:-1])
                clocks.append(self._clocks(windows[-1], keyed[-1], vocab_size))

            while len(drafts) < n_max and not self.eos_token_ids & set(drafts[-1:]):
                draft_laws.append(self._law(self.draft.feed((committed + drafts)[-1:]))[0])  # V
                add_row(draft_laws[-1].shape[0])
                drafts.append(int(race(clocks[-1], draft_laws[-1])))
            n_drafts = len(drafts)
            # The target always reads n_max + 1 tokens (padded after an EOS draft), so compiled decoding sees one shape.
            target_laws = self._law(self.target.feed([committed[-1]] + drafts + drafts[-1:] * (n_max - n_drafts))[: n_drafts + 1])
            vocab_size = min(target_laws.shape[1], draft_laws[0].shape[0])
            add_row(vocab_size)
            kept, last_token = residual_race(
                target_laws[:, :vocab_size], torch.stack([law[:vocab_size] for law in draft_laws]),
                torch.tensor(drafts, device=self.device), torch.stack([row[:vocab_size] for row in clocks]),
            )
            used_windows |= {window for window, is_keyed in zip(windows, keyed) if is_keyed}
            committed += drafts[:kept] + [last_token]
            stats = {"n_calls": stats["n_calls"] + 1, "proposed": stats["proposed"] + n_drafts, "accepted": stats["accepted"] + kept}
        generated = committed[len(prompt_ids):][:max_new_tokens]
        generated = generated[: next((ii + 1 for ii, t in enumerate(generated) if t in self.eos_token_ids), len(generated))]
        self.last_stats = {**stats, "n_tokens": len(generated)}
        return generated
