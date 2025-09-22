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

# ============================================
# steal.py
# Description: Implementation of STEAL algorithm
# ============================================

import torch
from functools import partial
import math
from ..base import BaseWatermark, BaseConfig
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    AutoTokenizer,
    AutoModelForMaskedLM,
)
from .countstore import CountStore

class STEALConfig(BaseConfig):
    """Config class for STEAL algorithm"""

    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        # Necessary Parameters
        self.prevctx_width = self.config_dict["prevctx_width"]
        self.counts_base = self.config_dict["counts_base"]
        self.counts_wm = self.config_dict["counts_wm"]

        # Control Parameters
        self.spoofer_strength = self.config_dict["spoofer_strength"]
        self.ordered = self.config_dict["ordered"]
        self.w_abcd = self.config_dict["w_abcd"]
        self.w_partials = self.config_dict["w_partials"]
        self.w_empty = self.config_dict["w_empty"]
        self.w_future = self.config_dict["w_future"]
        self.future_num_cands = self.config_dict["future_num_cands"]
        self.future_prefix_size = self.config_dict["future_prefix_size"]
        self.future_local_w_decay = self.config_dict["future_local_w_decay"]
        self.future_num_options_per_fillin = self.config_dict[
            "future_num_options_per_fillin"
        ]
        self.prevent_eos_if_zest_bad = self.config_dict["prevent_eos_if_zest_bad"]
        self.ppl_model_name_or_path = self.config_dict["ppl_model_name_or_path"]
        self.min_wm_mass_empty = self.config_dict["min_wm_mass_empty"]
        self.min_wm_count_nonempty = self.config_dict["min_wm_count_nonempty"]
        self.predict_gamma = self.config_dict["predict_gamma"]
        self.use_predict = self.config_dict["use_predict"]
        self.clip_at = self.config_dict["clip_at"]

        # Generate config
        self.end_inst_tag = "[/INST]"
        self.generation_tokenizer.period_token_id = self.generation_tokenizer(
            "Game over.", add_special_tokens=False
        ).input_ids[-1]

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return "STEAL"


class STEALUtils:
    """Utility class for STEAL algorithm, contains helper functions."""

    def __init__(self, config: STEALConfig, *args, **kwargs) -> None:
        """
        Initialize the STEAL utility class.

        Parameters:
            config (STEALConfig): Configuration for the STEAL algorithm.
        """
        self.config = config

    def get_counts(self, counts_file):
        """Load counts from the specified file."""
        dest_counts = CountStore(self.config.prevctx_width)
        dest_counts.init_from_file(counts_file)
        return dest_counts

class CustomNgramRepetitionPenaltyProcessor(LogitsProcessor):
    """Custom N-gram Repetition Penalty LogitsProcessor for the STEAL algorithm."""
    def __init__(self, *args: any, n: int, penalty: float, endinst_pattern: torch.Tensor, mode: str, **kwargs: any):
        """
        Initialize the Custom N-gram Repetition Penalty logits processor.
        Parameters:
            n (int): The size of the n-grams to consider for repetition penalty.
            penalty (float): The penalty factor to apply to repeated n-grams.
            endinst_pattern (torch.Tensor): Token pattern indicating the end of an instruction.
            mode (str): Mode of applying the penalty, either "divide" or "subtract".
        """
        super().__init__(*args, **kwargs)
        self.n = n
        self.penalty = penalty
        self.endinst_pattern = endinst_pattern
        self.mode = mode
        self.cached_response_start = [None for _ in range(2048)]

    # Confirm that input_ids[idx] is exactly the start of assistant response
    # TODO change annotation to LongTensor when torch bugs are understood
    def _is_response_start(self, input_ids: torch.Tensor, idx: int) -> bool:
        if self.endinst_pattern.device != input_ids.device:
            self.endinst_pattern = self.endinst_pattern.to(input_ids.device)
        return idx >= len(self.endinst_pattern) and bool(
            (
                input_ids[idx - len(self.endinst_pattern) : idx] == self.endinst_pattern
            ).all()
        )

    def __call__(
        self, input_ids: torch.LongTensor, logits: torch.Tensor
    ) -> torch.Tensor:
        # input_ids: (B, maxlen) | logits: (B, vocab_size)
        # Find bad tokens (those that would complete a repeated n-gram) in the response part (!)
        # Primitive string matching with no batching; improved with caching
        # NOTE Assumes: 1 occurence of endinst_pattern, chat models, single turn

        banned_toks = []
        for b in range(input_ids.shape[0]):
            # Find response start
            response_start = -1
            cached = self.cached_response_start[b]
            if cached is not None and self._is_response_start(input_ids[b], cached):
                response_start = cached
            else:
                response_start = input_ids.shape[1]
                while response_start > -1 and not self._is_response_start(
                    input_ids[b], response_start
                ):
                    response_start -= 1
            if response_start == -1:
                # Perhaps this is pre-response
                banned_toks.append(
                    torch.tensor([], dtype=torch.int, device=logits.device)
                )
                continue

            # Find banned tokens
            curr_banned_toks = []
            if self.n == 1:
                ctx = None  # GPT WM
            else:
                ctx = input_ids[b, -(self.n - 1) :]
            shifted_ids = [input_ids[b, response_start + i :] for i in range(self.n)]
            for ngram in zip(*shifted_ids):
                if (
                    ctx is None
                    or (torch.tensor(ngram[:-1], device=logits.device) == ctx).all()
                ):
                    curr_banned_toks.append(ngram[-1].item())
            banned_toks.append(
                torch.tensor(curr_banned_toks, dtype=torch.int, device=logits.device)
            )

        # Penalize the bad tokens
        for b, curr in enumerate(banned_toks):
            if self.mode == "divide":
                banned_slice = logits[b, curr]
                logits[b, curr] = torch.where(
                    banned_slice < 0,
                    banned_slice * self.penalty,
                    banned_slice / self.penalty,
                )
            elif self.mode == "subtract":
                logits[b, curr] -= self.penalty
            else:
                raise ValueError(
                    f"Uknown repetition penalty processor mode: {self.mode}"
                )

        return logits


class GracefulConclusionProcessor(LogitsProcessor):
    """Graceful Conclusion LogitsProcessor for the STEAL algorithm."""
    def __init__(self, *args: any, period_token: int, eos_token: int, panic_from: int, **kwargs: any):
        """
            Initialize the Graceful Conclusion logits processor.

            Parameters:
                period_token (int): Token ID for the period ('.') token.
                eos_token (int): Token ID for the end-of-sequence token.
                panic_from (int): Token position from which to enforce EOS if no period has been generated.
        """

        super().__init__(*args, **kwargs)
        self.period_token = period_token
        self.eos_token = eos_token
        self.panic_from = panic_from
        print(f"GracefulConclusionProcessor instantiated with: {self.__dict__}")

    def __call__(
        self, input_ids: torch.LongTensor, logits: torch.Tensor
    ) -> torch.Tensor:
        for b in range(input_ids.shape[0]):
            # Do not terminate the text unless we hit a period
            if input_ids[b][-1] != self.period_token:
                logits[b][self.eos_token] = -1e7
            else:
                # Period emitted, if it's after 700 toks we force terminate
                if input_ids.shape[1] > self.panic_from:
                    logits[b][self.eos_token] = 1e7
        return logits


class SpoofedProcessor(LogitsProcessor):
    """Spoofed LogitsProcessor for the STEAL algorithm."""
    def __init__(self, config: STEALConfig, utils: STEALUtils, *args: any, **kwargs: any) -> None:
        """
            Initialize the STEAL logits processor.

            Parameters:
                config (STEALConfig): Configuration for the KGW algorithm.
                utils (STEALUtils): Utility class for the KGW algorithm.
        """

        self.config = config
        self.utils = utils
        # print(f"SpoofedProcessor instantiated with: {cfg}")
        self.counts_base = self.utils.get_counts(self.config.counts_base)
        self.counts_wm = self.utils.get_counts(self.config.counts_wm)
        self.prevctx_width = self.config.prevctx_width
        self.vocab_size = self.config.vocab_size  # unused, see note below
        self.tokenizer = self.config.generation_tokenizer
        self.emptyctx = tuple([-1 for _ in range(self.prevctx_width)])
        if self.config.w_future > 0:
            self.ppl_model = AutoModelForMaskedLM.from_pretrained(
                self.config.ppl_model_name_or_path
            )
            self.ppl_tokenizer = AutoTokenizer.from_pretrained(
                self.config.ppl_model_name_or_path
            )

            # Check if tokenizer has pad token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            spec = self.tokenizer.special_tokens_map
            self.spec_ids = self.tokenizer(
                list(spec.values()), add_special_tokens=False
            )["input_ids"]
            self.spec_ids = list(set([x[0] for x in self.spec_ids]))

        # Cache for boosts
        self.boosts_cache: dict[str, torch.Tensor] = {}

        # Estimates
        self.green_estimates: list[list[float]] = []
        self.z_estimates: list[list[float]] = []

    def gen_clear(self):
        """
            Clear internal states. Should be called between different generations. 
        """
        # Cache for boosts
        self.boosts_cache: dict[str, torch.Tensor] = {}

        # Estimates
        self.green_estimates: list[list[float]] = []
        self.z_estimates: list[list[float]] = []
        torch.cuda.empty_cache()

    def _compute_future_token_ppl(
        self,
        pre_ids: torch.Tensor,
        post_ids: torch.Tensor,
        device: str,
        num_options_per_token: int = 10,
        num_tokens: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Given a batch of pre_ids and post_ids, compute the perplexity of the top num_options_per_token tokens for each masked token.
        Due to tokenization errors some may be masked away (probs=0) so it's not necessarily top num_options_per_token best choices;
        for similar reason they may not be sorted by probability, so use with care.

        Args:
            pre_ids: (batch_size, any_length)   # Generally assumed to be of equal length
            post_ids: (batch_size, any_length)  # Generally assumed to be of length 1 - The green token
            num_options_per_token: int          # Number of options to return for each masked token
            num_tokens: int                     # Number of masked tokens to consider

        Returns:
            resulting_tokens: (batch_size, num_tokens, num_options_per_token)    # The token ids of the top num_options_per_token tokens for each masked token
            resulting_probs: (batch_size, num_tokens, num_options_per_token)     # The probabilities of the top num_options_per_token tokens for each masked token
        """
        self.ppl_model.to(device)

        b = pre_ids.shape[0]

        pre_text = self.tokenizer.batch_decode(pre_ids, skip_special_tokens=True)
        post_text = self.tokenizer.batch_decode(post_ids, skip_special_tokens=True)

        input_txt = [x + "<mask>" * num_tokens + y for x, y in zip(pre_text, post_text)]

        inputs = self.ppl_tokenizer(input_txt, return_tensors="pt", padding=True).to(
            device
        )
        predictions = self.ppl_model(**inputs)[0]

        # Find the positions of the <mask> tokens
        mask = (inputs.input_ids == self.ppl_tokenizer.mask_token_id).unsqueeze(-1)

        selected_tokens = torch.masked_select(predictions, mask).reshape(
            (b, num_tokens, -1)
        )
        st_probs = torch.nn.functional.softmax(selected_tokens, dim=-1)
        # Format [batch_size, num_masked_tokens, logits]

        top_probs, top_tokens = torch.topk(
            st_probs, k=num_options_per_token, sorted=True, dim=-1
        )
        # Format [batch_size, num_masked_tokens, num_options_per_token]

        # Decode the tokens
        top_strings = self.ppl_tokenizer.batch_decode(
            top_tokens.reshape((-1, 1)), skip_special_tokens=True
        )
        assert len(top_strings) == b * num_tokens * num_options_per_token
        top_strings = [
            top_strings[i * num_options_per_token : (i + 1) * num_options_per_token]
            for i in range(b * num_tokens)
        ]

        # Re-tokenize with the other tokenizer
        top_tokenized = [
            self.tokenizer(
                x, return_tensors="pt", padding=True, add_special_tokens=False
            )
            for x in top_strings
        ]

        resulting_tokens = torch.zeros_like(top_probs, dtype=torch.long, device=device)
        resulting_probs = torch.zeros_like(top_probs, dtype=torch.float, device=device)

        spec_ids = torch.tensor(self.spec_ids, dtype=torch.long, device=device)
        for i in range(b):
            for j in range(num_tokens):
                comb_index = i * num_tokens + j
                curr_tokens = top_tokenized[comb_index].input_ids.to(
                    device
                )  # [options, 1?]

                # Kick out (set prob=0) those where
                # i) tokenizer didn't produce a single token
                # ii) the candidate is a special token
                good_mask = torch.ones(
                    (num_options_per_token,), dtype=torch.bool, device=device
                )
                if curr_tokens.shape[1] > 1:
                    expected_nb_pad = curr_tokens.shape[1] - 1
                    nb_pad = torch.sum(
                        curr_tokens == self.tokenizer.pad_token_id, dim=-1
                    )
                    good_mask &= nb_pad == expected_nb_pad
                good_mask &= ~torch.isin(curr_tokens[:, -1], spec_ids)

                # Extract and mask out bad ones
                resulting_tokens[i, j, :] = curr_tokens[:, -1]
                resulting_probs[i, j, :] = top_probs[i, j, :]
                resulting_probs[i, j, ~good_mask] = 0.0

        return resulting_tokens, resulting_probs

    # Given an ordered (tok|-1){3} or unordered (tok){0,3} context return a combined and
    # normalized boost vector with boosts from 0 to 1 for each token in the vocab
    def get_boosts(
        self,
        ctx: tuple,
        vocab_sz: int,
        ordered: bool,
        device: str,
        normalize: bool = True,
    ) -> torch.Tensor:
        # Get counts and mass for base and watermarked
        counts_base: dict[int, int] = self.counts_base.get(ctx, ordered)
        counts_base_tensor = torch.sparse_coo_tensor(
            torch.tensor([list(counts_base.keys())]),
            list(counts_base.values()),
            [vocab_sz],
            device=device,
        ).to_dense()
        mass_base = counts_base_tensor / (counts_base_tensor.sum() + 1e-6)

        counts_wm: dict[int, int] = self.counts_wm.get(ctx, ordered)
        counts_wm_tensor = torch.sparse_coo_tensor(
            torch.tensor([list(counts_wm.keys())]),
            list(counts_wm.values()),
            [vocab_sz],
            device=device,
        ).to_dense()
        mass_wm = counts_wm_tensor / (counts_wm_tensor.sum() + 1e-6)

        # NOTE: ft unused
        # counts_wm_ft: dict[int, int] = self.counts_wm_ft.get(ctx, ordered)

        # Compute ratios, small/0 is bad, large/0 is very good
        if (ordered and ctx == self.emptyctx) or (not ordered and len(ctx) == 0):
            min_data_thresh = round(
                self.config.min_wm_mass_empty * counts_base_tensor.sum().item()
            )
        else:
            min_data_thresh = self.config.min_wm_count_nonempty
        enough_data_mask = counts_wm_tensor >= min_data_thresh
        base_zero_mask = counts_base_tensor == 0

        # Get mass ratios, handles nonexistence nicely
        ratios = torch.zeros_like(mass_wm)  # stays 0 where not enough data
        core_mask = enough_data_mask & ~base_zero_mask
        ratios[core_mask] = mass_wm[core_mask] / mass_base[core_mask]
        ratios[enough_data_mask & base_zero_mask] = max(1, ratios.max().item()) + 1e-3

        if normalize:
            # Compute boost values by clipping
            ratios[ratios < 1] = 0
            ratios[ratios > self.config.clip_at] = self.config.clip_at
            ratios /= self.config.clip_at  # now 0 or 0.5-1
            # Tie break by wm counts (important for future)
            if ratios.max() > 0:
                ratios[ratios > 0] += (
                    counts_wm_tensor[ratios > 0] / counts_wm_tensor[ratios > 0].max()
                ) * 1e-4
                ratios = ratios / ratios.max()  # Should still be [0,1]
        else:
            if ratios.max() > 0:
                ratios = ratios / ratios.max()

        return ratios

    # Optimization: if needed boosts are cached skip the call to .get_boosts()
    # Esp. useful for empty tuple that never changes (but useful also for pairings)
    def _get_boosts_with_cache(
        self, ctx: tuple, vocab_sz: int, ordered: bool, device: str
    ) -> torch.Tensor:
        k = str((ctx, ordered))
        if k not in self.boosts_cache:
            self.boosts_cache[k] = self.get_boosts(ctx, vocab_sz, ordered, device)
        return self.boosts_cache[k]

    def __call__(  # noqa: C901
        self, input_ids: torch.LongTensor, logits: torch.Tensor
    ) -> torch.Tensor:
        # Check if there is enough input_ids to apply the processor
        if input_ids.shape[1] < self.prevctx_width:
            return logits

        # input_ids: (B, maxlen) | logits: (B, vocab_size)
        # NOTE: this can be >self.vocab_size (32128 vs 32100 for T5, last 28 ignored)
        device = str(logits.device)
        vocab_sz = logits.shape[1]
        ordered = self.config.ordered
        for b in range(input_ids.shape[0]):

            # Boosts per token are a weighted sum of several contribs
            boosts = torch.zeros((vocab_sz,), device=device)
            total_w = 0.0
            ctx = tuple()  # gptwm
            if self.prevctx_width > 0:
                ctx = tuple(input_ids[b][-self.prevctx_width :].cpu().tolist())

            # 1) {ABC}->D: most precise but generally sparse   abcd 4-gram abc->d
            boosts_abcd = self._get_boosts_with_cache(
                tuple(sorted(ctx)), vocab_sz, ordered, device
            )
            boosts += self.config.w_abcd * boosts_abcd
            total_w += self.config.w_abcd

            if self.config.w_partials > 0:
                # 2) Find S, the strongest among {ABC} by looking at {AB}->D and then add {S}->D
                # (so either D is weak and (S,D) is good or D is strong and (D,D) is good) -> both ok
                solo_boosts = []  # single token boosts, without any context
                for tok in ctx:
                    solo_boosts.append(
                        self._get_boosts_with_cache((tok,), vocab_sz, ordered, device)
                    )

                # pair 2-gram, a->d
                pair_boosts: list[list[torch.Tensor]] = [
                    [torch.tensor([]) for _ in range(len(ctx))] for _ in range(len(ctx))
                ]
                for i, toki in enumerate(ctx):
                    for delta, tokj in enumerate(ctx[i + 1 :]):
                        j = i + 1 + delta
                        pair_ctx = tuple(sorted([toki, tokj]))  # type: ignore
                        pair_boosts[i][j] = self._get_boosts_with_cache(
                            pair_ctx, vocab_sz, False, device
                        )
                        pair_boosts[j][i] = pair_boosts[i][j]

                # To be strong you need to have higher cossim with pair boost than all
                winner = -1
                cossim = F.cosine_similarity
                for i, toki in enumerate(ctx):
                    is_strong = True
                    for j, tokj in enumerate(ctx):
                        if i == j:
                            continue
                        cossim_i = cossim(
                            pair_boosts[i][j], solo_boosts[i], dim=0
                        ).item()
                        cossim_j = cossim(
                            pair_boosts[i][j], solo_boosts[j], dim=0
                        ).item()
                        is_strong &= cossim_i > cossim_j
                    if is_strong:
                        if winner == -1:
                            winner = i
                        else:
                            winner = -2
                            break

                # If there was a unique winner add its boost
                if winner > -1:
                    boosts += self.config.w_partials * solo_boosts[winner]
                    total_w += self.config.w_partials

            if self.config.w_empty > 0:
                # 3) Just use the empty context {}->D for an additional ctx-independent boost
                # (finding D that are strong + true)
                boosts_empty = self._get_boosts_with_cache(
                    tuple(), vocab_sz, ordered, device
                )
                boosts += self.config.w_empty * boosts_empty
                total_w += self.config.w_empty

            # future / lookahead
            # Q above: "what D to pick to maximize getting green on D"
            # Q below: "what D to pick to also maximize getting green on E/F/G" (should matter)
            # 1) Get some (wrt watermark only) possible choices for _E (based on BC_) or __F (C__) or ___G (___)
            # 2) Ask roberta to fill in the middle with a few choices --> boost the next token that leads to that

            # not use Future
            if self.config.w_future > 0:
                bsz = self.config.future_num_cands
                future_ctx = list(ctx)
                boosts_future = torch.zeros_like(boosts)
                total_local_w = 0.0
                local_w = 1.0

                # add +1 to range end to also consider [-1 -1 -1], now we dont
                for nb_skipped in range(1, self.prevctx_width):
                    # Get some possible choices from the counts -> if we have some evidence that
                    # in the future we can score we should use that info
                    future_ctx = future_ctx[1:] + [-1]  # add another token of future
                    tmp_boosts = self._get_boosts_with_cache(
                        tuple(future_ctx), vocab_sz, True, device
                    )
                    top_boosts, top_toks = torch.topk(tmp_boosts, k=bsz, sorted=True)
                    top_toks = top_toks[
                        top_boosts > 0
                    ]  # only candidates that got some boost
                    curr_bsz = top_toks.shape[0]  # may be < bsz

                    if curr_bsz == 0:
                        local_w *= self.config.future_local_w_decay
                        continue  # no meaningful candidates for BERT

                    # Prepare prefix/suffix pairs for BERT
                    prefix = input_ids[b][
                        -min(self.config.future_prefix_size, input_ids.shape[1]) :
                    ]
                    prefix_batch = prefix.repeat(curr_bsz, 1).to(device)
                    suffix_batch = top_toks.view(curr_bsz, 1).to(device)

                    # Get potential fillins (NB some probs may be 0 as invalid)
                    pot_tokens, pot_probs = self._compute_future_token_ppl(
                        prefix_batch,
                        suffix_batch,
                        device=device,
                        num_options_per_token=self.config.future_num_options_per_fillin,
                        num_tokens=nb_skipped,
                    )  # pot_tokens: shape [curr_bsz, nb_skipped, num_options_per_fillin]
                    curr_boosts = torch.sparse_coo_tensor(
                        pot_tokens[:, 0, :].ravel().view(1, -1),
                        pot_probs[:, 0, :].ravel().view(-1),
                        [vocab_sz],
                        device=device,
                    ).to_dense()  # same tokens get summed up --> so this can give you >1 boosts
                    curr_boosts[curr_boosts > 1] = 1  # clip

                    # Add up
                    boosts_future += local_w * curr_boosts
                    total_local_w += local_w
                    local_w *= self.config.future_local_w_decay
                boosts_future /= total_local_w

                # Add future
                boosts += self.config.w_future * boosts_future  # Add
                total_w += self.config.w_future

            # Average to get the final boosts
            boosts /= total_w
            new_logits = logits[b] + self.config.spoofer_strength * boosts

            # New Z-score estimation
            if self.config.use_predict:
                gamma = self.config.predict_gamma
                if len(self.z_estimates) < b + 1:
                    self.green_estimates.append([])
                    self.z_estimates.append([])
                green_probs = boosts * (1 - gamma) + gamma
                green_probs[green_probs > 1.0] = 1.0

                if (
                    "do_sample" in self.config.gen_kwargs.keys()
                    and self.config.gen_kwargs["do_sample"]
                ):
                    temperature = (
                        self.config.gen_kwargs["temperature"]
                        if "temperature" in self.config.gen_kwargs.keys()
                        else 0.7
                    )
                    warper = TemperatureLogitsWarper(temperature)
                    sampling_probs = warper(input_ids[b], logits[b]).softmax(0)
                else:
                    sampling_probs = torch.zeros_like(logits[b], device=logits.device)
                    sampling_probs[logits[b].argmax()] = 1.0

                total_green_prob = (green_probs * sampling_probs).sum().cpu().item()
                self.green_estimates[b].append(total_green_prob)
                N = sum(self.green_estimates[b])
                T = len(self.green_estimates[b])
                # TODO make a param if running on different gammas
                self.z_estimates[b].append(
                    ((N - gamma * T) / math.sqrt(gamma * (1 - gamma) * T))
                )

                # Prevent EOS if estimate is bad
                if self.config.prevent_eos_if_zest_bad:
                    if (
                        self.z_estimates[b][-1] < 5.0
                    ):  # TODO make a param if running on diff thresh
                        new_logits[self.tokenizer.eos_token_id] = -1e7

            # Finally apply the boosts
            logits[b] = new_logits

        return logits


class STEAL(BaseWatermark):
    """Top-level class for STEAL algorithm."""

    def __init__(
        self,
        algorithm_config: str | STEALConfig,
        transformers_config: TransformersConfig | None = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the STEAL algorithm.

        Parameters:
            algorithm_config (str | STEALConfig): Path to the algorithm configuration file or STEALConfig instance.
            transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if isinstance(algorithm_config, str):
            self.config = STEALConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, STEALConfig):
            self.config = algorithm_config
        else:
            raise TypeError(
                "algorithm_config must be either a path string or a STEALConfig instance"
            )

        self.utils = STEALUtils(self.config)
        cfg_gen = self.config.gen_kwargs
        # Build processor list
        processors = []
        if (
            "repetition_penalty" in cfg_gen.keys()
            and cfg_gen["repetition_penalty"] != 1
        ):
            endinst_pattern = self.config.generation_tokenizer(
                self.config.end_inst_tag, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]
            processors.append(
                CustomNgramRepetitionPenaltyProcessor(
                    n=self.config.prevctx_width + 1,
                    penalty=cfg_gen["repetition_penalty"],
                    endinst_pattern=endinst_pattern,
                    mode="divide",
                )
            )  # what if this goes first?
        if self.config.spoofer_strength != 0:
            processors.append(SpoofedProcessor(self.config, self.utils))
        if "panic_from" in cfg_gen.keys():
            # Always use graceful conclusion
            processors.append(
                GracefulConclusionProcessor(
                    period_token=self.config.generation_tokenizer.period_token_id,
                    eos_token=self.config.generation_tokenizer.eos_token_id,
                    panic_from=cfg_gen["panic_from"],
                )
            )
        self.logits_processor_list = LogitsProcessorList(processors)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text."""

        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=self.logits_processor_list,
            **self.config.gen_kwargs,
        )

        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.config.device)
        # Generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # Decode
        watermarked_text = self.config.generation_tokenizer.batch_decode(
            encoded_watermarked_text, skip_special_tokens=True
        )[0]

        for lp in self.logits_processor_list:
            if isinstance(lp, SpoofedProcessor):
                lp.gen_clear()

        return watermarked_text

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """STEAL does not have detection, to use the stolen watermark detector."""

        print("STEAL does not have detection, to use the stolen watermark detector.")

    def get_data_for_visualization(
        self, text: str, *args, **kwargs
    ) -> tuple[list[str], list[int]]:
        """STEAL does not have detection, to use the stolen watermark detector."""

        print("STEAL does not have detection, to use the stolen watermark detector.")
