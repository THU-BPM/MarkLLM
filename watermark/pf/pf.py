from typing import List
import torch
import numpy as np
from scipy.stats import gamma
from sympy.physics.units import temperature

from ..base import BaseConfig, BaseWatermark
from utils.transformers_config import TransformersConfig

# PF 方法的配置类
class PFConfig(BaseConfig):
    """Config class for PF algorithm, load config file and initialize parameters."""
    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.payload = self.config_dict['payload']
        self.salt_key = self.config_dict['salt_key']
        self.ngram = self.config_dict['ngram']
        self.seed = self.config_dict['seed']
        self.seeding = self.config_dict['seeding']
        self.max_seq_len = self.config_dict['max_seq_len']


    @property
    def algorithm_name(self) -> str:
        """Return algorithm name."""
        return 'PF'


class PFUtils:
    """Utility class for PF algorithm, contains helper functions."""
    def __init__(self, config: PFConfig, *args, **kwargs) -> None:
        """
            Initialize the PF utility class.

            Parameters:
                config (PFConfig): Configuration for the PF algorithm.
        """
        self.config = config
        self.pad_id = config.generation_tokenizer.pad_token_id if config.generation_tokenizer.pad_token_id is not None else config.generation_tokenizer.eos_token_id
        self.eos_id = config.generation_tokenizer.eos_token_id
        self.hashtable = torch.randperm(1000003)
        self.rng = torch.Generator(device=self.config.device)
        self.rng.manual_seed(self.config.seed)

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)]


    def get_seed_rng(self, input_ids: torch.LongTensor) -> int:
        """get a random seed according to input tokens."""
        if self.config.seeding == 'hash':
            seed = self.config.seed
            for i in input_ids:
                seed = (seed * self.config.salt_key + i.item()) % (2 ** 64 - 1)
        elif self.config.seeding == 'additive':
            seed = self.config.salt_key * torch.sum(input_ids).item()
            seed = self.hashint(seed)
        elif self.config.seeding == 'skip':
            seed = self.config.salt_key * input_ids[0].item()
            seed = self.hashint(seed)
        elif self.config.seeding == 'min':
            seed = self.hashint(self.config.salt_key * input_ids)
            seed = torch.min(seed).item()
        return seed

    def sample_next(
            self,
            logits: torch.FloatTensor,
            ngram_tokens: torch.LongTensor,
            temperature: float,
            top_p: float
    ) -> torch.LongTensor:
        """generate next token from logits."""
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            log_probs = probs_sort.log()

            seed = self.get_seed_rng(ngram_tokens)
            self.rng.manual_seed(seed)

            rs = torch.rand(self.config.vocab_size, generator=self.rng, device=self.rng.device)
            rs = rs.roll(-self.config.payload)
            rs = torch.Tensor(rs).to(probs_sort.device)
            rs = rs[probs_idx]

            log_probs = log_probs - rs.log()


            next_token = torch.argmax(log_probs, dim=-1, keepdim=True)
            next_token = torch.gather(probs_idx, -1, next_token)
        else:
            next_token = torch.argmax(logits, dim=-1)

        return next_token.reshape(-1)



    def score_tok(self, ngram_tokens: List[int], token_id: int) -> torch.Tensor:
        """calculate scores of each token."""
        seed = self.get_seed_rng(torch.tensor(ngram_tokens))
        self.rng.manual_seed(seed)
        rs = torch.rand(self.config.vocab_size, generator=self.rng, device=self.rng.device)
        # avoid log(0)
        rs[rs == 0] = 1e-4
        scores = -rs.log().roll(-token_id)
        return scores

    def get_threshold(self, n_tokens: int, alpha: float = 0.01) -> float:
        """calculate threshold for PF algorithm."""
        if n_tokens <= self.config.ngram:
            return float('inf')

        k = n_tokens - self.config.ngram
        threshold = gamma.ppf(1 - alpha, a=k, scale=1)
        return threshold

    def get_scores_by_t(
            self,
            text: str,
            scoring_method: str = "none",
            ntoks_max: int = None,
            payload_max: int = 0
    ) -> np.array :
        """
        Get score increment for each token in list of texts.
        Args:
            texts: list of texts
            scoring_method:
                'none': score all ngrams
                'v1': only score tokens for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            ntoks_max: maximum number of tokens
            payload_max: maximum number of messages
        Output:
            score_lists: list of [np array of score increments for every token and payload] for each text
        """
        tokens_id = self.config.generation_tokenizer.encode(text, add_special_tokens=False)
        if ntoks_max is not None:
            tokens_id = tokens_id[:ntoks_max]

        total_len = len(tokens_id)
        start_pos = self.config.ngram + 1
        rts = []
        seen_ntuples = set()

        for cur_pos in range(start_pos, total_len):
            ngram_tokens = tokens_id[cur_pos - self.config.ngram: cur_pos]
            if scoring_method == 'v1':
                tup_for_unique = tuple(ngram_tokens)
                if tup_for_unique in seen_ntuples:
                    continue
                seen_ntuples.add(tup_for_unique)
            elif scoring_method == 'v2':
                tup_for_unique = tuple(ngram_tokens + [tokens_id[cur_pos]])
                if tup_for_unique in seen_ntuples:
                    continue
                seen_ntuples.add(tup_for_unique)

            rt = self.score_tok(ngram_tokens, tokens_id[cur_pos])
            rt = rt[:payload_max + 1]
            rts.append(rt)

        return np.array([rt.cpu().numpy() for rt in rts])

    def get_scores(self,score_lists: np.array) -> float:
        """calculate sum of PF score."""
        if len(score_lists) == 0:
            return 0
        aggregated_score = sum(score_lists)
        return aggregated_score



class PF(BaseWatermark):
    def __init__(self, algorithm_config: str | PFConfig, transformers_config: TransformersConfig | None = None, *args,
                 **kwargs) -> None:
        """
            Initialize the PF algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if isinstance(algorithm_config, str):
            self.config = PFConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, PFConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be either a path string or a SIRConfig instance")

        self.utils = PFUtils(self.config)


    @torch.no_grad()
    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text using the PF algorithm."""

        prompt_tokens = self.config.generation_tokenizer.encode(prompt, add_special_tokens=False)
        min_prompt_size = len(prompt_tokens)
        total_len = min(self.config.max_seq_len, self.config.gen_kwargs["max_new_tokens"] + min_prompt_size)

        tokens = torch.full((total_len,), self.utils.pad_id).to(self.config.generation_model.device).long()
        tokens[: len(prompt_tokens)] = torch.tensor(prompt_tokens).long()
        input_text_mask = tokens != self.utils.pad_id

        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            outputs = self.config.generation_model.forward(
                tokens[prev_pos:cur_pos].unsqueeze(0), use_cache=True,
                past_key_values=outputs.past_key_values if prev_pos > 0 else None
            )
            ngram_tokens = tokens[cur_pos - self.config.ngram:cur_pos]
            temperature = self.config.gen_kwargs.get("temperature", 0.9)
            top_p = self.config.gen_kwargs.get("top_p", 1.0)
            next_toks = self.utils.sample_next(outputs.logits[:, -1, :], ngram_tokens, temperature, top_p)
            tokens[cur_pos] = torch.where(input_text_mask[cur_pos], tokens[cur_pos], next_toks)
            prev_pos = cur_pos

        tokens = tokens.tolist()

        tokens = tokens[: len(prompt_tokens) + self.config.gen_kwargs["max_new_tokens"]]

        try:
            tokens = tokens[: tokens.index(self.utils.eos_id)]
        except ValueError:
            pass

        return self.config.generation_tokenizer.decode(tokens)



    def detect_watermark(self, text: str, *args, **kwargs):
        """Detect watermark in the text."""

        scores = self.utils.get_scores_by_t(text)
        score = self.utils.get_scores(scores)
        alpha = self.config.gen_kwargs.get("alpha", 0.01)
        threshold = self.utils.get_threshold(len(scores), alpha)
        result = bool(score > threshold)
        score = float(score)
        threshold = float(threshold)
        return {"is_watermarked": result, "score": score, "threshold": threshold}






