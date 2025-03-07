from typing import List
import torch
import numpy as np
from scipy.stats import gamma

from ..base import BaseConfig, BaseWatermark
from utils.transformers_config import TransformersConfig

# PF 方法的配置类
class PFConfig(BaseConfig):
    """PF 算法的配置类"""
    def initialize_parameters(self) -> None:
        # 从配置字典中加载 PF 算法的参数
        self.payload = self.config_dict['payload']                 # 水印负载
        self.salt_key = self.config_dict['salt_key']               # 哈希密钥
        self.ngram = self.config_dict['ngram']                     # 用于生成种子的 ngram 大小
        self.seed = self.config_dict['seed']
        self.seeding = self.config_dict['seeding']
        self.max_seq_len = self.config_dict['max_seq_len']
        self.alpha = self.config_dict['alpha']
        self.temperature = self.config_dict['temperature']
        self.top_p = self.config_dict['top_p']


    @property
    def algorithm_name(self) -> str:
        """返回算法名称"""
        return 'PF'


class PFUtils:
    """PF 算法的工具类，包含辅助函数"""
    def __init__(self, config: PFConfig, *args, **kwargs) -> None:
        """
        初始化 PF 工具类。

        参数：
            config (PFConfig): PF 算法的配置实例。
        """
        self.config = config
        # 初始化随机数生成器，并以 hash_key 作为初始种子
        self.pad_id = config.generation_tokenizer.pad_token_id if config.generation_tokenizer.pad_token_id is not None else config.generation_tokenizer.eos_token_id
        self.eos_id = config.generation_tokenizer.eos_token_id
        self.hashtable = torch.randperm(1000003)
        self.rng = torch.Generator(device=self.config.device)
        self.rng.manual_seed(self.config.seed)

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)]


    def get_seed_rng(self, input_ids: torch.LongTensor) -> int:
        """
        根据输入的 tokens 获取一个随机数种子
        """
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
        """
        生成下一个 token（修改后适配单个文本，不再使用 batch 维度）
        """
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            log_probs = probs_sort.log()

            seed = self.get_seed_rng(ngram_tokens)  # 直接传入
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

        return next_token.reshape(-1)  # 保持输出格式一致



    def score_tok(self, ngram_tokens: List[int], token_id: int) -> torch.Tensor:
        """
        对文本中的每个 token 计算得分增量
        """
        seed = self.get_seed_rng(torch.tensor(ngram_tokens))
        self.rng.manual_seed(seed)
        rs = torch.rand(self.config.vocab_size, generator=self.rng, device=self.rng.device)
        rs[rs == 0] = 1e-4  # 避免 log(0)
        scores = -rs.log().roll(-token_id)  # 先取 log 再 roll
        return scores

    def get_threshold(self, n_tokens: int, alpha: float = 0.01) -> float:
        if n_tokens <= self.config.ngram:
            return float('inf')  # 如果文本太短，不进行检测

        k = n_tokens - self.config.ngram  # Gamma 分布的 shape 参数
        threshold = gamma.ppf(1 - alpha, a=k, scale=1)  # 计算阈值
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
            tokens_id = tokens_id[:ntoks_max]  # 限制最大 token 数量

        total_len = len(tokens_id)
        start_pos = self.config.ngram + 1
        rts = []
        seen_ntuples = set()

        for cur_pos in range(start_pos, total_len):
            ngram_tokens = tokens_id[cur_pos - self.config.ngram: cur_pos]  # 取 ngram
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
        """
        生成带水印的文本
        """
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
            next_toks = self.utils.sample_next(outputs.logits[:, -1, :], ngram_tokens, self.config.temperature, self.config.top_p)
            tokens[cur_pos] = torch.where(input_text_mask[cur_pos], tokens[cur_pos], next_toks)
            prev_pos = cur_pos

        tokens = tokens.tolist()

        # 截取最大生成长度
        tokens = tokens[: len(prompt_tokens) + self.config.gen_kwargs["max_new_tokens"]]

        # 如果有 EOS token，则截断
        try:
            tokens = tokens[: tokens.index(self.utils.eos_id)]
        except ValueError:
            pass

        # 直接返回解码后的文本
        return self.config.generation_tokenizer.decode(tokens)



    def detect_watermark(self, text: str, *args, **kwargs):
        scores = self.utils.get_scores_by_t(text)
        score = self.utils.get_scores(scores)
        threshold = self.utils.get_threshold(len(scores),self.config.alpha)
        result = bool(score > threshold)  # 转换布尔值
        score = float(score)  # 转换为 Python float
        threshold = float(threshold)
        return {"is_watermarked": result, "score": score, "threshold": threshold}






