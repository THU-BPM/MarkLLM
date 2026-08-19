# Copyright 2025 Amirhossein Dabiriaghdam.
# Copyright 2026 THU-BPM MarkLLM.
#
# Adapted from SimMark, released under the MIT License. See
# watermark/simmark/LICENSE for the original license and attribution.

"""MarkLLM integration of sentence-level SimMark watermarking."""

from dataclasses import dataclass
from functools import partial
import math
import re

import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import StoppingCriteria, StoppingCriteriaList

from exceptions.exceptions import AlgorithmNameMismatchError
from utils.transformers_config import TransformersConfig
from visualize.data_for_visualization import DataForVisualization
from watermark.base import BaseConfig, BaseWatermark


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_VISUALIZATION_TOKEN = re.compile(r"\S+\s*")


def split_sentences(text: str) -> list[str]:
    """Split text with Punkt when available and a no-download fallback."""
    text = text.strip()
    if not text:
        return []
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        sentences = _SENTENCE_BOUNDARY.split(text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


class SimMarkConfig(BaseConfig):
    """Load SimMark generation, embedding, and soft-z parameters."""

    def initialize_parameters(self) -> None:
        if self.config_dict["algorithm_name"] != self.algorithm_name:
            raise AlgorithmNameMismatchError(
                self.algorithm_name, self.config_dict["algorithm_name"]
            )

        self.embedding_model_name = self.config_dict["embedding_model_name"]
        self.embedding_prompt = self.config_dict.get(
            "embedding_prompt", "Represent the sentence for cosine similarity:"
        )
        self.interval_low = float(self.config_dict["interval_low"])
        self.interval_high = float(self.config_dict["interval_high"])
        self.expected_valid_fraction = float(
            self.config_dict["expected_valid_fraction"]
        )
        self.softness = float(self.config_dict.get("softness", 250.0))
        self.z_threshold = float(self.config_dict.get("z_threshold", 2.33))
        self.max_trials = int(self.config_dict.get("max_trials", 100))
        self.candidate_max_new_tokens = int(
            self.config_dict.get("candidate_max_new_tokens", 64)
        )
        self.ensure_sentence_punctuation = bool(
            self.config_dict.get("ensure_sentence_punctuation", True)
        )
        self.embedding_model = self.config_dict.get("embedding_model")

        if not -1 <= self.interval_low < self.interval_high <= 1:
            raise ValueError("SimMark cosine interval must lie within [-1, 1]")
        if not 0 < self.expected_valid_fraction < 1:
            raise ValueError("SimMark expected_valid_fraction must be in (0, 1)")
        if self.softness <= 0:
            raise ValueError("SimMark softness must be positive")
        if self.max_trials < 1:
            raise ValueError("SimMark max_trials must be positive")
        if self.candidate_max_new_tokens < 1:
            raise ValueError("SimMark candidate_max_new_tokens must be positive")

    @property
    def algorithm_name(self) -> str:
        return "SimMark"


@dataclass
class SimMarkSequenceScore:
    """Sentence similarities, soft evidence, and aggregate z-score."""

    z_score: float
    similarities: list[float]
    soft_scores: list[float]


class _CandidateSentenceCriteria(StoppingCriteria):
    """Stop after a complete sentence and the beginning of the next one."""

    def __init__(self, tokenizer, prompt_length: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        if input_ids.shape[0] != 1:
            raise ValueError(
                "SimMark sentence rejection sampling requires batch size 1"
            )
        candidate = self.tokenizer.decode(
            input_ids[0, self.prompt_length :], skip_special_tokens=True
        )
        return len(split_sentences(candidate)) > 1


class SimMarkUtils:
    """Sentence embedding, rejection, and soft-z detection helpers."""

    def __init__(self, config: SimMarkConfig) -> None:
        self.config = config
        self.embedder = config.embedding_model or SentenceTransformer(
            config.embedding_model_name, device=config.device
        )

    def _encode_sentences(self, sentences: list[str]) -> torch.Tensor:
        if not sentences:
            return torch.empty((0, 0))
        encode = partial(
            self.embedder.encode,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        try:
            embeddings = encode(sentences, prompt=self.config.embedding_prompt)
        except TypeError:
            embeddings = encode(sentences)
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.as_tensor(embeddings)
        return embeddings.float()

    def similarity(self, previous_sentence: str, candidate_sentence: str) -> float:
        embeddings = self._encode_sentences([previous_sentence, candidate_sentence])
        return float(F.cosine_similarity(embeddings[0], embeddings[1], dim=0).item())

    def in_interval(self, similarity: float) -> bool:
        return self.config.interval_low <= similarity <= self.config.interval_high

    def soft_score(self, similarity: float) -> float:
        if self.in_interval(similarity):
            return 1.0
        distance = min(
            abs(similarity - self.config.interval_low),
            abs(similarity - self.config.interval_high),
        )
        return math.exp(-self.config.softness * distance)

    def score_text(self, text: str) -> SimMarkSequenceScore:
        sentences = split_sentences(text)
        if len(sentences) < 2:
            return SimMarkSequenceScore(0.0, [], [])
        # SimMark's reference detector preserves the whitespace before every
        # sentence except the first one when computing its embeddings.
        embedding_sentences = [sentences[0], *(f" {item}" for item in sentences[1:])]
        embeddings = self._encode_sentences(embedding_sentences)
        similarities = (
            F.cosine_similarity(embeddings[:-1], embeddings[1:], dim=1).cpu().tolist()
        )
        soft_scores = [self.soft_score(value) for value in similarities]
        test_count = len(soft_scores)
        gamma = self.config.expected_valid_fraction
        numerator = sum(soft_scores) - gamma * test_count
        denominator = math.sqrt(test_count * gamma * (1 - gamma) + 1e-12)
        return SimMarkSequenceScore(
            numerator / denominator,
            similarities,
            soft_scores,
        )

    def _generate_candidate(self, text: str, remaining_tokens: int) -> str:
        tokenizer = self.config.generation_tokenizer
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(
            self.config.device
        )
        prompt_length = encoding["input_ids"].shape[1]
        generation_kwargs = dict(self.config.gen_kwargs)
        for key in (
            "max_length",
            "min_length",
            "max_new_tokens",
            "min_new_tokens",
            "stopping_criteria",
        ):
            generation_kwargs.pop(key, None)
        generation_kwargs.setdefault("do_sample", True)
        generation_kwargs["max_new_tokens"] = min(
            self.config.candidate_max_new_tokens, remaining_tokens
        )
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [_CandidateSentenceCriteria(tokenizer, prompt_length)]
        )
        with torch.inference_mode():
            output = self.config.generation_model.generate(
                **encoding, **generation_kwargs
            )
        sequences = output.sequences if hasattr(output, "sequences") else output
        candidate = tokenizer.decode(
            sequences[0, prompt_length:], skip_special_tokens=True
        ).strip()
        sentences = split_sentences(candidate)
        if sentences:
            candidate = sentences[0]
        if (
            candidate
            and self.config.ensure_sentence_punctuation
            and candidate[-1] not in ".!?"
        ):
            candidate += "."
        return candidate

    def generate(self, prompt: str) -> str:
        """Generate sentence candidates until they satisfy SimMark's interval."""
        tokenizer = self.config.generation_tokenizer
        initial_length = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        max_new_tokens = int(self.config.gen_kwargs.get("max_new_tokens", 200))
        text = prompt.strip()
        previous_sentences = split_sentences(text)
        previous_sentence = previous_sentences[-1] if previous_sentences else text

        while True:
            current_length = len(tokenizer(text, add_special_tokens=True)["input_ids"])
            generated_tokens = max(0, current_length - initial_length)
            remaining_tokens = max_new_tokens - generated_tokens
            if remaining_tokens <= 0:
                break

            accepted_candidate = None
            last_candidate = None
            for _ in range(self.config.max_trials):
                candidate = self._generate_candidate(text, remaining_tokens)
                if not candidate:
                    break
                last_candidate = candidate
                candidate_for_embedding = f" {candidate}"
                if self.in_interval(
                    self.similarity(previous_sentence, candidate_for_embedding)
                ):
                    accepted_candidate = candidate
                    break

            candidate = accepted_candidate or last_candidate
            if not candidate:
                break
            text = f"{text.rstrip()} {candidate.lstrip()}"
            previous_sentence = f" {candidate.lstrip()}"

        return text


class SimMark(BaseWatermark):
    """Sentence-level similarity rejection-sampling watermark."""

    def __init__(
        self,
        algorithm_config: str | SimMarkConfig,
        transformers_config: TransformersConfig | None = None,
        *args,
        **kwargs,
    ) -> None:
        if isinstance(algorithm_config, str):
            self.config = SimMarkConfig(algorithm_config, transformers_config, **kwargs)
        elif isinstance(algorithm_config, SimMarkConfig):
            self.config = algorithm_config
        else:
            raise TypeError(
                "algorithm_config must be a path string or SimMarkConfig instance"
            )
        self.utils = SimMarkUtils(self.config)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        return self.utils.generate(prompt)

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        result = self.utils.score_text(text)
        is_watermarked = bool(result.z_score > self.config.z_threshold)
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": result.z_score}
        return is_watermarked, result.z_score

    def get_data_for_visualization(
        self, text: str, *args, **kwargs
    ) -> DataForVisualization:
        sentences = split_sentences(text)
        result = self.utils.score_text(text)
        decoded_tokens = []
        values = []
        sentence_scores = [None, *result.soft_scores]
        for sentence, score in zip(sentences, sentence_scores):
            tokens = _VISUALIZATION_TOKEN.findall(sentence)
            decoded_tokens.extend(tokens)
            values.extend([score] * len(tokens))
        return DataForVisualization(decoded_tokens, values)
