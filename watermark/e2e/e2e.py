# Copyright 2025 Kahim Wong.
# Copyright 2026 THU-BPM MarkLLM.
#
# Adapted from E2E-LLM-Watermark, released under the MIT License. See
# watermark/e2e/LICENSE for the original license and attribution.

"""MarkLLM integration of the E2E-LLM-Watermark algorithm."""

from functools import partial
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessor, LogitsProcessorList

from exceptions.exceptions import AlgorithmNameMismatchError
from utils.transformers_config import TransformersConfig
from visualize.data_for_visualization import DataForVisualization
from watermark.base import BaseConfig, BaseWatermark

from .model import E2EDetector, E2EEncoder


class E2EConfig(BaseConfig):
    """Load and validate the E2E algorithm configuration."""

    def initialize_parameters(self) -> None:
        if self.config_dict["algorithm_name"] != self.algorithm_name:
            raise AlgorithmNameMismatchError(
                self.algorithm_name, self.config_dict["algorithm_name"]
            )

        self.delta = float(self.config_dict["delta"])
        self.top_k = int(self.config_dict["top_k"])
        self.window_size = int(self.config_dict["window_size"])
        self.detection_threshold = float(self.config_dict["detection_threshold"])
        self.checkpoint_path = self.config_dict["checkpoint_path"]
        self.reference_model_name = self.config_dict["reference_model_name"]
        self.reference_tokenizer_name = self.config_dict.get(
            "reference_tokenizer_name", self.reference_model_name
        )
        self.tokenizer_conversion_multiplier = int(
            self.config_dict.get("tokenizer_conversion_multiplier", 3)
        )
        self.mapper_layers = int(self.config_dict.get("mapper_layers", 5))
        self.hidden_dim = int(self.config_dict.get("hidden_dim", 64))
        self.detector_layers = int(self.config_dict.get("detector_layers", 3))
        self.reference_model = self.config_dict.get("reference_model")
        self.reference_tokenizer = self.config_dict.get("reference_tokenizer")

        if self.top_k < 2:
            raise ValueError("E2E top_k must be at least 2")
        if self.window_size < 1:
            raise ValueError("E2E window_size must be positive")
        if not 0 <= self.detection_threshold <= 1:
            raise ValueError("E2E detection_threshold must be in [0, 1]")
        if not Path(self.checkpoint_path).is_file():
            raise FileNotFoundError(
                f"E2E checkpoint not found at {self.checkpoint_path}. "
                "See watermark/e2e/README.md for download instructions."
            )

    @property
    def algorithm_name(self) -> str:
        return "E2E"


class E2EUtils:
    """Shared model loading, token conversion, and scoring utilities."""

    def __init__(self, config: E2EConfig) -> None:
        self.config = config
        self.reference_tokenizer = (
            config.reference_tokenizer
            or AutoTokenizer.from_pretrained(config.reference_tokenizer_name)
        )
        self.same_tokenizer = self._tokenizers_match(
            config.generation_tokenizer, self.reference_tokenizer
        )
        self.prefix_length = config.window_size
        if not self.same_tokenizer:
            self.prefix_length *= config.tokenizer_conversion_multiplier

        try:
            checkpoint = torch.load(
                config.checkpoint_path,
                map_location=config.device,
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
        self.input_dim = checkpoint["enc"]["mapper.0.weight"].shape[1]
        self.reference_embeddings = self._load_reference_embeddings()
        if self.reference_embeddings.embedding_dim != self.input_dim:
            raise ValueError(
                "E2E reference embedding dimension does not match the checkpoint: "
                f"expected {self.input_dim}, got "
                f"{self.reference_embeddings.embedding_dim}"
            )
        self.encoder = E2EEncoder(
            input_dim=self.input_dim,
            mapper_layers=config.mapper_layers,
            window_size=config.window_size,
            hidden_dim=config.hidden_dim,
        ).to(config.device)
        self.detector = E2EDetector(
            input_dim=self.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.detector_layers,
        ).to(config.device)
        self.encoder.load_state_dict(checkpoint["enc"])
        self.detector.load_state_dict(checkpoint["dec"])
        self.encoder.eval()
        self.detector.eval()

    def _load_reference_embeddings(self) -> torch.nn.Embedding:
        generation_embeddings = self.config.generation_model.get_input_embeddings()
        if (
            self.same_tokenizer
            and generation_embeddings.embedding_dim == self.input_dim
        ):
            embeddings = generation_embeddings
            embeddings.requires_grad_(False)
            return embeddings

        reference_model = self.config.reference_model
        if reference_model is None:
            reference_model = AutoModelForCausalLM.from_pretrained(
                self.config.reference_model_name,
                torch_dtype=torch.float16,
            )
        embeddings = reference_model.get_input_embeddings().to(self.config.device)
        embeddings.requires_grad_(False)
        return embeddings

    @staticmethod
    def _tokenizers_match(generation_tokenizer, reference_tokenizer) -> bool:
        try:
            return generation_tokenizer.get_vocab() == reference_tokenizer.get_vocab()
        except (AttributeError, NotImplementedError):
            return (
                len(generation_tokenizer) == len(reference_tokenizer)
                and generation_tokenizer.__class__ is reference_tokenizer.__class__
            )

    def _convert_candidate_ids(self, candidate_sequences: torch.Tensor) -> torch.Tensor:
        """Convert generation-tokenizer candidates to reference-tokenizer ids."""
        batch_size, candidate_count, _ = candidate_sequences.shape
        converted = []
        pad_token_id = self.reference_tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.reference_tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("The E2E reference tokenizer needs a pad or EOS token")

        for sequence in candidate_sequences.reshape(-1, candidate_sequences.shape[-1]):
            text = self.config.generation_tokenizer.decode(
                sequence.tolist(), skip_special_tokens=True
            )
            token_ids = self.reference_tokenizer(text, add_special_tokens=False)[
                "input_ids"
            ][-self.config.window_size :]
            token_ids = [pad_token_id] * (
                self.config.window_size - len(token_ids)
            ) + token_ids
            converted.append(token_ids)
        return torch.tensor(
            converted, device=self.config.device, dtype=torch.long
        ).reshape(batch_size, candidate_count, self.config.window_size)

    def candidate_embeddings(
        self, input_ids: torch.LongTensor, candidate_ids: torch.LongTensor
    ) -> torch.Tensor:
        """Build contextual reference embeddings for every top-k candidate."""
        context_length = self.prefix_length - 1
        previous_ids = (
            input_ids[:, -context_length:] if context_length else input_ids[:, :0]
        )
        previous_ids = previous_ids[:, None, :].expand(-1, candidate_ids.shape[1], -1)
        candidate_sequences = torch.cat(
            [previous_ids, candidate_ids.unsqueeze(-1)], dim=-1
        )
        if self.same_tokenizer:
            reference_ids = candidate_sequences
        else:
            reference_ids = self._convert_candidate_ids(candidate_sequences)
        return self.reference_embeddings(reference_ids).float()

    def encode_text(self, text: str) -> tuple[torch.LongTensor, torch.Tensor]:
        encoding = self.reference_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )
        input_ids = encoding["input_ids"][0].to(self.config.device)
        if input_ids.numel() == 0:
            raise ValueError("E2E cannot detect a watermark in empty text")
        embeddings = self.reference_embeddings(input_ids.unsqueeze(0)).float()
        return input_ids, embeddings

    def score_text(self, text: str) -> float:
        _, embeddings = self.encode_text(text)
        with torch.inference_mode():
            probability = torch.sigmoid(self.detector(embeddings)).squeeze()
        return float(probability.item())

    def score_prefixes(self, text: str) -> tuple[torch.LongTensor, list[float]]:
        input_ids, embeddings = self.encode_text(text)
        with torch.inference_mode():
            probabilities = (
                torch.sigmoid(self.detector(embeddings, return_all=True))
                .squeeze(0)
                .squeeze(-1)
            )
        return input_ids, probabilities.cpu().tolist()


class E2ELogitsProcessor(LogitsProcessor):
    """Apply learned, centered perturbations to the top-k token logits."""

    def __init__(self, config: E2EConfig, utils: E2EUtils) -> None:
        self.config = config
        self.utils = utils

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.utils.prefix_length:
            return scores

        candidate_count = min(self.config.top_k, scores.shape[-1])
        candidate_ids = torch.topk(scores, candidate_count, dim=-1).indices
        embeddings = self.utils.candidate_embeddings(input_ids, candidate_ids)
        with torch.inference_mode():
            candidate_bias = self.utils.encoder(embeddings)
        perturbation = torch.zeros_like(scores)
        perturbation.scatter_add_(
            1,
            candidate_ids,
            self.config.delta * candidate_bias.to(scores.dtype),
        )
        return scores + perturbation


class E2E(BaseWatermark):
    """End-to-end learned logits watermark and neural detector."""

    def __init__(
        self,
        algorithm_config: str | E2EConfig,
        transformers_config: TransformersConfig | None = None,
        *args,
        **kwargs,
    ) -> None:
        if isinstance(algorithm_config, str):
            self.config = E2EConfig(algorithm_config, transformers_config, **kwargs)
        elif isinstance(algorithm_config, E2EConfig):
            self.config = algorithm_config
        else:
            raise TypeError(
                "algorithm_config must be either a path string or an E2EConfig instance"
            )
        self.utils = E2EUtils(self.config)
        self.logits_processor = E2ELogitsProcessor(self.config, self.utils)

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
        score = self.utils.score_text(text)
        is_watermarked = bool(score > self.config.detection_threshold)
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": score}
        return is_watermarked, score

    def get_data_for_visualization(
        self, text: str, *args, **kwargs
    ) -> DataForVisualization:
        input_ids, probabilities = self.utils.score_prefixes(text)
        decoded_tokens = [
            self.reference_tokenizer.decode(int(token_id)) for token_id in input_ids
        ]
        return DataForVisualization(decoded_tokens, probabilities)

    @property
    def reference_tokenizer(self):
        return self.utils.reference_tokenizer
