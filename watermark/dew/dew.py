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
# dew.py
# Description: Implementation of DEW algorithm
# ============================================

import math
import os

import torch
import torch.nn.functional as F
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from ..base import BaseWatermark, BaseConfig
from utils.transformers_config import TransformersConfig
from visualize.data_for_visualization import DataForVisualization


def fit_whitening_pipeline(embedding_layer, n_components: int | None = None, output_path: str | None = None) -> dict:
    """Fit a StandardScaler + whitening-PCA pipeline on an input-embedding matrix.

    DEW whitens the token-embedding space so the keyed random projection acts on
    decorrelated, unit-variance features. The pipeline is specific to the model
    it was fit on; when ``output_path`` is given it is cached there for reuse.

    Parameters:
        embedding_layer (nn.Embedding): The model's input embedding layer.
        n_components (int | None): PCA components to keep (``None`` keeps all).
        output_path (str | None): If set, the fitted pipeline is saved here.

    Returns:
        dict: ``{"scaler": StandardScaler, "pca": PCA}``.
    """
    embeddings = embedding_layer.weight.detach().cpu().float().numpy()
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    pca = PCA(n_components=n_components, whiten=True, random_state=0)
    pca.fit(embeddings)

    pipeline = {"scaler": scaler, "pca": pca}
    if output_path:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        dump(pipeline, output_path)
    return pipeline


class DEWConfig(BaseConfig):
    """Config class for DEW (Dual-Embedding Watermarking) algorithm."""

    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.secret_key = self.config_dict['secret_key']
        self.n_vectors = self.config_dict['n_vectors']
        self.lambda_strength = self.config_dict['lambda_strength']
        self.top_k = self.config_dict['top_k']

        # Latent-score -> bias mapping and normalization behaviour.
        self.bias_use_tanh = self.config_dict['bias_use_tanh']
        self.tanh_scale = self.config_dict['tanh_scale']
        self.normalize_embeddings = self.config_dict['normalize_embeddings']
        self.normalize_projected_vectors = self.config_dict['normalize_projected_vectors']
        self.standardize_score_by_sqrt_n = self.config_dict['standardize_score_by_sqrt_n']

        # Context side.
        self.context_embedding_model = self.config_dict['context_embedding_model']
        self.context_window_size = self.config_dict['context_window_size']

        # Token-side whitening transform.
        self.token_embedding_transform_path = self.config_dict['token_embedding_transform_path']
        self.token_embedding_transform_n_components = self.config_dict.get('token_embedding_transform_n_components', None)

        self.detection_alpha = self.config_dict['detection_alpha']

        if not isinstance(self.context_window_size, int) or self.context_window_size <= 0:
            raise ValueError("context_window_size must be a positive integer")

        # Context encoder: a sentence model that embeds the preceding token window.
        try:
            self.context_model = AutoModel.from_pretrained(self.context_embedding_model).to(self.device)
        except Exception as e:
            raise ValueError(
                f"Failed to load context embedding model '{self.context_embedding_model}'. "
                f"Set config['context_embedding_model'] to a valid HuggingFace id or local path."
            ) from e
        self.context_model.eval()
        self.context_tokenizer = AutoTokenizer.from_pretrained(self.context_embedding_model)
        self.d_C = self.context_model.config.hidden_size

        # Token-side whitening pipeline (fit on demand and cached to disk).
        self.token_embedding_transform_scaler, self.token_embedding_transform_pca = self._load_token_transform()
        # Dimensionality feeding the token projection R_T: the PCA output size
        # when whitening is enabled, otherwise the raw embedding dimension.
        if self.token_embedding_transform_pca is not None:
            self.d_T = self.token_embedding_transform_pca.n_components_
        else:
            self.d_T = self.generation_model.config.hidden_size

    def _load_token_transform(self):
        """Load the token whitening pipeline, fitting and saving it if absent."""
        path = self.token_embedding_transform_path
        if path is None or str(path).lower() == "none":
            return None, None

        if os.path.exists(path):
            pipeline = load(path)
        else:
            pipeline = fit_whitening_pipeline(
                self.generation_model.get_input_embeddings(),
                n_components=self.token_embedding_transform_n_components,
                output_path=path,
            )

        # The transform is model specific; guard against reusing a stale file.
        raw_token_dim = self.generation_model.config.hidden_size
        fitted_dim = pipeline["scaler"].mean_.shape[0]
        if fitted_dim != raw_token_dim:
            raise ValueError(
                f"Token embedding transform at '{path}' was fit for embedding dim {fitted_dim}, "
                f"but the generation model has dim {raw_token_dim}. Use a separate "
                f"token_embedding_transform_path per generation model."
            )
        return pipeline["scaler"], pipeline["pca"]

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'DEW'


class DEWUtils:
    """Utility class for DEW algorithm, contains helper functions.

    DEW derives a per-token watermark signal from the agreement between two
    embeddings projected into a shared latent space R^n with keyed random
    matrices ``R_T`` (token side) and ``R_C`` (context side):

        p_T = normalize(whiten(token_embedding)   @ R_T)   # unit vector in R^n
        p_C = normalize(normalize(context_embedding) @ R_C)  # unit vector in R^n
        z   = sqrt(n) * <p_T, p_C>

    The ``sqrt(n)`` factor standardizes ``z`` to approximately N(0, 1) per token
    under the null hypothesis (unwatermarked text), which both bounds the
    generation bias and gives detection a closed-form Gaussian test.
    """

    def __init__(self, config: DEWConfig, *args, **kwargs) -> None:
        """
            Initialize the DEW utility class.

            Parameters:
                config (DEWConfig): Configuration for the DEW algorithm.
        """
        self.config = config
        self.sqrt_n = math.sqrt(self.config.n_vectors)
        self._init_transform_tensors()
        self.R_C, self.R_T = self._build_projection_matrices()
        self.token_projection_lookup = self._precompute_token_projections()

    def _init_transform_tensors(self) -> None:
        """Move the fitted whitening pipeline onto the compute device as tensors."""
        scaler = self.config.token_embedding_transform_scaler
        pca = self.config.token_embedding_transform_pca
        self.transform_enabled = scaler is not None and pca is not None
        if not self.transform_enabled:
            return

        device = self.config.device
        self.scaler_mean = torch.tensor(scaler.mean_, device=device, dtype=torch.float32)
        self.scaler_scale = torch.tensor(scaler.scale_, device=device, dtype=torch.float32)
        self.pca_mean = torch.tensor(pca.mean_, device=device, dtype=torch.float32)
        self.pca_components = torch.tensor(pca.components_, device=device, dtype=torch.float32)
        self.pca_whiten = bool(pca.whiten)
        if self.pca_whiten:
            self.pca_explained_variance = torch.tensor(pca.explained_variance_, device=device, dtype=torch.float32)

    def _build_projection_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the keyed random projection matrices R_C and R_T.

        A CPU generator seeded from ``secret_key`` keeps the matrices identical
        across devices (CUDA and CPU RNG streams differ), so generation and
        detection stay consistent regardless of where they run.
        """
        generator = torch.Generator(device='cpu').manual_seed(self.config.secret_key)
        R_C = F.normalize(torch.randn(self.config.d_C, self.config.n_vectors, generator=generator), dim=0)
        generator.manual_seed(self.config.secret_key + 1)
        R_T = F.normalize(torch.randn(self.config.d_T, self.config.n_vectors, generator=generator), dim=0)
        return R_C.to(self.config.device), R_T.to(self.config.device)

    def _whiten_token_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply the fitted StandardScaler + whitening-PCA transform, if enabled."""
        if not self.transform_enabled:
            return embeddings
        x = (embeddings - self.scaler_mean) / self.scaler_scale
        x = (x - self.pca_mean) @ self.pca_components.T
        if self.pca_whiten:
            x = x / torch.sqrt(self.pca_explained_variance)
        return x

    def _project_token_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Whiten, project with R_T, and (optionally) normalize -> p_T."""
        x = self._whiten_token_embeddings(embeddings)
        if self.config.normalize_embeddings:
            x = F.normalize(x, dim=-1, eps=1e-12)
        x = x @ self.R_T
        if self.config.normalize_projected_vectors:
            x = F.normalize(x, dim=-1, eps=1e-12)
        return x

    def _project_context_embeddings(self, e_C: torch.Tensor) -> torch.Tensor:
        """Normalize, project with R_C, and (optionally) normalize -> p_C."""
        if self.config.normalize_embeddings:
            e_C = F.normalize(e_C, dim=-1, eps=1e-12)
        p_C = e_C @ self.R_C
        if self.config.normalize_projected_vectors:
            p_C = F.normalize(p_C, dim=-1, eps=1e-12)
        return p_C

    def _precompute_token_projections(self) -> torch.Tensor:
        """Precompute the projected token vectors p_T for the whole vocabulary.

        The lookup table has shape ``(vocab_size, n_vectors)``. It is kept on the
        model device unless it would claim a large share of GPU memory, in which
        case it falls back to CPU.
        """
        device = self.config.device
        embedding_layer = self.config.generation_model.get_input_embeddings()

        lookup_bytes = self.config.vocab_size * self.config.n_vectors * 4  # float32
        lookup_device = device
        if torch.device(device).type == 'cuda' and lookup_bytes > 0.25 * torch.cuda.get_device_properties(device).total_memory:
            lookup_device = torch.device('cpu')

        lookup = torch.zeros((self.config.vocab_size, self.config.n_vectors), device=lookup_device)
        batch_size = 1024
        with torch.no_grad():
            for start in range(0, self.config.vocab_size, batch_size):
                end = min(start + batch_size, self.config.vocab_size)
                token_ids = torch.arange(start, end, device=device)
                embeddings = embedding_layer(token_ids).to(torch.float32)
                lookup[start:end] = self._project_token_embeddings(embeddings).to(lookup_device)
        return lookup

    def _fetch_token_projections(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up precomputed p_T vectors, returning them on the query device."""
        lookup = self.token_projection_lookup
        if lookup.device == token_ids.device:
            return lookup[token_ids]
        return lookup[token_ids.to(lookup.device)].to(token_ids.device)

    def _mean_pooling(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """Attention-masked mean pooling over the context encoder's token states."""
        token_embeddings = model_output[0]
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def embed_contexts(self, context_texts: list[str]) -> torch.Tensor:
        """Encode context strings into projected, normalized vectors p_C.

        Returns a tensor of shape ``(len(context_texts), n_vectors)``.
        """
        inputs = self.config.context_tokenizer(
            context_texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.config.device)
        with torch.no_grad():
            outputs = self.config.context_model(**inputs)
        e_C = self._mean_pooling(outputs, inputs["attention_mask"]).to(torch.float32)
        return self._project_context_embeddings(e_C)

    def context_text(self, input_ids: torch.Tensor) -> str:
        """Decode the last ``context_window_size`` tokens preceding the next token."""
        window = input_ids[0, -self.config.context_window_size:]
        return self.config.generation_tokenizer.decode(window, skip_special_tokens=True)

    def _standardize(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the sqrt(n) standardization to a raw latent score."""
        return z * self.sqrt_n if self.config.standardize_score_by_sqrt_n else z

    def latent_scores(self, p_T: torch.Tensor, p_C: torch.Tensor) -> torch.Tensor:
        """Standardized latent score z = sqrt(n) * <p_T, p_C> for paired rows."""
        return self._standardize((p_T * p_C).sum(dim=-1))

    def compute_biases(self, input_ids: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        """Compute additive logit biases for candidate tokens given the context.

        Parameters:
            input_ids (torch.Tensor): Sequence generated so far, shape (1, seq_len).
            candidate_ids (torch.Tensor): Candidate token ids, shape (num_candidates,).

        Returns:
            torch.Tensor: Bias per candidate, shape (num_candidates,).
        """
        p_C = self.embed_contexts([self.context_text(input_ids)])  # (1, n)
        p_T = self._fetch_token_projections(candidate_ids)          # (num_candidates, n)
        z = self._standardize(p_T @ p_C.squeeze(0))                 # (num_candidates,)
        # tanh keeps the per-token bias within (-lambda, lambda) so outlier
        # scores cannot overwhelm the language model's own logits.
        if self.config.bias_use_tanh:
            z = torch.tanh(self.config.tanh_scale * z)
        return self.config.lambda_strength * z


class DEWLogitsProcessor(LogitsProcessor):
    """Logits processor for DEW algorithm, biases top-k candidates toward the key."""

    def __init__(self, config: DEWConfig, utils: DEWUtils, *args, **kwargs) -> None:
        """
            Initialize the DEW logits processor.

            Parameters:
                config (DEWConfig): Configuration for the DEW algorithm.
                utils (DEWUtils): Utility class for the DEW algorithm.
        """
        self.config = config
        self.utils = utils

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Restrict sampling to the top-k tokens and add the DEW bias."""
        top_k = min(self.config.top_k, scores.size(-1))
        top_k_logits, top_k_indices = torch.topk(scores, top_k, dim=-1)

        new_scores = torch.full_like(scores, -float("inf"))
        for b_idx in range(input_ids.shape[0]):
            bias = self.utils.compute_biases(input_ids[b_idx:b_idx + 1], top_k_indices[b_idx])
            biased_logits = (top_k_logits[b_idx] + bias).to(dtype=new_scores.dtype)
            new_scores[b_idx].scatter_(0, top_k_indices[b_idx], biased_logits)
        return new_scores


class DEW(BaseWatermark):
    """Top-level class for DEW algorithm."""

    def __init__(self, algorithm_config: str | DEWConfig, transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        """
            Initialize the DEW algorithm.

            Parameters:
                algorithm_config (str | DEWConfig): Path to the algorithm configuration file or DEWConfig instance.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if isinstance(algorithm_config, str):
            self.config = DEWConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, DEWConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be either a path string or a DEWConfig instance")

        self.utils = DEWUtils(self.config)
        self.logits_processor = DEWLogitsProcessor(self.config, self.utils)

        # Generation requires a pad token; fall back to eos when unset.
        if self.config.generation_tokenizer.pad_token is None:
            self.config.generation_tokenizer.pad_token = self.config.generation_tokenizer.eos_token
            self.config.generation_model.config.pad_token_id = self.config.generation_model.config.eos_token_id

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text."""
        encoded_prompt = self.config.generation_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.config.device)

        encoded_watermarked_text = self.config.generation_model.generate(
            **encoded_prompt,
            pad_token_id=self.config.generation_tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs,
        )
        return self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]

    def _score_sequence(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute the per-token latent scores z_i for a tokenized sequence.

        Every token from the second onward is scored using the same context
        window the logits processor saw when that token was generated, so
        detection mirrors generation exactly. Detection uses the linear
        standardized score, independent of the tanh mapping used at generation.
        """
        seq_len = input_ids.size(1)

        context_texts = [
            self.utils.context_text(input_ids[:, :idx]) for idx in range(1, seq_len)
        ]
        target_ids = input_ids[0, 1:]

        # Encode contexts in batches to bound memory on long inputs.
        p_C_batches = []
        batch_size = 256
        for start in range(0, len(context_texts), batch_size):
            p_C_batches.append(self.utils.embed_contexts(context_texts[start:start + batch_size]))
        p_C = torch.cat(p_C_batches, dim=0)

        p_T = self.utils._fetch_token_projections(target_ids).to(p_C.device)
        return self.utils.latent_scores(p_T, p_C)

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect a DEW watermark via a one-sided Gaussian test on the mean score.

        Under H0 each per-token score z_i is approximately N(0, 1), so the mean
        z_bar over k tokens yields test statistic ``sqrt(k) * z_bar`` and
        ``p_value = 1 - Phi(sqrt(k) * z_bar)``. Text is flagged watermarked when
        ``p_value < detection_alpha``.
        """
        input_ids = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.config.device)

        if input_ids.size(1) < 2:
            result = {"is_watermarked": False, "score": 0.0, "p_value": 1.0, "num_tokens": 0}
            return result if return_dict else (result["is_watermarked"], result["score"])

        with torch.no_grad():
            z = self._score_sequence(input_ids)

        z_bar = z.mean().item()
        num_tokens = z.numel()
        test_stat = math.sqrt(num_tokens) * z_bar
        p_value = 0.5 * math.erfc(test_stat / math.sqrt(2.0))
        is_watermarked = bool(p_value < self.config.detection_alpha)

        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_bar, "p_value": p_value, "num_tokens": num_tokens}
        return (is_watermarked, z_bar)

    def get_data_for_visualization(self, text: str, *args, **kwargs) -> DataForVisualization:
        """Get data for visualization: per-token positive/negative latent-score flags."""
        input_ids = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.config.device)

        decoded_tokens = [self.config.generation_tokenizer.decode(token_id) for token_id in input_ids[0]]

        if input_ids.size(1) < 2:
            return DataForVisualization(decoded_tokens, [-1] * len(decoded_tokens))

        with torch.no_grad():
            z = self._score_sequence(input_ids)

        # The first token has no preceding context and is left unscored (-1).
        highlight_values = [-1] + [1 if score > 0 else 0 for score in z.tolist()]
        return DataForVisualization(decoded_tokens, highlight_values)
