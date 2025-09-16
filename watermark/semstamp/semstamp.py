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
# semstamp.py
# Description: Implementation of SEMSTAMP algorithm
# ============================================

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, models
from ..base import BaseWatermark, BaseConfig
from utils.transformers_config import TransformersConfig
from transformers import StoppingCriteria
from transformers.tokenization_utils import PreTrainedTokenizer
from nltk.tokenize import sent_tokenize
from transformers import StoppingCriteriaList
from typing import Callable, Iterator
from nearpy.hashes import RandomBinaryProjections
from scipy.spatial.distance import cosine
import torch.nn.functional as F


class SemStampConfig(BaseConfig):
    """Config class for SEMSTAMP algorithm.load config file and initialize parameters."""

    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.max_new_tokens = self.config_dict['max_new_tokens']
        self.min_new_tokens = self.config_dict['min_new_tokens']
        self.path_to_embedder = self.config_dict['path_to_embedder']
        self.N_max = self.config_dict['N_max']
        self.gamma = self.config_dict['gamma']
        self.margin_m = self.config_dict['margin_m']
        self.dimension_d = self.config_dict['dimension_d']
        self.prime_P = self.config_dict['prime_P']
        self.threshold = self.config_dict['threshold']

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return "SEMSTAMP"

# utils


class SemStampUtils:
    """Helper class for SEMSTAMP algorithm, contains helper functions."""

    def __init__(self, config: SemStampConfig, *args, **kwargs) -> None:
        """
            Initialize the SEMSTAMP utility class.

            Parameters:
                config (SemStampConfig): Configuration for the SEMSTAMP algorithm.
        """
        self.config = config
        self.rng = torch.Generator(device=self.config.device)

    class SBERTLSHModel:
        """Helper class for SBERTLSHModel"""

        def __init__(self, batch_size, lsh_dim, sbert_type='roberta', lsh_model_path=None, **kwargs):
            self.comparator: Callable[[np.ndarray, np.ndarray], float]
            self.do_lsh: bool = False
            self.dimension: int = -1
            self.batch_size: int = batch_size
            self.lsh_dim: int = lsh_dim
            print("initializing random projection LSH model")
            self.hasher = RandomBinaryProjections(
                'rbp_perm', projection_count=self.lsh_dim, rand_seed=1234)
            self.do_lsh = True
            self.comparator = lambda x, y: cosine(x, y)
            self.sbert_type = sbert_type
            self.dimension = 1024 if 'large' in self.sbert_type else 768

            print(f'loading SBERT {self.sbert_type} model...')
            if lsh_model_path is not None:
                word_embedding_model = models.Transformer(lsh_model_path)
                pooling_model = models.Pooling(
                    word_embedding_model.get_word_embedding_dimension(),
                    pooling_mode_mean_tokens=True
                )
                self.embedder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
                self.dimension = self.embedder.get_sentence_embedding_dimension()
            else:
                self.embedder = SentenceTransformer(
                    "sentence-transformers/all-mpnet-base-v1")
            self.embedder.eval()

            self.hasher.reset(dim=self.dimension)

        def get_embeddings(self, sents: Iterator[str]) -> np.ndarray:
            all_embeddings = self.embedder.encode(
                sents, batch_size=self.batch_size)
            return np.stack(all_embeddings)

        def get_hash(self, sents: Iterator[str]) -> Iterator[str]:
            embd = self.get_embeddings(sents)
            hash_strs = [self.hasher.hash_vector(e)[0] for e in embd]
            hash_ints = [int(s, 2) for s in hash_strs]
            return hash_ints

    @staticmethod
    def pairwise_cosine(data1, data2, device=torch.device('cpu')):
        data1, data2 = data1.to(device), data2.to(device)

        # N*1*M
        A = data1.unsqueeze(dim=1)

        # 1*N*M
        B = data2.unsqueeze(dim=0)

        # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
        A_normalized = A / A.norm(dim=-1, keepdim=True)
        B_normalized = B / B.norm(dim=-1, keepdim=True)

        cosine = A_normalized * B_normalized

        # return N*N matrix for pairwise distance
        cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
        return cosine_dis

    class SentenceEndCriteria(StoppingCriteria):
        """
        ONLY WORK WITH BATCH SIZE 1

        Stop generation whenever the generated string is **more than one** sentence (i.e. one full sentence + one extra token). this is determined by nltk sent_tokenize.
        Only stop if ALL sentences in the batch is at least two sentences

        Args:
            tokenizer (PreTrainedTokenizer):
            The exact tokenizer used for generation. MUST BE THE SAME!
        """

        def __init__(self, tokenizer: PreTrainedTokenizer):
            self.tokenizer = tokenizer
            self.current_num_sentences = 0

        def update(self, current_text):
            self.current_num_sentences = len(sent_tokenize(current_text))

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            assert input_ids.size(0) == 1
            text = self.tokenizer.decode(
                input_ids[0], skip_special_tokens=True)
            return len(sent_tokenize(text)) > self.current_num_sentences + 1

# main class


class SemStamp(BaseWatermark):
    """Top-level class for the SEMSTAMP algorithm."""

    def __init__(self, algorithm_config: str | SemStampConfig, transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        """
            Initialize the SEMSTAMP algorithm.

            Parameters:
                algorithm_config (str | SEMSTAMPConfig): Path to the algorithm configuration file or SEMSTAMPConfig instance.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if isinstance(algorithm_config, str):
            self.config = SemStampConfig(
                algorithm_config, transformers_config)
        elif isinstance(algorithm_config, SemStampConfig):
            self.config = algorithm_config
        else:
            raise TypeError(
                "algorithm_config must be either a path string or a SemStampConfig instance")

        self.utils = SemStampUtils(self.config)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text using the SEMSTAMP algorithm."""

        # get LSH
        lsh_model = self.utils.SBERTLSHModel(
            lsh_model_path=self.config.path_to_embedder, batch_size=1, lsh_dim=self.config.dimension_d, sbert_type='base'
        )

        # instantiate sentence end criteria
        sent_end_criteria = SemStampUtils.SentenceEndCriteria(
            self.config.generation_tokenizer)

        # lsh reject sampling

        # get valid mask
        lsh_seed = lsh_model.get_hash([prompt])[0]
        n_bins = 2**self.config.dimension_d
        n_accept = int(n_bins * self.config.gamma)
        self.utils.rng.manual_seed(self.config.prime_P * lsh_seed)
        # randomization
        vocab_permutation = torch.randperm(
            n_bins, device=self.config.device, generator=self.utils.rng)
        accept_mask = vocab_permutation[:n_accept]

        # start sampling
        text = prompt
        new_text = prompt
        text_ids = self.config.generation_tokenizer.encode(
            prompt, return_tensors='pt').to(self.config.device)
        prompt_length = len(text_ids[0])
        sent_end_criteria.update(new_text)

        total_trials = 0
        current_trials = 0
        maxedout_trials = 0
        while True:
            stopping_criteria = StoppingCriteriaList([sent_end_criteria])
            if "opt" in self.config.generation_model.config._name_or_path:
                num_candidates = 8
                outputs = self.config.generation_model.generate(text_ids,
                                                                max_new_tokens=self.config.max_new_tokens,
                                                                min_new_tokens=self.config.min_new_tokens,
                                                                do_sample=True,
                                                                temperature=0.7,
                                                                top_k=0,
                                                                repetition_penalty=1.05,
                                                                stopping_criteria=stopping_criteria,
                                                                )
                new_text_ids = outputs[:, :-1]
                new_text = self.config.generation_tokenizer.decode(
                    new_text_ids[0, text_ids.size(1):], skip_special_tokens=True)
            else:
                raise NotImplementedError("model type not supported")
            if new_text == '':
                print(
                    'WARNING: stopped generation because generated nothing (after discarding last generated token)', flush=True)
                break
            total_trials += 1
            current_trials += 1
            embeds = lsh_model.get_embeddings([new_text])
            embeds = torch.tensor(embeds, device=self.config.device)
            normals = torch.tensor(lsh_model.hasher.normals, device=self.config.device)
            # sims[i, j] is the cosine similarity between the ith generation and the jth normal vec
            sims=F.cosine_similarity(
        embeds.view(embeds.size(0), 1, embeds.size(1))
        .expand(embeds.size(0), normals.size(0), embeds.size(1))
        .contiguous()
        .view(-1, embeds.size(1)),
        normals.expand(embeds.size(0), normals.size(0), normals.size(1)).flatten(end_dim=1),
    ).view(embeds.size(0), normals.size(0))
            sims_abs = torch.abs(sims)
            min_sims = sims_abs.min(dim=1).values
            select = []
            for i in range(len(min_sims)):
                min_sim = min_sims[i].item()
                if (abs(min_sim) >= self.config.margin_m):
                    select.append(i)
            if(len(select)==0):
                continue
            [new_text] = np.array([new_text])[select]
            accepted_text = list(new_text)
            if (len(accepted_text) == 0 and current_trials < self.config.N_max):
                continue
            lsh_candidate = lsh_model.get_hash([new_text])[0]
            if lsh_candidate not in accept_mask:
                continue
            if (lsh_candidate in accept_mask) or current_trials >= self.config.N_max:
                if current_trials >= self.config.N_max:
                    print(
                        f'WARNING: desired semantic signature can\'t be sampled after max_trials {self.config.N_max}', flush=True)
                    print(f'CONTEXT: {text}', flush=True)
                    print(
                        f'NOTE: use regular (non-filtered-by-sig) continuation: {new_text}', flush=True)
                    maxedout_trials += 1
                current_trials = 0
                # passed, proceed to next sentence
                lsh_seed = lsh_candidate
                self.utils.rng.manual_seed(self.config.prime_P * lsh_seed)
                vocab_permutation = torch.randperm(
                    n_bins, device=self.config.device, generator=self.utils.rng)
                accept_mask = vocab_permutation[:n_accept]
                text += new_text
                text_ids = new_text_ids.to(self.config.device)
                sent_end_criteria.update(text)
                if (len(text_ids[0]) - prompt_length) >= self.config.max_new_tokens-1:
                    break
            watermarked_text = text.strip()
        return watermarked_text

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the input text."""
        # get embedder
        word_embedding_model = models.Transformer(self.config.path_to_embedder)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
        embedder = SentenceTransformer(
            modules=[word_embedding_model, pooling_model])

        sentences = sent_tokenize(text)
        n_sent = len(sentences)
        n_watermark = 0

        lsh_model = self.utils.SBERTLSHModel(
            lsh_model_path=self.config.path_to_embedder, batch_size=1, lsh_dim=self.config.dimension_d, sbert_type='base')
        lsh_seed = lsh_model.get_hash([sentences[0]])[0]
        n_bins = 2**self.config.dimension_d
        n_accept = int(n_bins * self.config.gamma)
        self.utils.rng.manual_seed(self.config.prime_P * lsh_seed)
        vocab_permutation = torch.randperm(n_bins, device=self.config.device, generator=self.utils.rng)
        accept_mask = vocab_permutation[:n_accept]
        
        for i in range(1, n_sent):
            lsh_candidate = lsh_model.get_hash([sentences[i]])[0]
            if lsh_candidate in accept_mask:
                n_watermark += 1
            lsh_seed = lsh_candidate
            self.utils.rng.manual_seed(self.config.prime_P * lsh_seed)
            vocab_permutation = torch.randperm(n_bins, device=self.config.device, generator=self.utils.rng)
            accept_mask = vocab_permutation[:n_accept]
        n_test_sent = n_sent - 1  # exclude the prompt and the ending
        num = n_watermark - self.config.gamma * (n_test_sent)
        denom = np.sqrt((n_test_sent) * self.config.gamma * (1-self.config.gamma))
        z_score = num / denom

        # Determine if the computed score exceeds the threshold for watermarking
        is_watermarked = z_score > self.config.threshold

        # Return results based on the `return_dict` flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)
