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
# cohemark.py
# Description: MarkLLM wrapper for CoheMark watermark
#   - Sentence-level semantic-space reject sampling (Fuzzy C-means membership)
#   - Detection via watermark_ratio
# ============================================

from __future__ import annotations

import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from transformers import GenerationConfig, StoppingCriteriaList

from ..base import BaseWatermark, BaseConfig
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError

# --- CoheMark dependencies (user-provided files) ---
from .utils import FuzzyModel
from .utils import SentenceEndCriteria, gen_sent
from .utils import threshold_detect

PUNCTS = ".!?"


class CoheMarkConfig(BaseConfig):
    """
    Config class for CoheMark algorithm.
    """

    def initialize_parameters(self) -> None:
        runtime_max_new = getattr(self.transformers_config, "max_new_tokens", None) if hasattr(self, "transformers_config") else None
        runtime_min_new = getattr(self.transformers_config, "min_new_tokens", None) if hasattr(self, "transformers_config") else None

        # max_new_tokens: prefer runtime value, then json, else default
        if runtime_max_new is not None:
            self.max_new_tokens = int(runtime_max_new)
        else:
            self.max_new_tokens = int(self.config_dict.get("max_new_tokens", 200))

        # min_new_tokens: CoheMark paper doesn't require it; default 0 unless runtime provided
        if runtime_min_new is not None:
            self.min_new_tokens = int(runtime_min_new)
        else:
            self.min_new_tokens = int(self.config_dict.get("min_new_tokens", 0))

        # reject sampling: max_trials (paper: stop if still cannot satisfy after some tries)
        self.max_trials = int(self.config_dict.get("max_trials", 30))

        # fuzzy model settings
        self.cluster_path = self.config_dict["cluster_path"]  # e.g., "centers-8-1.1-lfqa.pickle"
        self.m = float(self.config_dict.get("m", 1.1))
        self.K = int(self.config_dict.get("K", 8))
        self.embedder = self.config_dict.get("embedder", "sentence-transformers/all-mpnet-base-v1")

        # detection threshold on watermark_ratio (NOT z-score)
        self.ratio_threshold = float(self.config_dict.get("ratio_threshold", 0.9))

        # decoding params (keep close to your sampling scripts defaults)
        self.repetition_penalty = float(self.config_dict.get("repetition_penalty", 1.05))
        self.temperature = float(self.config_dict.get("temperature", 0.7))
        self.top_k = int(self.config_dict.get("top_k", 0))
        self.top_p = float(self.config_dict.get("top_p", 1.0))

        # build HF GenerationConfig
        self.gen_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            return_dict_in_generate=True,
        )

    @property
    def algorithm_name(self) -> str:
        return "CoheMark"


class CoheMarkUtils:
    """
    Helper class implementing CoheMark sentence-level reject sampling logic.
    """

    def __init__(self, config: CoheMarkConfig):
        self.config = config
        self.fuzzy_model = FuzzyModel(
            cluster_path=self.config.cluster_path,
            m=self.config.m,
            K=self.config.K,
            embedder=self.config.embedder,
        )

    @staticmethod
    def _is_one_sentence(text: str) -> bool:
        # your code: require ending punctuation and exactly one sentence after sent_tokenize
        if text == "":
            return False
        if text[-1] not in PUNCTS:
            return False
        return len(sent_tokenize(text)) == 1

    def _compute_green_ids(self, last_sent_membership_vec: np.ndarray, nums_samespace: int) -> list[int]:
        """
        Reproduce your K==8 logic.
        """
        sorted_indices = np.argsort(last_sent_membership_vec)

        if self.config.K == 8:
            if nums_samespace <= 5:
                # [-1] and [-3]
                return [int(sorted_indices[-1]), int(sorted_indices[-3])]
            else:
                # fallback mode with 4 green ids (and allow more trials in caller)
                return [
                    int(sorted_indices[-2]),
                    int(sorted_indices[-4]),
                    int(sorted_indices[-5]),
                    int(sorted_indices[-6]),
                ]

        # generic fallback: top1 and top3
        top1 = int(sorted_indices[-1])
        top3 = int(sorted_indices[-3]) if len(sorted_indices) >= 3 else top1
        return [top1, top3]

    def generate_watermarked(self, prompt: str, model, tokenizer, device: str | torch.device | None = None) -> str:
        """
        Generate prompt+completion, enforcing semantic-space watermark constraints per sentence.
        Stops when:
          - (generated_tokens - prompt_tokens) >= max_new_tokens - 1
          - OR cannot sample desired signature after max_trials (break)
        """
        if device is None:
            device = self.config.device
        device = str(device)

        sent_end_criteria = SentenceEndCriteria(tokenizer)

        text = prompt
        text_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_length = int(text_ids.size(1))

        # membership of last accepted sentence 
        last_sent_simi = self.fuzzy_model.calculate_membership([text]) 
        sent_end_criteria.update(prompt)

        current_trials = 0
        nums_samespace = 0

        # we dynamically expand max_trials when nums_samespace gets large
        local_max_trials = int(self.config.max_trials)

        while True:
            stopping_criteria = StoppingCriteriaList([sent_end_criteria])

            if "opt" in getattr(model.config, "_name_or_path", ""):
                new_sent, new_text_ids = gen_sent(
                    model=model,
                    tokenizer=tokenizer,
                    text_ids=text_ids,
                    gen_config=self.config.gen_config,
                    stopping_criteria=stopping_criteria,
                )
            else:
                raise NotImplementedError("CoheMark currently mirrors scripts: only OPT-like models supported.")

            # invalid sentence -> count trial and retry; stop if too many
            if (not new_sent) or (not self._is_one_sentence(new_sent)):
                current_trials += 1
                if current_trials >= local_max_trials:
                    # paper/code behavior: stop if cannot get a valid sample after max trials
                    break
                continue

            current_trials += 1

            # compute membership class of new sentence (argmax cluster id)
            new_sent_simi = self.fuzzy_model.calculate_membership([new_sent])  # (1,K)
            new_membership = int(np.argsort(new_sent_simi[0])[-1])

            # compute accept ids from last accepted sentence similarity distribution
            accept_ids = self._compute_green_ids(last_sent_simi[0], nums_samespace)

            # if we are in the "4-green-ids" mode, allow more trials (match your code)
            if self.config.K == 8 and len(accept_ids) == 4:
                local_max_trials = max(local_max_trials, 100)
            # stop if exceed trials budget
            if current_trials >= local_max_trials:
                break
            # reject if not in accept_ids
            if new_membership not in accept_ids:
                continue
            # accepted: update counters similar to your generate_response_* code
            if len(accept_ids) == 2 and new_membership == accept_ids[0]:
                nums_samespace += 1
            if len(accept_ids) == 4:
                nums_samespace = 0

            # commit sentence
            current_trials = 0
            last_sent_simi = self.fuzzy_model.calculate_membership([new_sent])
            text += new_sent
            text_ids = new_text_ids
            sent_end_criteria.update(text)

            # stop by max_new_tokens (paper/code: only max token is used)
            if (int(text_ids.size(1)) - prompt_length) >= (self.config.max_new_tokens - 1):
                break

        return text

    def detect_ratio(self, text: str) -> float | None:
        """
        Compute watermark_ratio via threshold_detect.
        Now using the first sentence as prompt, rest as completion. Please change the prompt extraction logic in different scenarios.
        """
        sents = sent_tokenize(text)
        if len(sents) < 2:
            return None

        prompt_gen = sents[0]
        completion_sents = sents[1:]

        # threshold_detect returns watermark_ratio or None
        return threshold_detect(sents=completion_sents, fuzzy_model=self.fuzzy_model, prompt_gen=prompt_gen)


class CoheMark(BaseWatermark):

    def __init__(self, algorithm_config: str | CoheMarkConfig, transformers_config: TransformersConfig | None = None, *args, **kwargs):
        if isinstance(algorithm_config, str):
            self.config = CoheMarkConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, CoheMarkConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be either a path string or a CoheMarkConfig instance")

        self.utils = CoheMarkUtils(self.config)
        self.max_trials = self.config.max_trials

    @classmethod
    def load(cls, algorithm_config: str, transformers_config: TransformersConfig | None = None, *args, **kwargs) -> "CoheMark":
        """
        Optional convenience load, consistent with some watermark implementations.
        """
        cfg_dict = load_config_file(algorithm_config)
        if cfg_dict.get("algorithm_name", "CoheMark") != "CoheMark":
            raise AlgorithmNameMismatchError("Config algorithm_name mismatch for CoheMark")
        return cls(algorithm_config, transformers_config, *args, **kwargs)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        return self.utils.generate_watermarked(
            prompt=prompt,
            model=self.config.generation_model,
            tokenizer=self.config.generation_tokenizer
        )

    def detect_watermark(self, text: str, *args, **kwargs) -> dict:
        """
        Detect watermark using watermark_ratio (NOT z-score).
        """
        ratio = self.utils.detect_ratio(text)
        if ratio is None:
            return {
                "is_watermarked": False,
                "score": None,
                "threshold": self.config.ratio_threshold,
                "meta": {"reason": "insufficient_sentences_or_detection_failed"},
            }

        return {
            "is_watermarked": bool(ratio >= self.config.ratio_threshold),
            "score": float(ratio),
            "threshold": self.config.ratio_threshold,
            "meta": {
                "detector": "watermark_ratio",
                "note": "Original CoheMark paper reports detection performance using AUC; threshold is scenario-dependent. You may also choose other cluster files based on your data domain. The default cluster is trained on the LFQA dataset. The min_length is not specified/used in the original CoheMark paper."
            },
        }

    def get_data_for_visualization(self, text: str, *args, **kwargs) -> dict:
        """
        Optional: provide minimal visualization payload.
        """
        ratio = self.utils.detect_ratio(text)
        return {
            "algorithm": "CoheMark",
            "watermark_ratio": ratio,
            "ratio_threshold": self.config.ratio_threshold,
        }
