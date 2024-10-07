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

# ==============================================
# xsir.py
# Description: Implementation of X-SIR algorithm
# ==============================================

import os
import json
import torch
import numpy as np
from typing import Union
from functools import partial
from ..base import BaseWatermark
from utils.utils import load_config_file
from .transform_model import TransformModel
from sentence_transformers import SentenceTransformer
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from transformers import LogitsProcessor, LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization
from .generate_semantic_mappings import generate_semantic_mappings


class XSIRConfig:
    """Config class for X-SIR algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the X-SIR configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/XSIR.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'XSIR':
            raise AlgorithmNameMismatchError('XSIR', config_dict['algorithm_name'])

        # Load config
        self.delta = config_dict['delta']
        self.chunk_length = config_dict['chunk_length']
        self.scale_dimension = config_dict['scale_dimension']
        self.z_threshold = config_dict['z_threshold']
        self.transform_model_input_dim = config_dict['transform_model_input_dim']
        self.transform_model_name = config_dict['transform_model_name']
        self.embedding_model_path = config_dict['embedding_model_path']
        self.mapping_name = config_dict['mapping_name']
        self.dictionary = config_dict.get('dictionary', None)

        # Load transformer model's configuration
        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs


class XSIRUtils:
    """Utility class for X-SIR algorithm, contains helper functions."""

    def __init__(self, config: XSIRConfig, *args, **kwargs) -> None:
        """
            Initialize the X-SIR utility class.

            Parameters:
                config (XSIRConfig): Configuration for the SIR algorithm.
        """
        self.config = config
        # import ipdb; ipdb.set_trace()
        # Load the embedding model: mapping a sentence to an embedding
        self.embedding_model = SentenceTransformer(self.config.embedding_model_path).to(self.config.device)

        # Load the transform model: mapping a sentence embedding to logit bias
        self.transform_model = self._get_transform_model(
            model_name=self.config.transform_model_name,
            input_dim=config.transform_model_input_dim
        ).to(self.config.device, dtype=torch.float32)
        self.mapping = self._get_mapping(self.config.mapping_name)

    def get_embedding(self, sentence: str) -> torch.FloatTensor:
        """Get the embedding of the input sentence."""
        emb = self.embedding_model.encode(sentence, show_progress_bar=False, convert_to_tensor=True).to(self.config.device)
        return emb[None, :]

    def get_text_split(self, sentence: str) -> list[list[str]]:
        """Get the input sentence split into chunks."""
        tokens = self.config.generation_tokenizer.tokenize(sentence, add_special_tokens=False)
        return [tokens[x: x + self.config.chunk_length] for x in range(0, len(tokens), self.config.chunk_length)]

    def scale_vector(self, v: np.array) -> np.array:
        """Scale the input vector."""
        mean = np.mean(v)
        v_minus_mean = v - mean
        v_minus_mean = np.tanh(1000 * v_minus_mean)
        return v_minus_mean

    def get_context_sentence(self, input_ids: torch.LongTensor) -> str:
        """Get the context sentence from the input_ids."""
        input_sentence = self.config.generation_tokenizer.decode(input_ids, skip_special_tokens=True)
        input_tokens = self.config.generation_tokenizer.tokenize(input_sentence, add_special_tokens=False)

        # Split the input sentence into chunks
        chunks = [input_tokens[x: x + self.config.chunk_length] for x in range(0, len(input_tokens), self.config.chunk_length)]

        if len(chunks) == 0:
            return ""
        
        # If the last chunk is of the same length as the chunk length, return the whole sentence
        if len(chunks[-1]) == self.config.chunk_length:
            return input_sentence
        else:
            # Otherwise, return the sentence without the last chunk
            return self.config.generation_tokenizer.convert_tokens_to_string([tok for group in chunks[:-1] for tok in group])

    def _get_mapping(self, mapping_name: str) -> list[int]:
        """Get the mapping."""

        # try loading mapping from the provided mapping path
        try:
            with open(mapping_name, 'r') as f:
                mapping = json.load(f)

        # if the file does not exist, create a new mapping and save it to the provided mapping path
        except Exception as e:
            print(f"Error loading provided mapping file ({mapping_name}): {e}")
            print(f"Try to generate new mapping file ...")

            assert self.config.dictionary is not None and os.path.isfile(self.config.dictionary), \
                "Please provide a valid dictionary file for generating the mapping."
            assert self.config.scale_dimension is not None and self.config.scale_dimension > 0, \
                "Please provide a valid scale dimension for generating the mapping."

            mapping = generate_semantic_mappings(
                tokenizer=self.config.generation_tokenizer,
                dictionary=self.config.dictionary,
                output_file=mapping_name,
                dimension=self.config.scale_dimension,
                vocab_size=self.config.vocab_size
            )

            print(f"Generated new mapping file: {mapping_name}")
        return mapping

    def _get_transform_model(self, model_name: str, input_dim: int) -> TransformModel:
        """Get the transform model from the provided model name."""
        model = TransformModel(input_dim=input_dim)
        model.load_state_dict(torch.load(model_name))
        return model


class XSIRLogitsProcessor(LogitsProcessor):
    """Logits processor for X-SIR algorithm."""

    def __init__(self, config: XSIRConfig, utils: XSIRUtils, *args, **kwargs):
        """
            Initialize the X-SIR logits processor.

            Parameters:
                config (XSIRConfig): Configuration for the X-SIR algorithm.
                utils (XSIRUtils): Utility class for the X-SIR algorithm.
        """
        self.config = config
        self.utils = utils

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        batched_bias = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            current_bias = self._get_bias(input_ids[b_idx])
            batched_bias[b_idx] = current_bias

        scores = self._bias_logits(scores=scores, batched_bias=batched_bias)
        return scores

    def _bias_logits(self, scores: torch.Tensor, batched_bias: list[np.ndarray]) -> torch.Tensor:
        """Bias the scores for the greenlist tokens."""
        batched_bias_np = np.array(batched_bias) 
        batched_bias_tensor = torch.Tensor(batched_bias_np).to(self.config.device)
        scores = scores + batched_bias_tensor * self.config.delta
        return scores

    def _get_bias(self, input_ids: torch.LongTensor) -> np.ndarray:
        """Calculate the bias for the given input_ids."""
        context_sentence = self.utils.get_context_sentence(input_ids)
        context_embedding = self.utils.get_embedding(context_sentence)
        output = self.utils.transform_model(context_embedding).cpu()[0].numpy()
        similarity_array = self.utils.scale_vector(output)[self.utils.mapping]
        return -similarity_array


class XSIR(BaseWatermark):
    """Top-level class for X-SIR algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the X-SIR algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = XSIRConfig(algorithm_config, transformers_config)
        self.utils = XSIRUtils(self.config)
        self.logits_processor = XSIRLogitsProcessor(self.config, self.utils)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs):
        """Generate watermarked text."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]), 
            **self.config.gen_kwargs
        )

        # encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)

        # generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)

        # decode
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]

        return watermarked_text

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[dict[str, Union[bool, float]], tuple[bool, float]]:
        """Detect watermark in the input text."""
        z_score = None
        is_watermarked = None

        all_value = [] # store all bias values
        chunks = self.utils.get_text_split(text) # split text into chunks

        if len(chunks) <= 1:
            print("Text is too short to detect watermark. Returning False.")
            is_watermarked, z_score = False, 0

        # calculate bias for each token
        for cid in range(1, len(chunks)):
            context_sentence = self.config.generation_tokenizer.convert_tokens_to_string([t for c in chunks[0: cid] for t in c])
            context_embedding = self.utils.get_embedding(context_sentence)
            output = self.utils.transform_model(context_embedding).cpu().detach()[0].numpy()
            similarity_array = self.utils.scale_vector(output)[self.utils.mapping]

            tokens = chunks[cid]
            token_ids = self.config.generation_tokenizer.convert_tokens_to_ids(tokens)

            for tok_ids in token_ids:
                all_value.append(-float(similarity_array[tok_ids]))

        # z_score is the mean of all bias values
        z_score = np.mean(all_value)

        # Determine if the z_score indicates a watermark
        is_watermarked = z_score > self.config.z_threshold # TODO: what if we have no `z_threshold`?

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)

    def get_data_for_visualization(self, text: str, *args, **kwargs):
        """Get data for visualization."""
        highlight_values = []
        decoded_tokens = []
        chunks = self.utils.get_text_split(text) # split text into chunks

        if len(chunks) <= 1:
            # If the sentence is shorter than required, append highlight -1 for each token
            decoded_tokens = chunks[0]
            highlight_values = [-1] * len(decoded_tokens)

        # calculate bias for each token
        for cid in range(1, len(chunks)):
            context_sentence = self.config.generation_tokenizer.convert_tokens_to_string([t for c in chunks[0: cid] for t in c])
            context_embedding = self.utils.get_embedding(context_sentence)
            output = self.utils.transform_model(context_embedding).cpu().detach()[0].numpy()
            similarity_array = self.utils.scale_vector(output)[self.utils.mapping]

            tokens = chunks[cid]
            token_ids = self.config.generation_tokenizer.convert_tokens_to_ids(tokens)

            for tok_ids in token_ids:
                similarity_value = -float(similarity_array[tok_ids])
                highlight_values.append(1 if similarity_value > 0 else 0)
                decoded_tokens.append(self.config.generation_tokenizer.convert_ids_to_tokens(tok_ids))

        return DataForVisualization(decoded_tokens, highlight_values)