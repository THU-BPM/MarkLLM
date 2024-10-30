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
# kgw.py
# Description: Implementation of KGW algorithm
# ============================================


import torch
import scipy.stats
from math import sqrt
from functools import partial
from watermark.base import BaseWatermark
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from transformers import LogitsProcessor, LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization
# from MarkLLM.watermark.TS.ts_config import TSConfig
from transformers import AutoModel, OPTForCausalLM, AutoTokenizer, LogitsProcessorList
from watermark.ts.TS_networks import DeltaNetwork, GammaNetwork

class TSConfig:
    """Config class for KGW algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the KGW configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """

        if algorithm_config is None:
            config_dict = load_config_file('config/TS.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'TS':
            raise AlgorithmNameMismatchError('TS', config_dict['algorithm_name'])

 ################### TS_Watermark ###################################

        self.hash_key = config_dict['hash_key']

        self.seeding_scheme = config_dict['seeding_scheme']

        self.ckpt_path = config_dict['ckpt_path']

        self.gamma = config_dict['gamma']

        self.delta = config_dict['delta']

        self.prefix_length = config_dict['prefix_length']


        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.z_threshold = config_dict['z_threshold']
        self.device = transformers_config.device

        self.gen_kwargs = transformers_config.gen_kwargs



 ################### TS_Watermark ###################################

class TSUtils:
    """Utility class for KGW algorithm, contains helper functions."""

    def __init__(self, config: TSConfig, *args, **kwargs) -> None:
        """
            Initialize the KGW utility class.

            Parameters:
                config (KGWConfig): Configuration for the KGW algorithm.
        """
        self.config = config
        self.device = self.config.device
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.config.hash_key)
        self.ckpt_path = self.config.ckpt_path
        self.seeding_scheme = self.config.seeding_scheme
        self.hash_key = self.config.hash_key
        self.vocab_size = self.config.vocab_size

        model = OPTForCausalLM.from_pretrained(
            "facebook/opt-1.3b",
            torch_dtype=torch.float,
            device_map='auto'
        )


        self.tokenizer_llama = self.config.generation_tokenizer
        self.tokenizer_opt = AutoTokenizer.from_pretrained("facebook/opt-1.3b", padding_side="left")

        self.embed_matrix = model.get_input_embeddings().weight.to(self.device)

        ckpt_path = self.config.ckpt_path


        ############################ TS_Watermark ####################333#####

        if len(ckpt_path) > 0:
            print("checkpoint_load")
            checkpoint = torch.load(ckpt_path)
            layer_delta = sum(1 for key in checkpoint['delta_state_dict'] if "weight" in key)  # Counting only weight keys as layers
            layer_gamma = sum(1 for key in checkpoint['gamma_state_dict'] if "weight" in key)  # Counting only weight keys as layers

            self.gamma_network = GammaNetwork(input_dim=self.embed_matrix.shape[1], layers=layer_gamma).to(self.device)
            self.delta_network = DeltaNetwork(input_dim=self.embed_matrix.shape[1], layers=layer_delta).to(self.device)

            self.delta_network.load_state_dict(checkpoint['delta_state_dict'])
            self.gamma_network.load_state_dict(checkpoint['gamma_state_dict'])

            for name, param in self.delta_network.named_parameters():
                param.requires_grad = False
            for name, param in self.gamma_network.named_parameters():
                param.requires_grad = False
            self.delta_network.eval()
            self.gamma_network.eval()


        else:
            self.gamma = torch.tensor([gamma]).to(device)
            self.delta = torch.tensor([delta]).to(device)



        self.gamma_list = torch.empty(0, dtype=torch.float).to(self.device)
        self.delta_list = torch.empty(0, dtype=torch.float).to(self.device)




    def _seed_rng(self, input_ids: torch.LongTensor, seeding_scheme: str = None) -> None:

        # can optionally override the seeding scheme,
        # but uses the instance attr by default
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme


        # using the last token and hash_key to generate random seed
        if seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1, f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            prev_token = input_ids[-1].item()
            self.rng.manual_seed(self.hash_key * prev_token)
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
        return

    def _get_greenlist_ids(self, input_ids: torch.LongTensor,process) -> list[int]:
        # Always use ids given by OPT model
        # since our gamma/delta network is trained on the embedding matrix of OPT model
        # seed the rng using the previous tokens/prefix according to the seeding_scheme
        self._seed_rng(input_ids)

        # use last token to get gamma value and delta value
        if len(self.ckpt_path) > 0:
            gamma = self.gamma_network(self.embed_matrix[input_ids[-1].item()])
            delta = self.delta_network(self.embed_matrix[input_ids[-1].item()])
        else:
            delta = self.delta
            gamma = self.gamma

        if process == 'process':
        # get every token's gamma value and delta value
          self.gamma_list = torch.cat([self.gamma_list, gamma])
          self.delta_list = torch.cat([self.delta_list, delta])



        # generate greenlist, every token have different greenlist_id
        greenlist_size = int(self.vocab_size * gamma)
        vocab_permutation = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)

        greenlist_ids = vocab_permutation[:greenlist_size]

        gamma_len=len(self.gamma_list)

        return greenlist_ids, gamma, delta,gamma_len



    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        var = torch.sum(self.gamma_list * (1 - self.gamma_list))
        mean = torch.sum(self.gamma_list)
        z = (observed_count - mean)/torch.sqrt(var)
        return z

    def _score_sequence(
        self,
        input_ids: torch.Tensor,

    )-> tuple[float, list[int]]:

        num_tokens_scored = len(input_ids) - self.config.prefix_length
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                    f"the first min_prefix_len={self.config.prefix_length} tokens required by the seeding scheme."
                )
            )

        green_token_count = 0
        # green_token_mask = []
        green_token_mask = [-1 for _ in range(self.config.prefix_length)]

        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx]
            if "opt" in self.config.generation_model.name_or_path.lower():

                greenlist_ids, gamma, delta,gamma_len = self._get_greenlist_ids(input_ids[:idx],"detect")


            else:
                llama_str = self.tokenizer_llama.decode(input_ids[max(idx-5, 0):idx], add_special_tokens=False)
                ids_opt = self.tokenizer_opt(llama_str, add_special_tokens=False)['input_ids']
                if len(ids_opt) == 0:
                    green_token_mask.append(False)
                    continue
                greenlist_ids, gamma, delta,gamma_len = self._get_greenlist_ids(torch.tensor(ids_opt).to(self.device),"detect")


            # exit()
            if curr_token in greenlist_ids:
                green_token_count += 1
                green_token_mask.append(True)
            else:
                green_token_mask.append(False)

        self.gamma_list=self.gamma_list[self.config.prefix_length:]

        z_score = self._compute_z_score(green_token_count, num_tokens_scored)



        return z_score, green_token_mask




class TSLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for TS algorithm, process logits to add watermark."""

    def __init__(self, config: TSConfig, utils: TSUtils, *args, **kwargs) -> None:

        self.config = config
        self.utils = utils

################### TS_Watermark ###################################
        self.tokenizer_llama = self.config.generation_tokenizer
        self.tokenizer_opt = AutoTokenizer.from_pretrained("facebook/opt-1.3b", padding_side="left")

        if "opt" not in self.config.generation_model.name_or_path.lower():
            self.vocab_size = len(self.tokenizer_llama)




    def _calc_greenlist_mask(self, logits: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        green_tokens_mask = torch.zeros_like(logits)
        green_tokens_mask[greenlist_token_ids] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        device = input_ids.device
        if self.utils.rng is None:
            self.utils.rng = torch.Generator(device=device)

        if "opt" not in self.config.generation_model.name_or_path.lower():
            llama_str = self.tokenizer_llama.batch_decode(input_ids[:,-5:], add_special_tokens=False)
            ids_opt = self.tokenizer_opt(llama_str, add_special_tokens=False)['input_ids']

        for b_idx in range(input_ids.shape[0]):

            if "opt" not in self.config.generation_model.name_or_path.lower():
                greenlist_ids, gamma, delta,gamma_len = self.utils._get_greenlist_ids(torch.tensor(ids_opt[b_idx]).to(device),'process')

            else:
                greenlist_ids, gamma, delta,gamma_len = self.utils._get_greenlist_ids(input_ids[b_idx],'process')

            # get greenlist token mask and add bias on the each logits base on greenlist mask



            green_tokens_mask = self._calc_greenlist_mask(logits=scores[b_idx], greenlist_token_ids=greenlist_ids)




            delta = delta.to(dtype=scores.dtype)
            scores[b_idx][green_tokens_mask] = scores[b_idx][green_tokens_mask] + delta

        return scores




class TS(BaseWatermark):
    """Top-level class for KGW algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the KGW algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = TSConfig(algorithm_config, transformers_config)
        self.utils = TSUtils(self.config)
        self.logits_processor = TSLogitsProcessor(self.config, self.utils)


    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs
        )


        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)

        # print(encoded_prompt)

        prefix_length = encoded_prompt['input_ids'].shape[1]

        # Generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # Decode
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text[:,prefix_length:], skip_special_tokens=True)[0]
        return watermarked_text

    def generate_unwatermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate unwatermarked text."""


        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)

        prefix_length = encoded_prompt['input_ids'].shape[1]

        # Generate unwatermarked text
        encoded_unwatermarked_text = self.config.generation_model.generate(**encoded_prompt, **self.config.gen_kwargs)
        # Decode
        unwatermarked_text = self.config.generation_tokenizer.batch_decode(encoded_unwatermarked_text[:,prefix_length:], skip_special_tokens=True)[0]
        return unwatermarked_text




    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the text."""

        # Encode the text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)

        # Compute z_score using a utility method
        z_score, _ = self.utils._score_sequence(encoded_text)

        # Determine if the z_score indicates a watermark
        is_watermarked = z_score > self.config.z_threshold

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)

    def get_data_for_visualization(self, text: str, *args, **kwargs) -> tuple[list[str], list[int]]:
        """Get data for visualization."""

        # Encode text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)

        # Compute z-score and highlight values
        z_score, highlight_values = self.utils._score_sequence(encoded_text)

        # decode single tokens
        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)

        return DataForVisualization(decoded_tokens, highlight_values)