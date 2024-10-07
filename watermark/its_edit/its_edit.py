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

# ================================================
# exp_edit.py
# Description: Implementation of EXPEdit algorithm
# ================================================

import torch
import numpy as np
from math import log
from ..base import BaseWatermark
from .mersenne import MersenneRNG
from utils.utils import load_config_file
from .cython_files.levenshtein import levenshtein
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from visualize.data_for_visualization import DataForVisualization


class ITSEditConfig:
    """Config class for EXPEdit algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the EXPEdit configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/ITSEdit.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'ITSEdit':
            raise AlgorithmNameMismatchError('ITSEdit', config_dict['algorithm_name'])

        self.pseudo_length = config_dict['pseudo_length']
        self.sequence_length = config_dict['sequence_length']
        self.n_runs = config_dict['n_runs']
        self.p_threshold = config_dict['p_threshold']
        self.key = config_dict['key']
        self.top_k = config_dict['top_k']

        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs


class ITSEditUtils:
    """Utility class for EXPEdit algorithm, contains helper functions."""

    def __init__(self, config: ITSEditConfig, *args, **kwargs) -> None:
        """
            Initialize the EXPEdit utility class.

            Parameters:
                config (EXPEditConfig): Configuration for the EXPEdit algorithm.
        """
        self.config = config
        torch.manual_seed(self.config.key)
        self.rng = MersenneRNG(self.config.key)

    def transform_sampling(self,probs,pi,xi):
        """Sample token using inverse transform method."""
        cdf = torch.cumsum(torch.gather(probs, 1, pi), 1)
        return torch.gather(pi, 1, torch.searchsorted(cdf, xi))
    
    def transform_key_func(self,generator,n,vocab_size):
        """Generate key for watermark."""
        pi = torch.randperm(vocab_size, generator=generator)
        xi = torch.rand((n,1), generator=generator)
        return xi,pi
    
    def value_transformation(self, value: float) -> float:
        """Transform value to range [0, 1]."""
        return value/(value + 1)

    def phi(self,tokens,n,k,generator,key_func,vocab_size,dist,null=False,normalize=False):
        '''Compute the test statistic for the watermark detection algorithm.'''
        eff_vocab_size = vocab_size
        xi,pi = key_func(generator,n,vocab_size)
        tokens = torch.argsort(pi)[tokens]
        if normalize:
            tokens = tokens.float() / vocab_size
        
        A = self.adjacency(tokens,xi,dist,k)
        closest = torch.min(A,axis=1)[0]
        min_value, min_index = torch.min(closest, 0)
        return min_value, min_index.item()

    def adjacency(self,tokens,xi,dist,k):
        '''Compute the adjacency matrix for the test statistic.'''
        m = len(tokens)
        n = len(xi)
        A = torch.empty(size=(m-(k-1),n))
        for i in range(m-(k-1)):
            for j in range(n):
                A[i][j] = dist(tokens[i:i+k],xi[(j+torch.arange(k))%n])

        return A

class ITSEdit(BaseWatermark):
    """Top-level class for the EXPEdit algorithm."""
    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        self.config = ITSEditConfig(algorithm_config, transformers_config)
        self.utils = ITSEditUtils(self.config)
    
    def generate(self,model,prompts,vocab_size,n,m,seeds,key_func,sampler,random_offset=False):
        '''Generate watermark tokens.'''
        batch_size = len(prompts)
        generator = torch.Generator()
        xis,pis = [],[]
        # Generate keys
        for seed in seeds:
            generator.manual_seed(int(seed))
            xi,pi = key_func(generator,n,vocab_size)
            xis.append(xi.unsqueeze(0))
            pis.append(pi.unsqueeze(0))
        xis = torch.vstack(xis)
        pis = torch.vstack(pis)
        offset = torch.zeros(size=(batch_size,),dtype=torch.int64)
        inputs = prompts.to(model.device)
        attn = torch.ones_like(inputs)
        past = None
        # Generate tokens
        for i in range(m):
            with torch.no_grad():
                if past:
                    output = model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
                else:
                    output = model(inputs)
            probs = torch.nn.functional.softmax(output.logits[:,-1], dim=-1).cpu()
            tokens = sampler(probs, pis, xis[torch.arange(batch_size),(offset.squeeze()+i)%n]).to(model.device)
            inputs = torch.cat([inputs, tokens], dim=-1)

            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

        return inputs.detach().cpu()
    
    def clip(self,tokens):
        '''Clip tokens to remove padding.'''
        eos = torch.where(tokens == 2)[0] # find instances of the EOS token
        if len(eos) > 0:
            # truncate after the first EOS token (end of the response)
            tokens = tokens[:eos[0]]

        return tokens
    
    def generate_unwatermarked_text(self, prompt: str, *args, **kwargs) -> str:
        '''Generate unwatermarked text.'''
        def generate_rnd(prompts,m,model):
            inputs = prompts.to(model.device)
            attn = torch.ones_like(inputs)
            past = None
            for i in range(m):
                torch.manual_seed(self.config.key)
                with torch.no_grad():
                    output = model(inputs)

                probs = torch.nn.functional.softmax(output.logits[:,-1], dim=-1)
                tokens = torch.multinomial(probs,1)
                inputs = torch.cat([inputs, tokens], dim=1)

                past = output.past_key_values
                attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
            
            return inputs.detach().cpu()
        encoded_prompt = self.config.generation_tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=2048)[0].to(self.config.device)
        prompt_tokens = len(encoded_prompt)
        unwatermarked_prompt=generate_rnd(encoded_prompt.unsqueeze(0),self.config.sequence_length,self.config.generation_model)[0,prompt_tokens:]
        return self.config.generation_tokenizer.decode(unwatermarked_prompt, skip_special_tokens=True)
    
    def generate_watermarked_text(self, prompt: str, *args, **kwargs):
        """Generate watermarked text."""
        seeds = torch.randint(2**32, (self.config.sequence_length,))
        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=2048)[0].to(self.config.device)
        prompt_tokens = len(encoded_prompt)
        generate_watermark = lambda prompt,seed : self.generate(self.config.generation_model,
                                                       prompt,
                                                       self.config.vocab_size,
                                                       self.config.pseudo_length,
                                                       self.config.sequence_length,
                                                       seed,
                                                       self.utils.transform_key_func,
                                                       self.utils.transform_sampling,
                                                       random_offset=True)
        watermark_sample=generate_watermark(encoded_prompt.unsqueeze(0), seeds[0].unsqueeze(0))[0,prompt_tokens:]
        watermarked_sample = self.clip(watermark_sample)
        return self.config.generation_tokenizer.decode(watermarked_sample, skip_special_tokens=True)
    
    def permutation_test(self,tokens,vocab_size,n,k,seed,test_stat,n_runs=100,max_seed=100000):
        '''Permutation test for watermark detection.'''
        torch.manual_seed(self.config.key)
        generator = torch.Generator()
        generator.manual_seed(int(seed))

        test_result,_= test_stat(tokens=tokens,
                                n=n,
                                k=k,
                                generator=generator,
                                vocab_size=vocab_size)
        p_val = 0
        for run in range(n_runs):
            pi = torch.randperm(vocab_size)
            tokens = torch.argsort(pi)[tokens]
            
            seed = torch.randint(high=max_seed,size=(1,)).item()
            generator.manual_seed(int(seed))
            null_result,_ = test_stat(tokens=tokens,
                                    n=n,
                                    k=k,
                                    generator=generator,
                                    vocab_size=vocab_size,
                                    null=True)
            # assuming lower test values indicate presence of watermark
            p_val += (null_result <= test_result).float() / n_runs
        
        return p_val
    
    def transform_edit_score(self,tokens,xi,gamma=1):
        '''Compute the edit score between two sequences.'''
        return levenshtein(tokens.numpy(),xi.squeeze().numpy(),gamma)
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the text."""
        torch.manual_seed(self.config.key)
        seeds = torch.randint(2**32, (self.config.sequence_length,))
        encoded_text = self.config.generation_tokenizer.encode(text, return_tensors='pt', truncation=True,max_length=2048)[0]
        test_stat = lambda tokens,n,k,generator,vocab_size,null=False : self.utils.phi(tokens=tokens,
                                                                        n=n,
                                                                        k=k,
                                                                        generator=generator,
                                                                        key_func=self.utils.transform_key_func,
                                                                        vocab_size=self.config.vocab_size,
                                                                        dist=lambda x,y : self.transform_edit_score(x,y,1),
                                                                        null=False,
                                                                        normalize=True)
        test = lambda tokens,seed : self.permutation_test(tokens,
                                             self.config.vocab_size,
                                             self.config.pseudo_length,
                                             len(tokens),
                                             seed,
                                             test_stat,
                                             n_runs=self.config.n_runs,)
        pval_watermark = test(encoded_text, seeds[0])
        if return_dict:
            return {"is_watermarked": (pval_watermark<=self.config.p_threshold).item(), "score": pval_watermark.item()}
        else:
            return ((pval_watermark<=self.config.p_threshold).item(), pval_watermark.item())
    
    def get_data_for_visualization(self, text: str, *args, **kwargs):
        """Get data for visualization."""

        # Encode text
        encoded_text = self.config.generation_tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).numpy()[0]

        # Find best match
        generator = torch.Generator()
        generator.manual_seed(int(self.config.key))
        _, index = self.utils.phi(encoded_text,self.config.pseudo_length,self.config.sequence_length, generator, self.utils.transform_key_func, self.config.vocab_size, lambda x,y : self.utils.transform_edit_score(x,y,1), null=False, normalize=True)
        random_numbers = self.utils.xi[(index + np.arange(len(encoded_text))) % len(self.utils.xi)]
        
        highlight_values = []

        # Compute highlight values
        for i in range(0, len(encoded_text)):
            r = random_numbers[i][encoded_text[i]]
            v = log(1/(1 - r))
            v = self.utils.value_transformation(v)
            highlight_values.append(v)

        # Decode each token id to its corresponding string token
        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)
        
        return DataForVisualization(decoded_tokens, highlight_values)