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
# k_semstamp.py
# Description: Implementation of KSEMSTAMP algorithm
# ============================================

import pickle
import queue
import torch
import os
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp
from sentence_transformers import SentenceTransformer,models
from ..base import BaseWatermark, BaseConfig
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from datasets import load_from_disk
from kmeans_pytorch import *
from transformers import StoppingCriteria
from transformers.tokenization_utils import PreTrainedTokenizer
from nltk.tokenize import sent_tokenize
from transformers import StoppingCriteriaList,GenerationConfig
from tqdm import trange

class KSemStampConfig(BaseConfig):
    """Config class for KSEMSTAMP algorithm.load config file and initialize parameters."""
    
    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.domain_data= self.config_dict['domain_data']
        self.max_new_tokens = self.config_dict['max_new_tokens']
        self.min_new_tokens = self.config_dict['min_new_tokens']
        self.path_to_embedder = self.config_dict['path_to_embedder']
        self.N_max = self.config_dict['N_max']
        self.gamma = self.config_dict['gamma']
        self.margin_m = self.config_dict['margin_m']
        self.k = self.config_dict['k']
        self.prime_P = self.config_dict['prime_P']
        self.threshold = self.config_dict['threshold']
        self.path_to_centroids = self.config_dict['path_to_centroids']
        # self.temperature = getattr(self.transformers_config, 'temperature', 0.7)
        # self.top_k = getattr(self.transformers_config, 'top_k', 0)
        # self.repetition_penalty= getattr(self.transformers_config, '', 1.05)

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return "KSEMSTAMP"

# utils
class KSemStampUtils:
    """Helper class for K-SemStamp algorithm, contains helper functions."""

    def __init__(self, config: KSemStampConfig,*args, **kwargs) -> None:
        """
            Initialize the K_SEMSTAMP utility class.

            Parameters:
                config (KSemStampConfig): Configuration for the K_SEMSTAMP algorithm.
        """
        self.config = config
        self.rng = torch.Generator(device=self.config.device)

    @staticmethod
    def worker(rank, text_chunk, embedder_path, queue, encode_batch_size):
        """
        Worker function to process a text chunk and generate embeddings on a specific GPU.
        """
        device = f'cuda:{rank}'
        word_embedding_model = models.Transformer(embedder_path)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
        embedder = SentenceTransformer(modules=[word_embedding_model, pooling_model],device=device)

        sent_embeds = []

        # Progress bar for each worker
        with tqdm(total=len(text_chunk), desc=f"Worker {rank} Encoding", position=rank) as pbar:
            for i in range(0, len(text_chunk), encode_batch_size):
                batch_texts = text_chunk[i:i + encode_batch_size]
                batch_embeds = embedder.encode(batch_texts, convert_to_tensor=True)
                sent_embeds.extend(batch_embeds)
                pbar.update(len(batch_texts))

        # Put all embeddings into the queue
        cpu_embeds = [emb.cpu().numpy() for emb in sent_embeds]
        queue.put(cpu_embeds)

    # get domain data and embed
    @staticmethod
    def embed_gen_list(dataset_path, embedder_path, encode_batch_size=32, num_gpus=torch.cuda.device_count()):
        """
        Parallelized embedding generation for the dataset with progress bars.
        """
        from multiprocessing import Process, Queue
        mp.set_start_method('spawn', force=True)
        dataset = load_from_disk(dataset_path)
        texts = dataset['text']

        # Total progress bar
        total_progress = tqdm(total=len(texts), desc="Total Progress", position=num_gpus)

        # Split the dataset into chunks for each GPU
        text_chunks = [texts[i::num_gpus] for i in range(num_gpus)]

        # Queue to collect embeddings from workers
        queue = Queue()

        processes = []
        for rank, text_chunk in enumerate(text_chunks):
            p = Process(target=KSemStampUtils.worker, args=(rank, text_chunk, embedder_path, queue, encode_batch_size))
            p.start()
            processes.append(p)

        # Collect embeddings from workers
        all_embeds = []
        while any(p.is_alive() for p in processes) or not queue.empty():
            while not queue.empty():
                chunk_embeds = queue.get()
                chunk_embeds = [torch.tensor(arr).to('cuda') for arr in chunk_embeds]
                all_embeds.extend(chunk_embeds)
                total_progress.update(len(chunk_embeds))

        total_progress.close()

        for p in processes:
            p.join()

        # Save embeddings to a single pickle file
        name = os.path.join(dataset_path, "embeds.pkl")
        with open(name, 'wb') as f:
            pickle.dump({'text': all_embeds}, f)

        print(f"Embeddings saved to {name}")
        return name

    # load embeddings
    def load_embeds(self,embed_path):
        with open(embed_path, 'rb') as f:
            d = pickle.load(f)
        # move all embeddings to the same device
        gen_embeds = torch.stack([torch.tensor(t).to(self.config.device) if not isinstance(t, torch.Tensor) else t.to(self.config.device) for t in d['text']])

        return gen_embeds

    def generate_cluster(self):
        """
        1. get domain data
        2. embed domain data
        3. K-Means training
        4. save centroids to self.config.path_to_centroids
        """
        embed_path = KSemStampUtils.embed_gen_list(self.config.domain_data, self.config.path_to_embedder)
        embeds = self.utils.load_embeds(embed_path)
        cluster_ids, cluster_centers = kmeans(
            embeds,
            num_clusters=self.config.k,
            distance="cosine",
            device=embeds.device
        )
        torch.save(cluster_centers, self.config.path_to_centroids)
        print(f"cluster centers saved to {self.config.path_to_centroids}")
        return cluster_centers

    def pairwise_cosine(self,data1, data2):
        data1, data2 = data1.to(self.config.device), data2.to(self.config.device)

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

    def get_cluster_id(self,text, cluster_centers, embedder):
        embedding = embedder.encode(text, convert_to_tensor=True)
        embedding = embedding.reshape(1, -1)
        # convert to float
        embedding = embedding.float()
        # transfer to device
        embedding = embedding.to(self.config.device)
        dis =self.pairwise_cosine(embedding, cluster_centers)
        choice_cluster = torch.argmin(dis, dim=-1)
        cluster_id=choice_cluster.cpu()
        return cluster_id

    def get_cluster_mask(self,curr_cluster_id, k_dim, lmbd):
        self.rng.manual_seed(curr_cluster_id.item() * self.config.prime_P)
        num_accept = int(k_dim * lmbd)
        mask = torch.randperm(k_dim, device=self.config.device, generator=self.rng)[:num_accept]
        return mask.to(self.config.device)
    
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
            text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            return len(sent_tokenize(text)) > self.current_num_sentences + 1

    def kmeans_reject_overlap(self,text, embedder, cluster_centers, margin):
        gen_embed = embedder.encode(text, convert_to_tensor=True)
        gen_embed = gen_embed.reshape(1, -1)
        cluster_centers = torch.tensor(np.array(cluster_centers))
        dis = self.pairwise_cosine(gen_embed, cluster_centers)

        # each row of ranking corresponds to the cluster distance closeness of a generation
        ranked_dis = torch.argsort(dis, dim=-1)
        closest = ranked_dis[0]

        # second nearest cluster
        second_closest = ranked_dis[1]

        first_dis = dis[closest]

        sec_dis = dis[second_closest]

        if ((sec_dis - first_dis).item() > margin):
            return text, closest.clone().detach()
        else:
            return None, closest.clone().detach()
        
    
# main class
class KSemStamp(BaseWatermark):
    """Top-level class for the KSEMSTAMP algorithm."""

    def __init__(self, algorithm_config: str | KSemStampConfig, transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        """
            Initialize the KSEMSTAMP algorithm.

            Parameters:
                algorithm_config (str | KSEMSTAMPConfig): Path to the algorithm configuration file or KSEMSTAMPConfig instance.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if isinstance(algorithm_config, str):
            self.config = KSemStampConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, KSemStampConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be either a path string or a KSemStampConfig instance")

        self.utils = KSemStampUtils(self.config)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text using the KSEMSTAMP algorithm."""
        # get cluster
        cluster_centers=torch.load(self.config.path_to_centroids)
        #cluster_centers = self.utils.generate_cluster()

        # get embedder
        word_embedding_model = models.Transformer(self.config.path_to_embedder)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
        embedder = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        # instantiate sentence end criteria
        sent_end_criteria = KSemStampUtils.SentenceEndCriteria(self.config.generation_tokenizer)

        # get current cluster id
        curr_cluster_id = self.utils.get_cluster_id(prompt, cluster_centers, embedder)
        # get cluster mask
        mask = self.utils.get_cluster_mask(curr_cluster_id, self.config.k, self.config.gamma)

        text = prompt
        new_text = prompt
        text_ids = self.config.generation_tokenizer.encode(prompt, return_tensors='pt').to(self.config.generation_model.device)
        prompt_length = len(text_ids[0])
        sent_end_criteria.update(new_text)

        current_trials = 0
        # debug_text_segments = [(prompt, text_ids.size(1), curr_cluster_id)]
        cluster_id_sequence = [curr_cluster_id.item()]
    
        while True:
            stopping_criteria = StoppingCriteriaList([sent_end_criteria])

            outputs = self.config.generation_model.generate(text_ids, stopping_criteria=stopping_criteria,**self.config.gen_kwargs)
            outputs = outputs[:, :-1]
            new_text_ids = outputs
            new_text = self.config.generation_tokenizer.decode(new_text_ids[0, text_ids.size(1):], skip_special_tokens=True)
            if new_text == '':
                print('WARNING: stopped generation because generated nothing (after discarding last generated token)', flush=True)
                break

            current_trials += 1

            accepted_text, curr_cluster_id = self.utils.kmeans_reject_overlap(text=new_text, embedder=embedder, cluster_centers=cluster_centers, margin=self.config.margin_m)

            if (accepted_text == None and current_trials < self.config.N_max):
                continue
            else:
                new_text = accepted_text if accepted_text != None else new_text
            cluster_id_sequence.append(curr_cluster_id.item())

            # valid or max trials reached
            if (curr_cluster_id in mask) or current_trials >= self.config.N_max:
                if current_trials >= self.config.N_max:
                    print(f'WARNING: desired semantic signature can\'t be sampled after max_trials {self.config.N_max}')
                # debug_text_segments.append(
                #     (new_text, new_text_ids.size(1) - text_ids.size(1), curr_cluster_id))
                current_trials = 0
                mask = self.utils.get_cluster_mask(curr_cluster_id, self.config.k, self.config.gamma)
                text += new_text
                text_ids = new_text_ids.to(self.config.generation_model.device)
                sent_end_criteria.update(text)
                if (len(text_ids[0]) - prompt_length) >= self.config.max_new_tokens-1:
                    break
        watermarked_text=text.strip()  
        return watermarked_text
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the input text."""
        # get cluster
        cluster_centers = torch.load(self.config.path_to_centroids)
        # get embedder
        word_embedding_model = models.Transformer(self.config.path_to_embedder)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
        embedder = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        sentences = sent_tokenize(text)
        n_sent = len(sentences)
        n_watermark = 0
        curr_cluster_id = self.utils.get_cluster_id(sentences[0], embedder=embedder, cluster_centers=cluster_centers)
        cluster_mask = self.utils.get_cluster_mask(curr_cluster_id, self.config.k, self.config.gamma)
        for i in range(1, n_sent):
            curr_cluster_id = self.utils.get_cluster_id(sentences[i], embedder=embedder, cluster_centers=cluster_centers)
            if curr_cluster_id in cluster_mask:
                n_watermark += 1
            cluster_mask = self.utils.get_cluster_mask(curr_cluster_id, self.config.k, self.config.gamma)
        n_test_sent = n_sent - 1  # exclude the prompt
        num = n_watermark - self.config.gamma * (n_test_sent)
        denom = np.sqrt((n_test_sent) * self.config.gamma * (1-self.config.gamma))
        z_score=num / denom

        # Determine if the computed score exceeds the threshold for watermarking
        is_watermarked = z_score > self.config.threshold

        # Return results based on the `return_dict` flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)