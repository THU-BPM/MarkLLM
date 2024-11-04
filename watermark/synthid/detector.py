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
# detector.py
# Description: Implementation of SynthID watermark detectors
# ============================================

import os
import abc
import tqdm
import json
import torch
import numpy as np
from .detector_bayesian_torch import RawBayesianDetector
from evaluation.dataset import C4Dataset


class SynthIDDetector(abc.ABC):
    """Base class for SynthID watermark detectors.
    
    This class defines the interface that all SynthID watermark detectors must implement.
    Subclasses should override the detect() method to implement specific detection algorithms.
    """

    def __init__(self):
        """Initialize the detector."""
        pass

    @abc.abstractmethod
    def detect(self, g_values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Detect watermark presence in the given g-values.

        Args:
            g_values: Array of shape [batch_size, seq_len, watermarking_depth] containing
                the g-values computed from the text.
            mask: Binary array of shape [batch_size, seq_len] indicating which g-values
                should be used in detection. g-values with mask value 0 are discarded.

        Returns:
            Array of shape [batch_size] containing detection scores, where higher values
            indicate stronger evidence of watermarking.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement detect()")

class MeanDetector(SynthIDDetector):
    
    def detect(self, g_values, mask):
        """
        Args:
            g_values: shape [batch_size, seq_len, watermarking_depth]
            mask: shape [batch_size, seq_len]
        Returns:
            scores: shape [batch_size]
        """
        watermarking_depth = g_values.shape[-1]
        num_unmasked = np.sum(mask, axis=1)  # shape [batch_size]
        return np.sum(g_values * np.expand_dims(mask, 2), axis=(1, 2)) / (
                watermarking_depth * num_unmasked
        )
    

class WeightedMeanDetector(SynthIDDetector):
    
    def detect(
        self,
        g_values: np.ndarray,
        mask: np.ndarray,
        weights: np.ndarray = None,
    ) -> np.ndarray:
        """Computes the Weighted Mean score.

        Args:
            g_values: g-values of shape [batch_size, seq_len, watermarking_depth]
            mask: A binary array shape [batch_size, seq_len] indicating which g-values
                should be used. g-values with mask value 0 are discarded
            weights: array of non-negative floats, shape [watermarking_depth]. The
                weights to be applied to the g-values. If not supplied, defaults to
                linearly decreasing weights from 10 to 1

        Returns:
            Weighted Mean scores, of shape [batch_size]. This is the mean of the
            unmasked g-values, re-weighted using weights.
        """
        watermarking_depth = g_values.shape[-1]

        if weights is None:
            weights = np.linspace(start=10, stop=1, num=watermarking_depth)

        # Normalise weights so they sum to watermarking_depth
        weights *= watermarking_depth / np.sum(weights)

        # Apply weights to g-values
        g_values = g_values * np.expand_dims(weights, axis=(0, 1))

        num_unmasked = np.sum(mask, axis=1)  # shape [batch_size]
        return np.sum(g_values * np.expand_dims(mask, 2), axis=(1, 2)) / (
            watermarking_depth * num_unmasked
        )
    

class BayesianDetector(SynthIDDetector):
    def __init__(self, logits_processor):
        # Truncate outputs to this length for training.
        self.pos_truncation_length = 200
        self.neg_truncation_length = 200
        # Pad trucated outputs to this length for equal shape across all batches.
        self.max_padded_length = 1000
        self.trained_detector = None
        self.logits_processor = logits_processor
        self.device = torch.device(logits_processor.device)
        self.tokenizer = logits_processor.config.generation_tokenizer
        self.train_detector()
    
    def detect(self, g_values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Detect watermark using Bayesian detector.
        
        Args:
            g_values: g-values of shape [batch_size, seq_len, watermarking_depth]
            mask: A binary array shape [batch_size, seq_len] indicating which g-values
                should be used. g-values with mask value 0 are discarded
                
        Returns:
            Detection scores of shape [batch_size]
        """
        if self.trained_detector is None:
            raise ValueError("Detector not trained. Call train_detector() first.")
        g_values = torch.tensor(g_values, dtype=torch.float32, device=self.device)
        mask = torch.tensor(mask, device=self.device)
        return self.trained_detector.detector_module(g_values, mask)
    

    def tokenize_and_batch_texts(self, texts, tokenizer, device, padded_length=2500, batch_size=16):
        tokenized_texts = []
        batched = []
        
        for text in tqdm.tqdm(texts):
            inputs = tokenizer(
                text,
                return_tensors='pt',
                add_special_tokens=False,
                padding=True,
            ).to(device)
            
            line = inputs['input_ids'].cpu().numpy()[0].tolist()
            if len(line) >= padded_length:
                line = line[:padded_length]
            else:
                line = line + [tokenizer.eos_token_id] * (padded_length - len(line))
            
            batched.append(torch.tensor(line, dtype=torch.long, device=device)[None, :])
            if len(batched) == batch_size:
                tokenized_texts.append(torch.cat(batched, dim=0))
                batched = []
        
            # Handle the last incomplete batch
        if batched:
            tokenized_texts.append(torch.cat(batched, dim=0))
        
        return tokenized_texts

    def get_data_for_training(self, dataset_name: str='c4', dataset_path: str='dataset/c4/processed_c4.json', num_samples: int=1000):
        unwatermarked_dataset = C4Dataset(dataset_path, max_samples=num_samples)
        watermarked_dataset = self.get_watermarked_data(unwatermarked_dataset)
        tokenized_natural_texts = self.tokenize_and_batch_texts(
            unwatermarked_dataset.natural_texts, 
            self.tokenizer, 
            self.device
        )

        # Process watermarked texts
        watermarked_texts = [item['watermarked_text'] for item in watermarked_dataset]
        tokenized_watermarked_texts = self.tokenize_and_batch_texts(
            watermarked_texts, 
            self.tokenizer, 
            self.device
        )
        return tokenized_natural_texts, tokenized_watermarked_texts


    def get_watermarked_data(self, dataset):
        """Generate and cache watermarked data from dataset prompts.
        
        Args:
            dataset: Dataset object containing prompts
            
        Returns:
            list: List of dictionaries containing prompts and their watermarked generations
        """
        output_dir = 'watermark/synthid/generated_texts'
        output_path = os.path.join(output_dir, 'watermarked_texts.json')
        
        # If cached data exists, load it
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate watermarked texts
        generated_data = []
        for prompt in tqdm.tqdm(dataset.prompts):
            # Generate watermarked text
            watermarked_text = self.logits_processor.generate_watermarked_text(prompt)
            
            # Extract only the generated part (excluding prompt)
            generated_part = watermarked_text[len(prompt):].strip()
            
            # Store in list
            generated_data.append({
                'prompt': prompt,
                'watermarked_text': generated_part
            })

        # Save generated texts to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, ensure_ascii=False, indent=2)
            
        return generated_data
        

    def train_detector(self):
        """Train the Bayesian detector.
        
        Args:
            wm_outputs: Watermarked outputs
            uwm_outputs: Unwatermarked outputs
            logits_processor: SynthID logits processor
            tokenizer: Tokenizer
            device: PyTorch device
        """
        uwm_outputs, wm_outputs = self.get_data_for_training()
        bayesian_detector, test_loss = (
            RawBayesianDetector.train_best_detector(
                tokenized_wm_outputs=wm_outputs,
                tokenized_uwm_outputs=uwm_outputs,
                logits_processor=self.logits_processor,
                tokenizer=self.tokenizer,
                torch_device=self.device,
                max_padded_length=self.max_padded_length,
                pos_truncation_length=self.pos_truncation_length,
                neg_truncation_length=self.neg_truncation_length,
                verbose=True,
                learning_rate=3e-3,
                n_epochs=50,
                l2_weights=np.zeros((1,)),
            )
        )
        self.trained_detector = bayesian_detector

def get_detector(detector_name: str, logits_processor):
    if detector_name == 'mean':
        return MeanDetector()
    elif detector_name == 'weighted_mean':
        return WeightedMeanDetector()
    elif detector_name == 'bayesian':
        return BayesianDetector(logits_processor)
    else:
        raise ValueError(f"Detector {detector_name} not found.")
