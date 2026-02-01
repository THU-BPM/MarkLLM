import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import StoppingCriteria
import torch
from nltk.tokenize import sent_tokenize

class FuzzyModel:

    def __init__(self, cluster_path, m, K, embedder="sentence-transformers/all-mpnet-base-v1"):
        # open and read pickle file
        with open(cluster_path, 'rb') as file:
            # load the centers
            centers = pickle.load(file)

        self.centers = centers
        self.model = SentenceTransformer(embedder)
        self.m = m
        self.K = K

    def calculate_membership(self, text):
        """
        Calculate the membership of a new data point to each cluster.
        
        Parameters:
        new_data (np.array): The new data point.
        cluster_centers (list of np.array): List containing cluster centers.
        m (float): The fuzzy parameter.
        
        Returns:
        list: A list of memberships for each cluster.
        """

        embeddings = self.model.encode(text)
        new_data = embeddings
        num = len(new_data)
        # Ensure new_data is a numpy array
        new_data = np.array(new_data)
        data = new_data
        centers = self.centers
        alpha = self.m

        distance = np.zeros([num, self.K])
        for i in range(num):
            for j in range(self.K):
                distance[i, j] = np.linalg.norm(data[i, :] - centers[j, :], ord=2)

        u_new = np.zeros([num, self.K])
        for i in range(num):
            for j in range(self.K):
                u_new[i, j] = 1. / np.sum((distance[i, j] / distance[i, :]) ** (2 / (alpha - 1)))

        return u_new
    
def gen_sent(model, tokenizer, text_ids, gen_config, stopping_criteria):
    outputs = model.generate(
            text_ids,
            gen_config,
            stopping_criteria=stopping_criteria,
        )
    new_text_ids = outputs.sequences[:, :-1]
    new_text = tokenizer.decode(
        new_text_ids[0, text_ids.size(1):], skip_special_tokens=True)
    return new_text, new_text_ids

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
    
# calculate watermark ratio
def threshold_detect(sents, fuzzy_model, prompt_gen):
    nums_samespace = 0
    # implement K==8 logic
    if fuzzy_model.K == 8:
        n_sent = len(sents) # number of sentences
        n_watermark = 0
        last_sent_simi = fuzzy_model.calculate_membership([prompt_gen])
        sorted_indices = np.argsort(last_sent_simi[0])
        first_largest_index = sorted_indices[-1]
        third_largest_index = sorted_indices[-3]
        greenids = [first_largest_index, third_largest_index]
        accept_ids = greenids
        for i in range(0, len(sents)):
            new_sent_simi = fuzzy_model.calculate_membership([sents[i]])
            new_sent_sorted_indices = np.argsort(new_sent_simi[0])
            new_sent_membership = new_sent_sorted_indices[-1]
            if new_sent_membership == accept_ids[0] and len(accept_ids)==2:
                nums_samespace = nums_samespace + 1
            if new_sent_membership in accept_ids:
                n_watermark += 1
            last_sent = sents[i]
            last_sent_simi = fuzzy_model.calculate_membership([last_sent])
            sorted_indices = np.argsort(last_sent_simi[0])
            if nums_samespace <= 5:
                first_largest_index = sorted_indices[-1]
                third_largest_index = sorted_indices[-3]
                greenids = [first_largest_index, third_largest_index]
                accept_ids = greenids
            else:
                second_largest_index = sorted_indices[-2]
                fourth_largest_index = sorted_indices[-4]
                fifth_largest_index = sorted_indices[-5]
                sixth_largest_index = sorted_indices[-6]
                greenids = [second_largest_index, fourth_largest_index, fifth_largest_index, sixth_largest_index]
                accept_ids = greenids
                nums_samespace = 0
        n_test_sent = n_sent  # exclude the prompt and the ending
        if n_test_sent == 0:
            return None
        watermark_ratio = n_watermark / n_test_sent
        return watermark_ratio