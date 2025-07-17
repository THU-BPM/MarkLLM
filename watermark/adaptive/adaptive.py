import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import random
from torch.nn import functional as F

from ..base import BaseWatermark, BaseConfig
from .semantic_model import SemanticModel
from utils.transformers_config import TransformersConfig


class AdaptiveConfig(BaseConfig):
    """Config class for Adaptive algorithm, load config file and initialize parameters."""

    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.threshold = self.config_dict['threshold']   # threshold for detecting watermark
        self.max_new_tokens = self.config_dict['max_new_tokens']
        self.min_new_tokens = self.config_dict['min_new_tokens']
        self.alpha = self.config_dict['alpha']   # threshold for measuring the next token entropy
        self.top_k = self.config_dict['top_k']
        self.top_p = self.config_dict['top_p']
        self.repetition_penalty = self.config_dict['repetition_penalty']
        self.no_repeat_ngram_size = self.config_dict['no_repeat_ngram_size']
        self.secret_string = self.config_dict['secret_string']
        self.measure_threshold = self.config_dict['measure_threshold']
        self.delta_0 = self.config_dict['delta_0']
        self.delta = self.config_dict['delta']
        self.seed = self.config_dict['seed']
        self.measurement_model = self.config_dict['measurement_model']
        self.embedding_model = self.config_dict['embedding_model']
        self.transform_model_path = self.config_dict['transform_model_path']
        self.transform_model_output_dim = self.config_dict['transform_model_output_dim']

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'Adaptive'
    

class AdaptiveUtils:
    """Utility class for Adaptive algorithm, contains helper functions."""

    def __init__(self, config: AdaptiveConfig, *args, **kwargs) -> None:
        """
            Initialize the Adaptive utility class.

            Parameters:
                config (AdaptiveConfig): Configuration for the Adaptive algorithm.
        """

        self.config = config
        self.measure_model, self.measure_tokenizer = self._load_model(self.config.measurement_model)
        self.embedding_model = self._load_embedding_model(self.config.embedding_model)
        self.transform_model = self._load_semantic_model(self.config.transform_model_path)
        self.mapping_list = self._vocabulary_mapping(self.config.vocab_size, self.config.transform_model_output_dim, self.config.seed)

    
    def _load_model(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
        model.eval()
        return model, tokenizer
    
    def _load_embedding_model(self, model_name: str):
        embedding_model = SentenceTransformer(model_name).to(self.config.device)
        embedding_model.eval()
        return embedding_model
    
    def _load_semantic_model(self, model_path: str):
        transform_model = SemanticModel()
        transform_model.load_state_dict(torch.load(model_path))
        transform_model.to(self.config.device)
        transform_model.eval()
        return transform_model
        
    def _vocabulary_mapping(self, vocab_size, model_output_dim, seed):
        random.seed(seed)
        return [random.randint(0, model_output_dim-1) for _ in range(vocab_size)]
    
    def calc_banned_ngram_tokens(self, prev_input_ids: torch.Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
        """Copied from fairseq for no_repeat_ngram in beam_search"""
        if cur_len + 1 < no_repeat_ngram_size:
            # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            return [[] for _ in range(num_hypos)]
        generated_ngrams = [{} for _ in range(num_hypos)]
        for idx in range(num_hypos):
            gen_tokens = prev_input_ids[idx].tolist()
            generated_ngram = generated_ngrams[idx]
            for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

        def _get_generated_ngrams(hypo_idx):
            # Before decoding the next token, prevent decoding of ngrams that have already appeared
            start_idx = cur_len + 1 - no_repeat_ngram_size
            ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
            return generated_ngrams[hypo_idx].get(ngram_idx, [])

        banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
        return banned_tokens
    
    def postprocess_next_token_scores(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty, no_repeat_ngram_size):
        # _enforce_repetition_penalty
        if repetition_penalty != 1.0:
            for i in range(batch_size * num_beams):
                for previous_token in set(prev_output_tokens[i].tolist()):
                    # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    if lprobs[i, previous_token] < 0:
                        lprobs[i, previous_token] *= repetition_penalty
                    else:
                        lprobs[i, previous_token] /= repetition_penalty
        
        # lower eos token prob to zero if min_length is not reached
        if prev_output_tokens.size(1) < self.config.min_new_tokens:
            lprobs[:, self.config.generation_tokenizer.eos_token_id] = -float("Inf")
        
        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            num_batch_hypotheses = batch_size * num_beams
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            banned_batch_tokens = self.calc_banned_ngram_tokens(
                prev_output_tokens, num_batch_hypotheses, no_repeat_ngram_size, prev_output_tokens.size(1)
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                lprobs[i, banned_tokens] = -float("inf")

    def top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ) -> torch.Tensor:
        """ 
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

    def stopping_criteria(self, ids, tokenizer):
        stop_words = ["word.", "word!", "word?", "word...", "word;"]
        stop_words_ids = [tokenizer.encode(stop_word, return_tensors='pt', add_special_tokens=False)[0][-1].to(self.config.device) for stop_word in stop_words]
        
        if ids[0][-1] == tokenizer.eos_token_id:
            return True

        if ids[0][-1] in stop_words_ids:
            if len(ids[0]) > self.config.min_new_tokens:
                return True
        return False

    def next_token_entropy(self, input_text, model, tokenizer, device):
        input_ids = tokenizer.encode(input_text, return_tensors='pt', add_special_tokens=False).to(device)
        outputs = model(input_ids)
        probs = torch.nn.functional.softmax(outputs.logits[0, -1, :], dim=-1)
        mask = probs > 0
        entropy = -torch.sum(probs[mask] * torch.log(probs[mask]))
        return entropy
    
    def bias_logits(self, logits, v_embedding, delta):
        logits = torch.mul(logits, (1 + delta*v_embedding))
        return logits
    
    def watermarking(self, ids, logits, secret_string, measure_threshold):
        if len(ids[0]) <= measure_threshold:
            embedding = self.embedding_model.encode(secret_string, convert_to_tensor=True)
            embedding = embedding.clone().detach()
            with torch.no_grad():
                t_embedding = self.transform_model(embedding).tolist()
            t_embedding = [1.0 if x>0.0 else 0.0 for x in t_embedding]
            v_embedding = torch.tensor([t_embedding[i] for i in self.mapping_list], device=self.config.device)
            logits[0] = self.bias_logits(logits[0], v_embedding, self.config.delta_0)
        elif len(ids[0]) > measure_threshold:
            measure_text = self.config.generation_tokenizer.decode(ids[-1])
            measure_entroy = self.next_token_entropy(measure_text, self.measure_model, self.measure_tokenizer, self.config.device)
            if measure_entroy >= self.config.alpha:
                embedding = self.embedding_model.encode(measure_text, convert_to_tensor=True)
                embedding = embedding.clone().detach()
                with torch.no_grad():
                    t_embedding = self.transform_model(embedding).tolist()   # torch([])
                t_embedding = [1.0 if x>0.0 else 0.0 for x in t_embedding]
                v_embedding = torch.tensor([t_embedding[i] for i in self.mapping_list], device=self.config.device)
                logits[0] = self.bias_logits(logits[0], v_embedding, self.config.delta)
        return logits
    



class Adaptive(BaseWatermark):
    """Top-level class for the Adaptive algorithm."""

    def __init__(self, algorithm_config: str | AdaptiveConfig, transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        if isinstance(algorithm_config, str):
            self.config = AdaptiveConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, AdaptiveConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be either a path string or a AdaptiveUtils instance")
        
        self.utils = AdaptiveUtils(self.config)

    # Adaptive watermark text generation
    def generate_watermarked_text(self, prompt: str, *args, **kwargs):
        input_ids = self.config.generation_tokenizer.encode(prompt, return_tensors='pt').to(self.config.device)

        output_ids = torch.tensor([[]], dtype=torch.int64, device=self.config.device)
        attn = torch.ones_like(input_ids)
        past = None
        for i in range(self.config.max_new_tokens):
            with torch.no_grad():
                if past:
                    output = self.config.generation_model(input_ids[:,-1:], attention_mask=attn, past_key_values=past)
                else:
                    output = self.config.generation_model(input_ids)
            
            logits = output.logits[:,-1, :]
            self.utils.postprocess_next_token_scores(logits, 1, 1, output_ids, repetition_penalty=self.config.repetition_penalty, no_repeat_ngram_size=self.config.no_repeat_ngram_size)
            logits = self.utils.watermarking(output_ids, logits, self.config.secret_string, self.config.measure_threshold)   # watermarking
            logits = self.utils.top_k_top_p_filtering(logits, top_k=self.config.top_k, top_p=self.config.top_p)   # top-k, top-p filtering
            probs = torch.nn.functional.softmax(logits, dim=-1)   # softmax
            next_id = torch.multinomial(probs, num_samples=1)   # sampling

            input_ids = torch.cat((input_ids, next_id), dim=-1)   # update input_ids
            output_ids = torch.cat((output_ids, next_id), dim=-1)   # update output_ids

            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

            # stopping criteria
            stop = self.utils.stopping_criteria(output_ids, self.config.generation_tokenizer)
            if stop:
                output_text = self.config.generation_tokenizer.decode(output_ids[0].tolist())
                return output_text
        
        output_text = self.config.generation_tokenizer.decode(output_ids[0])
        
        return output_text
    

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        watermark_ids = self.config.generation_tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(self.config.device)
        
        e = self.utils.embedding_model.encode(self.config.secret_string, convert_to_tensor=True, device=self.config.device)
        e = e.clone().detach()
        with torch.no_grad():
            te = self.utils.transform_model(e).tolist()
        te = [1.0 if x>0.0 else 0.0 for x in te]
        ve = torch.tensor([te[i] for i in self.utils.mapping_list], device=self.config.device)

        score = []
        for i in range(len(watermark_ids[0])):
            if i <= self.config.measure_threshold:
                s = ve[watermark_ids[0][i]]
                score.append(s)
            elif i > self.config.measure_threshold:
                measure_text = self.config.generation_tokenizer.decode(watermark_ids[0][:i])
                measure_entroy = self.utils.next_token_entropy(measure_text, self.utils.measure_model, self.utils.measure_tokenizer, self.config.device)
                if measure_entroy >= self.config.alpha:
                    e = self.utils.embedding_model.encode(measure_text, convert_to_tensor=True, device=self.config.device)
                    e = e.clone().detach()
                    with torch.no_grad():
                        te = self.utils.transform_model(e).tolist()
                    te = [1.0 if x>0.0 else 0.0 for x in te]
                    ve = torch.tensor([te[i] for i in self.utils.mapping_list], device=self.config.device)
                    s = ve[watermark_ids[0][i]]
                    score.append(s)
        
        normalized_score = sum(score)/len(score)
        normalized_score = normalized_score.item()

        is_watermarked = normalized_score > self.config.threshold

        if return_dict:
            return {
                "is_watermarked": is_watermarked,
                "score": normalized_score
            }
        else:
            return (is_watermarked, normalized_score)

        
