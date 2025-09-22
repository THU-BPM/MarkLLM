import os
import torch
from math import sqrt
from transformers import AutoModel, AutoTokenizer
from functools import partial
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from transformers import LogitsProcessor, LogitsProcessorList
from .model import Classifier, Unified_Feature_Translator
from .navigator import Navigator
from ..base import BaseConfig, BaseWatermark

class IEConfig(BaseConfig):
    """Config class for IE algorithm, load config file and initialize parameters."""
    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.gamma = self.config_dict['gamma']
        self.delta = self.config_dict['delta']
        self.hash_key = self.config_dict['hash_key']
        self.z_threshold = self.config_dict['z_threshold']
        self.prefix_length = self.config_dict['prefix_length']
        self.embedding_model_path = self.config_dict['embedding_model_path']
        self.entropy_tagger_path = self.config_dict['entropy_tagger_path']
        self.entropy_threshold = self.config_dict['entropy_threshold']
        self.start_entropy = self.config_dict['start_entropy']
        self.interval = self.config_dict['interval']
        self.end_entropy = self.config_dict['end_entropy']
        self.direction = self.config_dict['direction']
    
    @property
    def algorithm_name(self) -> str:
        """Return algorithm name."""
        return "IE"
    
class IEUtils:
    """Utility class for IE algorithm, contains helper functions."""

    def __init__(self, config: IEConfig, *args, **kwargs):
        self.config = config
        self.rng = torch.Generator(device=self.config.device)

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed the random number generator with the last prefix_length tokens of the input_ids."""
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        prev_token = time_result % self.config.vocab_size
        self.rng.manual_seed(self.config.hash_key * prev_token)
        return

    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids."""
        self._seed_rng(input_ids)
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)
        greenlist_ids = vocab_permutation[:greenlist_size] 
        return greenlist_ids
    
    def calculate_entropy(self, model, feature_extractor, entropy_tagger, tokenized_text: torch.Tensor):
        """Calculate entropy for each token in the tokenized_text."""
        entropy = []
        for batch_feature in feature_extractor.feature_extractor_for_bert(tokenized_text, 512, 32):
            batch_feature = batch_feature.float().to(tokenized_text.device)
            with torch.no_grad():
                scores = torch.softmax(entropy_tagger(batch_feature), dim=1)
            entropy += scores[:, 1].tolist()
        entropy = [1.0] + entropy[:-1]
        return entropy
        # with torch.no_grad():
        #     output = model(torch.unsqueeze(tokenized_text, 0), return_dict=True)
        #     probs = torch.softmax(output.logits, dim=-1)
        #     entropy = -torch.where(probs > 0, probs * probs.log(), probs.new([0.0])).sum(dim=-1)
        #     entropy = entropy[0].cpu().tolist()
        #     entropy.insert(0, -10000.0)
        #     return entropy[:-1]

    def _compute_z_score(self, observed_count: int, T: int) -> float: 
        """Compute z-score for the observed count of green tokens."""
        expected_count = self.config.gamma
        numer = observed_count - expected_count * T 
        denom = sqrt(T * expected_count * (1 - expected_count))  
        z = numer / denom
        return z

    def score_sequence(self, input_ids: torch.Tensor, entropy_list: list[float]) -> tuple[float, list[int], list[int]]:
        """Score the input_ids based on the greenlist and entropy."""
        num_tokens_scored = (len(input_ids) - self.config.prefix_length - 
                             len([e for e in entropy_list[self.config.prefix_length:] if e > 0.5]))
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                )
            )

        green_token_flags = [-1 for _ in range(self.config.prefix_length)]
        weights = [-1 for _ in range(self.config.prefix_length)]

        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx]
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
            if curr_token in greenlist_ids:
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
            if entropy_list[idx] < 0.5:
                weights.append(1)
            else:
                weights.append(0)

        # calculate number of green tokens where weight is 1
        green_token_count = sum([1 for i in range(len(green_token_flags)) if green_token_flags[i] == 1 and weights[i] == 1])
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        
        return z_score, green_token_flags, weights
    
class IE(BaseWatermark):
    """Top-level class for IE algorithm."""

    def __init__(self, algorithm_config: str | IEConfig, transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        """
            Initialize the IE algorithm.

            Parameters:
                algorithm_config (str | IEConfig): Path to the algorithm configuration file or IEConfig instance.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if isinstance(algorithm_config, str):
            self.config = IEConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, IEConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be either a path string or a IEConfig instance")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.utils = IEUtils(self.config)
        # load tagger and embed model
        self.embedding_model = AutoModel.from_pretrained(self.config.embedding_model_path, torch_dtype=torch.float16).eval().to(self.device)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.config.embedding_model_path)
        self.entropy_tagger = Classifier(input_dim=self.embedding_model.config.hidden_size)
        tagger_path = os.path.join(self.config.entropy_tagger_path, 'entropy_tagger_' + str(self.config.entropy_threshold).replace('.', '_') + '.pt')
        self.entropy_tagger.load_state_dict(torch.load(tagger_path, map_location=self.device, weights_only=True))
        self.entropy_tagger.eval()
        self.entropy_tagger.to(self.device)
        self.feature_extractor = Unified_Feature_Translator(self.config.generation_tokenizer, self.embedding_tokenizer, self.embedding_model)
        self.logits_processor = IELogitsProcessor(self.config, self.utils, self.feature_extractor, self.entropy_tagger)
        

    def generate_watermarked_text(self, prompt: str, *args, **kwargs):
        """Generate watermarked text."""

        # Configure generate_with_watermark
        self.config.generation_model.to(self.device)
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]), 
            **self.config.gen_kwargs
        )
        
        # encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
        # generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # decode
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the text."""

        # encode text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)

        entropy_list = self.utils.calculate_entropy(self.config.generation_model, self.feature_extractor, self.entropy_tagger, encoded_text)
        
        # compute z_score
        z_score, _, _ = self.utils.score_sequence(encoded_text, entropy_list)

        # Determine if the z_score indicates a watermark
        is_watermarked = z_score > self.config.z_threshold

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)
        
    def navigate(self, dataset_name, beam_size, machine_evaluations, human_evaluations):
        self.navigator = Navigator(
            dataset_name,
            beam_size,
            range(self.config.start_entropy, self.config.end_entropy + self.config.interval, self.config.interval),
            machine_evaluations,
            human_evaluations
        )
        return self.navigator.print_result()

    def get_data_for_visualization(self, text: str, *args, **kwargs):
        """Get data for visualization."""
        
        # encode text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.generation_model.device)

        # calculate entropy
        entropy_list = self.utils.calculate_entropy(self.config.generation_model, encoded_text)
        
        # compute z-score, highlight_values, and weights
        z_score, highlight_values, weights = self.utils.score_sequence(encoded_text, entropy_list)
        
        # decode single tokens
        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)
        
        return DataForVisualization(decoded_tokens, highlight_values, weights)  
    
class IELogitsProcessor(LogitsProcessor):
    """Logits processor for IE algorithm, contains the logic to bias the logits."""

    def __init__(self, config: IEConfig, utils: IEUtils, feature_extractor, classifier, *args, **kwargs) -> None:
        """
            Initialize the IE logits processor.

            Parameters:
                config (IEConfig): Configuration for the IE algorithm.
                utils (IEUtils): Utility class for the IE algorithm.
        """
        self.config = config
        self.utils = utils
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids: torch.LongTensor) -> torch.BoolTensor:
        """Calculate greenlist mask for the given scores and greenlist token ids."""
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        """Bias the scores for the greenlist tokens."""
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        # get entropy
        features = self.feature_extractor.feature_extractor_next_token(input_ids, max_seq_len=512)
        with torch.no_grad():
            entropy = torch.softmax(self.classifier(features.float().to(input_ids.device)), dim=1)[:, 1]
        entropy_mask = (entropy < 0.5).view(-1, 1).to(green_tokens_mask.device)
        
        green_tokens_mask = green_tokens_mask * entropy_mask

        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.config.delta)
        return scores