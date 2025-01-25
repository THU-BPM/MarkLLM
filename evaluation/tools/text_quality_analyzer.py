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

# =======================================================
# text_quality_analyzer.py
# Description: Analyze text quality using various metrics
# =======================================================

import math
import torch
import sacrebleu
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from utils.openai_utils import OpenAIAPI
from exceptions.exceptions import CodeExecutionError, InvalidAnswerError


class TextQualityAnalyzer:
    """Base class for text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str):
        pass


class DirectTextQualityAnalyzer(TextQualityAnalyzer):
    """Base class for direct text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str):
        pass


class ReferencedTextQualityAnalyzer(TextQualityAnalyzer):
    """Base class for referenced text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str, reference):
        pass


class ExternalDiscriminatorTextQualityAnalyzer(TextQualityAnalyzer):
    """Base class for external discriminator text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text1: str, text2: str, description: str):
        pass


class PPLCalculator(DirectTextQualityAnalyzer):
    """Perplexity calculator for text quality analysis."""

    def __init__(self, model, tokenizer, device='cuda') -> None:
        """
            Initialize the perplexity calculator.

            Parameters:
                model: The language model for perplexity calculation.
                tokenizer: The tokenizer for the language model.
                device (str): The device to use for the calculation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def analyze(self, text: str):
        """Calculate the perplexity of the given text."""
        criterion = torch.nn.CrossEntropyLoss()
        encoded_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
        logits = self.model(torch.unsqueeze(encoded_text, 0), return_dict=True).logits[0]
        loss = criterion(logits[:-1], encoded_text[1:])
        ppl = torch.exp(loss)
        return ppl.item()


class LogDiversityAnalyzer(DirectTextQualityAnalyzer):
    """Log diversity analyzer for text quality analysis."""
    
    def __init__(self) -> None:
        super().__init__()

    def _eval_text(self, text: str, ngram: int):
        """Evaluate text to compute the number of unique and total n-grams."""
        tokens = text.split()
        ngram_set = set()
        total_ngrams = 0

        for i in range(len(tokens) - ngram + 1):
            ngram_set.add(" ".join(tokens[i:i + ngram]))
            total_ngrams += 1

        return len(ngram_set), total_ngrams

    def _eval_one_instance(self, text: str, ngram_list: list):
        """Evaluate a single text instance for multiple n-gram lengths."""
        results = {}
        for n in ngram_list:
            unique, total = self._eval_text(text, n)
            results[n] = {"unique": unique, "total": total}
        unique_tokens = set(text.split())
        return results, unique_tokens

    def analyze(self, text: str):
        """Analyze text to compute log diversity based on n-gram uniqueness."""
        ngram_list = [2, 3, 4]
        prediction_results = {n: {"unique": 0, "total": 0} for n in ngram_list}
        unique_token_set = set()

        stripped_text = text.strip()
        ngram_results, unique_tokens = self._eval_one_instance(stripped_text, ngram_list)

        unique_token_set.update(unique_tokens)

        for n in ngram_list:
            prediction_results[n]["unique"] += ngram_results[n]["unique"]
            prediction_results[n]["total"] += ngram_results[n]["total"]

        # Compute diversity scores for each n-gram length
        diversity_scores = [
            1 - (prediction_results[n]["unique"] / prediction_results[n]["total"])
            for n in ngram_list
        ]

        # Overall diversity is the product of individual n-gram diversities
        overall_diversity = (1 - diversity_scores[0] / 100) * (1 - diversity_scores[1] / 100) * (1 - diversity_scores[2] / 100)
        log_diversity = -math.log(max(1 - overall_diversity, math.exp(-20)))

        return log_diversity


class BLEUCalculator(ReferencedTextQualityAnalyzer):
    """BLEU calculator for text quality analysis."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str, reference: str):
        """Calculate the BLEU score of the given text with the reference."""
        b = sacrebleu.corpus_bleu([text], [[reference]]).score
        return b


class ROUGE1Calculator(ReferencedTextQualityAnalyzer):
    """ROUGE1 calculator for text quality analysis."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str, reference: str):
        """Calculate the ROUGE-1 score of the given text with the reference."""
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        scores = scorer.score(text, reference)
        return scores['rouge1'].fmeasure


class ROUGE2Calculator(ReferencedTextQualityAnalyzer):
    """ROUGE2 calculator for text quality analysis."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str, reference: str):
        """Calculate the ROUGE-2 score of the given text with the reference."""
        scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
        scores = scorer.score(text, reference)
        return scores['rouge2'].fmeasure


class ROUGELCalculator(ReferencedTextQualityAnalyzer):
    """ROUGEL calculator for text quality analysis."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str, reference: str):
        """Calculate the ROUGE-L score of the given text with the reference."""
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(text, reference)
        return scores['rougeL'].fmeasure
    

class BERTScoreCalculator(ReferencedTextQualityAnalyzer):
    """BERTScore calculator for text quality analysis."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.bert_scorer = BERTScorer(
            model_type=model_path,
            num_layers=8,
            batch_size=32,
            nthreads=4,
            all_layers=False,
            idf=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            rescale_with_baseline=False,
            lang="en"
        )
    
    def analyze(self, text: str, reference: str):
        """Calculate the BERTScore of the given text with the reference."""
        P, R, F1 = self.bert_scorer.score([text], [reference])
        return F1.tolist()[0]


class PassOrNotJudger(ReferencedTextQualityAnalyzer):
    """Pass or not judger for text quality analysis."""
    def __init__(self) -> None:
        pass

    def _check_correctness(self, prompt: str, completion: str, test: str, entry_point: str):
        """Check the correctness of the code.""" 
        check_program = (
            prompt + '\n' + completion + "\n" +
            test + "\n" +
            f"check({entry_point})"
        )
        # print(check_program)
        try:
            exec_globals = {}
            exec(check_program, exec_globals)
            return 1
        except BaseException as e:
            return 0

    def analyze(self, text: str, reference: dict):
        """Check if the text passes the correctness test."""
        passed = self._check_correctness(reference['task'], text, reference['test'], reference['entry_point'])
        return passed
    

class GPTTextDiscriminator(ExternalDiscriminatorTextQualityAnalyzer):
    """GPT text discriminator for text quality analysis."""

    def __init__(self, openai_model: str, task_description: str) -> None:
        """
            Initialize the GPT text discriminator.

            Parameters:
                openai_model (str): The OpenAI model to use for text discrimination.
                task_description (str): The description of the task for text discrimination.
        """
        self.openai_model = openai_model
        self.task_description = task_description
    
    def _get_query(self, text1: str, text2: str, question: str):
        """Get the query for text discrimination."""

        query = f"Task Description: {self.task_description}\n"
        query += f"Question: {question}\n"
        query += f"Answer 1: {text1}\n"
        query += f"Answer 2: {text2}\n"
        query += f"Which anwser is better? Only return a number."
        query += f"Return 1 if the first text is better, 2 if the second text is better, 0 if they are equal."
        return query

    def analyze(self, text1: str, text2: str, question: str):
        """Analyze the text to determine which one is better."""
        openai_util = OpenAIAPI(model=self.openai_model, temperature=0.2, 
                                system_content="You are a helpful assistant to determine which of the two answers is better based on the given task description.")
        query = self._get_query(text1, text2, question)
        answer = openai_util.get_result(query)
        # validate answer
        if answer not in ['0', '1', '2']:
            raise InvalidAnswerError
        return eval(answer)