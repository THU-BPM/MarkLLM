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

# ==========================================================
# success_rate_calculator.py
# Description: Calculate success rate of watermark detection
# ==========================================================

from typing import List, Dict, Union
from exceptions.exceptions import TypeMismatchException, ConfigurationError


class DetectionResult:
    """Detection result."""

    def __init__(self, gold_label: bool, detect_result: Union[bool, float]) -> None:
        """
            Initialize the detection result.

            Parameters:
                gold_label (bool): The expected watermark presence.
                detect_result (Union[bool, float]): The detection result.
        """
        self.gold_label = gold_label
        self.detect_result = detect_result


class BaseSuccessRateCalculator:
    """Base class for success rate calculator."""
    def __init__(self, labels: List[str] = ['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']) -> None:
        """
            Initialize the success rate calculator.

            Parameters:
                labels (List[str]): The list of metric labels to include in the output.
        """
        self.labels = labels
    
    def _check_instance(self, data: List[Union[bool, float]], expected_type: type):
        """Check if the data is an instance of the expected type."""
        for d in data:
            if not isinstance(d, expected_type):
                raise TypeMismatchException(expected_type, type(d))
    
    def _filter_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Filter metrics based on the provided labels."""
        return {label: metrics[label] for label in self.labels if label in metrics}
    
    def calculate(self, watermarked_result: List[Union[bool, float]], non_watermarked_result: List[Union[bool, float]]) -> Dict[str, float]:
        """Calculate success rates based on provided results."""
        pass


class FundamentalSuccessRateCalculator(BaseSuccessRateCalculator):
    """
        Calculator for fundamental success rates of watermark detection.

        This class specifically handles the calculation of success rates for scenarios involving
        watermark detection after fixed thresholding. It provides metrics based on comparisons
        between expected watermarked results and actual detection outputs.

        Use this class when you need to evaluate the effectiveness of watermark detection algorithms
        under fixed thresholding conditions.
    """

    def __init__(self, labels: List[str] = ['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC']) -> None:
        """
            Initialize the fundamental success rate calculator.

            Parameters:
                labels (List[str]): The list of metric labels to include in the output.
        """
        super().__init__(labels)
    
    def _compute_metrics(self, inputs: List[DetectionResult]) -> Dict[str, float]:
        """Compute metrics based on the provided inputs."""
        TP = sum(1 for d in inputs if d.detect_result and d.gold_label)
        TN = sum(1 for d in inputs if not d.detect_result and not d.gold_label)
        FP = sum(1 for d in inputs if d.detect_result and not d.gold_label)
        FN = sum(1 for d in inputs if not d.detect_result and d.gold_label)

        TPR = TP / (TP + FN) if TP + FN else 0.0
        FPR = FP / (FP + TN) if FP + TN else 0.0
        TNR = TN / (TN + FP) if TN + FP else 0.0
        FNR = FN / (FN + TP) if FN + TP else 0.0
        P = TP / (TP + FP) if TP + FP else 0.0
        R = TP / (TP + FN) if TP + FN else 0.0
        F1 = 2 * (P * R) / (P + R) if P + R else 0.0
        ACC = (TP + TN) / (len(inputs)) if inputs else 0.0

        return {
            'TPR': TPR, 'TNR': TNR, 'FPR': FPR, 'FNR': FNR,
            'P': P, 'R': R, 'F1': F1, 'ACC': ACC
        }

    def calculate(self, watermarked_result: List[bool], non_watermarked_result: List[bool]) -> Dict[str, float]:
        """calculate success rates of watermark detection based on provided results."""
        self._check_instance(watermarked_result, bool)
        self._check_instance(non_watermarked_result, bool)

        inputs = [DetectionResult(True, x) for x in watermarked_result] + [DetectionResult(False, x) for x in non_watermarked_result]
        metrics = self._compute_metrics(inputs)
        return self._filter_metrics(metrics)


class DynamicThresholdSuccessRateCalculator(BaseSuccessRateCalculator):
    """
        Calculator for success rates of watermark detection with dynamic thresholding.

        This class calculates success rates for watermark detection scenarios where the detection
        thresholds can dynamically change based on varying conditions. It supports evaluating the
        effectiveness of watermark detection algorithms that adapt to different signal or noise conditions.

        Use this class to evaluate detection systems where the threshold for detecting a watermark
        is not fixed and can vary.
    """

    def __init__(self, 
                 labels: List[str] = ['TPR', 'TNR', 'FPR', 'FNR', 'P', 'R', 'F1', 'ACC'], 
                 rule='best', 
                 target_fpr=None,
                 reverse=False) -> None:
        """
            Initialize the dynamic threshold success rate calculator.

            Parameters:
                labels (List[str]): The list of metric labels to include in the output.
                rule (str): The rule for determining the threshold. Choose from 'best' or 'target_fpr'.
                target_fpr (float): The target false positive rate to achieve.
                reverse (bool): Whether to reverse the sorting order of the detection results.
                                True: higher values are considered positive.
                                False: lower values are considered positive.
        """
        super().__init__(labels)
        self.rule = rule
        self.target_fpr = target_fpr
        self.reverse = reverse
        
        # Validate rule configuration
        if self.rule not in ['best', 'target_fpr']:
            raise ConfigurationError(f"Invalid rule specified: {self.rule}. Choose from 'best' or 'target_fpr'.")

        # Validate target_fpr configuration based on the rule
        if self.rule == 'target_fpr':
            if self.target_fpr is None:
                raise ConfigurationError("target_fpr must be set when rule is 'target_fpr'.")
            if not isinstance(self.target_fpr, (float, int)) or not (0 <= self.target_fpr <= 1):
                raise ConfigurationError("target_fpr must be a float or int within the range [0, 1].")

    def _find_best_threshold(self, inputs: List[DetectionResult]) -> float:
        """Find the best threshold that maximizes F1."""
        best_threshold = 0
        best_metrics = None
        for i in range(len(inputs) - 1):
            threshold = (inputs[i].detect_result + inputs[i + 1].detect_result) / 2
            metrics = self._compute_metrics(inputs, threshold)
            if best_metrics is None or metrics['F1'] > best_metrics['F1']:
                best_threshold = threshold
                best_metrics = metrics
        return best_threshold

    def _find_threshold_by_fpr(self, inputs: List[DetectionResult]) -> float:
        """Find the threshold that achieves the target FPR."""
        threshold = 0
        for i in range(len(inputs) - 1):
            threshold = (inputs[i].detect_result + inputs[i + 1].detect_result) / 2
            metrics = self._compute_metrics(inputs, threshold)
            if metrics['FPR'] <= self.target_fpr:
                break
        return threshold

    def _find_threshold(self, inputs: List[DetectionResult]) -> float:
        """Find the threshold based on the specified rule."""
        sorted_inputs = sorted(inputs, key=lambda x: x.detect_result, reverse=self.reverse)
        
        # If the rule is to find the best threshold by maximizing accuracy
        if self.rule == 'best':
            return self._find_best_threshold(sorted_inputs)
        else:
            # If the rule is to find the threshold that achieves the target FPR
            return self._find_threshold_by_fpr(sorted_inputs)

    def _compute_metrics(self, inputs: List[DetectionResult], threshold: float) -> Dict[str, float]:
        """Compute metrics based on the provided inputs and threshold."""
        if not self.reverse:
            TP = sum(1 for x in inputs if x.detect_result >= threshold and x.gold_label)
            FP = sum(1 for x in inputs if x.detect_result >= threshold and not x.gold_label)
            TN = sum(1 for x in inputs if x.detect_result < threshold and not x.gold_label)
            FN = sum(1 for x in inputs if x.detect_result < threshold and x.gold_label)
        else:
            TP = sum(1 for x in inputs if x.detect_result <= threshold and x.gold_label)
            FP = sum(1 for x in inputs if x.detect_result <= threshold and not x.gold_label)
            TN = sum(1 for x in inputs if x.detect_result > threshold and not x.gold_label)
            FN = sum(1 for x in inputs if x.detect_result > threshold and x.gold_label)

        metrics = {
            'TPR': TP / (TP + FN) if TP + FN else 0,
            'FPR': FP / (FP + TN) if FP + TN else 0,
            'TNR': TN / (TN + FP) if TN + FP else 0,
            'FNR': FN / (FN + TP) if FN + TP else 0,
            'P': TP / (TP + FP) if TP + FP else 0,
            'R': TP / (TP + FN) if TP + FN else 0,
            'F1': 2 * TP / (2 * TP + FP + FN) if 2 * TP + FP + FN else 0,
            'ACC': (TP + TN) / (len(inputs)) if inputs else 0
        }
        return metrics

    def calculate(self, watermarked_result: List[float], non_watermarked_result: List[float]) -> Dict[str, float]:
        """Calculate success rates based on provided results."""
        self._check_instance(watermarked_result + non_watermarked_result, float)

        inputs = [DetectionResult(True, x) for x in watermarked_result] + [DetectionResult(False, x) for x in non_watermarked_result]
        threshold = self._find_threshold(inputs)
        metrics = self._compute_metrics(inputs, threshold)
        return self._filter_metrics(metrics)