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
# exceptions.py
# Description: Custom exceptions for the project
# ==============================================

class LengthMismatchError(Exception):
    """Exception raised when the expected and actual lengths do not match."""
    def __init__(self, expected, actual):
        message = f"Expected length: {expected}, but got {actual}."
        super().__init__(message)


class InvalidTextSourceModeError(ValueError):
    """Exception raised when an invalid text source mode is provided."""
    def __init__(self, mode):
        message = f"'{mode}' is not a valid text source mode. Choose 'natural' or 'generated'."
        super().__init__(message)


class AlgorithmNameMismatchError(ValueError):
    """Exception raised when the algorithm name in the config does not match the expected watermark algorithm class."""
    def __init__(self, expected, actual):
        message = f"Config algorithm name '{actual}' does not match expected algorithm name '{expected}'."
        super().__init__(message)


class InvalidDirectAnalyzerTypeError(Exception):
    """Exception raised when an invalid text quality analyzer type is provided."""
    def __init__(self, message="Analyzer must be a type of DirectTextQualityAnalyzer"):
        super().__init__(message)


class InvalidReferencedAnalyzerTypeError(Exception):
    """Exception raised when an invalid referenced text quality analyzer type is provided."""
    def __init__(self, message="Analyzer must be a type of ReferencedTextQualityAnalyzer"):
        super().__init__(message)


class InvalidAnswerError(ValueError):
    """Exception raised for an invalid answer input."""
    def __init__(self, answer):
        super().__init__(f"Invalid answer: {answer}")


class TypeMismatchException(Exception):
    """Exception raised when a type mismatch is found in the data."""
    def __init__(self, expected_type: type, found_type: type, message: str = ""):
        self.expected_type = expected_type
        self.found_type = found_type
        self.message = message
        super().__init__(self.message or f"Expected all items to be of type {self.expected_type.__name__}, but found type {self.found_type.__name__}.")


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration of success rate calculators."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class OpenAIModelConfigurationError(Exception):
    """Exception raised for errors in the OpenAI model configuration."""
    def __init__(self, message: str):
        super().__init__(message)


class DiversityValueError(Exception):
    """Exception raised when the diversity values are not within the expected range."""
    def __init__(self, diversity_type: str):
        message = f"{diversity_type} diversity must be one of 0, 20, 40, 60, 80, 100."
        super().__init__(message)


class CodeExecutionError(Exception):
    """Exception raised when there is an error in code execution during tests."""
    def __init__(self, message="Error during code execution"):
        self.message = message
        super().__init__(self.message)

class InvalidDetectModeError(Exception):
    """Exception raised for errors in the input detect mode."""
    def __init__(self, mode, message="Invalid detect mode configuration"):
        self.mode = mode
        self.message = message
        super().__init__(f"{message}: {mode}")

class InvalidWatermarkModeError(Exception):
    """Exception raised for errors in the input watermark mode."""
    def __init__(self, mode, message="Invalid watermark mode configuration"):
        self.mode = mode
        self.message = message
        super().__init__(f"{message}: {mode}")