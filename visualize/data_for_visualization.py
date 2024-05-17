# =========================================
# data_for_visualization.py
# Description: Data class for visualization
# =========================================

from exceptions.exceptions import LengthMismatchError


class DataForVisualization:
    """Data class for visualization."""

    def __init__(self, decoded_tokens: list[str], highlight_values: list, weights: list=None) -> None:
        """
            Initialize the data for visualization.

            Parameters:
                decoded_tokens (list[str]): The decoded tokens.
                highlight_values (list): The highlight values.
                weights (list): The weights.
        """
        self.decoded_tokens = decoded_tokens
        self.highlight_values = highlight_values
        self.weights = weights
        if highlight_values is not None:
            if len(self.decoded_tokens) != len(self.highlight_values):
                raise LengthMismatchError(len(self.decoded_tokens), len(self.highlight_values))
        if weights is not None:
            if len(self.decoded_tokens) != len(self.weights):
                raise LengthMismatchError(len(self.decoded_tokens), len(self.weights))