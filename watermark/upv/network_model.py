# ============================================
# network_model.py
# Description: Network model for UPV algorithm
# ============================================

import torch
from torch import nn


class UPVSubNet(nn.Module):
    """Sub-network for UPV algorithm."""

    def __init__(self, input_dim: int, num_layers: int, hidden_dim: int = 64) -> None:
        """
            Initialize the UPV sub-network.

            Parameters:
                input_dim (int): Input dimension.
                num_layers (int): Number of layers.
                hidden_dim (int): Hidden dimension.
        """
        super(UPVSubNet, self).__init__()

        # Define layers
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class UPVGenerator(nn.Module):
    """Watermark Generator for UPV algorithm."""

    def __init__(self, input_dim: int, window_size: int, num_layers: int = 5, hidden_dim: int = 64):
        """
            Initialize the UPV generator.

            Parameters:
                input_dim (int): Input dimension.
                window_size (int): Window size.
                num_layers (int): Number of layers.
                hidden_dim (int): Hidden dimension.
        """
        super(UPVGenerator, self).__init__()

        # subnet
        self.sub_net = UPVSubNet(input_dim, num_layers, hidden_dim)
        self.window_size = window_size
        self.relu = nn.ReLU()

        # linear layer and sigmoid layer after merging features
        self.combine_layer = nn.Linear(window_size*hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the UPV generator."""

        # x is expected to be of shape (batch_size, window_size, input_dim)
        batch_size = x.shape[0]
        # Reshape x to (-1, input_dim) so it can be passed through the sub_net in one go
        x = x.view(-1, x.shape[-1])
        sub_net_output = self.sub_net(x)
        
        # Reshape sub_net_output back to (batch_size, window_size*hidden_dim)
        sub_net_output = sub_net_output.view(batch_size, -1) 
        combined_features = self.combine_layer(sub_net_output)
        combined_features = self.relu(combined_features)
        output = self.output_layer(combined_features)
        output = self.sigmoid(output)

        return output


class UPVDetector(nn.Module):
    """Watermark Detector for UPV algorithm."""

    def __init__(self, bit_number: int, b_layers: int = 5, input_dim: int = 64, 
                 hidden_dim: int = 128, num_classes: int = 1, num_layers: int = 2):
        """
            Initialize the UPV detector.

            Parameters:
                bit_number (int): Number of bits.
                b_layers (int): Number of layers for the binary classifier.
                input_dim (int): Input dimension.
                hidden_dim (int): Hidden dimension.
                num_classes (int): Number of classes.
                num_layers (int): Number of layers for the classifier.
        """
        super(UPVDetector, self).__init__()
        self.binary_classifier = UPVSubNet(bit_number, b_layers)
        self.classifier = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the UPV detector."""

        # Reshape the input to process through the binary classifier
        x1 = x.view(-1, x.shape[-1])
        features = self.binary_classifier(x1)
        # Reshape the features to be LSTM compatible
        features = features.view(x.size(0), x.size(1), -1)

        # Process features through the LSTM
        lstm_output, _ = self.classifier(features)
        # Select the last output for classification
        last_lstm_output = lstm_output[:, -1, :]

        # Pass through the hidden fully connected layer
        hidden_output = self.fc_hidden(last_lstm_output)
        hidden_output = self.sigmoid(hidden_output) 

        # Final classification
        final_output = self.fc(hidden_output)
        return self.sigmoid(final_output) 