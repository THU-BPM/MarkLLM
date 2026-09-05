# Copyright 2025 Kahim Wong.
# Copyright 2026 THU-BPM MarkLLM.
#
# The model architecture is adapted from E2E-LLM-Watermark, released under
# the MIT License. See watermark/e2e/LICENSE for the original license.

"""Neural encoder and detector used by E2E-LLM-Watermark."""

import torch
from torch import nn


class E2EEncoder(nn.Module):
    """Score top-k generation candidates from their contextual embeddings."""

    def __init__(
        self,
        input_dim: int,
        mapper_layers: int = 5,
        window_size: int = 10,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.mapper = nn.ModuleList([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
        for _ in range(mapper_layers - 1):
            self.mapper.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.mapper.append(nn.Linear(hidden_dim, hidden_dim))

        self.fc1 = nn.Linear(window_size * hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return a centered score in ``[-1, 1]`` for every candidate."""
        batch_size, candidate_count, _, _ = embeddings.shape
        for layer in self.mapper:
            embeddings = layer(embeddings)
        embeddings = embeddings.reshape(batch_size, candidate_count, -1)
        logits = self.fc2(self.relu(self.fc1(embeddings))).squeeze(-1)
        centered_logits = logits - logits.mean(dim=1, keepdim=True)
        return torch.tanh(1000 * centered_logits)


class E2EDetector(nn.Module):
    """Detect the learned watermark from a sequence of token embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 1,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_1 = nn.Linear(hidden_dim, num_classes)

    def forward(
        self, embeddings: torch.Tensor, return_all: bool = False
    ) -> torch.Tensor:
        """Return the final logit, or one logit per prefix for visualization."""
        hidden_states, _ = self.lstm(embeddings)
        if not return_all:
            hidden_states = hidden_states[:, -1, :]
        return self.fc_1(self.relu(self.fc_2(hidden_states)))
