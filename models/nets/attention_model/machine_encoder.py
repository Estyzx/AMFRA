"""AMFRA machine encoder.

Encodes the 13-D machine features [B, K, 13] (MachineFeatureBuilder) into
[B, K, embedding_dim] machine embeddings (the machine-side query source in
AMFRA attention).

Structure: Linear → GELU → Linear → LayerNorm, plus a fixed sinusoidal machine
positional encoding (a buffer, so the parameter count is independent of K and
the encoder is permutation equivariant up to the fixed encoding). Invalid
machines (mask=False) output an exactly-zero embedding.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class MachineEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = 13,
        embedding_dim: int = 128,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_position_encoding: bool = True,
    ):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else embedding_dim
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.use_position_encoding = use_position_encoding

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        if use_position_encoding:
            # Precomputed sinusoidal encodings for up to 64 machines (buffer, not a parameter)
            max_machines = 64
            pe = torch.zeros(max_machines, embedding_dim)
            position = torch.arange(0, max_machines, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("position_encoding", pe)  # [max_machines, embedding_dim]

    def forward(
        self,
        machine_features: torch.Tensor,
        machine_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """machine_features [B, K, feature_dim], machine_valid_mask [B, K] bool
        → machine embeddings [B, K, embedding_dim] (invalid machines zeroed)."""
        B, K, F = machine_features.shape
        assert F == self.feature_dim
        assert machine_valid_mask.shape == (B, K)

        embeddings = self.net(machine_features)
        if self.use_position_encoding:
            assert K <= self.position_encoding.size(0)
            embeddings = embeddings + self.position_encoding[:K].unsqueeze(0)

        # Invalid machines output exact zero vectors
        return embeddings * machine_valid_mask.to(embeddings.dtype).unsqueeze(-1)
