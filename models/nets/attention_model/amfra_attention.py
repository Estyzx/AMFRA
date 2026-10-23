"""AMFRA machine-field relation attention (optional dynamic gating).

Machine-field pair logits combine:
1. scaled dot-product matching (machine_query · field_key / sqrt(d))
2. a relation bias learned from the 12 pairwise relation features by an MLP
   (last layer zero-initialized so the initial bias ≈ 0)
3. an optional dynamic gate ∈ [0,1] scaling the relation bias (A3)

Invalid (machine, field) pairs are filled with -inf; softmax/sampling happens
externally in the Agent layer.
"""

import math
from typing import NamedTuple, Optional

import torch
import torch.nn as nn


class AMFRAAttentionOutput(NamedTuple):
    pair_logits: torch.Tensor
    base_logits: torch.Tensor
    relation_bias: torch.Tensor
    relation_gate: torch.Tensor
    gated_relation_bias: torch.Tensor
    effective_pair_mask: torch.Tensor


class AsynchronousMachineFieldAttention(nn.Module):
    """Machine-field relation attention producing pair logits [B, K, N].

    - Wq/Wk are shared across machines and fields; no positional encoding, so the
      module is permutation equivariant over the K and N dims;
    - the relation MLP last layer and the gate output bias are zero-initialized,
      so training starts smoothly from pure match scoring with gate ≈ 0.5;
    - no softmax inside this module.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        attention_dim: Optional[int] = None,
        relation_dim: int = 12,
        relation_hidden_dim: int = 64,
        mask_value: float = float("-inf"),
        use_gate: bool = False,
        gate_hidden_dim: int = 64,
        use_relation_bias: bool = True,
    ):
        super().__init__()
        attention_dim = attention_dim or embedding_dim
        self.embedding_dim = embedding_dim
        self.relation_dim = relation_dim
        self.mask_value = mask_value
        self.use_gate = use_gate
        self.use_relation_bias = use_relation_bias
        self.scale = math.sqrt(attention_dim)

        self.Wq = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, attention_dim, bias=False)

        self.relation_mlp = nn.Sequential(
            nn.Linear(relation_dim, relation_hidden_dim),
            nn.GELU(),
            nn.Linear(relation_hidden_dim, 1),
        )
        nn.init.zeros_(self.relation_mlp[2].weight)
        nn.init.zeros_(self.relation_mlp[2].bias)

        if use_gate:
            self.gate_machine_proj = nn.Linear(embedding_dim, gate_hidden_dim, bias=False)
            self.gate_field_proj = nn.Linear(embedding_dim, gate_hidden_dim, bias=False)
            self.gate_relation_proj = nn.Linear(relation_dim, gate_hidden_dim, bias=True)
            self.gate_output = nn.Linear(gate_hidden_dim, 1, bias=True)
            nn.init.zeros_(self.gate_output.bias)

    def forward(
        self,
        machine_embeddings: torch.Tensor,
        field_embeddings: torch.Tensor,
        relation_features: torch.Tensor,
        pair_mask: torch.Tensor,
        machine_valid_mask: torch.Tensor,
        field_valid_mask: torch.Tensor,
    ) -> AMFRAAttentionOutput:
        """machine_embeddings [B,K,D], field_embeddings [B,N,D],
        relation_features [B,K,N,R], pair_mask [B,K,N] bool,
        machine_valid_mask [B,K] bool, field_valid_mask [B,N] bool."""
        B, K, D = machine_embeddings.shape
        N = field_embeddings.shape[1]
        assert relation_features.shape == (B, K, N, self.relation_dim)
        assert pair_mask.shape == (B, K, N)
        assert machine_valid_mask.shape == (B, K)
        assert field_valid_mask.shape == (B, N)

        # Base match score: scaled dot product of machine queries and field keys
        machine_query = self.Wq(machine_embeddings)  # [B, K, attention_dim]
        field_key = self.Wk(field_embeddings)        # [B, N, attention_dim]
        base_logits = torch.bmm(machine_query, field_key.transpose(1, 2)) / self.scale

        # Validity = hard-constraint pair_mask ∧ machine active ∧ field real
        effective_pair_mask = (
            pair_mask
            & machine_valid_mask.unsqueeze(2)
            & field_valid_mask.unsqueeze(1)
        )

        if not self.use_relation_bias:
            # Degenerate case (never used by the registered ablations): match score only
            relation_bias = torch.zeros_like(base_logits)
            relation_gate = effective_pair_mask.to(base_logits.dtype)
            gated_relation_bias = torch.zeros_like(base_logits)
            raw_logits = base_logits
        else:
            # Relation bias: relation features → MLP → one scalar per pair
            relation_bias = self.relation_mlp(relation_features).squeeze(-1)  # [B, K, N]
            if not self.use_gate:
                # A2: logits = base + relation_bias
                relation_gate = effective_pair_mask.to(base_logits.dtype)
                gated_relation_bias = relation_bias
                raw_logits = base_logits + relation_bias
            else:
                # A3: additive-projection gate, logits = base + gate * relation_bias
                m_gate = self.gate_machine_proj(machine_embeddings)  # [B, K, H]
                f_gate = self.gate_field_proj(field_embeddings)      # [B, N, H]
                r_gate = self.gate_relation_proj(relation_features)  # [B, K, N, H]
                gate_hidden = torch.nn.functional.gelu(
                    m_gate.unsqueeze(2) + f_gate.unsqueeze(1) + r_gate)
                relation_gate = torch.sigmoid(
                    self.gate_output(gate_hidden).squeeze(-1))       # [B, K, N]
                # Force the gate to 0 on invalid pairs
                relation_gate = torch.where(
                    effective_pair_mask, relation_gate, torch.zeros_like(relation_gate))
                gated_relation_bias = relation_gate * relation_bias
                raw_logits = base_logits + gated_relation_bias

        pair_logits = torch.where(
            effective_pair_mask,
            raw_logits,
            torch.full_like(raw_logits, self.mask_value),
        )
        return AMFRAAttentionOutput(
            pair_logits=pair_logits,
            base_logits=base_logits,
            relation_bias=relation_bias,
            relation_gate=relation_gate,
            gated_relation_bias=gated_relation_bias,
            effective_pair_mask=effective_pair_mask,
        )
