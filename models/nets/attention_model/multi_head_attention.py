import math

import torch
from torch import nn


class AttentionScore(nn.Module):
    r"""Attention alignment scorer (parameter-free).

    .. math::
        u_j =
        \begin{cases}
             &C\mathrm{tanh}\left(\frac{q^Tk_j}{\sqrt{\smash{d_q}}}\right) & \text{ if } j \notin \mathrm{mask}\\
             &-\infty & \text{ otherwise. }
        \end{cases}

    (without tanh clipping when ``use_tanh`` is False)
    """

    def __init__(self, use_tanh=False, C=10):
        super(AttentionScore, self).__init__()
        self.use_tanh = use_tanh
        self.C = C

    def forward(self, query, key, mask=torch.zeros([], dtype=torch.bool)):
        """query/key: [..., 1, h_dim] / [..., graph_size, h_dim]; mask True = -inf."""
        u = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if self.use_tanh:
            logits = torch.tanh(u) * self.C
        else:
            logits = u
        return logits.masked_fill(mask.expand_as(logits), float("-inf"))


class MultiHeadAttention(nn.Module):
    r"""Multi-head attention: softmax(AttentionScore) over value heads, then W^O projection."""

    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.attentionScore = AttentionScore()
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, query, key, value, mask):
        """query : [batch, n_queries, embedding_dim]; key/value : [batch, n_keys, embedding_dim];
        mask : [batch, 1, n_keys] bool. Returns [batch, n_queries, embedding_dim]."""
        query_heads = self._make_heads(query)
        key_heads = self._make_heads(key)
        value_heads = self._make_heads(value)
        # [n_heads, batch, n_queries, n_keys]
        compatibility = self.attentionScore(query_heads, key_heads, mask)
        attention_weights = torch.softmax(compatibility, dim=-1)
        out_heads = torch.matmul(attention_weights, value_heads)
        return self.project_out(self._unmake_heads(out_heads))

    def _make_heads(self, v):  # [batch, nkeys, n_heads*head_dim] -> [n_heads, batch, nkeys, head_dim]
        batch_size, nkeys, h_dim = v.shape
        return v.reshape(batch_size, nkeys, self.n_heads, h_dim // self.n_heads).movedim(-2, 0)

    def _unmake_heads(self, v):  # [n_heads, batch, ..., head_dim] -> [batch, ..., n_heads*head_dim]
        return v.movedim(0, -2).flatten(-2)
