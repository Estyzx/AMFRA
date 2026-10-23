import torch
from torch import nn

from ...nets.attention_model.multi_head_attention import MultiHeadAttention


class Normalization(nn.Module):
    """BatchNorm normalization (flattened over nodes before normalizing)."""
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.BatchNorm1d(embedding_dim, affine=True)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(input.size())


class StandardSelfAttention(nn.Module):
    """Standard Transformer self-attention (Q=K=V come from the nodes)."""

    def __init__(self, embedding_dim, n_heads):
        super().__init__()
        if embedding_dim % n_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.query_encoder = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key_encoder = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value_encoder = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.multi_head_attention = MultiHeadAttention(embedding_dim, n_heads)

    def forward(self, node_embeddings, node_mask=None):
        query = self.query_encoder(node_embeddings)
        key = self.key_encoder(node_embeddings)
        value = self.value_encoder(node_embeddings)
        if node_mask is None:
            node_mask = torch.zeros((), dtype=torch.bool, device=node_embeddings.device)
        # Masked nodes do not participate in self-attention (not a look-ahead constraint)
        return self.multi_head_attention(query, key, value, node_mask)


class MultiHeadAttentionLayer(nn.Module):
    r"""Transformer encoder layer: self-attention + MLP, each with a residual
    connection and BatchNorm.

    .. math::
        \begin{aligned}
        \pmb{x}_1&=\mathrm{BatchNorm}(\pmb{x}+\mathrm{MultiHeadAttention}(q,\pmb{k},\pmb{v})) \\
        \pmb{h}&=\mathrm{BatchNorm}(\pmb{x}_1+\mathrm{MLP}(\pmb{x}_1))
        \end{aligned}
    """

    def __init__(self, n_heads, embedding_dim, feed_forward_hidden=512):
        super().__init__()
        self.attention = StandardSelfAttention(embedding_dim, n_heads)
        self.attention_normalization = Normalization(embedding_dim)
        self.feed_forward_normalization = Normalization(embedding_dim)
        self.feed_forward = (
            nn.Sequential(
                nn.Linear(embedding_dim, feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(feed_forward_hidden, embedding_dim),
            )
            if feed_forward_hidden > 0
            else nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, node_embeddings, node_mask=None):
        attention_output = self.attention(node_embeddings, node_mask=node_mask)
        normalized_attention = self.attention_normalization(
            node_embeddings + attention_output)
        ff_out = self.feed_forward(normalized_attention)
        return self.feed_forward_normalization(normalized_attention + ff_out)


class GraphAttentionEncoder(nn.Module):
    r"""Graph attention encoder: ``n_layers`` self-attention layers over the nodes.

    Inputs
    ------
    node_embeddings : [batch, graph_size, embedding_dim]
    mask : [batch, graph_size] bool or None; True = masked node (e.g. a FUTURE
        task slot that must not take part in encoder attention).

    Outputs
    -------
    (h, h_mean) : h [batch, graph_size, embedding_dim]
    """

    def __init__(self, n_heads, embed_dim, n_layers, feed_forward_hidden=512):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.layers = nn.ModuleList(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden)
            for _ in range(n_layers)
        )

    def forward(self, node_embeddings, mask=None):
        if mask is not None:
            assert mask.ndim == 2 and mask.shape == node_embeddings.shape[:2]
            mask = mask.to(device=node_embeddings.device, dtype=torch.bool)
            mask = mask.unsqueeze(1)  # [B, 1, n_keys]
        encoded = node_embeddings
        for layer in self.layers:
            encoded = layer(encoded, node_mask=mask)
        return encoded, encoded.mean(dim=1)
