"""Static node embeddings for the harvest problem."""

import torch
import torch.nn as nn


class HarvestEmbedding(nn.Module):
    """Static field/depot node embedding (6-dimensional field features).

    input : dict with 'depot' [batch, 2], 'field_features' [batch, n_fields, 6]
    output : [batch, n_fields + 1, embedding_dim] (depot node first)
    """

    def __init__(self, embedding_dim):
        super(HarvestEmbedding, self).__init__()
        node_dim = 6  # x, y, service_time, waiting, is_pending, is_active
        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.init_embed_depot = nn.Linear(2, embedding_dim)

    def forward(self, input):
        depot_embedding = self.init_embed_depot(input["depot"])[:, None, :]
        node_embeddings = self.init_embed(input["field_features"])
        return torch.cat((depot_embedding, node_embeddings), 1)
