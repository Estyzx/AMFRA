import torch
from torch import nn


from ...nets.attention_model.multi_head_attention import (
    AttentionScore,
    MultiHeadAttention,
)
from ...nets.attention_model.machine_features import MachineFeatureBuilder
from ...nets.attention_model.machine_encoder import MachineEncoder
from ...nets.attention_model.machine_field_relations import MachineFieldRelationBuilder
from ...nets.attention_model.amfra_attention import AsynchronousMachineFieldAttention


class Decoder(nn.Module):
    r"""Decoder of the Attention Model (harvest problem only).

    Keys and values are first precomputed for the encoder embeddings :math:`\pmb{h}`:

    .. math::
        \pmb{k}, \pmb{v}, \pmb{k}^\prime = W^K\pmb{h}, W^V\pmb{h}, W^{K^\prime}\pmb{h}

    together with the graph-embedding projection
    :math:`W_{gc}\bar{\pmb{h}}` for
    :math:`\bar{\pmb{h}} = \frac{1}{N}\sum\nolimits_i \pmb{h}_i`.

    Each decoding step then runs one of two paths (autoregressive sampling and
    mask updates happen in the Agent layer):

    - ``decoder_mode="joint"`` (A0/A2/A3): the AMFRA joint pipeline — machine
      features [B,K,13] → machine embeddings [B,K,D] → relation features
      [B,K,N,12] → pair logits [B,K,N]; the per-machine return logits are
      concatenated as an extra (N+1)-th column, giving ``static_logits``
      [B,K,N+1];
    - ``decoder_mode="field_pointer"`` (E0): plain AM pointer attention over the
      visible fields giving ``field_logits`` [B,N]; the Agent layer turns the
      chosen field into a (machine, field) action with the deterministic
      "nearest feasible machine" adapter.
    """

    def __init__(
        self,
        embedding_dim,
        n_heads,
        tanh_clipping,
        amfra_use_gate=True,
        amfra_map_size=10.0,
        use_relation_bias=True,
        decoder_mode="joint",
    ):
        super(Decoder, self).__init__()
        if decoder_mode not in ("joint", "field_pointer"):
            raise ValueError(
                f"decoder_mode must be 'joint'/'field_pointer', got {decoder_mode!r}"
            )
        self.decoder_mode = decoder_mode
        self.use_amfra = decoder_mode == "joint"

        # Each node is projected into three parts (glimpse key, glimpse value, logit key)
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # glimpse/pointer are only used by the E0 field-pointer path
        self.glimpse = MultiHeadAttention(embedding_dim=embedding_dim, n_heads=n_heads)
        self.pointer = AttentionScore(use_tanh=True, C=tanh_clipping)

        if self.use_amfra:
            self.amfra_feature_builder = MachineFeatureBuilder()
            self.amfra_machine_encoder = MachineEncoder(
                feature_dim=13, embedding_dim=embedding_dim
            )
            self.amfra_relation_builder = MachineFieldRelationBuilder(map_size=amfra_map_size)
            self.amfra_attention = AsynchronousMachineFieldAttention(
                embedding_dim=embedding_dim,
                relation_dim=12,
                use_gate=amfra_use_gate and self.use_amfra,
                use_relation_bias=use_relation_bias and self.use_amfra,
            )
        # STOP logit head (zero-initialized; consumed by the E0 path, kept in the
        # joint path so that all released checkpoints share one state_dict layout)
        self.stop_head = nn.Linear(embedding_dim, 1)
        nn.init.zeros_(self.stop_head.weight)
        nn.init.zeros_(self.stop_head.bias)
        # Per-machine "return" logit head (zero-initialized); input = machine
        # embedding ⊕ graph context ⊕ remaining-field fraction ⊕ nearest feasible
        # field normalized distance
        self.return_head = nn.Linear(2 * embedding_dim + 2, 1)
        nn.init.zeros_(self.return_head.weight)
        nn.init.zeros_(self.return_head.bias)

    def _precompute(self, embeddings, keep_mask=None):
        # The fixed-context projection of the graph embedding is computed only once (efficiency)
        if keep_mask is None:
            graph_embed = embeddings.mean(1)
        else:
            # No look-ahead pooling: average only over visible nodes (depot included);
            # FUTURE/padding slots stay out of the graph context
            keep = keep_mask.to(embeddings.dtype).unsqueeze(-1)  # [B,n,1]
            count = keep.sum(dim=1).clamp(min=1.0)               # [B,1]
            graph_embed = (embeddings * keep).sum(1) / count
        graph_context = self.project_fixed_context(graph_embed).unsqueeze(-2)

        glimpse_key, glimpse_val, logit_key = self.project_node_embeddings(embeddings).chunk(
            3, dim=-1
        )
        return (embeddings, graph_context, glimpse_key, glimpse_val, logit_key)

    def advance(self, cached_embeddings, state):
        """Single decoding step: returns (bundle, graph_context); the bundle is
        {"static_logits": [B,K,N+1]} for joint decoding or
        {"field_logits": [B,N], "stop_logit": [B,1]} for the E0 field pointer."""
        node_embeddings, graph_context, glimpse_K, glimpse_V, logit_K = cached_embeddings

        if self.decoder_mode == "joint":
            bundle = self._compute_amfra_joint(node_embeddings, graph_context, state)
        else:
            bundle = self._compute_field_pointer(
                node_embeddings, graph_context,
                glimpse_K, glimpse_V, logit_K, state,
            )
        return bundle, graph_context

    def _compute_amfra_joint(self, node_embeddings, graph_context, state):
        """AMFRA joint decoding forward pipeline (called once per decision step).

        Task embeddings reuse the graph-encoder cache (static); machine features
        and relation features are recomputed from the current environment state
        (dynamic — this is what "asynchronous" means).

        Returns {"static_logits": [B,K,N+1]}: field columns are pair logits with
        -inf on infeasible pairs; the last column is the per-machine return logit.
        Legality of the return slot depends on which fields earlier machines have
        already taken, so Agent._static_step re-masks sequentially per machine
        (every step legal, no duplicate assignment).
        """
        s = state.states
        machine_positions = s["machine_positions"]       # [B, K, 2]
        machine_status = s["machine_status"]             # [B, K]
        field_work_used = s["field_work_used"]           # [B, K]
        max_field_work = s["max_field_work"]             # [B, K]
        sorties_used = s["sorties_used"]                 # [B, K]
        available_at = s["available_at"]                 # [B, K]
        global_time = s["global_time"]                   # [B, 1]
        work_hours = s["work_hours"]                     # [B, 1]
        travel_speed = s["travel_speed"]                 # [B, K]
        service_times = s["service_times"]               # [B, K, N]
        visible_field_mask = s["visible_field_mask"]     # [B, N] bool
        feasible_pair_mask = s["feasible_pair_mask"]     # [B, K, N] bool
        field_coords = s["observations"]["loc"]          # [B, N, 2]
        harvest_efficiencies = s["machine_harvest_efficiencies"]  # [B, K]
        move_costs = s["machine_move_costs"]                      # [B, K]
        activation_costs = s["machine_activation_costs"]          # [B, K]
        compatibility_mask = s["compatibility_mask"]              # [B, K, N]
        cumulative_time = s["cumulative_time"]                    # [B, K]

        # 1. Machine features → machine embeddings; MachineEncoder treats idle_mask as the
        #    validity mask: BUSY/RETURNING are zeroed and serve only as observation context
        mf = self.amfra_feature_builder(
            machine_positions=machine_positions,
            machine_status=machine_status,
            field_work_used=field_work_used,
            max_field_work=max_field_work,
            sorties_used=sorties_used,
            available_at=available_at,
            global_time=global_time,
            work_hours=work_hours,
            harvest_efficiencies=harvest_efficiencies,
            move_costs=move_costs,
            activation_costs=activation_costs,
            compatibility_mask=compatibility_mask,
            cumulative_time=cumulative_time,
        )
        machine_embeddings = self.amfra_machine_encoder(mf.features, mf.idle_mask)

        # 2. Relation features (real coordinates/times/budgets, never embeddings); pair_mask is
        #    the environment's feasible_pair_mask passed straight through, keeping the model
        #    exactly in sync with the environment rules
        rel = self.amfra_relation_builder(
            machine_positions=machine_positions,
            field_positions=field_coords,
            field_work_used=field_work_used,
            max_field_work=max_field_work,
            idle_mask=mf.idle_mask,
            service_times=service_times,
            feasible_pair_mask=feasible_pair_mask,
            valid_field_mask=visible_field_mask,
            global_time=global_time,
            work_hours=work_hours,
            travel_speed=travel_speed,
            move_costs=move_costs,
        )

        # 3. AMFRA pair scoring: pair_logits [B,K,N] (infeasible pairs = -inf)
        amfra_out = self.amfra_attention(
            machine_embeddings=machine_embeddings,
            field_embeddings=node_embeddings[:, 1:, :],
            relation_features=rel.features,
            pair_mask=rel.pair_mask,
            machine_valid_mask=mf.idle_mask,
            field_valid_mask=visible_field_mask,
        )
        pair_logits = amfra_out.pair_logits

        # 4. Static variant: return logit + [B,K,N+1] concatenation. The return-logit
        #    input is a deterministic function of the current obs and independent of any
        #    within-step prefix, preserving the rollout/replay mask contract; zero
        #    initialization keeps the initial return logit at exactly 0.
        Bs, Ks, Ds = machine_embeddings.shape
        remaining_frac = (
            visible_field_mask.to(pair_logits.dtype).sum(
                dim=-1, keepdim=True) / max(visible_field_mask.size(-1), 1)
        )  # [B, 1]
        feasible_dist = torch.where(
            rel.pair_mask, rel.distance,
            torch.full_like(rel.distance, float("inf")),
        )
        has_field = feasible_dist.isfinite().any(dim=-1)     # [B,K]
        nearest_norm = feasible_dist.min(dim=-1).values / self.amfra_relation_builder.max_map_distance
        nearest_norm = torch.where(
            has_field, nearest_norm.clamp(0.0, 1.0),
            torch.ones_like(nearest_norm),                   # no candidate = farthest
        )  # [B,K]
        graph_ctx = graph_context.squeeze(-2)                # [B,D]
        return_input = torch.cat([
            machine_embeddings,                              # [B,K,D]
            graph_ctx.unsqueeze(1).expand(Bs, Ks, Ds),
            remaining_frac.unsqueeze(-1).expand(Bs, Ks, 1),  # [B,K,1]
            nearest_norm.unsqueeze(-1),                      # [B,K,1]
        ], dim=-1)                                           # [B,K,2D+2]
        return_logits = self.return_head(return_input).squeeze(-1)   # [B,K]
        static_logits = torch.cat(
            [pair_logits, return_logits.unsqueeze(-1)], dim=-1)      # [B,K,N+1]
        return {"static_logits": static_logits}

    def _compute_field_pointer(
        self, node_embeddings, graph_context, glimpse_K, glimpse_V, logit_K, state,
    ):
        """E0 plain Encoder–Decoder field-pointer forward pass (Kool et al. 2019).

        The graph context (projection of the mean over visible nodes) acts as the
        query → a multi-head glimpse attends over the visible fields → a
        single-head pointer produces the field logits [B,N]; the STOP logit comes
        from stop_head on the graph context. No machine-field joint pair scoring,
        relation bias or gating is used. Returns:
          field_logits : [B, N], logits of the visible fields, -inf for invisible ones;
          stop_logit   : [B, 1].
        """
        s = state.states
        visible_field_mask = s["visible_field_mask"].to(torch.bool)  # [B,N]
        # Field embeddings take nodes 1..N (depot node 0 excluded); the depot is not a harvestable task
        field_mask = (~visible_field_mask).unsqueeze(1)              # [B,1,N]
        glimpse = self.glimpse(
            graph_context,
            glimpse_K[:, 1:, :],
            glimpse_V[:, 1:, :],
            field_mask,
        )                                                           # [B,1,D]
        field_logits = self.pointer(
            glimpse, logit_K[:, 1:, :], field_mask
        ).squeeze(1)                                                # [B,N]
        stop_logit = self.stop_head(graph_context.squeeze(-2))      # [B,1]
        return {"field_logits": field_logits, "stop_logit": stop_logit}
