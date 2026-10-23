import numpy as np
import torch
import torch.nn as nn

from .nets.attention_model.decoder import Decoder
from .nets.attention_model.embedding import HarvestEmbedding
from .nets.attention_model.encoder import GraphAttentionEncoder


class Backbone(nn.Module):
    """Attention model backbone: graph encoder + decoder.

    encode() embeds the field/depot graph and caches the glimpse K/V and the
    graph context; decode() produces one decoding step via Decoder.advance
    (AMFRA joint logits [B,K,N+1], or E0 field-pointer logits).
    """

    def __init__(
        self,
        embedding_dim=128,
        n_encode_layers=3,
        tanh_clipping=10.0,
        n_heads=8,
        device="cpu",
        amfra_use_gate=True,
        amfra_map_size=10.0,
        use_relation_bias=True,
        decoder_mode="joint",
    ):
        super(Backbone, self).__init__()
        self.device = device
        self.embedding = HarvestEmbedding(embedding_dim)
        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=n_encode_layers,
        )
        self.decoder = Decoder(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            tanh_clipping=tanh_clipping,
            amfra_use_gate=amfra_use_gate,
            amfra_map_size=amfra_map_size,
            use_relation_bias=use_relation_bias,
            decoder_mode=decoder_mode,
        )

    def _node_mask(self, state):
        """Encoder masking mask [B, n_nodes] (True = masked): FUTURE/padding task
        slots must not take part in encoder attention; the depot (node 0) is
        always visible."""
        visible = state.states["visible_field_mask"].to(torch.bool)   # [B, N]
        B = visible.size(0)
        depot_visible = torch.ones(B, 1, dtype=torch.bool, device=visible.device)
        keep = torch.cat([depot_visible, visible], dim=1)             # [B, N+1]
        return ~keep  # True = masked

    def encode(self, obs):
        """Encode the observation and precompute the decoding cache."""
        state = stateWrapper(obs, device=self.device)
        input = state.states["observations"]  # field coordinates, depot coordinates, field features
        embedding = self.embedding(input)     # [batch, n_fields+1, embedding_dim]
        node_mask = self._node_mask(state)
        encoded_inputs, _ = self.encoder(embedding, mask=node_mask)
        return self.decoder._precompute(
            encoded_inputs,
            keep_mask=None if node_mask is None else (~node_mask),
        )

    def decode(self, obs, cached_embeddings):
        """Decode one step using the cache, returning (bundle, graph_context)."""
        state = stateWrapper(obs, device=self.device)
        return self.decoder.advance(cached_embeddings, state)


class Agent(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        device="cpu",
        n_encode_layers=3,
        amfra_use_gate=True,
        amfra_map_size=10.0,
        use_relation_bias=True,
        decoder_mode="joint",
    ):
        super().__init__()
        self.device = device
        self.decoder_mode = decoder_mode
        self.backbone = Backbone(
            embedding_dim=embedding_dim,
            device=device,
            n_encode_layers=n_encode_layers,
            amfra_use_gate=amfra_use_gate,
            amfra_map_size=amfra_map_size,
            use_relation_bias=use_relation_bias,
            decoder_mode=decoder_mode,
        )

    def get_joint_action_and_value_cached(
        self, x, action_seq=None, state=None, return_logits=False, greedy=False
    ):
        """Joint action interface (shared by rollout sampling and training replay).

        Joint action = per-machine selection sequence; the joint log-probability
        is the sum over the machines, and on replay the saved order is replayed
        step by step. Each per-step mask is a deterministic function of
        (state, selected prefix), so the rollout/replay joint log-prob is exactly
        identical.

        Parameters
        ----
        x : obs dict.
        action_seq : [B, K] long or None. None = sampling; otherwise exact replay.
        state : encoding cache (output of Backbone.encode) or None.
        greedy : during sampling use argmax instead of multinomial sampling (evaluation path).

        Returns
        ----
        action_seq [B,K], joint_logprob [B], entropy [B], state,
        sub_valid_mask [B,K], and (when return_logits) the E0 field-selection
        sequence (or None) appended at the end.
        """
        if state is None:
            state = self.backbone.encode(x)
        bundle, _ = self.backbone.decode(x, state)
        if "static_logits" in bundle:
            # Main path: per-machine sequential categorical [B,K,N+1] (without-replacement mask)
            sel, lp, ent, sub_valid = self._static_step(
                bundle, x, action=action_seq, greedy=greedy)
            if return_logits:
                return sel, lp, ent, state, sub_valid, None
            return sel, lp, ent, state, sub_valid
        # E0 ablation: pick the field first, the nearest feasible machine executes → [K,N+1] machine action.
        # The return_logits slot returns the field-selection sequence (required by the replay contract).
        sel, lp, ent, sub_valid, fseq = self._static_step_field_pointer(
            bundle, x, action=action_seq, greedy=greedy)
        if return_logits:
            return sel, lp, ent, state, sub_valid, fseq
        return sel, lp, ent, state, sub_valid

    def _static_step(self, bundle, x, action=None, greedy=False):
        """Static-version single step: sequential machine selection (independent categorical [N+1] per machine).

        Sequential masking contract (aligned with the heterogeneous env's action_mask):
        - use the env ``action_mask`` directly as the base mask (it already encodes
          the three constraint layers: field-level area, trip-level H_k,
          machine-level T_max + standby logic);
        - field candidates get the without-replacement constraint on top (the taken
          set is updated step by step inside the loop);
        - legality of the return-to-depot/standby slot is guaranteed by the env
          action_mask: in trip → may return; at depot → may stand by;
          T_max exhausted → standby only;
        - **forced dispatch**: while serviceable fields remain and nothing has been
          dispatched this round, the "standby" slot of **at-depot machines** is
          masked off (bit-for-bit consistent with the force rule of the env step),
          which rules out the "all harvesters stand by" stall; the "return" of
          in-trip machines is untouched (returning is normal multi-trip behavior);
        - therefore every machine keeps at least one legal action per step (the
          Categorical never gets an all -inf row), and the same field cannot be
          assigned twice to two machines within one step;
        - on replay, the same mask sequence is replayed machine by machine in stored
          order, so rollout/replay log-probs match exactly (the mask is a
          deterministic function of (state, selected prefix)).

        Returns a 4-tuple:
        (action [B,K], joint_lp [B], entropy [B], sub_valid).
        """
        logits = bundle["static_logits"]          # [B,K,N+1]
        B, K, Np1 = logits.shape
        N = Np1 - 1
        device = logits.device
        # Use the env action_mask [B,K,N+1] as the base mask (three constraint layers already included)
        action_mask = torch.as_tensor(x["action_mask"], dtype=torch.bool,
                                      device=device)                  # [B,K,N+1]
        taken = torch.zeros(B, N, dtype=torch.bool, device=device)
        in_trip = torch.as_tensor(x["in_trip"], dtype=torch.bool,
                                  device=device)                        # [B,K]
        # Forced dispatch: serviceable fields remain and nothing dispatched this round → mask the standby slot of at-depot machines (kept in sync with env)
        any_serviceable = action_mask[:, :, :N].any(dim=1).any(dim=1)  # [B]
        dispatched = torch.zeros(B, dtype=torch.bool, device=device)
        sel_out = torch.zeros(B, K, dtype=torch.long, device=device)
        lp_sum = torch.zeros(B, device=device)
        ent_sum = torch.zeros(B, device=device)
        replay = action is not None
        for k in range(K):
            lk = logits[:, k, :]                                    # [B, N+1]
            # Field candidates = env-feasible AND not already taken by earlier machines in the same step
            fk = action_mask[:, k, :N] & ~taken                     # [B, N]
            ok = torch.zeros(B, Np1, dtype=torch.bool, device=device)
            ok[:, :N] = fk
            # Return/standby slot: use the env action_mask directly (in_trip/standby/T_max already encoded)
            ok[:, N] = action_mask[:, k, N]
            # Serviceable tasks remain & nothing dispatched this round & this machine is at depot with a free field → standby forbidden
            # (in-trip return is never blocked; only the "standing by at depot" stall is)
            ok[:, N] = ok[:, N] & ~(
                any_serviceable & ~dispatched & ~in_trip[:, k] & fk.any(dim=1))
            lk = torch.where(ok, lk, torch.full_like(lk, float("-inf")))
            pk = torch.distributions.Categorical(logits=lk)
            if replay:
                sk = torch.as_tensor(action, dtype=torch.long,
                                     device=device)[:, k]
            elif greedy:
                sk = lk.argmax(dim=-1)
            else:
                sk = pk.sample()
            lp_sum = lp_sum + pk.log_prob(sk)
            ent_sum = ent_sum + pk.entropy()
            sel_out[:, k] = sk
            took = sk < N
            dispatched = dispatched | took
            if took.any():
                rows = took.nonzero(as_tuple=True)[0]
                taken[rows, sk[rows]] = True
        sub_valid = torch.ones(B, K, dtype=torch.bool, device=device)
        return sel_out, lp_sum, ent_sum, sub_valid

    def _static_step_field_pointer(self, bundle, x, action=None, greedy=False):
        """E0 static-version single step: field pointer (pick the field first, nearest machine executes).

        The pointer scores fields (machine-agnostic [B,N] field_logits) and picks
        the next field one at a time; each selected field is deterministically
        assigned to the nearest available (unassigned and feasible) machine (zero
        log-prob, not part of the learned policy). Selection continues until every
        machine has a field or no candidate field remains (the static version has
        no time cost, an at-depot machine with work available must depart, and the
        STOP slot does not participate); remaining unassigned machines: in trip →
        return to depot, at depot → standby no-op (both legal). Replay feeds back
        the same masks one by one from the stored field-selection sequence, so
        rollout/replay log-probs agree.

        action (replay) shape [B, K]: the field-selection sequence (slot q stores
        the index of the field selected q-th, unused slots are arbitrary and are
        only read at slots that had candidates).

        Returns a 5-tuple:
        (machine action [B,K], joint_lp [B], entropy [B], sub_valid, field-selection sequence).
        """
        field_logits = bundle["field_logits"]                   # [B, N]
        B, N = field_logits.shape
        device = field_logits.device
        # Support both input kinds: numpy obs (legacy env path) and GPU tensor obs (vec env):
        # tensors use .to() directly (zero-copy or device transfer), numpy goes through as_tensor
        def _obs_t(key, dtype):
            v = x[key]
            if isinstance(v, torch.Tensor):
                return v.to(device=device, dtype=dtype)
            return torch.as_tensor(np.asarray(v), dtype=dtype, device=device)

        mpos = _obs_t("machine_positions", field_logits.dtype)   # [B,K,2]
        fcoords = _obs_t("observations", field_logits.dtype)     # [B,N,2]
        feasible = _obs_t("feasible_pair_mask", torch.bool)      # [B,K,N]
        in_trip = _obs_t("in_trip", torch.bool)                  # [B,K]
        K = feasible.shape[1]
        dist = torch.norm(mpos.unsqueeze(2) - fcoords.unsqueeze(1),
                          dim=-1)                               # [B,K,N]
        machine_free = torch.ones(B, K, dtype=torch.bool, device=device)
        taken = torch.zeros(B, N, dtype=torch.bool, device=device)
        sel_out = torch.full((B, K), N, dtype=torch.long, device=device)
        fseq_out = torch.full((B, K), N, dtype=torch.long, device=device)
        lp_sum = torch.zeros(B, device=device)
        ent_sum = torch.zeros(B, device=device)
        replay = action is not None
        if replay:
            fseq = torch.as_tensor(action, dtype=torch.long, device=device)

        for q in range(K):
            cand = (feasible & machine_free.unsqueeze(2)).any(dim=1) & ~taken  # [B,N]
            row_mask = cand.any(dim=1)                           # [B]
            if not row_mask.any():
                break
            lk = torch.where(cand, field_logits,
                             torch.full_like(field_logits, float("-inf")))
            # Rows without candidates get a placeholder finite logit (their lp/ent contribution is
            # masked out by row_mask), avoiding an all -inf row that trips torch 1.13 Categorical validation
            if not row_mask.all():
                lk[~row_mask, 0] = 0.0
            pk = torch.distributions.Categorical(logits=lk)
            if replay:
                sk = fseq[:, q]                                  # [B]
            elif greedy:
                sk = lk.argmax(dim=-1)
            else:
                sk = pk.sample()
            sk_safe = torch.where(row_mask, sk, torch.zeros_like(sk))
            lp_sum = lp_sum + pk.log_prob(sk_safe) * row_mask.to(field_logits.dtype)
            ent_sum = ent_sum + pk.entropy() * row_mask.to(field_logits.dtype)
            # Nearest **feasible** available machine (deterministic assignment, zero log-prob)
            rows = row_mask.nonzero(as_tuple=True)[0]
            col = dist[rows, :, sk_safe[rows]]                   # [R,K]
            ok_mach = machine_free[rows] & feasible[rows, :, sk_safe[rows]]
            col = torch.where(ok_mach, col,
                              torch.full_like(col, float("inf")))
            kk = col.argmin(dim=1)                               # [R]
            sel_out[rows, kk] = sk_safe[rows]
            machine_free[rows, kk] = False
            taken[rows, sk_safe[rows]] = True
            fseq_out[rows, q] = sk_safe[rows]
        # Remaining unassigned machines: in trip → return to depot; at depot → standby no-op (both legal)
        sub_valid = torch.ones(B, K, dtype=torch.bool, device=device)
        # 7th position: field-selection sequence (replay contract)
        return sel_out, lp_sum, ent_sum, sub_valid, fseq_out


class stateWrapper:
    """Convert a dict of numpy arrays into an object exposing functions and data."""

    def __init__(self, states, device):
        self.device = device
        # Tensors already on the target device are reused with zero copies via .to(): the vec
        # pipeline rebuilds the wrapper at every decode step, so re-copying all ~30 obs tensors is
        # pure overhead; obs fields are fresh env-side clones and the decoder only reads, never
        # writes in place, so sharing is safe. numpy inputs still copy through torch.tensor, keeping the old semantics.
        self.states = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor)
                else torch.tensor(v, device=self.device))
            for k, v in states.items()
        }
        self._build_harvest_v2_observations()

    def _build_harvest_v2_observations(self):
        """Assemble the model input (non-look-ahead, no time windows).

        field_features [N, 6] (future/padding slots all zero):
            0,1: normalized coordinates; 2: normalized work hours; 3: hours already waited;
            4: PENDING flag; 5: visible flag.
        Raw machine-side state (machine_status/field_work_used/…) is kept in
        states for the AMFRA joint decoding pipeline to read.
        """
        s = self.states
        loc = s["observations"]                                  # [B, N, 2]
        wh = torch.clamp(s["work_hours"], min=1e-6)              # [B, 1]
        t_now = s["global_time"]                                 # [B, 1]
        visible = s["visible_field_mask"].to(torch.bool)         # [B, N]
        vis_f = visible.to(loc.dtype)

        waiting = torch.clamp(
            (t_now - s["arrival_times"]) / wh, min=0.0
        ) * vis_f
        pending = s["pending_mask"].to(torch.bool).to(loc.dtype) * vis_f

        field_features = torch.stack([
            loc[..., 0], loc[..., 1],
            s["demand"] * vis_f,
            waiting, pending, vis_f,
        ], dim=-1)                                               # [B, N, 6]

        s["observations"] = {
            "loc": loc,
            "depot": s["depot"].squeeze(-1) if s["depot"].dim() > 2 else s["depot"],
            "demand": s["demand"],
            "field_features": field_features,
        }
        # Cast boolean masks uniformly to torch.bool (gym MultiBinary becomes int8 after vectorized stacking)
        for mask_key in ("visible_field_mask", "pending_mask",
                         "completed_mask", "feasible_pair_mask",
                         "idle_machine_mask"):
            if mask_key in s:
                s[mask_key] = s[mask_key].to(torch.bool)
