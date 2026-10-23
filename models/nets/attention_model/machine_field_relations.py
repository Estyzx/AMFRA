"""AMFRA machine-field relation feature builder.

Computes the 12-dimensional pairwise machine-field relation features (no
trainable parameters) feeding the AMFRA relation bias / gate.

- No time windows: relation features are 12-dimensional;
- the hard constraint is "field work budget feasible": ``w_k + p_i <= H_k``;
- ``pair_mask`` directly takes the environment's ``feasible_pair_mask``, so the
  model mask stays strictly consistent with the environment legality rules
  rather than re-deriving them;
- load features keep the field work budget ratio semantics ``w_k / H_k``;
- heterogeneous support: ``service_times`` is [B, K, N] (machine-dependent work
  time), ``travel_speed`` / ``move_costs`` are [B, K] (per-machine attributes).

Distance semantics match the environment:
``distance_km = ||machine_pos - field_pos||_2 * map_size``.
"""

import math
from typing import NamedTuple

import torch
import torch.nn as nn


class MachineFieldRelations(NamedTuple):
    features: torch.Tensor    # [B, K, N, 12]
    pair_mask: torch.Tensor   # [B, K, N] bool = feasible_pair_mask ∧ idle ∧ visible
    distance: torch.Tensor    # [B, K, N] km


class MachineFieldRelationBuilder(nn.Module):
    """Stateless machine-field relation feature builder.

    ``pair_mask`` passes through the environment's ``feasible_pair_mask``
    without re-deriving it, to avoid drift between model and environment rules.
    """

    #: Move-cost normalization reference (CNY/km), shared with machine feature dimension 9
    MOVE_COST_REF = 20.0

    def __init__(self, map_size: float = 10.0):
        super().__init__()
        assert map_size > 0
        self.map_size = map_size
        self.max_map_distance = map_size * math.sqrt(2.0)

    def forward(
        self,
        machine_positions: torch.Tensor,     # [B, K, 2]
        field_positions: torch.Tensor,       # [B, N, 2]
        field_work_used: torch.Tensor,       # [B, K] w_k (h)
        max_field_work: torch.Tensor,        # [B, K] H_k (h)
        idle_mask: torch.Tensor,             # [B, K] bool
        service_times: torch.Tensor,         # [B, K, N] p_ik (h)
        feasible_pair_mask: torch.Tensor,    # [B, K, N] bool (environment E_n)
        valid_field_mask: torch.Tensor,      # [B, N] bool
        global_time: torch.Tensor,           # [B, 1]
        work_hours: torch.Tensor,            # [B, 1]
        travel_speed: torch.Tensor,          # [B, K] (or [B, 1], broadcast)
        move_costs: torch.Tensor,            # [B, K] (or [B, 1], broadcast)
    ) -> MachineFieldRelations:
        B, K, _ = machine_positions.shape
        N = field_positions.shape[1]

        st = service_times
        assert st.shape == (B, K, N)
        ts_full = travel_speed.expand(B, K)
        mc_full = move_costs.expand(B, K)
        assert (ts_full > 0).all(), "travel_speed must be positive"

        mp = machine_positions.unsqueeze(2)       # [B, K, 1, 2]
        fp = field_positions.unsqueeze(1)         # [B, 1, N, 2]
        w = field_work_used.unsqueeze(2)          # [B, K, 1]
        h = torch.clamp(max_field_work, min=1e-6).unsqueeze(2)  # [B, K, 1]
        wh = torch.clamp(work_hours, min=1e-6).unsqueeze(2)     # [B, 1, 1]
        ts = ts_full.unsqueeze(2)                 # [B, K, 1]
        t_now = global_time.unsqueeze(2)          # [B, 1, 1]

        valid_pair_base = (
            idle_mask.unsqueeze(2) & valid_field_mask.unsqueeze(1)
        )  # [B, K, N]

        # Distance (km) / travel time (h)
        diff = mp - fp                            # [B, K, N, 2]
        distance = torch.norm(diff, dim=-1) * self.map_size
        travel_time = distance / ts

        # Budget margin: H_k - w_k - p_i
        budget_margin = h - w - st

        # Load semantics (budget ratio)
        current_load = torch.clamp(w / h, 0.0, 1.0)             # [B,K,1]
        post_assignment_load = (w + st) / h                     # [B,K,N]

        # Mean current load of idle machines (only idle ones participate)
        idle_f = idle_mask.to(current_load.dtype)               # [B, K]
        idle_count = idle_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B,1]
        mean_current_load = (
            current_load.squeeze(2) * idle_f
        ).sum(dim=1, keepdim=True) / idle_count                 # [B, 1]

        load_deviation = post_assignment_load - mean_current_load.unsqueeze(2)
        abs_load_deviation = torch.abs(load_deviation)

        # Distance competitive advantage (only meaningful for visible tasks)
        safe_distance = torch.where(
            valid_pair_base, distance, torch.full_like(distance, float("inf"))
        )
        active_distance = torch.where(
            idle_mask.unsqueeze(2).expand_as(distance), distance,
            torch.zeros_like(distance)
        )
        sum_distance = active_distance.sum(dim=1, keepdim=True)   # [B,1,N]
        mean_distance = sum_distance / idle_count.unsqueeze(2)
        distance_adv_mean = mean_distance - distance

        # Nearest competitor distance (excluding self): global min + second min to handle ties
        global_min = safe_distance.min(dim=1, keepdim=True).values
        is_min = (safe_distance == global_min) & idle_mask.unsqueeze(2).expand_as(distance)
        min_count = is_min.sum(dim=1, keepdim=True)
        dist_without_min = torch.where(
            is_min, torch.full_like(distance, float("inf")), safe_distance
        )
        second_min = dist_without_min.min(dim=1, keepdim=True).values
        self_is_unique_min = is_min & (min_count == 1)
        min_other_per_field = torch.where(
            self_is_unique_min, second_min.expand_as(distance),
            global_min.expand_as(distance)
        )
        has_competitor = torch.isfinite(min_other_per_field)
        distance_adv_min = torch.where(
            has_competitor, min_other_per_field - distance,
            torch.zeros_like(distance)
        )

        # Budget consumption ratio: p_i / (H_k - w_k)
        remaining = torch.clamp(h - w, min=1e-6)
        budget_consumption_ratio = torch.clamp(st / remaining, 0.0, 2.0)

        # Normalized features [B, K, N, 12]
        norm_distance = torch.clamp(distance / self.max_map_distance, 0.0, 1.0)
        norm_travel_time = travel_time / wh
        norm_service_time = st / wh  # st is [B,K,N], wh is [B,1,1]
        norm_budget_margin = torch.clamp(budget_margin / wh, -2.0, 2.0)
        feat_current_load = current_load.expand_as(distance)
        norm_dist_adv_mean = torch.clamp(
            distance_adv_mean / self.max_map_distance, -1.0, 1.0)
        norm_dist_adv_min = torch.clamp(
            distance_adv_min / self.max_map_distance, -1.0, 1.0)
        # Pairing move cost: c_k × d_ki / (cost reference × maximum map distance)
        norm_move_cost = torch.clamp(
            mc_full.unsqueeze(2) * distance
            / (self.MOVE_COST_REF * self.max_map_distance), 0.0, 1.0)

        features = torch.stack([
            norm_distance,
            norm_travel_time,
            norm_service_time,
            norm_budget_margin,
            feat_current_load,
            post_assignment_load,
            load_deviation,
            abs_load_deviation,
            norm_dist_adv_mean,
            norm_dist_adv_min,
            budget_consumption_ratio,
            norm_move_cost,
        ], dim=-1)  # [B, K, N, 12]

        # Zero out invalid positions (including FUTURE task slots)
        features = torch.where(
            valid_pair_base.unsqueeze(-1), features, torch.zeros_like(features))

        pair_mask = feasible_pair_mask & valid_pair_base
        return MachineFieldRelations(features=features, pair_mask=pair_mask,
                                     distance=distance)
