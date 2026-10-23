"""AMFRA machine feature builder.

Builds the 13-dimensional feature vector per machine from the environment
state (no trainable parameters). Machines share the global clock
``global_time``; idle/busy state is expressed by ``machine_status``
(DEPOT_IDLE/FIELD_IDLE/BUSY/RETURNING), and the budget is the per-sortie field
work time ``H_k``; progress over the global planning cycle is given by
``cumulative_time / T_max``.

Feature order (Fm=13):
    0,1: normalized position x/y
    2: status_code / 3 (0=DEPOT_IDLE, 1=FIELD_IDLE, 2=BUSY, 3=RETURNING)
    3: depot_idle indicator (idle at the depot, available to start a sortie)
    4: field_idle indicator (idle in the field)
    5: w_k / H_k (share of the current sortie's field work time)
    6: sorties_used / 5, clipped to [0,1]
    7: normalized wait = (global_time - available_at) / work_hours (>= 0)
    8: harvest efficiency ρ_k / 10
    9: move cost c_k^d / 20 (CNY/km)
    10: activation cost c_k^f / 200 (CNY)
    11: compatibility ratio (share of servable fields)
    12: cumulative_time / work_hours (share of the global horizon T_max consumed;
        the T_max constraint itself is enforced by the environment mask, this
        feature lets the model anticipate it)

BUSY/RETURNING machines are still encoded (read-only context) but do not
appear in feasible pairs (masked out by feasible_pair_mask).
"""

from typing import NamedTuple

import torch
import torch.nn as nn


class MachineFeatures(NamedTuple):
    features: torch.Tensor    # [B, K, 13]
    idle_mask: torch.Tensor   # [B, K] bool, True = DEPOT_IDLE or FIELD_IDLE


class MachineFeatureBuilder(nn.Module):
    """Stateless, permutation-equivariant machine feature builder (no trainable
    parameters, no machine ID)."""

    def forward(
        self,
        machine_positions: torch.Tensor,          # [B, K, 2]
        machine_status: torch.Tensor,             # [B, K] 0=DEPOT_IDLE..3=RETURNING
        field_work_used: torch.Tensor,            # [B, K] w_k (h)
        max_field_work: torch.Tensor,             # [B, K] H_k (h)
        sorties_used: torch.Tensor,               # [B, K]
        available_at: torch.Tensor,               # [B, K]
        global_time: torch.Tensor,                # [B, 1]
        work_hours: torch.Tensor,                 # [B, 1] normalization reference (T_max)
        harvest_efficiencies: torch.Tensor,       # [B, K] ρ_k
        move_costs: torch.Tensor,                 # [B, K] c_k^d
        activation_costs: torch.Tensor,           # [B, K] c_k^f
        compatibility_mask: torch.Tensor,         # [B, K, N] bool
        cumulative_time: torch.Tensor,            # [B, K]
    ) -> MachineFeatures:
        safe_wh = torch.clamp(work_hours, min=1e-6)          # [B, 1]
        safe_h = torch.clamp(max_field_work, min=1e-6)       # [B, K]

        pos_x = machine_positions[..., 0]                    # [B, K]
        pos_y = machine_positions[..., 1]

        status = machine_status.to(machine_positions.dtype)
        status_norm = status / 3.0
        depot_idle = (status == 0).to(machine_positions.dtype)
        field_idle = (status == 1).to(machine_positions.dtype)
        idle_mask = (status == 0) | (status == 1)            # [B, K] bool

        work_used_ratio = torch.clamp(field_work_used / safe_h, 0.0, 1.0)
        sorties_norm = torch.clamp(
            sorties_used.to(machine_positions.dtype) / 5.0, 0.0, 1.0)
        # 0 for idle machines; clipped to >= 0 so it only reflects "already waited"
        wait = torch.clamp(
            (global_time - available_at) / safe_wh, 0.0, 2.0)

        # Heterogeneous attributes. Normalization ranges (10 / 20 / 200) cover the
        # training distribution (efficiency 3-8, move cost 10-18, activation 50-150)
        dt = machine_positions.dtype
        harvest_eff_norm = torch.clamp(harvest_efficiencies.to(dt) / 10.0, 0.0, 1.0)
        move_cost_norm = torch.clamp(move_costs.to(dt) / 20.0, 0.0, 1.0)
        activation_cost_norm = torch.clamp(activation_costs.to(dt) / 200.0, 0.0, 1.0)

        # Share of fields each machine can serve
        compat = compatibility_mask.to(dt)
        compat_ratio = compat.sum(dim=-1) / compat.size(-1)  # [B, K]

        horizon_used = torch.clamp(cumulative_time.to(dt) / safe_wh, 0.0, 1.0)

        features = torch.stack([
            pos_x,
            pos_y,
            status_norm,
            depot_idle,
            field_idle,
            work_used_ratio,
            sorties_norm,
            wait,
            harvest_eff_norm,
            move_cost_norm,
            activation_cost_norm,
            compat_ratio,
            horizon_used,
        ], dim=-1)  # [B, K, 13]

        return MachineFeatures(features=features, idle_mask=idle_mask)
