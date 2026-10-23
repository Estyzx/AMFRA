# -*- coding: utf-8 -*-
"""Batch-vectorised torch GPU version of the static **heterogeneous** multi-trip
harvester scheduling environment.

Bit-for-bit equivalent to ``HarvestStaticEnv`` (numpy) in
``envs/harvest_static_env.py`` under float64: observations, masks, rewards, costs and
done all match. Design points:

- State is float64 tensors of shape ``[B, ...]`` living on ``device``. Environment-side
  work is elementwise/indexing on small tensors, so the cost of float64 on GPU is
  negligible next to the Python/numpy per-environment loop, and it buys masks and costs
  that are bit-for-bit identical to the numpy float64 path.
- Model-side input stays float32 (converted by the trainer before feeding the model, as
  in the old path's ``prep``); the float64 observations returned by the environment are
  used only for internal settlement and mask decisions.
- ``step(rows, actions)`` updates only the environments in ``rows``, mirroring the numpy
  path's semantics of "step only the active environments"; the full-batch observation is
  returned every time (the state of non-stepped rows stays frozen).
- The environment's ``feasible_pair_mask`` / ``action_mask`` are the single source of
  truth consumed by the model-side ``_static_step``, so the masks must be bit-for-bit
  identical to the numpy version (same formula, same epsilon).

Contract (exactly the numpy version's, see the ``harvest_static_env.py`` docstring):
- action ``j∈[0,N-1]`` = go to field j, ``j=N`` = return to depot (in trip) or standby
  (at depot);
- within one step the machines are processed in order and fields are taken without
  replacement (a field already taken by an earlier machine is infeasible for later ones);
- an illegal action costs -10 and is ignored;
- once all fields are done, the return cost of in-trip machines is settled automatically
  and their trips are reset;
- permanently unserviceable fields (all horizons exhausted and nothing in trip) →
  mark-everything-completed fallback;
- **heterogeneous**: every machine has its own speed, efficiency, move cost, activation
  cost and compatibility;
- **three-level constraints**: field-level area, trip-level H_k (single trip), machine-level
  T_max (global); every machine—field pair must satisfy both ``work ≤ H_k − trip_used``
  and ``cumulative + travel + work ≤ T_max`` (the same double constraint applies to the
  first field from the depot and to appends while in trip, bit-for-bit identical to the
  numpy version).
"""

import torch


class HarvestStaticEnvVec:
    """B static heterogeneous harvester scheduling environments, all state held as GPU/CPU tensors, stepped batch-vectorised."""

    def __init__(
        self,
        num_envs,
        device,
        max_tasks=50,
        n_machines=15,
        travel_speed=25.0,        # default homogeneous speed (overridden by machine_attrs when heterogeneous)
        harvest_speed=6.0,        # default homogeneous efficiency (overridden by machine_attrs when heterogeneous)
        move_cost_coef=1.0,       # default homogeneous unit-distance move cost
        map_size=50.0,
        start_cost=68.0,
        machine_activation_cost=0.0,
        T_max=24.0,               # global scheduling horizon cap (h)
        trip_max_field_work=8.0,  # max field-work duration per single trip (h)
        seed=None,
    ):
        self.B = int(num_envs)
        self.device = torch.device(device)
        self.max_tasks = int(max_tasks)
        self.K = int(n_machines)
        self.N = self.max_tasks
        # default homogeneous parameters (the heterogeneous load_instances overrides them)
        self._default_travel_speed = float(travel_speed)
        self._default_harvest_speed = float(harvest_speed)
        self._default_move_cost_coef = float(move_cost_coef)
        self.map_size = float(map_size)
        self.start_cost = float(start_cost)
        self.machine_activation_cost_default = float(machine_activation_cost)
        self.T_max = float(T_max)
        self.trip_max_field_work = float(trip_max_field_work)
        self.seed = seed
        self._alloc_state()
        self.done = torch.zeros(self.B, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    # State allocation
    # ------------------------------------------------------------------
    def _alloc_state(self):
        dev = self.device
        B, K, N = self.B, self.K, self.N
        # fields
        self.coords = torch.zeros(B, N, 2, dtype=torch.float64, device=dev)
        self.areas = torch.zeros(B, N, dtype=torch.float64, device=dev)
        self.completed = torch.zeros(B, N, dtype=torch.bool, device=dev)
        self.served = torch.zeros(B, N, dtype=torch.bool, device=dev)
        # machine positions and state
        self.positions = torch.zeros(B, K, 2, dtype=torch.float64, device=dev)
        self.in_trip = torch.zeros(B, K, dtype=torch.bool, device=dev)
        self.trip_used = torch.zeros(B, K, dtype=torch.float64, device=dev)
        self.sorties = torch.zeros(B, K, dtype=torch.float64, device=dev)
        self.cumulative_time = torch.zeros(B, K, dtype=torch.float64, device=dev)
        self.machine_activated = torch.zeros(B, K, dtype=torch.bool, device=dev)
        self.total_cost = torch.zeros(B, dtype=torch.float64, device=dev)
        self.depot = torch.zeros(B, 2, dtype=torch.float64, device=dev)
        # trip-level constraints
        self.max_field_work = torch.full((B, K), self.trip_max_field_work,
                                         dtype=torch.float64, device=dev)
        # --- heterogeneous attributes [B, K] or [B, K, N] ---
        self.machine_travel_speeds = torch.full((B, K), self._default_travel_speed,
                                                dtype=torch.float64, device=dev)
        self.machine_harvest_efficiencies = torch.full(
            (B, K), self._default_harvest_speed, dtype=torch.float64, device=dev)
        self.machine_move_costs = torch.full((B, K), self._default_move_cost_coef,
                                             dtype=torch.float64, device=dev)
        self.machine_activation_costs = torch.full(
            (B, K), self.machine_activation_cost_default, dtype=torch.float64, device=dev)
        self.compatibility = torch.ones(B, K, N, dtype=torch.bool, device=dev)
        # service time matrix [B, K, N]: s_ik = area_i / rho_k
        self.service_times_matrix = torch.zeros(B, K, N, dtype=torch.float64, device=dev)
        self.done = torch.zeros(B, dtype=torch.bool, device=dev)

    # ------------------------------------------------------------------
    # Instance loading
    # ------------------------------------------------------------------
    def load_instances(self, tasks, depot, max_field_work, areas,
                       machine_attrs=None):
        """Load a batch of static heterogeneous instances.

        tasks: [B, N, 2] float64 (x, y), coordinates already normalised to [0,1];
        depot: [B, 2] float64;
        max_field_work: [B, K] float64, max field-work duration per single trip H_k;
        areas: [B, N] float64 field areas (mu);
        machine_attrs: dict, optional, containing:
            - travel_speeds: [B, K] (km/h)
            - harvest_efficiencies: [B, K] (mu/h)
            - move_costs: [B, K] (CNY/km)
            - activation_costs: [B, K] (CNY)
            - compatibility: [B, K, N] int8/bool (0/1)
        """
        dev = self.device
        tasks = torch.as_tensor(tasks, dtype=torch.float64, device=dev)
        depot = torch.as_tensor(depot, dtype=torch.float64, device=dev)
        max_field_work = torch.as_tensor(
            max_field_work, dtype=torch.float64, device=dev)
        areas = torch.as_tensor(areas, dtype=torch.float64, device=dev)
        B, K, N = self.B, self.K, self.N

        assert tasks.shape == (B, N, 2), tasks.shape
        assert depot.shape == (B, 2), depot.shape
        assert max_field_work.shape == (B, K), max_field_work.shape
        assert areas.shape == (B, N), areas.shape

        self.coords = tasks.clone()
        self.depot = depot.clone()
        self.max_field_work = max_field_work.clone()
        self.areas = areas.clone()

        # heterogeneous machine attributes
        if machine_attrs is not None:
            for key, attr_name in [
                ("travel_speeds", "machine_travel_speeds"),
                ("harvest_efficiencies", "machine_harvest_efficiencies"),
                ("move_costs", "machine_move_costs"),
                ("activation_costs", "machine_activation_costs"),
            ]:
                val = torch.as_tensor(
                    machine_attrs[key], dtype=torch.float64, device=dev)
                assert val.shape == (B, K), f"{key} shape {val.shape} != ({B},{K})"
                setattr(self, attr_name, val.clone())

            if "compatibility" in machine_attrs:
                compat = torch.as_tensor(
                    machine_attrs["compatibility"], dtype=torch.bool, device=dev)
                assert compat.shape == (B, K, N), f"compatibility shape {compat.shape}"
                self.compatibility = compat.clone()
            else:
                self.compatibility = torch.ones(B, K, N, dtype=torch.bool, device=dev)
        else:
            # default homogeneous values
            self.machine_travel_speeds = torch.full(
                (B, K), self._default_travel_speed, dtype=torch.float64, device=dev)
            self.machine_harvest_efficiencies = torch.full(
                (B, K), self._default_harvest_speed, dtype=torch.float64, device=dev)
            self.machine_move_costs = torch.full(
                (B, K), self._default_move_cost_coef, dtype=torch.float64, device=dev)
            self.machine_activation_costs = torch.full(
                (B, K), self.machine_activation_cost_default,
                dtype=torch.float64, device=dev)
            self.compatibility = torch.ones(B, K, N, dtype=torch.bool, device=dev)

        # build the service time matrix [B, K, N]: s_ik = area_i / rho_k
        # areas: [B, N] -> [B, 1, N]; efficiencies: [B, K] -> [B, K, 1]
        rho = self.machine_harvest_efficiencies.clamp(min=1e-6)
        self.service_times_matrix = (
            self.areas.unsqueeze(1) / rho.unsqueeze(2))  # [B, K, N]

        self._reset_rollout_state()

    def _reset_rollout_state(self):
        B, K, N = self.B, self.K, self.N
        dev = self.device
        self.completed = torch.zeros(B, N, dtype=torch.bool, device=dev)
        self.served = torch.zeros(B, N, dtype=torch.bool, device=dev)
        self.positions = self.depot.unsqueeze(1).expand(B, K, 2).clone()
        self.in_trip = torch.zeros(B, K, dtype=torch.bool, device=dev)
        self.trip_used = torch.zeros(B, K, dtype=torch.float64, device=dev)
        self.sorties = torch.zeros(B, K, dtype=torch.float64, device=dev)
        self.cumulative_time = torch.zeros(B, K, dtype=torch.float64, device=dev)
        self.machine_activated = torch.zeros(B, K, dtype=torch.bool, device=dev)
        self.total_cost = torch.zeros(B, dtype=torch.float64, device=dev)
        self.done = torch.zeros(B, dtype=torch.bool, device=dev)

    def reset(self):
        self._reset_rollout_state()
        return self.obs_dict()

    # ------------------------------------------------------------------
    # Observations (bit-for-bit identical to numpy `_obs`)
    # ------------------------------------------------------------------
    def obs_dict(self):
        B, K, N = self.B, self.K, self.N
        dev = self.device

        # mask construction (bit-for-bit identical to the numpy version)
        mask = torch.zeros(B, K, N + 1, dtype=torch.bool, device=dev)

        # remaining single-trip budget [B, K]
        trip_rem = self.max_field_work - self.trip_used
        # remaining global horizon [B, K]
        horizon_rem = self.T_max - self.cumulative_time

        # in-trip machines: may go to a compatible, budget-feasible field, or return to the depot
        in_trip_expanded = self.in_trip.unsqueeze(2)  # [B, K, 1]
        # service time matrix [B, K, N]
        st_mat = self.service_times_matrix
        # travel time [B, K, N]: d(pos_k, coord_i) * map_size / v_k (shared by both branches)
        diff = self.positions.unsqueeze(2) - self.coords.unsqueeze(1)  # [B, K, N, 2]
        dist = torch.norm(diff, dim=3) * self.map_size  # [B, K, N]
        travel_t = dist / self.machine_travel_speeds.unsqueeze(2).clamp(min=1e-6)
        work_t = st_mat  # [B, K, N]
        total_t = travel_t + work_t  # [B, K, N]
        # pair feasibility: compatible ∧ not completed ∧ trip-level (work ≤ remaining H_k)
        # ∧ machine-level (travel + work ≤ remaining horizon) — one and the same double
        # constraint at the depot and in trip (at the depot trip_used=0, so trip_rem is the full H_k budget of a new trip)
        feas_pair = ((~self.completed.unsqueeze(1)) & self.compatibility
                     & (work_t <= trip_rem.unsqueeze(2) + 1e-9)
                     & (total_t <= horizon_rem.unsqueeze(2) + 1e-9))  # [B, K, N]
        mask[:, :, :N] = mask[:, :, :N] | (in_trip_expanded & feas_pair)
        # return slot: in-trip machines may return to the depot
        mask[:, :, N] = mask[:, :, N] | self.in_trip

        # machines at the depot: check the horizon is large enough
        at_base = ~self.in_trip  # [B, K]
        # horizon exhausted: standby only
        horizon_exhausted = horizon_rem < 1e-9  # [B, K]
        # standby slot: always selectable for a machine at the depot
        mask[:, :, N] = mask[:, :, N] | at_base  # standby is always selectable for a machine at the depot

        # when the horizon suffices: feasible fields = double-constraint pair feasibility (same formula as in trip)
        feas_at_base = feas_pair
        mask[:, :, :N] = mask[:, :, :N] | (at_base.unsqueeze(2) & ~horizon_exhausted.unsqueeze(2) & feas_at_base)

        mean_mfw = self.max_field_work.mean(dim=1, keepdim=True).clamp(min=1e-6)
        mean_eff = self.machine_harvest_efficiencies.mean(dim=1, keepdim=True).clamp(min=1e-6)
        ones_f = torch.ones(B, K, dtype=torch.float64, device=dev)
        zeros_f = torch.zeros(B, K, dtype=torch.float64, device=dev)

        return {
            "observations": self.coords.clone(),                 # [B,N,2]
            "demand": (self.areas / (mean_eff * mean_mfw)).clone(),   # [B,N]
            "service_times": st_mat.clone(),                     # [B,K,N] heterogeneous service time matrix
            "completed_mask": self.completed.clone(),            # [B,N] bool
            "machine_positions": self.positions.clone(),         # [B,K,2]
            "machine_status": torch.where(self.in_trip, ones_f, zeros_f),  # [B,K]
            "field_work_used": self.trip_used.clone(),           # [B,K]
            "sorties_used": self.sorties.clone(),                # [B,K]
            "available_at": self.cumulative_time.clone(),        # [B,K]
            "global_time": self.cumulative_time.min(dim=1, keepdim=True)[0],  # [B,1]
            "work_hours": torch.full((B, 1), self.T_max, dtype=torch.float64, device=dev),
            "visible_field_mask": (~self.completed).to(torch.int8),
            "pending_mask": (~self.completed).to(torch.int8),
            "task_status": torch.where(
                self.completed,
                torch.full_like(self.areas, 3.0),
                torch.full_like(self.areas, 1.0)),
            "idle_machine_mask": torch.ones(B, K, dtype=torch.int8, device=dev),
            "remaining_field_work": (self.max_field_work - self.trip_used).clone(),
            "feasible_pair_mask": mask[:, :, :N].to(torch.int8),  # [B,K,N] int8
            "arrival_times": torch.zeros(B, N, dtype=torch.float64, device=dev),
            "travel_speed": self.machine_travel_speeds.clone(),  # [B,K] heterogeneous
            "in_trip": self.in_trip.clone(),                     # [B,K] bool
            "trip_used": self.trip_used.clone(),                 # [B,K]
            "max_field_work": self.max_field_work.clone(),       # [B,K]
            "action_mask": mask.clone(),                         # [B,K,N+1] bool
            "depot": self.depot.clone(),                         # [B,2]
            # --- new: heterogeneous + T_max ---
            "cumulative_time": self.cumulative_time.clone(),     # [B,K]
            "remaining_horizon": horizon_rem.clone(),            # [B,K]
            "machine_activated": self.machine_activated.to(torch.int8),  # [B,K]
            "machine_travel_speeds": self.machine_travel_speeds.clone(),  # [B,K]
            "machine_harvest_efficiencies": self.machine_harvest_efficiencies.clone(),
            "machine_move_costs": self.machine_move_costs.clone(),  # [B,K]
            "machine_activation_costs": self.machine_activation_costs.clone(),
            "compatibility_mask": self.compatibility.to(torch.int8),  # [B,K,N]
            "areas": self.areas.clone(),                         # [B,N]
        }

    # ------------------------------------------------------------------
    # Step (bit-for-bit identical to numpy `step`; updates only the rows given)
    # ------------------------------------------------------------------
    def step(self, rows, actions):
        """rows: LongTensor [R] (indices of the environments to step); actions: [R,K] long.

        Returns (full obs dict, rewards [R] float64, done [R] bool,
              {"total_cost": [R], "cumulative_time": [R,K], "machine_activated": [R,K]}).
        """
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        R = rows.numel()
        K, N = self.K, self.N
        dev = self.device
        assert actions.shape == (R, K), actions.shape

        reward = torch.zeros(R, dtype=torch.float64, device=dev)
        mask = self.obs_dict()["action_mask"]                    # [B,K,N+1]
        taken = torch.zeros(self.B, N, dtype=torch.bool, device=dev)
        # forced dispatch (serviceable fields remain and nothing has been dispatched this round)
        any_serviceable = mask[:, :, :N].any(dim=1).any(dim=1)   # [B]
        dispatched = torch.zeros(R, dtype=torch.bool, device=dev)

        j = actions                                               # [R,K]
        jc = j.clamp(max=N - 1)                                   # safe indexing
        is_field = (j < N)                                        # [R,K]

        for k in range(K):
            jk = j[:, k]
            jck = jc[:, k]
            ifk = is_field[:, k]
            m_valid = mask[rows, k].gather(1, jk.unsqueeze(1)).squeeze(1)  # [R]
            # forced dispatch: this machine is at the depot, a free feasible field is left and
            # nothing was dispatched this round -> standby is illegal (an in-trip machine's
            # return is never blocked, only the "stand by at depot" stall is, as in the numpy version)
            fk_avail = mask[rows, k, :N] & (~taken[rows])                 # [R, N]
            force = (any_serviceable[rows] & (~dispatched)
                     & (~self.in_trip[rows, k]) & fk_avail.any(dim=1))    # [R]
            field_valid = m_valid & ifk & (~taken[rows, jck])              # [R]
            ret_valid = m_valid & (~ifk) & (~force)                        # [R]
            valid = field_valid | ret_valid

            # illegal action penalty (numpy: reward -= 10.0)
            reward = reward - 10.0 * (~valid).to(reward.dtype)

            # machine-specific attributes
            v_k = self.machine_travel_speeds[rows, k]            # [R]
            c_k_d = self.machine_move_costs[rows, k]             # [R]

            # ---- return to depot (legal return & in trip): settle the return distance + time, reset the trip ----
            rr = ret_valid & self.in_trip[rows, k]
            d_ret = (torch.norm(self.positions[rows, k] - self.depot[rows], dim=1)
                     * self.map_size)                             # [R]
            travel_t_ret = d_ret / v_k.clamp(min=1e-6)            # [R]
            reward = reward - d_ret * c_k_d * rr.to(reward.dtype)
            # cumulative time (return travel time)
            self.cumulative_time[rows, k] = torch.where(
                rr, self.cumulative_time[rows, k] + travel_t_ret,
                self.cumulative_time[rows, k])
            self.positions[rows, k] = torch.where(
                rr.unsqueeze(1), self.depot[rows], self.positions[rows, k])
            self.in_trip[rows, k] = torch.where(
                rr, torch.zeros_like(self.in_trip[rows, k]), self.in_trip[rows, k])
            self.trip_used[rows, k] = torch.where(
                rr, torch.zeros_like(self.trip_used[rows, k]), self.trip_used[rows, k])

            # ---- go to a field ----
            ff = field_valid
            dispatched = dispatched | ff
            d_fld = (torch.norm(self.positions[rows, k] - self.coords[rows, jck], dim=1)
                     * self.map_size)                             # [R]
            travel_t_fld = d_fld / v_k.clamp(min=1e-6)            # [R]
            work_t = self.service_times_matrix[rows, k, jck]      # [R]

            # move cost
            reward = reward - d_fld * c_k_d * ff.to(reward.dtype)

            # activation cost (first sortie)
            first_activation = ff & (~self.machine_activated[rows, k])
            c_k_f = self.machine_activation_costs[rows, k]        # [R]
            reward = reward - c_k_f * first_activation.to(reward.dtype)
            self.machine_activated[rows, k] = self.machine_activated[rows, k] | ff

            # trip start cost
            start = ff & (~self.in_trip[rows, k])
            reward = reward - self.start_cost * start.to(reward.dtype)
            self.sorties[rows, k] = torch.where(
                start, self.sorties[rows, k] + 1.0, self.sorties[rows, k])

            # update state
            self.cumulative_time[rows, k] = torch.where(
                ff, self.cumulative_time[rows, k] + travel_t_fld + work_t,
                self.cumulative_time[rows, k])
            self.trip_used[rows, k] = torch.where(
                ff, self.trip_used[rows, k] + work_t, self.trip_used[rows, k])
            self.positions[rows, k] = torch.where(
                ff.unsqueeze(1), self.coords[rows, jck], self.positions[rows, k])
            self.in_trip[rows, k] = torch.where(
                ff, torch.ones_like(self.in_trip[rows, k]), self.in_trip[rows, k])
            self.completed[rows, jck] = self.completed[rows, jck] | ff
            self.served[rows, jck] = self.served[rows, jck] | ff
            taken[rows, jck] = taken[rows, jck] | ff

        # ---- once all fields are done, auto-settle the return cost of the in-trip machines ----
        completed_all = self.completed[rows].all(dim=1)            # [R]
        in_trip_any = self.in_trip[rows].any(dim=1)                # [R]
        auto = completed_all & in_trip_any
        for k in range(K):
            rr2 = auto & self.in_trip[rows, k]
            v_k = self.machine_travel_speeds[rows, k]
            c_k_d = self.machine_move_costs[rows, k]
            d_ret = (torch.norm(self.positions[rows, k] - self.depot[rows], dim=1)
                     * self.map_size)
            travel_t_ret = d_ret / v_k.clamp(min=1e-6)
            reward = reward - d_ret * c_k_d * rr2.to(reward.dtype)
            self.cumulative_time[rows, k] = torch.where(
                rr2, self.cumulative_time[rows, k] + travel_t_ret,
                self.cumulative_time[rows, k])
            self.positions[rows, k] = torch.where(
                rr2.unsqueeze(1), self.depot[rows], self.positions[rows, k])
            self.in_trip[rows, k] = torch.where(
                rr2, torch.zeros_like(self.in_trip[rows, k]), self.in_trip[rows, k])
            self.trip_used[rows, k] = torch.where(
                rr2, torch.zeros_like(self.trip_used[rows, k]), self.trip_used[rows, k])

        # ---- permanently unserviceable fallback: mark everything completed (served stays unset for those fields -> counted as unfinished) ----
        dead = (~completed_all) & (~in_trip_any)
        if bool(dead.any().item()):
            # recompute the masks from the post-action state (after a machine returns its trip
            # budget is reset, so feasibility may change)
            fresh = self.obs_dict()["feasible_pair_mask"]        # [B,K,N]
            any_feas = fresh[rows].any(dim=1).any(dim=1)          # [R]
            mark = dead & (~any_feas)
            self.completed[rows] = self.completed[rows] | mark.unsqueeze(1)

        # ---- settlement ----
        self.total_cost[rows] = self.total_cost[rows] + (-reward)
        done = (self.completed[rows].all(dim=1)
                & (~self.in_trip[rows].any(dim=1)))                # [R]
        self.done[rows] = done
        return self.obs_dict(), reward, done, {
            "total_cost": self.total_cost[rows].clone(),
            "cumulative_time": self.cumulative_time[rows].clone(),
            "machine_activated": self.machine_activated[rows].clone(),
            "served_all": self.served[rows].all(dim=1),
            "unfinished": (~self.served[rows]).sum(dim=1),
        }
