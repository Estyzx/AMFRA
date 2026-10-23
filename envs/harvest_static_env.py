# -*- coding: utf-8 -*-
"""Static **heterogeneous** multi-trip harvester scheduling environment.

Problem definition
------------------
N farmland tasks are all known in advance; K **heterogeneous** harvesters start
from the dispatch centre. Each machine k has its own attributes:

- v_k: travel speed (km/h)
- ρ_k: field work efficiency (mu/h)
- c_k^d: unit-distance transfer cost (CNY/km)
- c_k^f: machine activation fixed cost (CNY)
- e_ik ∈ {0,1}: whether farmland i can be served (compatibility)

Three-level resource constraints:

1. **Field level**: area_i — farmland area (mu)
2. **Trip level**: H_k — max continuous field-work hours per single trip (h), reset after returning to the depot
3. **Machine level**: T_max — cap on the whole scheduling horizon (h), never reset

Field-work time of machine k on farmland i: s_ik = area_i / ρ_k
Travel time of machine k from position a to b: t_k = d(a,b) / v_k
Cumulative time of machine k: cumulative_time[k] = Σ(travel + work), never reset

Objective
---------
min J = Σ transfer costs + Σ trip start costs + Σ machine activation costs

- Transfer cost: c_k^d × distance (settled on every move)
- Trip start cost: c_start (settled whenever a new trip is started)
- Machine activation cost: c_k^f (settled once, on machine k's first dispatch)

Decision interface
------------------
- Action space MultiDiscrete([N+1] × K): j∈[0,N-1] go to farmland j; j=N return to depot / standby
- Mask [K, N+1] (for any machine to go to any farmland, the trip-level and the machine-level
  constraint must hold at the same time: work ≤ H_k − trip_used and
  cumulative + travel + work ≤ T_max):
  - in-trip machine: compatible farmlands satisfying both constraints, or return to depot
  - at-depot machine: compatible farmlands satisfying both constraints (the first field must fit
    wholly inside the fresh budget H_k), or standby
  - at-depot machine with cumulative_time[k] ≥ T_max: standby only
- Termination: all farmlands completed and all machines at the depot (farmlands that are
  permanently unserviceable are dropped by the fallback, served is not set →
  judged as unfinished)

Usage: compatible with gym 0.23 (reset()→obs; step(action)→(obs, reward, done, info)).
"""

import numpy as np
import gym
from gym import spaces


class HarvestStaticEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        max_tasks_per_day=50,
        n_machines=3,
        travel_speed=25.0,        # default homogeneous speed (overridden by machine_attrs when heterogeneous)
        map_size=10.0,
        start_cost=5.0,           # trip start cost per trip
        harvest_speed=6.0,        # default homogeneous efficiency (overridden by machine_attrs when heterogeneous)
        move_cost_coef=1.0,       # default homogeneous unit-distance cost
        T_max=24.0,               # cap on the global scheduling horizon (h)
        trip_max_field_work=8.0,  # max field-work hours per single trip (h)
        machine_activation_cost=0.0,  # default machine activation cost
        seed=None,
    ):
        super().__init__()
        self.max_tasks = int(max_tasks_per_day)
        self.K = int(n_machines)
        self.map_size = float(map_size)
        self.start_cost = float(start_cost)
        self.T_max = float(T_max)
        self.trip_max_field_work = float(trip_max_field_work)
        self.machine_activation_cost_default = float(machine_activation_cost)
        # Default homogeneous parameters (overridden by the heterogeneous set_instance)
        self._default_travel_speed = float(travel_speed)
        self._default_harvest_speed = float(harvest_speed)
        self._default_move_cost_coef = float(move_cost_coef)
        self.rng = np.random.default_rng(seed)

        # obs space (includes the heterogeneous fields)
        self.observation_space = spaces.Dict({
            "observations": spaces.Box(0, 1, shape=(self.max_tasks, 2), dtype=np.float64),
            "demand": spaces.Box(0, 1, shape=(self.max_tasks,), dtype=np.float64),
            "service_times": spaces.Box(0, 100, shape=(self.K, self.max_tasks), dtype=np.float64),
            "completed_mask": spaces.MultiBinary(self.max_tasks),
            "machine_positions": spaces.Box(0, 1, shape=(self.K, 2), dtype=np.float64),
            "machine_status": spaces.Box(0, 3, shape=(self.K,), dtype=np.float64),
            "field_work_used": spaces.Box(0, 100, shape=(self.K,), dtype=np.float64),
            "sorties_used": spaces.Box(0, 100, shape=(self.K,), dtype=np.float64),
            "available_at": spaces.Box(0, 1000, shape=(self.K,), dtype=np.float64),
            "global_time": spaces.Box(0, 1000, shape=(1,), dtype=np.float64),
            "work_hours": spaces.Box(0, 1000, shape=(1,), dtype=np.float64),
            "visible_field_mask": spaces.MultiBinary(self.max_tasks),
            "pending_mask": spaces.MultiBinary(self.max_tasks),
            "task_status": spaces.Box(0, 4, shape=(self.max_tasks,), dtype=np.float64),
            "idle_machine_mask": spaces.MultiBinary(self.K),
            "remaining_field_work": spaces.Box(0, 100, shape=(self.K,), dtype=np.float64),
            "feasible_pair_mask": spaces.MultiBinary([self.K, self.max_tasks]),
            "arrival_times": spaces.Box(0, 1000, shape=(self.max_tasks,), dtype=np.float64),
            "travel_speed": spaces.Box(0, 1000, shape=(self.K,), dtype=np.float64),
            "in_trip": spaces.MultiBinary(self.K),
            "trip_used": spaces.Box(0, 100, shape=(self.K,), dtype=np.float64),
            "max_field_work": spaces.Box(0, 100, shape=(self.K,), dtype=np.float64),
            "action_mask": spaces.MultiBinary([self.K, self.max_tasks + 1]),
            "depot": spaces.Box(0, 1, shape=(2,), dtype=np.float64),
            # --- new: heterogeneous + T_max ---
            "cumulative_time": spaces.Box(0, 1000, shape=(self.K,), dtype=np.float64),
            "remaining_horizon": spaces.Box(0, 1000, shape=(self.K,), dtype=np.float64),
            "machine_activated": spaces.MultiBinary(self.K),
            "machine_travel_speeds": spaces.Box(0, 1000, shape=(self.K,), dtype=np.float64),
            "machine_harvest_efficiencies": spaces.Box(0, 1000, shape=(self.K,), dtype=np.float64),
            "machine_move_costs": spaces.Box(0, 1000, shape=(self.K,), dtype=np.float64),
            "machine_activation_costs": spaces.Box(0, 10000, shape=(self.K,), dtype=np.float64),
            "compatibility_mask": spaces.MultiBinary([self.K, self.max_tasks]),
            "areas": spaces.Box(0, 1000, shape=(self.max_tasks,), dtype=np.float64),
        })
        self.action_space = spaces.MultiDiscrete([self.max_tasks + 1] * self.K)
        self._reset_state()

    # ------------------------------------------------------------------
    # Instance loading
    # ------------------------------------------------------------------

    def set_instance(self, tasks, depot, max_field_work, areas,
                     machine_attrs=None):
        """Load a static instance.

        Parameters
        ----------
        tasks : list of (x, y)
            Coordinates normalized to [0,1].
        depot : array [2]
            Normalized coordinates of the dispatch centre.
        max_field_work : array [K]
            Max field-work hours per single trip H_k (h).
        areas : array [N]
            Farmland areas (mu).
        machine_attrs : dict, optional
            Heterogeneous machine attributes, containing:
            - travel_speeds : array [K] (km/h)
            - harvest_efficiencies : array [K] (mu/h)
            - move_costs : array [K] (CNY/km)
            - activation_costs : array [K] (CNY)
            - compatibility : array [K, N] int8 (0/1)
            If not given, the default homogeneous parameters are used.
        """
        self._tasks = list(tasks)
        self._depot = np.asarray(depot, dtype=np.float64)
        self._max_field_work = np.asarray(max_field_work, dtype=np.float64)

        # Areas
        n_actual = len(tasks)
        self._areas = np.asarray(areas, dtype=np.float64)

        # Machine attributes
        if machine_attrs is not None:
            self._machine_travel_speeds = np.asarray(
                machine_attrs["travel_speeds"], dtype=np.float64)
            self._machine_harvest_efficiencies = np.asarray(
                machine_attrs["harvest_efficiencies"], dtype=np.float64)
            self._machine_move_costs = np.asarray(
                machine_attrs["move_costs"], dtype=np.float64)
            self._machine_activation_costs = np.asarray(
                machine_attrs["activation_costs"], dtype=np.float64)
            self._compatibility = np.asarray(
                machine_attrs["compatibility"], dtype=np.int8)
        else:
            # Default homogeneous
            self._machine_travel_speeds = np.full(
                self.K, self._default_travel_speed)
            self._machine_harvest_efficiencies = np.full(
                self.K, self._default_harvest_speed)
            self._machine_move_costs = np.full(
                self.K, self._default_move_cost_coef)
            self._machine_activation_costs = np.full(
                self.K, self.machine_activation_cost_default)
            self._compatibility = np.ones(
                (self.K, self.max_tasks), dtype=np.int8)

        # Compute the per-machine service-time matrix [K, N]
        self._service_times_matrix = np.zeros((self.K, self.max_tasks))
        for k in range(self.K):
            rho = self._machine_harvest_efficiencies[k]
            if rho > 0:
                self._service_times_matrix[k, :n_actual] = (
                    self._areas[:n_actual] / rho)

    def set_random_instance(self):
        """Generate a random heterogeneous instance (used for training)."""
        n_clusters = 2
        centers = self.rng.uniform(0.25, 0.75, (n_clusters, 2))
        n_tasks = self.max_tasks

        # Farmlands: coordinates + areas
        tasks = []
        areas = np.zeros(n_tasks)
        for i in range(n_tasks):
            c = centers[int(self.rng.integers(n_clusters))]
            x = float(np.clip(c[0] + self.rng.normal(0, 0.12), 0.05, 0.95))
            y = float(np.clip(c[1] + self.rng.normal(0, 0.12), 0.05, 0.95))
            # Area (mu): lognormal distribution, clipped to 0.5-20 mu
            area = float(np.clip(self.rng.lognormal(1.0, 0.8), 0.5, 20.0))
            areas[i] = area
            tasks.append((x, y))

        # Heterogeneous machine attributes
        # Speed: 20-35 km/h
        speeds = self.rng.uniform(20.0, 35.0, self.K)
        # Efficiency: 3-8 mu/h
        efficiencies = self.rng.uniform(3.0, 8.0, self.K)
        # Unit-distance cost: 10-18 CNY/km
        move_costs = self.rng.uniform(10.0, 18.0, self.K)
        # Activation cost: 50-150 CNY
        act_costs = self.rng.uniform(50.0, 150.0, self.K)
        # Compatibility: fully compatible by default (could be extended to type restrictions later)
        compat = np.ones((self.K, n_tasks), dtype=np.int8)

        mfw = np.full(self.K, self.trip_max_field_work)

        self.set_instance(
            tasks=tasks,
            depot=np.array([0.5, 0.5]),
            max_field_work=mfw,
            areas=areas,
            machine_attrs={
                "travel_speeds": speeds,
                "harvest_efficiencies": efficiencies,
                "move_costs": move_costs,
                "activation_costs": act_costs,
                "compatibility": compat,
            },
        )
        return n_tasks


    # ------------------------------------------------------------------
    # Read-only accessors for the instance data (external tooling reads these)
    # ------------------------------------------------------------------

    @property
    def depot(self):
        return self._depot

    @property
    def machine_travel_speeds(self):
        return self._machine_travel_speeds

    @property
    def machine_harvest_efficiencies(self):
        return self._machine_harvest_efficiencies

    @property
    def machine_move_costs(self):
        return self._machine_move_costs

    @property
    def machine_activation_costs(self):
        return self._machine_activation_costs

    @property
    def max_field_work_per_machine(self):
        return self._max_field_work

    @property
    def service_times_matrix(self):
        return self._service_times_matrix

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _reset_state(self):
        self._tasks = []
        self._depot = np.array([0.5, 0.5])
        self._max_field_work = np.full(self.K, self.trip_max_field_work)
        self._areas = np.zeros(self.max_tasks)
        self._machine_travel_speeds = np.full(self.K, self._default_travel_speed)
        self._machine_harvest_efficiencies = np.full(self.K, self._default_harvest_speed)
        self._machine_move_costs = np.full(self.K, self._default_move_cost_coef)
        self._machine_activation_costs = np.full(self.K, self.machine_activation_cost_default)
        self._compatibility = np.ones((self.K, self.max_tasks), dtype=np.int8)
        self._service_times_matrix = np.zeros((self.K, self.max_tasks))

        self.coords = np.zeros((self.max_tasks, 2))
        self.completed = np.zeros(self.max_tasks, dtype=bool)
        self.served = np.zeros(self.max_tasks, dtype=bool)
        self._n_actual = 0
        self.positions = np.tile(self._depot, (self.K, 1))
        self.in_trip = np.zeros(self.K, dtype=bool)
        self.trip_used = np.zeros(self.K)
        self.sorties = np.zeros(self.K)
        self.cumulative_time = np.zeros(self.K)
        self.machine_activated = np.zeros(self.K, dtype=bool)
        self.total_cost = 0.0
        self.done = False

    def _apply_instance(self):
        n_actual = len(self._tasks)
        self.coords = np.zeros((self.max_tasks, 2))
        for i, t in enumerate(self._tasks):
            self.coords[i] = (t[0], t[1])

        # Padding slots default to "completed"
        self.completed = np.zeros(self.max_tasks, dtype=bool)
        self.completed[n_actual:] = True
        # served marks only farmlands actually finished by a real dispatch; the fallback drop
        # never sets it → used to decide whether all tasks are really completed
        self.served = np.zeros(self.max_tasks, dtype=bool)
        self._n_actual = n_actual

        self.positions = np.tile(self._depot, (self.K, 1))
        self.in_trip = np.zeros(self.K, dtype=bool)
        self.trip_used = np.zeros(self.K)
        self.sorties = np.zeros(self.K)
        self.cumulative_time = np.zeros(self.K)
        self.machine_activated = np.zeros(self.K, dtype=bool)
        self.total_cost = 0.0
        self.done = False

    def reset(self):
        self._apply_instance()
        return self._obs()

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def all_served(self):
        """Whether every real farmland has been served by an actual dispatch (fields dropped by the fallback do not count)."""
        return bool(self.served[:self._n_actual].all())

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _obs(self):
        mask = np.zeros((self.K, self.max_tasks + 1), dtype=bool)
        for k in range(self.K):
            # Remaining budget of the current trip
            trip_rem = self._max_field_work[k] - self.trip_used[k]
            # Remaining global horizon
            horizon_rem = self.T_max - self.cumulative_time[k]

            if self.in_trip[k]:
                # In trip: compatible and unfinished farmlands must satisfy both the trip-level
                # and the machine-level constraint: work ≤ remaining trip budget (H_k) and
                # travel + work ≤ remaining global horizon (T_max)
                for i in range(self.max_tasks):
                    if (not self.completed[i]
                            and self._compatibility[k, i]):
                        d = float(np.linalg.norm(
                            self.positions[k] - self.coords[i])) * self.map_size
                        travel_t = d / max(self._machine_travel_speeds[k], 1e-6)
                        work_t = self._service_times_matrix[k, i]
                        if (work_t <= trip_rem + 1e-9
                                and travel_t + work_t <= horizon_rem + 1e-9):
                            mask[k, i] = True
                mask[k, self.max_tasks] = True  # Return to depot
            else:
                # At depot: going to farmland i must satisfy both the machine-level constraint
                # (travel + work ≤ remaining horizon) and the trip-level constraint (the first
                # field's whole work time ≤ the fresh per-trip budget H_k)
                if horizon_rem < 1e-9:
                    # T_max exhausted, standby only
                    mask[k, self.max_tasks] = True
                else:
                    # Need to estimate: total time to reach farmland i = travel + work
                    for i in range(self.max_tasks):
                        if (not self.completed[i]
                                and self._compatibility[k, i]):
                            d = float(np.linalg.norm(
                                self.positions[k] - self.coords[i])) * self.map_size
                            travel_t = d / max(self._machine_travel_speeds[k], 1e-6)
                            work_t = self._service_times_matrix[k, i]
                            if (travel_t + work_t <= horizon_rem + 1e-9
                                    and work_t <= trip_rem + 1e-9):
                                mask[k, i] = True
                    # Standby is always selectable (an at-depot machine may always decline to dispatch)
                    mask[k, self.max_tasks] = True

        return {
            "observations": self.coords.copy(),
            "demand": (self._areas / np.clip(
                self._machine_harvest_efficiencies.mean() * self._max_field_work.mean(),
                1e-6, None)).copy(),
            "service_times": self._service_times_matrix.copy(),  # [K, N]
            "completed_mask": self.completed.copy(),
            "machine_positions": self.positions.copy(),
            "machine_status": np.where(self.in_trip, 1.0, 0.0),
            "field_work_used": self.trip_used.copy(),
            "sorties_used": self.sorties.copy(),
            "available_at": self.cumulative_time.copy(),
            "global_time": np.array([self.cumulative_time.min()]),
            "work_hours": np.full(1, self.T_max),
            "visible_field_mask": (~self.completed).astype(np.int8),
            "pending_mask": (~self.completed).astype(np.int8),
            "task_status": np.where(self.completed, 3.0, 1.0),
            "idle_machine_mask": np.ones(self.K, dtype=np.int8),
            "remaining_field_work": (self._max_field_work - self.trip_used),
            "feasible_pair_mask": mask[:, : self.max_tasks].astype(np.int8),
            "arrival_times": np.zeros(self.max_tasks),
            "travel_speed": self._machine_travel_speeds.copy(),  # [K] heterogeneous
            "in_trip": self.in_trip.copy(),
            "trip_used": self.trip_used.copy(),
            "max_field_work": self._max_field_work.copy(),
            "action_mask": mask,
            "depot": self._depot.copy(),
            # --- new ---
            "cumulative_time": self.cumulative_time.copy(),
            "remaining_horizon": (self.T_max - self.cumulative_time),
            "machine_activated": self.machine_activated.astype(np.int8),
            "machine_travel_speeds": self._machine_travel_speeds.copy(),
            "machine_harvest_efficiencies": self._machine_harvest_efficiencies.copy(),
            "machine_move_costs": self._machine_move_costs.copy(),
            "machine_activation_costs": self._machine_activation_costs.copy(),
            "compatibility_mask": self._compatibility.copy(),
            "areas": self._areas.copy(),
        }

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self, action):
        """action: [K] int, each machine picks j∈[0,N-1]=go to farmland or N=return to depot/standby.

        Machines are processed in order (matching the model-side sequential without-replacement
        masking): within one step a farmland can be assigned to at most one machine, and fields
        already picked by earlier machines are infeasible for the later ones.
        """
        action = np.asarray(action, dtype=np.int64)
        mask = self._obs()["action_mask"]
        reward = 0.0
        taken = np.zeros(self.max_tasks, dtype=bool)
        # Forced dispatch: as long as serviceable farmlands remain in the current state,
        # at least one harvester must be sent out this round
        any_serviceable = bool(mask[:, :self.max_tasks].any())
        dispatched = False

        for k in range(self.K):
            j = int(action[k])
            valid = bool(mask[k, j]) and (j == self.max_tasks or not taken[j])
            # Serviceable farmlands left & nothing dispatched this round & this at-depot machine
            # has an available field → standby is illegal
            # (returning to the depot is normal multi-trip behaviour and is NOT restricted;
            # only the stalling caused by "standing by at the depot" is forbidden)
            force = (any_serviceable and not dispatched and not self.in_trip[k]
                     and bool((mask[k, :self.max_tasks] & ~taken).any()))
            if valid and j == self.max_tasks and force:
                valid = False
            if not valid:
                reward -= 10.0
                continue

            v_k = self._machine_travel_speeds[k]
            c_k_d = self._machine_move_costs[k]

            if j == self.max_tasks:
                if self.in_trip[k]:
                    # Return to depot: settle the return distance cost + time
                    d = float(np.linalg.norm(
                        self.positions[k] - self._depot)) * self.map_size
                    travel_t = d / max(v_k, 1e-6)
                    reward -= d * c_k_d
                    self.cumulative_time[k] += travel_t
                    self.positions[k] = self._depot.copy()
                    self.in_trip[k] = False
                    self.trip_used[k] = 0.0
                # else: standby is a no-op, no cost and no state change

            else:
                # Go to farmland j
                d = float(np.linalg.norm(
                    self.positions[k] - self.coords[j])) * self.map_size
                travel_t = d / max(v_k, 1e-6)
                work_t = self._service_times_matrix[k, j]

                # Transfer cost
                reward -= d * c_k_d

                # Activation cost (first dispatch)
                if not self.machine_activated[k]:
                    reward -= self._machine_activation_costs[k]
                    self.machine_activated[k] = True

                # Trip start cost
                if not self.in_trip[k]:
                    reward -= self.start_cost
                    self.sorties[k] += 1

                # Update state
                self.cumulative_time[k] += travel_t + work_t
                self.trip_used[k] += work_t
                self.positions[k] = self.coords[j].copy()
                self.in_trip[k] = True
                self.completed[j] = True
                self.served[j] = True
                taken[j] = True
                dispatched = True

        # Once everything is done, auto-settle the return trip of the machines still in transit
        if self.completed.all() and self.in_trip.any():
            for k in range(self.K):
                if self.in_trip[k]:
                    d = float(np.linalg.norm(
                        self.positions[k] - self._depot)) * self.map_size
                    travel_t = d / max(self._machine_travel_speeds[k], 1e-6)
                    reward -= d * self._machine_move_costs[k]
                    self.cumulative_time[k] += travel_t
                    self.positions[k] = self._depot.copy()
                    self.in_trip[k] = False
                    self.trip_used[k] = 0.0

        # Fallback: the remaining farmlands are unserviceable to every machine → mark them completed
        # only (served stays unset for these fields, i.e. they count as unfinished, and the trainer
        # applies the infeasibility penalty)
        if not self.completed.all() and not self.in_trip.any():
            fresh_mask = self._obs()["action_mask"]
            if not fresh_mask[:, :self.max_tasks].any():
                self.completed = np.ones(self.max_tasks, dtype=bool)

        self.total_cost += -reward
        done = bool(self.completed.all() and not self.in_trip.any())
        self.done = done
        return self._obs(), float(reward), np.bool_(done), {
            "total_cost": self.total_cost,
            "cumulative_time": self.cumulative_time.copy(),
            "machine_activated": self.machine_activated.copy(),
            "served_all": bool(self.served[:self._n_actual].all()),
            "unfinished": int((~self.served[:self._n_actual]).sum()),
        }
