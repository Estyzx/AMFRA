# -*- coding: utf-8 -*-
"""Greedy inference on a static instance, plus nearest-neighbour / random baselines.

Instance: --data CSV (id,x_km,y_km,area_mu); without it a random 50-field instance is used.
Cost: J = C_move + C_start + C_act, the same accounting for model / NN / random.
Fleet: homogeneous by default; --fleet three-type for three sizes in equal numbers;
--fleet custom --fleet-spec <json> for any mix and any per-type parameters;
--hetero-seed S for random per-machine multipliers U(0.7, 1.3).
Output: trips per machine, total cost, baseline comparison, per-trip work cap check.

Usage
-----
    python simulate.py                                    # random instance, baselines only
    python simulate.py --data fields.csv --checkpoint runs/<run>/ckpt/<update>.pt \
        --map-size 50 --n-machines 15 --max-field-work 14 --t-max 24 \
        --start-cost 68 --move-cost 13.6 --save-schedule schedule.json
    python simulate.py --data fields.csv --fleet custom --fleet-spec fleet.json
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.harvest_static_env import HarvestStaticEnv
from models.attention_model_wrapper import Agent


# Built-in preset behind --fleet three-type:
# (name, travel speed km/h, harvest efficiency mu/h, unit-distance cost CNY/km,
# activation cost CNY).
THREE_TYPE_FLEET = (
    ("large", 30.0, 20.0, 18.0, 150.0),
    ("medium", 25.0, 15.0, 13.0, 100.0),
    ("small", 20.0, 5.0, 8.0, 50.0),
)
# Fields every --fleet-spec entry must provide; "name" and "max_field_work" are optional.
FLEET_SPEC_FIELDS = ("count", "travel_speed", "harvest_efficiency",
                     "move_cost", "activation_cost")


def three_type_types(counts):
    """Turn the built-in preset into the same dict shape a --fleet-spec entry uses."""
    return [{"name": name, "count": n, "travel_speed": v,
             "harvest_efficiency": rho, "move_cost": c, "activation_cost": a,
             "max_field_work": None}
            for (name, v, rho, c, a), n in zip(THREE_TYPE_FLEET, counts)]


def parse_fleet_spec(spec):
    """Read --fleet-spec (a JSON file path or inline JSON) into a list of machine types.

    One entry per type; the machines are laid out type by type in the given order, so a
    fleet of any size and any mix can be described. Every required numeric field must be
    present, finite and positive, and count must be a positive integer. Unknown fields are
    rejected so a typo cannot silently fall back to a default.
    """
    if spec is None:
        raise SystemExit("--fleet custom 需要 --fleet-spec（JSON 文件路径或内联 JSON）")
    text = spec.strip()
    if not text.startswith(("[", "{")):
        if not os.path.isfile(text):
            raise SystemExit(f"--fleet-spec 文件不存在: {text}")
        with open(text, encoding="utf-8-sig") as f:
            text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        raise SystemExit(f"--fleet-spec 不是合法 JSON: {err}")
    if isinstance(data, dict):
        data = data.get("types")
        if data is None:
            raise SystemExit('--fleet-spec 的对象形式需要 "types" 列表')
    if not isinstance(data, list) or not data:
        raise SystemExit("--fleet-spec 至少需要一个机型")
    types = []
    for idx, item in enumerate(data, 1):
        if not isinstance(item, dict):
            raise SystemExit(f"--fleet-spec 第 {idx} 项必须是 JSON 对象")
        unknown = sorted(set(item) - set(FLEET_SPEC_FIELDS)
                         - {"name", "max_field_work"})
        if unknown:
            raise SystemExit(f"--fleet-spec 第 {idx} 项含未知字段: {', '.join(unknown)}")
        missing = [k for k in FLEET_SPEC_FIELDS if k not in item]
        if missing:
            raise SystemExit(f"--fleet-spec 第 {idx} 项缺少字段: {', '.join(missing)}")
        entry = {"name": str(item.get("name") or f"type{idx}")}
        for key in FLEET_SPEC_FIELDS:
            try:
                value = float(item[key])
            except (TypeError, ValueError):
                raise SystemExit(f"--fleet-spec 第 {idx} 项的 {key} 不是数字")
            if not np.isfinite(value) or value <= 0:
                raise SystemExit(f"--fleet-spec 第 {idx} 项的 {key} 必须为有限正数")
            entry[key] = value
        if abs(entry["count"] - round(entry["count"])) > 1e-9:
            raise SystemExit(f"--fleet-spec 第 {idx} 项的 count 必须为整数")
        entry["count"] = int(round(entry["count"]))
        h_value = item.get("max_field_work")
        if h_value is None:
            entry["max_field_work"] = None
        else:
            try:
                h_value = float(h_value)
            except (TypeError, ValueError):
                raise SystemExit(f"--fleet-spec 第 {idx} 项的 max_field_work 不是数字")
            if not np.isfinite(h_value) or h_value <= 0:
                raise SystemExit(
                    f"--fleet-spec 第 {idx} 项的 max_field_work 必须为有限正数")
            entry["max_field_work"] = h_value
        types.append(entry)
    return types


def resolve_fleet_args(args):
    """Settle --n-machines / --fleet-spec after parsing, so bad combinations fail early."""
    explicit_n = args.n_machines is not None
    if args.fleet == "custom":
        args.fleet_types = parse_fleet_spec(args.fleet_spec)
        total = sum(t["count"] for t in args.fleet_types)
        if explicit_n and total != args.n_machines:
            raise SystemExit(f"--fleet-spec 各机型台数之和为 {total}，"
                             f"与 --n-machines {args.n_machines} 不一致")
        args.n_machines = total
    else:
        if args.fleet_spec is not None:
            raise SystemExit("--fleet-spec 只能与 --fleet custom 一起使用")
        if not explicit_n:
            args.n_machines = 15
    if args.n_machines <= 0:
        raise SystemExit("--n-machines 必须为正整数")
    if args.hetero_seed is not None and args.fleet != "homogeneous":
        raise SystemExit(f"--hetero-seed 与 --fleet {args.fleet} 不能同时使用")
    return args


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="模型 checkpoint 路径；缺省时仅运行最近邻/随机基线")
    ap.add_argument("--data", type=str, default=None,
                    help="地块 CSV：id,x_km,y_km,area_mu；缺省用随机实例演示")
    ap.add_argument("--map-size", type=float, default=10.0,
                    help="地图尺度（公里），坐标归一化分母，需 ≥ 坐标最大值")
    ap.add_argument("--depot-x", type=float, default=None,
                    help="调度中心 x（公里），缺省为地图中心")
    ap.add_argument("--depot-y", type=float, default=None,
                    help="调度中心 y（公里），缺省为地图中心")
    ap.add_argument("--n-machines", type=int, default=None,
                    help="农机台数；缺省 15（--fleet custom 时缺省为 --fleet-spec 台数之和）")
    ap.add_argument("--max-field-work", type=float, default=8.0,
                    help="每台农机单趟田间作业上限（小时），默认全机群统一；"
                         "--fleet-spec 内可用 max_field_work 按机型覆盖该值")
    ap.add_argument("--harvest-speed", type=float, default=6.0,
                    help="收割速度（亩/小时），作业时长 = 面积 / 速度")
    ap.add_argument("--travel-speed", type=float, default=25.0,
                    help="同构机群行驶速度（km/h）")
    ap.add_argument("--start-cost", type=float, default=5.0,
                    help="每趟行程启动成本（元/趟）")
    ap.add_argument("--move-cost", type=float, default=1.0,
                    help="每公里转移成本（元/km）")
    ap.add_argument("--work-hours", type=float, default=24.0,
                    help="保留旧命令参数；单趟限制与报告均以 --max-field-work 为准")
    ap.add_argument("--t-max", type=float, default=24.0,
                    help="全局调度规划周期上限 T_max（小时），不随返场重置；"
                         "与训练器 --t-max 应保持一致（缺省 24 为环境默认）")
    ap.add_argument("--seed", type=int, default=2026,
                    help="随机基线种子，同时决定机群启用成本的抽取")
    ap.add_argument("--hetero-seed", type=int, default=None,
                    help="启用随机倍率异构机群（速度/作业效率/单位距离成本 × U(0.7,1.3)）；"
                         "与 --fleet custom/three-type 互斥")
    ap.add_argument("--fleet", type=str, default="homogeneous",
                    choices=["homogeneous", "three-type", "custom"],
                    help="机群配置：homogeneous 为同构（统一参数）；"
                         "three-type 为三型异构（大/中/小各 n_machines/3 台）；"
                         "custom 为自定义异构（需 --fleet-spec）")
    ap.add_argument("--fleet-spec", type=str, default=None,
                    help="自定义机群规格：JSON 文件路径或内联 JSON 列表，每项含 "
                         "count/travel_speed/harvest_efficiency/move_cost/activation_cost"
                         "（可选 name、max_field_work）；仅与 --fleet custom 一起使用")
    ap.add_argument("--save-schedule", type=str, default=None,
                    help="可选：调度方案 JSON 输出路径")
    args = ap.parse_args()
    return resolve_fleet_args(args)


def load_instance(args):
    """Build (tasks, depot, max_field_work, areas, machine_attrs) from the CSV; return None when there is no data, falling back to a random instance."""
    if not args.data:
        return None
    for name in ("map_size", "harvest_speed", "travel_speed", "max_field_work", "t_max"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (args.depot_x is None) != (args.depot_y is None):
        raise SystemExit("--depot-x 与 --depot-y 必须同时提供")
    tasks = []
    areas = []
    with open(args.data, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            x, y = float(row["x_km"]), float(row["y_km"])
            area = float(row["area_mu"])
            if not np.isfinite([x, y, area]).all() or area <= 0:
                raise ValueError("CSV coordinates must be finite and area_mu must be positive")
            if x > args.map_size or y > args.map_size:
                print(f"[警告] {row['id']} 坐标 ({x},{y}) 超过 map_size="
                      f"{args.map_size}，归一化后 >1 超出模型训练分布")
            # new heterogeneous interface: tasks store only (x, y), the areas are passed separately
            tasks.append((x / args.map_size, y / args.map_size))
            areas.append(area)
    if not tasks:
        raise SystemExit("地块 CSV 至少需要一行数据")
    depot = (np.array([args.depot_x / args.map_size,
                       args.depot_y / args.map_size], dtype=float)
             if args.depot_x is not None else np.array([0.5, 0.5]))
    areas_arr = np.array(areas, dtype=np.float64)
    # Machine attributes: every machine carries its own values, so the presets and a fully
    # custom --fleet-spec both go through the same builder.
    rng = np.random.default_rng(args.seed)
    fleet = getattr(args, "fleet", "homogeneous")
    if fleet == "custom":
        types = args.fleet_types
    elif fleet == "three-type":
        # Preset: three sizes in equal numbers, hence the divisibility requirement.
        if args.n_machines % 3 != 0:
            raise SystemExit("--fleet three-type 需要 --n-machines 为 3 的倍数；"
                             "台数不等时请用 --fleet custom --fleet-spec")
        types = three_type_types([args.n_machines // 3] * 3)
    else:
        types = None

    if types is not None:
        machine_attrs = {
            "travel_speeds": np.concatenate(
                [np.full(t["count"], t["travel_speed"]) for t in types]),
            "harvest_efficiencies": np.concatenate(
                [np.full(t["count"], t["harvest_efficiency"]) for t in types]),
            "move_costs": np.concatenate(
                [np.full(t["count"], t["move_cost"]) for t in types]),
            "activation_costs": np.concatenate(
                [np.full(t["count"], t["activation_cost"]) for t in types]),
        }
        max_field_work = np.concatenate(
            [np.full(t["count"], args.max_field_work if t["max_field_work"] is None
                     else t["max_field_work"]) for t in types])
        if len(machine_attrs["travel_speeds"]) != args.n_machines:
            raise SystemExit("机群规格台数与 --n-machines 不一致")
        print(f"异构机群 {args.n_machines} 台：" + "；".join(
            f"{t['name']}×{t['count']}（{t['travel_speed']:g} km/h，"
            f"{t['harvest_efficiency']:g} 亩/h，{t['move_cost']:g} 元/km，"
            f"启用 {t['activation_cost']:g} 元）" for t in types))
        overrides = [t for t in types if t["max_field_work"] is not None]
        if overrides:
            print("单趟作业上限按机型：" + "；".join(
                f"{t['name']} {t['max_field_work']:g}h" for t in overrides))
    else:
        machine_attrs = {
            "travel_speeds": np.full(args.n_machines, args.travel_speed),
            "harvest_efficiencies": np.full(args.n_machines, args.harvest_speed),
            "move_costs": np.full(args.n_machines, args.move_cost),
            "activation_costs": rng.uniform(50.0, 150.0, args.n_machines),  # aligned with the training distribution
        }
        max_field_work = np.full(args.n_machines, args.max_field_work)
        # Random-multiplier heterogeneity: per-machine multipliers drawn in this
        # fixed order (speed, work efficiency, unit-distance cost). The work cap
        # H_k and the activation-cost draw stay unchanged.
        if getattr(args, "hetero_seed", None) is not None:
            frng = np.random.default_rng(args.hetero_seed)
            factor_v = frng.uniform(0.7, 1.3, args.n_machines)
            factor_rho = frng.uniform(0.7, 1.3, args.n_machines)
            factor_cd = frng.uniform(0.7, 1.3, args.n_machines)
            machine_attrs["travel_speeds"] = machine_attrs["travel_speeds"] * factor_v
            machine_attrs["harvest_efficiencies"] = (
                machine_attrs["harvest_efficiencies"] * factor_rho)
            machine_attrs["move_costs"] = machine_attrs["move_costs"] * factor_cd
            print(f"随机倍率异构机群已启用（hetero_seed={args.hetero_seed}，倍率 U(0.7,1.3)）")
    # Compatibility: a machine can serve a field only if the field fits wholly
    # inside that machine's own per-trip field-work cap H_k (area / rho_k <= H_k).
    # With a homogeneous fleet this is normally all-ones; it bites only when a
    # slow machine cannot finish a large field within one trip.
    machine_attrs["compatibility"] = (
        areas_arr[None, :] / machine_attrs["harvest_efficiencies"][:, None]
        <= max_field_work[:, None] + 1e-9).astype(np.int8)
    print(f"已加载 {len(tasks)} 块田（{args.data}）")
    return tasks, depot, max_field_work, areas_arr, machine_attrs


def build_agent(ckpt_path, device, map_size):
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt.get("metadata", {})
    mode = meta.get("decoder_mode", "joint")
    abl = meta.get("amfra_ablation", "A2")
    expected_modes = {"E0": "field_pointer", "A0": "joint", "A2": "joint", "A3": "joint"}
    if abl not in expected_modes or mode != expected_modes[abl]:
        raise ValueError(f"unsupported checkpoint decoder configuration: {mode}/{abl}")
    if mode == "field_pointer":
        n_layers, gate, rel = 3, False, False
    else:
        n_layers, gate, rel = 3, (abl == "A3"), (abl in ("A2", "A3"))
    print(f"模型: decoder={mode} ablation={abl} n_layers={n_layers} "
          f"update={ckpt.get('update')}")
    agent = Agent(
        embedding_dim=128, device=device,
        n_encode_layers=n_layers,
        amfra_use_gate=gate, amfra_map_size=map_size,
        use_relation_bias=rel,
        decoder_mode=mode,
    ).to(device)
    # Old checkpoints may carry keys that no longer exist in the model (e.g. the
    # removed critic); surplus keys are filtered and printed (a *missing* key
    # still fails the load — it means a live parameter was deleted by mistake).
    want = set(agent.state_dict().keys())
    raw = ckpt["model_state_dict"]
    surplus = sorted(set(raw) - want)
    if surplus:
        print(f"忽略 {len(surplus)} 个已删除的参数键（来自 {os.path.basename(ckpt_path)}）:")
        for key in surplus:
            print(f"  - {key}")
    agent.load_state_dict({k: v for k, v in raw.items() if k in want})
    agent.eval()
    return agent


def prep(obs):
    return {k: (v.astype(np.float32) if np.issubdtype(v.dtype, np.floating) else v)
            for k, v in obs.items()}


def baseline_step(env, obs, mode, rng):
    """Nearest-neighbour / random baseline: sequential selection with within-step taken de-duplication (consistent with the env's sequential semantics)."""
    N, K = env.max_tasks, env.K
    feas = obs["feasible_pair_mask"]
    a = np.full(K, N, dtype=np.int64)
    taken = np.zeros(N, dtype=bool)
    for k in range(K):
        fk = np.flatnonzero(feas[k] & ~taken)
        if fk.size:
            if mode == "random":
                a[k] = int(rng.choice(fk))
            else:
                d = np.linalg.norm(env.positions[k] - env.coords[fk], axis=1)
                a[k] = int(fk[np.argmin(d)])
            taken[a[k]] = True
    return a


def rollout(env, agent, mode, rng, trace=False):
    """Roll out until done; with trace=True record every step's per-machine actions for schedule extraction.

    Returns (total cost J, action log, finished). finished says whether all the real
    fields were served (env.all_served()); an unfinished run (timeout / abandoned
    deadlock) counts as an invalid trajectory.
    """
    obs = env.reset()
    cost, steps = 0.0, 0
    log = []
    while not env.done and steps < 2000:
        if mode == "model":
            b = prep({k: np.stack([obs[k]]) for k in obs})
            with torch.no_grad():
                cache = agent.backbone.encode(b)
                act, _, _, _, _ = agent.get_joint_action_and_value_cached(
                    b, state=cache, greedy=True)
            a = act[0].cpu().numpy()
        else:
            a = baseline_step(env, obs, mode, rng)
        if trace:
            log.append((steps, a.copy(), obs["machine_positions"].copy(),
                        obs["in_trip"].copy()))
        obs, r, done, info = env.step(a)
        cost += -r
        steps += 1
    finished = env.all_served()
    return cost, log, finished


def extract_schedule(env, log, args):
    """Extract each machine's trips and relocation distances from the action trajectory (including returns and the end-of-episode auto-return).

    Additionally returns activated: whether each machine ever sortied (used to settle
    the machine activation cost).
    """
    trips = {k: [] for k in range(env.K)}
    current = {k: [] for k in range(env.K)}
    dist = {k: 0.0 for k in range(env.K)}
    last_target = {k: None for k in range(env.K)}
    activated = np.zeros(env.K, dtype=bool)
    for _, act, pos, in_trip in log:
        for k, j in enumerate(act):
            if j < env.max_tasks:
                dist[k] += float(np.linalg.norm(pos[k] - env.coords[j])) \
                    * args.map_size
                current[k].append(f"f{j + 1}")
                last_target[k] = j
                activated[k] = True
            else:
                # return to depot (for an at-depot standby no-op the position is already the depot, so the distance is 0)
                dist[k] += float(np.linalg.norm(pos[k] - env._depot)) \
                    * args.map_size
                if current[k]:
                    trips[k].append(current[k])
                    current[k] = []
                last_target[k] = None
    # End-of-episode auto-return (settled inside the env, not recorded in trace): a machine
    # whose last action was a field returns from the target field, not from its pre-action position
    for k in range(env.K):
        if last_target[k] is not None:
            dist[k] += float(np.linalg.norm(env.coords[last_target[k]]
                                            - env._depot)) * args.map_size
            if current[k]:
                trips[k].append(current[k])
                current[k] = []
    # per machine: total work / total relocation / per-trip work durations (multi-trip semantics, single trip ≤ work_hours)
    work_h, use, trip_work = [], [], []
    # heterogeneous env: service_times_matrix is [K, N], different for every machine
    for k in range(env.K):
        serv = sum(env._service_times_matrix[k, int(f[1:]) - 1] for t in trips[k] for f in t)
        drive = dist[k] / env._machine_travel_speeds[k]
        use.append(serv + drive)
        work_h.append(serv)
        trip_work.append([sum(env._service_times_matrix[k, int(f[1:]) - 1] for f in t)
                          for t in trips[k]])
    return trips, dist, work_h, use, trip_work, activated


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)
    agent = build_agent(args.checkpoint, device, args.map_size) if args.checkpoint else None
    rng = np.random.default_rng(args.seed)

    inst = load_instance(args)
    env = HarvestStaticEnv(max_tasks_per_day=len(inst[0]) if inst else 50,
                           n_machines=args.n_machines,
                           map_size=args.map_size, start_cost=args.start_cost,
                           harvest_speed=args.harvest_speed,
                           move_cost_coef=args.move_cost,
                           T_max=args.t_max,
                           trip_max_field_work=args.max_field_work, seed=7)
    if inst:
        tasks, depot, mfw, areas, machine_attrs = inst
        env.set_instance(tasks, depot, mfw, areas=areas, machine_attrs=machine_attrs)
        dk = (f"({args.depot_x:.2f}, {args.depot_y:.2f})"
              if args.depot_x is not None
              else f"({args.map_size / 2:.2f}, {args.map_size / 2:.2f})")
        print(f"调度中心: {dk} km | 单趟上限: " + (
            f"{float(mfw[0]):g}h" if np.allclose(mfw, mfw[0])
            else "各机不同 [" + ", ".join(f"{h:g}" for h in mfw) + "] h"))
    else:
        print("[演示模式] 未提供 --data，使用随机实例（50 块田）")
        if getattr(args, "fleet", "homogeneous") != "homogeneous":
            print(f"[提示] --fleet {args.fleet} 需要 --data；随机实例使用环境自带的异构机群属性")
        env.set_random_instance()

    if agent is None:
        # No checkpoint: the learned constructor cannot be evaluated, but the two
        # weight-free baselines still exercise the instance, env and schedule rules.
        print("[提示] 未提供 --checkpoint：仅运行最近邻与随机基线")
        cost_nn, _, fin_nn = rollout(env, agent, "nn", rng)
        cost_rd, _, fin_rd = rollout(env, agent, "random", rng)
        print("\n=== 基线结果（成本单位：元；口径 J=C_move+C_start+C_act）===")
        print(f"最近邻基线 J: {cost_nn:.2f}（全部完成={fin_nn}）")
        print(f"随机基线 J:   {cost_rd:.2f}（全部完成={fin_rd}）")
        return

    cost_m, log, finished_m = rollout(env, agent, "model", rng, trace=True)
    cost_nn, _, finished_nn = rollout(env, agent, "nn", rng)
    cost_rd, _, finished_rd = rollout(env, agent, "random", rng)
    trips, dist, work_h, use, trip_work, activated = extract_schedule(env, log, args)

    n_trips = sum(len(t) for t in trips.values())
    move = sum(dist.values())
    start = n_trips * args.start_cost
    n_activated = int(activated.sum())
    activate = float(sum(env._machine_activation_costs[k]
                         for k in range(env.K) if activated[k]))
    # Relocation cost is settled per machine with its own unit-distance cost, so
    # the decomposition stays exact for a heterogeneous fleet (for a homogeneous
    # fleet this equals distance × --move-cost).
    move_cost_total = float(sum(dist[k] * float(env._machine_move_costs[k])
                                for k in range(env.K)))
    # unified three-part cost accounting: J = C_move + C_start + C_act (identical for model/NN/random)
    j_decomposed = move_cost_total + start + activate
    # penalty = env total cost − the three physical cost terms; should be ≈0 (residual
    # illegal-action penalties caused by the gap between the model's step-wise mask and the env
    # semantics; a proper greedy run must keep it at 0)
    penalty = cost_m - j_decomposed

    def _comparison(base, finished):
        if not finished_m or not finished:
            return "不可行方案不参与成本比较"
        pct = 100 * (base - cost_m) / base if base > 0 else 0.0
        return f"模型较优 {pct:+.1f}%"

    print("\n=== 仿真结果（成本单位：元；口径统一 J=C_move+C_start+C_act）===")
    print(f"模型 greedy 总成本 J: {cost_m:.2f}（全部完成={finished_m}）")
    print(f"  ├ 转场 {move:.1f}km（按各机单位距离成本）= {move_cost_total:.2f}")
    print(f"  ├ 启动 {n_trips}趟×{args.start_cost} = {start:.2f}")
    print(f"  └ 启用 {n_activated}台 = {activate:.2f} → 三项合计 {j_decomposed:.2f}（惩罚 {penalty:.2f}）")
    print(f"最近邻基线 J: {cost_nn:.2f}（全部完成={finished_nn}），{_comparison(cost_nn, finished_nn)}")
    print(f"随机基线 J:   {cost_rd:.2f}（全部完成={finished_rd}），{_comparison(cost_rd, finished_rd)}")
    print(f"总转场距离: {move:.1f} km | 总行程数: {n_trips}"
          f" | 启动成本: {start:.1f} | 启用成本: {activate:.1f}")
    print("\n=== 单趟上限核算（每趟作业 ≤ H，回中心后可再出勤）===")
    ok = finished_m
    for k in range(env.K):
        tw = trip_work[k]
        mx = max(tw) if tw else 0.0
        limit = float(env._max_field_work[k])
        flag = "OK" if mx <= limit + 1e-9 else "超时"
        ok &= mx <= limit + 1e-9
        print(f"农机{k + 1:2d}: {len(tw)}趟 总作业 {work_h[k]:6.2f}h "
              f"+ 转场 {dist[k] / env._machine_travel_speeds[k]:5.2f}h | 最大单趟 {mx:6.2f}h"
              f" / {limit}h [{flag}]")
    print(f"结论: {'全部农田完成，且各机单趟作业均在限内' if ok else '方案不可行：农田未全部完成或存在超趟作业'}")

    if args.save_schedule:
        sched = {f"machine_{k + 1}": [{"trip": t, "work_h": hours}
                                      for t, hours in zip(trips[k], trip_work[k])]
                 for k in range(env.K)}
        with open(args.save_schedule, "w", encoding="utf-8") as f:
            json.dump({"cost": cost_m, "finished": finished_m,
                       "move": move_cost_total, "start": start,
                       "activate": activate, "penalty": penalty,
                       "trips": sched}, f, ensure_ascii=False, indent=2)
        print(f"\n调度方案已保存: {args.save_schedule}")


if __name__ == "__main__":
    main()
