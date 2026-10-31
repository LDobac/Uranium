"""RL learner updates for the bench suite (060-R2, 042 learner optimisation, 032-R19 ⑥): the
learner's `update` without games, on one fixed synthetic batch of segments (`SyntheticSegments`,
the layout of `_core.SegmentBatch.group`) from a checkpoint's weights (dropout 0, as `python -m
uranium rl`). Lane group sizes and the learner before the optimisations (`reference`) are swept in
one run from the same weights and batch, so the times compare and the per-update statistics are a
parity log (`bench rl-parity`: losses, KL, gradient norm per update, 042 lane-group equivalence).
The first update warms kernels up (cuDNN benchmark per shape) and is not counted in the medians.
Process settings as in an RL run: `configure(backend)` without the BC trainer's defaults."""

import copy
import json
import math
import statistics
import time

import numpy as np
import torch

from uranium.agent.infer import leaves
from uranium.bench.inputs import random_obs
from uranium.bench.timing import peak_gib, peak_rss_gib, reset_peak, sync
from uranium.model.net import ENTITY_KEYS, LABEL_KEYS, OBS_KEYS

# 033 reward weights of config/reward.toml (initial values, without the league's decay).
WEIGHTS = [1.0, 0.3, 0.3, 0.05]
STAT_KEYS = ("loss", "policy", "value", "kl", "win_open", "rho_mean", "adv_mean", "grad_norm", "kl_heads", "entropy_heads")
PAD = 64  # entity padding of learner batches (uranium_league::batch::PAD_ENTITIES, 030-R7)
# `--off` names: the new learner's attribute each one clears.
OFF = {"trim": "trim_pad", "pinned": "pinned", "spatial_bf16": "spatial_bf16"}


class SyntheticSegments:
    """A fixed RL batch of `lanes` segments of `t` steps in the layout of `_core.SegmentBatch`
    (`lanes`, `trained`, `versions`, `group`), with real-game-like sizes: each lane takes one of
    `templates` synthetic step blocks (random features, `entities` = (lo, hi) own and enemy
    entities, padded to 64 per lane group like the Rust collate) whose labels the network sampled
    under the masks (legal actions; behaviour log-probs from that sampling, so ρ is near but not 1).
    Every lane but the first has a core_state0 (mid-game memory, outer counters at varied phases);
    1 lane in 8 is terminal with a shorter trained length, the others train T-1 steps (the last is
    the bootstrap step). Same seed, same batch."""

    def __init__(self, net, lanes: int, t: int, entities: tuple[int, int] = (64, 192), templates: int = 8, seed: int = 1, device=None):
        from uranium import _core

        device = torch.device(device or next(net.parameters()).device)
        self.lanes, self.t = lanes, t
        max_select = net.cfg.heads.max_select
        # The Rust collate's dtypes per array.
        ref = _core.test_segments(1, 1).group(0, 1, 1, max_select)
        self.dtypes = {k: v.dtype for k, v in ref.items() if isinstance(v, np.ndarray)}
        gen = torch.Generator().manual_seed(seed)
        rng = np.random.default_rng(seed)
        p = ref["privileged"].shape[1]
        self.blocks = []
        for i in range(max(1, min(templates, lanes))):
            n = int(rng.integers(entities[0], entities[1] + 1))
            n_pad = -(-n // PAD) * PAD
            obs = random_obs(t, n_pad, net.cfg.input, torch.device("cpu"), gen, valid=n)
            dev_obs = {k: v.to(device) for k, v in obs.items()}
            was = net.training
            net.eval()
            with torch.no_grad():
                out = net.act(dev_obs, net.init_state(t, device), 1.0, torch.Generator(device=device).manual_seed(seed * 1000 + i))
            net.train(was)
            labels = {k: v.cpu() for k, v in out["action"].items()}
            block = {k: v.numpy() for k, v in obs.items()}
            block |= {k: labels[k].numpy() for k in LABEL_KEYS}
            block["privileged"] = (torch.randn(t, p, generator=gen) * 0.1).numpy()
            block["behaviour"] = out["log_prob"].float().cpu().numpy()
            self.blocks.append((n_pad, block))
        state = net.init_state(1, torch.device("cpu"))
        self.leaves = [(tuple(x.shape), x.dtype) for x in leaves(state)]
        self.outer = net.outer is not None
        self.lane = []
        for i in range(lanes):
            terminal = i % 8 == 7
            trained = t - 1 - int(rng.integers(0, max(1, t // 4))) if terminal else t - 1
            r = rng.normal(0.0, 0.01, (t, 4)).astype(np.float32)
            if terminal:
                r[trained - 1, 0] += 1.0 if i % 16 == 7 else -1.0
            self.lane.append({"block": i % len(self.blocks), "terminal": terminal, "trained": trained, "rewards": r, "state0": None if i == 0 else self._state0(rng, i)})

    def _state0(self, rng, lane: int) -> list:
        """One lane's core_state0 as the Rust group gives it: (shape, float32 values) per state
        tensor; memory masks all set; the outer counter at phase `13·lane mod 32` of a later segment."""
        parts = []
        for k, (shape, dtype) in enumerate(self.leaves):
            if dtype == torch.bool:
                v = np.ones(shape, np.float32)
            elif self.outer and k == len(self.leaves) - 1:
                v = np.full(shape, 64.0 + (13 * lane) % 32, np.float32)
            else:
                v = (rng.standard_normal(shape) * 0.1).astype(np.float32)
            parts.append((shape, v.ravel()))
        return parts

    @property
    def trained(self) -> list[int]:
        return [x["trained"] for x in self.lane]

    @property
    def versions(self) -> list[int]:
        return [0] * self.lanes

    def group(self, start: int, end: int, t: int, max_select: int) -> dict:
        assert t == self.t and 0 <= start < end <= self.lanes, (start, end, t)
        lanes = self.lane[start:end]
        n = max(self.blocks[x["block"]][0] for x in lanes)
        cols: dict = {k: [] for k in (*OBS_KEYS, *LABEL_KEYS, "privileged", "behaviour")}
        for x in lanes:
            n_b, block = self.blocks[x["block"]]
            for k in cols:
                v = block[k]
                if k in ENTITY_KEYS and n_b < n:
                    v = np.pad(v, [(0, 0), (0, n - n_b)] + [(0, 0)] * (v.ndim - 2))
                elif k == "selected":
                    v = np.where(v >= n_b, n, v)  # END is the group's entity count
                cols[k].append(v)
        g = {k: np.concatenate(v).astype(self.dtypes.get(k, v[0].dtype), copy=False) for k, v in cols.items()}
        g["mask"] = np.concatenate([(np.arange(t) < x["trained"]).astype(np.float32) for x in lanes]).astype(self.dtypes["mask"])
        g["rewards"] = np.concatenate([x["rewards"] for x in lanes]).astype(self.dtypes["rewards"])
        g["terminal"] = np.array([x["terminal"] for x in lanes], dtype=self.dtypes["terminal"])
        g["trained"] = np.array([x["trained"] for x in lanes], dtype=self.dtypes["trained"])
        g["versions"] = np.zeros(len(lanes), dtype=self.dtypes["versions"])
        g["state0"] = [x["state0"] for x in lanes]
        g |= {"b": len(lanes), "t": t, "n": n}
        return g


def profile_update(learner, batch, device: torch.device, n_ops: int, trace: str | None = None) -> dict:
    """The operators with the most device time (CPU time on CPU) in one update, and the host
    syncs and copies of it (CUDA runtime calls)."""
    acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with torch.profiler.profile(activities=acts) as prof:
        learner.update(batch, WEIGHTS)
        sync(device)
    avg = prof.key_averages()
    key = "self_device_time_total" if device.type == "cuda" else "self_cpu_time_total"
    ev = sorted(avg, key=lambda e: getattr(e, key, 0.0), reverse=True)[:n_ops]
    if trace:
        prof.export_chrome_trace(trace)
    calls = {e.key: e.count for e in avg if e.key.startswith(("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy", "hipStreamSynchronize", "hipDeviceSynchronize", "hipEventSynchronize", "hipMemcpy"))}
    return {"ops": [{"name": e.key, "calls": e.count, "self_ms": round(getattr(e, key, 0.0) / 1e3, 3), "cpu_ms": round(e.self_cpu_time_total / 1e3, 3)} for e in ev], "runtime_calls": calls}


def run_learner(kind: str, group: int, net, cfg: dict, batch, device: torch.device, precision: str, algo: str, updates: int, phases: bool, n_ops: int = 0, trace: str | None = None, log=print, off: tuple = ()) -> dict:
    """`updates` + 1 updates of a fresh learner (`new` or `reference`) from `net`'s weights;
    `off`: staging steps of the new learner left out (`OFF`), to measure each alone."""
    from uranium.rl.learner import Learner
    from uranium.rl.reference import ReferenceLearner

    cls = {"new": Learner, "reference": ReferenceLearner}[kind]
    l = cls(copy.deepcopy(cfg), copy.deepcopy(net), copy.deepcopy(net), device, precision, algo, micro_lanes=group)
    l.phases = phases
    for k in off if kind == "new" else ():
        setattr(l, OFF[k], False)
    reset_peak(device)
    res: list[dict] = []
    for step in range(updates + 1):
        sync(device)
        u = time.perf_counter()
        s = l.update(batch, WEIGHTS)
        sync(device)
        wall = time.perf_counter() - u
        res.append({"update": step, "wall_s": round(wall, 4), "stats": {k: s[k] for k in STAT_KEYS if k in s}} | ({"phases": s["phases"]} if "phases" in s else {}))
        log(f"[bench rl] {kind} group {group} update {step}: {wall:.2f}s loss {s['loss']:.6f} kl {s['kl']:.3e}")
    walls = [x["wall_s"] for x in res[1:]]
    out = {
        "learner": kind, "group": group, "off": list(off) if kind == "new" else [], "lanes": batch.lanes, "seq_len": batch.t, "algo": algo, "updates": res,
        "median_s": statistics.median(walls) if walls else None, "mean_s": statistics.fmean(walls) if walls else None,
        "p95_s": sorted(walls)[min(len(walls) - 1, math.ceil(0.95 * len(walls)) - 1)] if walls else None,
        "peak_gib": peak_gib(device), "peak_rss_gib": peak_rss_gib(),
    }
    if walls and phases:
        names = sorted({k for x in res[1:] for k in x.get("phases", {})})
        out["phase_median_s"] = {k: round(statistics.median(x["phases"].get(k, 0.0) for x in res[1:]), 4) for k in names}
    if n_ops:
        l.phases = False
        out |= profile_update(l, batch, device, n_ops, trace)
    del l
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def compare(base: dict, cand: dict) -> dict:
    """Parity of two `bench rl` runs (042 lane-group equivalence, 032-R19 ⑤): per statistic, the
    largest absolute and relative difference over the common updates."""
    ub = {u["update"]: u["stats"] for u in base["updates"]}
    uc = {u["update"]: u["stats"] for u in cand["updates"]}
    steps = sorted(set(ub) & set(uc))
    out: dict = {"updates": len(steps)}
    flat = lambda s, k: [s[k]] if not isinstance(s[k], list) else s[k]  # noqa: E731
    for k in STAT_KEYS:
        diffs = []
        for i in steps:
            if k not in ub[i] or k not in uc[i] or ub[i][k] is None or uc[i][k] is None:
                continue
            for a, b in zip(flat(uc[i], k), flat(ub[i], k)):
                diffs.append((abs(a - b), abs(a - b) / max(abs(b), 1e-8)))
        if diffs:
            out[k] = {"max_abs": max(d[0] for d in diffs), "max_rel": max(d[1] for d in diffs)}
    out["finite"] = all(math.isfinite(uc[i]["loss"]) for i in steps)
    return out


def markdown(runs: list[dict], env: dict, command: str) -> str:
    from uranium.bench.report import fmt

    out = [f"Generated by `{command}` (spec 060-R2, 042 RL learner updates, synthetic segments).", "", "Environment: " + ", ".join(f"{k} {v}" for k, v in env.items()) + ".", ""]
    out += ["Peak GPU: the learner's (net, BC reference, Adam, one update), not the inference servers'. Peak RSS: the peak of the process that ran it up to its end (the synthetic batch and every earlier run of that process included; `--isolate` gives each run its own process). Times with `--phases` synchronise the device at every mark: not for speed comparisons.", ""]
    out += ["| learner | lane group | updates | median s (p95) | mean s | peak GPU GiB | peak RSS GiB | last loss |", "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        if r.get("failed"):
            out.append(f"| {r['learner']} | {r['group']} | – | **failed**: `{r['failed']}` | | | | |")
            continue
        last = r["updates"][-1]["stats"]["loss"] if r["updates"] else None
        name = r["learner"] + (f" (off: {', '.join(r['off'])})" if r.get("off") else "")
        out.append(f"| {name} | {r['group']} | {max(len(r['updates']) - 1, 0)} | {fmt(r['median_s'], '.2f')} ({fmt(r['p95_s'], '.2f')}) | {fmt(r['mean_s'], '.2f')} | {fmt(r['peak_gib'], '.2f')} | {fmt(r['peak_rss_gib'], '.2f')} | {fmt(last, '.6f')} |")
    phased = [r for r in runs if r.get("phase_median_s")]
    if phased:
        names = sorted({k for r in phased for k in r["phase_median_s"]})
        out += ["", "Phase medians, s (device synchronised at each mark):", "", "| learner | group | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        out += [f"| {r['learner']} | {r['group']} | " + " | ".join(fmt(r["phase_median_s"].get(k), ".4f") for k in names) + " |" for r in phased]
    for r in runs:
        if r.get("ops"):
            out += ["", f"Operators by device time, {r['learner']} lane group {r['group']} (one update); runtime calls: {r.get('runtime_calls')}", "", "| op | calls | self ms | cpu ms |", "|---|---|---|---|"]
            out += [f"| `{o['name']}` | {o['calls']} | {o['self_ms']} | {o['cpu_ms']} |" for o in r["ops"]]
    return "\n".join(out) + "\n"
