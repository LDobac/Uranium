import copy
import json
import math
import statistics
import time

import numpy as np
import torch

from uranium.bench.inputs import random_obs
from uranium.bench.timing import peak_gib, peak_rss_gib, reset_peak, sync
from uranium.model.net import ENTITY_KEYS, LABEL_KEYS, OBS_KEYS
from uranium.model.state import leaves

WEIGHTS = [1.0, 0.3, 0.3, 0.05]
STAT_KEYS = ("loss", "policy", "value", "kl", "win_open", "rho_mean", "adv_mean", "grad_norm", "kl_heads", "entropy_heads", "aux_priv", "aux_jepa")
PAD = 64
OFF = {"pinned": "pinned", "spatial_bf16": "spatial_bf16"}


class SyntheticSegments:
    def __init__(self, net, lanes: int, t: int, entities: tuple[int, int] = (64, 192), templates: int = 8, seed: int = 1, device=None):
        from uranium import _core

        device = torch.device(device or next(net.parameters()).device)
        self.lanes, self.t = lanes, t
        max_select = net.cfg.heads.max_select
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

    @property
    def real_steps(self) -> list[int]:
        return [self.t] * self.lanes

    @property
    def priv_trained(self) -> list[int]:
        return self.trained

    @property
    def ref_races(self) -> list[int]:
        return [i % 4 for i in range(self.lanes)]

    def group(self, start: int, end: int, t: int, max_select: int) -> dict:
        assert 0 <= start < end <= self.lanes, (start, end)
        return self.group_lanes(list(range(start, end)), t, max_select)

    def group_lanes(self, idx: list, t: int, max_select: int) -> dict:
        assert t == self.t and idx and all(0 <= i < self.lanes for i in idx), (idx, t)
        lanes = [self.lane[i] for i in idx]
        n = max(self.blocks[x["block"]][0] for x in lanes)
        cols: dict = {k: [] for k in (*OBS_KEYS, *LABEL_KEYS, "privileged", "behaviour")}
        for x in lanes:
            n_b, block = self.blocks[x["block"]]
            for k in cols:
                v = block[k]
                if k in ENTITY_KEYS and n_b < n:
                    v = np.pad(v, [(0, 0), (0, n - n_b)] + [(0, 0)] * (v.ndim - 2))
                elif k == "selected":
                    v = np.where(v >= n_b, n, v)
                cols[k].append(v)
        g = {k: np.concatenate(v).astype(self.dtypes.get(k, v[0].dtype), copy=False) for k, v in cols.items()}
        g["mask"] = np.concatenate([(np.arange(t) < x["trained"]).astype(np.float32) for x in lanes]).astype(self.dtypes["mask"])
        g["rewards"] = np.concatenate([x["rewards"] for x in lanes]).astype(self.dtypes["rewards"])
        g["terminal"] = np.array([x["terminal"] for x in lanes], dtype=self.dtypes["terminal"])
        g["trained"] = np.array([x["trained"] for x in lanes], dtype=self.dtypes["trained"])
        g["versions"] = np.zeros(len(lanes), dtype=self.dtypes["versions"])
        g["r_value_policy"] = np.ones(len(lanes), dtype=self.dtypes["r_value_policy"])
        g["pseudo_scale"] = np.ones((len(lanes), 2), dtype=self.dtypes["pseudo_scale"])
        g["sil_return"] = np.full(len(lanes), np.nan, dtype=self.dtypes["sil_return"])
        g["ref_race"] = np.array([i % 4 for i in idx], dtype=self.dtypes["ref_race"])
        g["real"] = np.ones(len(lanes) * t, dtype=self.dtypes["real"])
        g["priv_present"] = np.ones(len(lanes) * t, dtype=self.dtypes["priv_present"])
        g["state0"] = [x["state0"] for x in lanes]
        g |= {"b": len(lanes), "t": t, "n": n}
        return g


def profile_update(learner, batch, device: torch.device, n_ops: int, trace: str | None = None) -> dict:
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


def race_refs(references: int) -> tuple[int, list | None]:
    if references <= 1:
        return 0, None
    k = min(references, 4)
    return k - 1, [i + 1 if i + 1 < k else 0 for i in range(4)]


def run_learner(kind: str, group: int, net, cfg: dict, batch, device: torch.device, precision: str, algo: str, updates: int, phases: bool, n_ops: int = 0, trace: str | None = None, log=print, off: tuple = (), trim: bool = False, references: int = 1) -> dict:
    from uranium.rl.learner import Learner
    from uranium.rl.reference import ReferenceLearner

    cls = {"new": Learner, "reference": ReferenceLearner}[kind]
    extra, race_ref = race_refs(references if kind == "new" else 1)
    refs = {"refs": [copy.deepcopy(net) for _ in range(extra)], "race_ref": race_ref} if race_ref is not None else {}
    l = cls(copy.deepcopy(cfg), copy.deepcopy(net), copy.deepcopy(net), device, precision, algo, micro_lanes=group, **refs)
    l.phases = phases
    for k in off if kind == "new" else ():
        setattr(l, OFF[k], False)
    if kind == "new":
        l.trim_pad = trim
    reset_peak(device)
    res: list[dict] = []
    frames = compiled_frames if kind == "new" and l.compiling else None
    base = frames() if frames else 0
    for step in range(updates + 1):
        sync(device)
        u = time.perf_counter()
        s = l.update(batch, WEIGHTS)
        sync(device)
        wall = time.perf_counter() - u
        res.append({"update": step, "wall_s": round(wall, 4), "stats": {k: s[k] for k in STAT_KEYS if k in s}} | ({"phases": s["phases"]} if "phases" in s else {}) | ({"frames": frames() - base} if frames else {}))
        log(f"[bench rl] {kind} group {group} update {step}: {wall:.2f}s loss {s['loss']:.6f} kl {s['kl']:.3e}")
    walls = [x["wall_s"] for x in res[1:]]
    out = {
        "learner": kind, "group": group, "off": list(off) if kind == "new" else [], "trim": trim and kind == "new", "references": 1 + extra, "lanes": batch.lanes, "seq_len": batch.t, "algo": algo, "updates": res,
        "median_s": statistics.median(walls) if walls else None, "mean_s": statistics.fmean(walls) if walls else None,
        "p95_s": sorted(walls)[min(len(walls) - 1, math.ceil(0.95 * len(walls)) - 1)] if walls else None,
        "peak_gib": peak_gib(device), "peak_rss_gib": peak_rss_gib(),
    }
    if frames:
        out["compile"] = {"first_s": res[0]["wall_s"], "frames_first": res[0]["frames"], "frames_later": res[-1]["frames"] - res[0]["frames"],
                          "later_at": [x["update"] for a, x in zip(res, res[1:]) if x["frames"] > a["frames"]], "graphs": l.graphs}
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


def compiled_frames() -> int:
    import torch._dynamo.utils as du

    return du.counters["frames"]["ok"]


def compare(base: dict, cand: dict) -> dict:
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
        name = r["learner"] + (f" (off: {', '.join(r['off'])})" if r.get("off") else "") + (" (trim)" if r.get("trim") else "") + (f" ({r['references']} KL references)" if r.get("references", 1) > 1 else "")
        out.append(f"| {name} | {r['group']} | {max(len(r['updates']) - 1, 0)} | {fmt(r['median_s'], '.2f')} ({fmt(r['p95_s'], '.2f')}) | {fmt(r['mean_s'], '.2f')} | {fmt(r['peak_gib'], '.2f')} | {fmt(r['peak_rss_gib'], '.2f')} | {fmt(last, '.6f')} |")
    for r in runs:
        c = r.get("compile")
        if c:
            out += ["", f"torch.compile, {r['learner']} lane group {r['group']}: first update {c['first_s']:.1f} s with {c['frames_first']} compiled frames, {c['frames_later']} more in later updates {c['later_at']}" + (", CUDA graphs" if c["graphs"] else "") + "."]
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
