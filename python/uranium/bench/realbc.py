"""Real-data BC updates for the bench suite (060-R2, 041): the same initial weights (seeded) and
the same batches (a fresh Rust feeder) for every candidate, so the update times compare and the
per-update losses are a parity curve (`bench loss`: a candidate must keep the curve of the
baseline within noise, AMB-000-4 step 2). Lane group sizes (041-R3, AMB-041-7) are swept in one
run. The first update warms kernels up and is not counted in the medians."""

import copy
import json
import math
import os
import statistics
import time

import torch

from uranium.bench.timing import peak_gib, peak_rss_gib, reset_peak

LOSS_KEYS = ("total", "action", "delay", "queued", "selected", "target_unit", "target_loc", "value", "win_open")


def profile_update(trainer, batch, lr: float, step: int, device: torch.device, n_ops: int, trace: str | None = None) -> list[dict]:
    """The operators with the most device time (CPU time on CPU) in one update."""
    acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with torch.profiler.profile(activities=acts) as prof:
        trainer.train_batch(batch, lr, step)
    key = "self_device_time_total" if device.type == "cuda" else "self_cpu_time_total"
    ev = sorted(prof.key_averages(), key=lambda e: getattr(e, key, 0.0), reverse=True)[:n_ops]
    if trace:
        prof.export_chrome_trace(trace)
    return [{"name": e.key, "calls": e.count, "self_ms": round(getattr(e, key, 0.0) / 1e3, 3), "cpu_ms": round(e.self_cpu_time_total / 1e3, 3)} for e in ev]


def run_group(a, group: int, device: torch.device, precision: str, log=print) -> dict:
    from uranium import _core
    from uranium.bc.trainer import BcTrainer, new_net_config
    from uranium.model import UraniumNet

    cfg = copy.deepcopy(json.loads(_core.bc_config(a.config, a.stage)[0]))
    cfg["micro_lanes"] = group
    dirs = _core.dataset_dirs(a.dataset, a.shards)
    feeder = _core.Feeder(a.config, dirs, stage=a.stage, synthetic_maps=a.synthetic_maps)
    torch.manual_seed(a.seed)
    net = UraniumNet(new_net_config(a.core, cfg["value"].get("win_open") is not None))
    chunk = int(os.environ.get("URANIUM_BC_ENTITY_CHUNK", "0") or 0)
    t = BcTrainer(cfg, net, device, precision, chunk)
    reset_peak(device)
    updates = []
    for step in range(a.updates + 1):
        w = time.perf_counter()
        b = feeder.next()
        if b is None:
            break
        wait = time.perf_counter() - w
        u = time.perf_counter()
        out = json.loads(t.train_batch(b, _core.lr_at(a.config, a.stage, step), step))
        wall = time.perf_counter() - u
        updates.append({"step": step, "wall_s": round(wall, 4), "wait_s": round(wait, 4), "loss": {k: out[k] for k in LOSS_KEYS if k in out}} | ({"phases": out["phases"]} if "phases" in out else {}))
        log(f"[bench bc] group {group} update {step}: {wall:.2f}s (wait {wait:.2f}s) loss {out['total']:.5f}")
    walls = [x["wall_s"] for x in updates[1:]]
    res = {
        "group": group, "lanes": cfg["lanes"], "seq_len": cfg["seq_len"], "entity_chunk": chunk, "updates": updates,
        "median_s": statistics.median(walls) if walls else None, "mean_s": statistics.fmean(walls) if walls else None,
        "p95_s": sorted(walls)[min(len(walls) - 1, math.ceil(0.95 * len(walls)) - 1)] if walls else None,
        "wait_median_s": statistics.median(x["wait_s"] for x in updates[1:]) if walls else None,
        "peak_gib": peak_gib(device), "peak_rss_gib": peak_rss_gib(),
    }
    if a.ops:
        b = feeder.next()
        if b is not None:
            res["ops"] = profile_update(t, b, _core.lr_at(a.config, a.stage, a.updates + 1), a.updates + 1, device, a.ops, a.trace)
    del t, net, feeder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def compare_losses(base: dict, cand: dict, window: int = 20) -> dict:
    """Parity of two `bench bc` runs of one group size: per-update and window-mean relative
    differences of the total loss (the 1-6 comparison, AMB-000-4)."""
    lb = {u["step"]: u["loss"]["total"] for u in base["updates"]}
    lc = {u["step"]: u["loss"]["total"] for u in cand["updates"]}
    steps = sorted(set(lb) & set(lc))
    rel = lambda x, y: abs(x - y) / max(abs(y), 1e-6)  # noqa: E731
    wins = []
    for i in range(0, len(steps), window):
        s = steps[i:i + window]
        mb, mc = statistics.fmean(lb[k] for k in s), statistics.fmean(lc[k] for k in s)
        wins.append({"from": s[0], "to": s[-1], "base": mb, "cand": mc, "rel": rel(mc, mb)})
    return {
        "updates": len(steps), "first_rel": rel(lc[steps[0]], lb[steps[0]]) if steps else None,
        "max_rel": max((rel(lc[k], lb[k]) for k in steps), default=None), "window_max_rel": max((w["rel"] for w in wins), default=None),
        "windows": wins, "finite": all(math.isfinite(lc[k]) for k in steps),
    }
