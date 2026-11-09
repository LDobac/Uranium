"""`python -m uranium diag spatial`: dead-channel check (032-R25, reports/jepa_survey.md 10.4). Per
checkpoint, teacher-forced passes over a few batches of real steps: every ReLU-bearing point's
positive fraction and never-active channel count (`uranium.model.monitor` with `points="all"`:
spatial stages incl. the bottleneck / s8, entity FFN, head ResMlp, core FFN), the near-constant S8
channels and s_pool dimensions (`const`, std below `--const-threshold`), and the spatial convs'
weight sums (a filter whose weights sum far below zero on a non-negative input is what killed s8).
The same games for every checkpoint (a fresh stream each)."""

import json
import time
from pathlib import Path

import numpy as np


def add_args(p) -> None:
    p.add_argument("--checkpoint", action="append", required=True, help="checkpoint folder (repeatable)")
    p.add_argument("--dataset", action="append", default=[])
    p.add_argument("--shards", action="append", default=[])
    p.add_argument("--split", default="val")
    p.add_argument("--config", default="config/bc.toml")
    p.add_argument("--backend", choices=["cuda", "rocm", "cpu"])
    p.add_argument("--precision", choices=["auto", "f32"], default="auto", help="auto = the backend's (032-R8); f32 = no autocast")
    p.add_argument("--batches", type=int, default=2, help="batches recorded")
    p.add_argument("--skip-batches", type=int, default=2, help="batches run first without recording (recurrent state warm-up)")
    p.add_argument("--lanes-per-group", type=int, default=8)
    p.add_argument("--no-inner", action="store_true", help="skip the ResBlocks' inner ReLU (one conv recomputed per block)")
    p.add_argument("--const-threshold", type=float, default=1e-4, help="032-R25 ③ const: std below this (AMB-041-14)")
    p.add_argument("--out", required=True, help="output JSON")
    p.add_argument("--maps-dir", help="map folder (else from .env)")
    p.add_argument("--synthetic-maps", action="store_true", help="tests: synthetic maps")


def weight_stats(net) -> dict:
    sp = net.spatial
    named = [("stem", sp.stem)]
    for i, d in enumerate(sp.downs):
        named += [(f"down{i}.conv", d.conv), (f"down{i}.res.c1", d.res.c1), (f"down{i}.res.c2", d.res.c2)]
    for j, r in enumerate(sp.bottleneck):
        named += [(f"bottleneck{j}.c1", r.c1), (f"bottleneck{j}.c2", r.c2)]
    out = {}
    for name, m in named:
        w, b = m.weight.detach().float(), m.bias.detach().float()
        s = w.flatten(1).sum(1)
        out[name] = {"w_absmean": float(w.abs().mean()), "w_sum_mean": float(s.mean()), "w_sum_frac_neg": float((s < 0).float().mean()), "b_mean": float(b.mean())}
    return out


def run(a) -> int:
    import torch

    from uranium import _core, ckpt
    from uranium.device import autocast, configure, resolve
    from uranium.diag.dump import redact
    from uranium.diag.stream import lane_groups, real_rows
    from uranium.model import to_obs
    from uranium.model.monitor import ROWS_KEY, ReluMonitor, stats_to_host

    dirs = [str(p) for p in _core.dataset_dirs(a.dataset, a.shards)]
    backend, device, precision = resolve(a.backend)
    configure(backend)
    torch.backends.cudnn.benchmark = False  # a short read-only pass: no per-shape search
    cfg = json.loads(_core.bc_config(a.config, "s1")[0])
    T, LANES = cfg["seq_len"], cfg["lanes"]
    G = min(a.lanes_per_group, LANES)
    prec = "f32" if a.precision == "f32" else precision
    res = {"data": [redact(d) for d in dirs], "split": a.split, "batches": a.batches, "skip_batches": a.skip_batches, "backend": backend,
           "precision": prec, "const_threshold": a.const_threshold, "checkpoints": {}}
    for spec in a.checkpoint:
        t0 = time.time()
        cdir = Path(_core.resolve_checkpoint(spec))
        net, _ = ckpt.load_model(cdir, device)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        # the same games for every checkpoint: a fresh stream each, one lane group in memory at a time
        stream = _core.EvalStream(a.config, dirs, split=a.split, games=None, seed=1, has_open=False, maps_dir=a.maps_dir, synthetic_maps=a.synthetic_maps)
        states = [None] * -(-LANES // G)
        rows = 0
        recording = False
        with ReluMonitor(net, inner=not a.no_inner, points="all", threshold=a.const_threshold) as mon:
            for bi, b, gi, s, steps, hg in lane_groups(stream, LANES, G, T, cfg["pad_entities"], net.cfg.heads.max_select, states, a.skip_batches + a.batches):
                if bi >= a.skip_batches and not recording:
                    mon.reset()  # the warm-up batches only carry the recurrent state
                    recording = True
                m = real_rows(hg, steps, s, T)
                keep = np.asarray(hg["keep"]).astype(bool)
                obs = to_obs(hg, device)
                obs[ROWS_KEY] = torch.as_tensor(m, device=device)
                st = states[gi]
                s0 = net.init_state(hg["b"], device) if st is None else net.mask_state(st, torch.as_tensor(keep, device=device))
                ctx = autocast(device, precision) if a.precision == "auto" else torch.autocast(device.type, enabled=False)
                with torch.no_grad(), ctx:
                    out = net.forward_seq(obs, net.labels(hg, device), torch.as_tensor(hg["privileged"]).to(device, torch.float32), s0, T)
                states[gi] = out["state"]
                del out, obs
                if recording:
                    rows += int(m.sum())
            stream.finish()
            if not recording:
                raise SystemExit(f"{spec}: fewer than {a.skip_batches + 1} batches in the split")
            st = stats_to_host(mon.stats())
        print(f"[{a.skip_batches} warm-up + {a.batches} batches] {time.time() - t0:.1f}s", flush=True)
        res["checkpoints"][redact(cdir)] = {"sha256": ckpt.policy_version(cdir), "rows": rows, "precision": prec, "points": st, "weights": weight_stats(net)}
        s8 = st["spatial.s8"]
        c8, cp = st["spatial.s8.const"], st["spatial.pool.const"]
        print(f"{redact(cdir)}: s8 pos_frac {s8['pos_frac']:.6f} never_active {s8['never_active']}/{s8['channels']} | const s8 {c8['const']}/{c8['channels']} "
              f"pool {cp['const']}/{cp['channels']} (std < {a.const_threshold:g}, {c8['dtype']}) rows {rows}", flush=True)
        for k, v in st.items():
            if "pos_frac" in v:
                print(f"  {k:28} pos {v['pos_frac']:.4f} never {v['never_active']}/{v['channels']}")
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[out] {a.out}")
    return 0
