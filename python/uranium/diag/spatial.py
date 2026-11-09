"""`python -m uranium diag spatial`: dead-channel check (reports/jepa_survey.md 10.4). Per checkpoint,
teacher-forced passes over a few batches of real rows: every ReLU-bearing point's positive fraction
and never-active channel count (`uranium.model.monitor`: spatial stages incl. the bottleneck / s8,
entity FFN, head ResMlp, core FFN), plus the spatial convs' weight sums (a filter whose weights sum
far below zero on a non-negative input is what killed s8)."""

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
    from uranium.model import to_obs
    from uranium.model.monitor import ROWS_KEY, ReluMonitor, stats_to_host

    dirs = [str(p) for p in _core.dataset_dirs(a.dataset, a.shards)]
    backend, device, precision = resolve(a.backend)
    configure(backend)
    cfg = json.loads(_core.bc_config(a.config, "s1")[0])
    T, LANES = cfg["seq_len"], cfg["lanes"]
    G = min(a.lanes_per_group, LANES)
    t0 = time.time()
    stream = _core.EvalStream(a.config, dirs, split=a.split, games=None, seed=1, has_open=False, maps_dir=a.maps_dir, synthetic_maps=a.synthetic_maps)
    data = []  # per batch: per group (hg, real rows, keep) or None
    nb = 0
    while (b := stream.next()) is not None and nb < a.skip_batches + a.batches:
        nb += 1
        steps = b.lane_steps()
        groups = []
        for gi in range(-(-LANES // G)):
            s, e = gi * G, min((gi + 1) * G, LANES)
            if all(steps[l] == 0 for l in range(s, e)):
                groups.append(None)
                continue
            hg = b.group(s, e, T, cfg["pad_entities"], 64)
            m = np.zeros(hg["b"] * T, bool)
            for li in range(hg["b"]):
                m[li * T:li * T + steps[s + li]] = True
            groups.append((hg, m, np.asarray(hg["keep"]).astype(bool)))
        data.append(groups)
    stream.finish()
    print(f"[data] {len(data)} batches ({a.skip_batches} warm-up) in {time.time() - t0:.1f}s", flush=True)
    res = {"data": [redact(d) for d in dirs], "split": a.split, "batches": a.batches, "skip_batches": a.skip_batches, "backend": backend,
           "precision": "f32" if a.precision == "f32" else precision, "checkpoints": {}}
    for spec in a.checkpoint:
        cdir = Path(_core.resolve_checkpoint(spec))
        net, _ = ckpt.load_model(cdir, device)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        states = [None] * len(data[0])
        mon = ReluMonitor(net, inner=not a.no_inner)
        rows = 0
        for bi, groups in enumerate(data):
            record = bi >= a.skip_batches
            if record and not mon.handles:
                mon.__enter__()
            for gi, g in enumerate(groups):
                if g is None:
                    states[gi] = None
                    continue
                hg, m, keep = g
                obs = to_obs(hg, device)
                obs[ROWS_KEY] = torch.as_tensor(m, device=device)
                st = states[gi]
                s0 = net.init_state(hg["b"], device) if st is None else net.mask_state(st, torch.as_tensor(keep, device=device))
                ctx = autocast(device, precision) if a.precision == "auto" else torch.autocast(device.type, enabled=False)
                with torch.no_grad(), ctx:
                    out = net.forward_seq(obs, net.labels(hg, device), torch.as_tensor(hg["privileged"]).to(device, torch.float32), s0, T)
                states[gi] = out["state"]
                if record:
                    rows += int(m.sum())
        st = stats_to_host(mon.stats())
        mon.close()
        res["checkpoints"][redact(cdir)] = {"sha256": ckpt.policy_version(cdir), "rows": rows, "points": st, "weights": weight_stats(net)}
        s8 = st.get("spatial.s8", {})
        print(f"{redact(cdir)}: s8 pos_frac {s8.get('pos_frac', float('nan')):.6f} never_active {s8.get('never_active')}/{s8.get('channels')} rows {rows}", flush=True)
        for k, v in st.items():
            print(f"  {k:28} pos {v['pos_frac']:.4f} never {v['never_active']}/{v['channels']}")
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[out] {a.out}")
    return 0
