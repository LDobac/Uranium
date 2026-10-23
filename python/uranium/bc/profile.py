"""`python -m uranium bc-profile` (060-AC4): where a BC update's time goes, on real S1 batches with
a fresh network. Per update: data wait, collate, host-to-device copy, forward, loss, backward,
optimizer (device synchronised at each mark), then the operators with the most device time from
`torch.profiler`. The first batch only warms kernels and autotuning up and is not counted."""

import argparse
import json
import time


def add_parser(sub) -> None:
    p = sub.add_parser("bc-profile", help="time per phase and operator of BC updates (060-AC4)")
    p.add_argument("--dataset", action="append", default=[])
    p.add_argument("--shards", action="append", default=[])
    p.add_argument("--batches", type=int, default=3, help="measured updates after one warm-up update")
    p.add_argument("--backend", choices=["cuda", "rocm", "cpu"])
    p.add_argument("--config", default="config/bc.toml")
    p.add_argument("--core", choices=["lstm", "gtrxl"], default="gtrxl")
    p.add_argument("--ops", type=int, default=25, help="operators listed (0 = no torch.profiler pass)")
    p.add_argument("--trace", help="also write a Chrome trace of the profiled update here")
    p.add_argument("--synthetic-maps", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(fn=main)


def main(a) -> int:
    import os

    import torch

    from uranium import _core
    from uranium.bc.trainer import BcTrainer, new_net_config
    from uranium.bench.realbc import profile_update
    from uranium.device import configure, resolve
    from uranium.model import UraniumNet

    os.environ["URANIUM_BC_PROFILE"] = "1"
    backend, device, precision = resolve(a.backend)
    configure(backend)
    cfg_json, _ = _core.bc_config(a.config, "s1")
    cfg = json.loads(cfg_json)
    dirs = _core.dataset_dirs(a.dataset, a.shards)
    feeder = _core.Feeder(a.config, dirs, stage="s1", synthetic_maps=a.synthetic_maps)
    net = UraniumNet(new_net_config(a.core, cfg["value"].get("win_open") is not None))
    chunk = int(os.environ.get("URANIUM_BC_ENTITY_CHUNK", "0") or 0)
    t = BcTrainer(cfg, net, device, precision, chunk)
    rows = []
    for step in range(a.batches + 1):
        w = time.perf_counter()
        b = feeder.next()
        if b is None:
            break
        wait = time.perf_counter() - w
        out = json.loads(t.train_batch(b, cfg["lr"], step))
        if step > 0:
            rows.append({"wait": round(wait, 4), **out["phases"]})
    keys = sorted({k for r in rows for k in r})
    mean = {k: round(sum(r.get(k, 0.0) for r in rows) / max(len(rows), 1), 4) for k in keys}
    report = {"backend": backend, "precision": precision, "core": a.core, "entity_chunk": chunk, "updates": rows, "mean": mean}
    if device.type == "cuda":
        report["max_memory_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
    if a.ops:
        b = feeder.next()
        if b is not None:
            report["ops"] = profile_update(t, b, cfg["lr"], a.batches + 1, device, a.ops, a.trace)
    print(json.dumps(report, indent=2))
    return 0
