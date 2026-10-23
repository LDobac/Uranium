"""BC parity (AMB-000-4 step 1-6, temporary, removed with Burn).

  convert <export> <out> [--dropout X]   Burn weights (`burn-parity --export`) -> a PyTorch
                                         checkpoint for `python -m uranium bc --init <out>`
  compare <burn.log> <torch.log>         per-update loss of the two trainers, side by side

Both trainers start from the same weights (`--init`: step 0, same schedule) and read the same
batches (the Rust feeder), so with dropout 0 their losses differ only by float rounding, which
grows slowly over the updates."""

import argparse
import json
import math
from pathlib import Path


def convert(a) -> None:
    import torch
    from safetensors.torch import load_file

    from uranium import ckpt
    from uranium.model import UraniumNet, UraniumNetConfig
    from uranium.model.burn import burn_to_torch

    meta = json.loads((Path(a.export) / "meta.json").read_text())
    cfg = UraniumNetConfig.from_dict(meta["config"])
    if a.dropout is not None:
        cfg.entity.dropout = a.dropout
    net = UraniumNet(cfg)
    n = sum(p.numel() for p in net.parameters())
    assert n == meta["num_params"], f"parameter count {n} != Burn {meta['num_params']}"
    net.load_state_dict(burn_to_torch(net, load_file(str(Path(a.export) / "weights.safetensors"))))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt.save_model(out, net)
    ckpt.save_meta(out, cfg, {"burn_export": str(a.export)}, "f32")
    print(f"{n} params -> {out} (dropout {cfg.entity.dropout})")


def losses(path: str) -> dict[int, dict]:
    out = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not (line.startswith("{") and '"step"' in line and '"loss"' in line):
            continue
        d = json.loads(line)
        out[d["step"]] = d["loss"]
    return out


def compare(a) -> None:
    b, t = losses(a.burn), losses(a.torch)
    steps = sorted(set(b) & set(t))
    print(f"updates: burn {len(b)}, torch {len(t)}, common {len(steps)}")
    keys = ["total", "action", "delay", "queued", "selected", "target_unit", "target_loc", "value"]
    rel = lambda x, y: abs(x - y) / max(abs(y), 1e-6)  # noqa: E731
    for s in steps[:5]:
        print(s, {k: f"{t[s][k]:.5f}/{b[s][k]:.5f}" for k in keys[:3]}, f"rel {rel(t[s]['total'], b[s]['total']):.2e}")
    w = a.window
    print(f"window means of total (torch / burn), window {w}:")
    worst = 0.0
    for i in range(0, len(steps), w):
        win = steps[i:i + w]
        mt = sum(t[s]["total"] for s in win) / len(win)
        mb = sum(b[s]["total"] for s in win) / len(win)
        r = rel(mt, mb)
        worst = max(worst, r)
        print(f"  {win[0]:5d}-{win[-1]:5d}  {mt:.5f}  {mb:.5f}  rel {r:.2e}")
    first = [rel(t[s]["total"], b[s]["total"]) for s in steps[: a.first]]
    print(json.dumps({
        "first_updates": a.first, "first_max_rel": max(first) if first else math.nan,
        "window_max_rel": worst, "finite": all(math.isfinite(t[s]["total"]) for s in steps),
    }))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("export")
    c.add_argument("out")
    c.add_argument("--dropout", type=float)
    c.set_defaults(fn=convert)
    m = sub.add_parser("compare")
    m.add_argument("burn")
    m.add_argument("torch")
    m.add_argument("--window", type=int, default=50)
    m.add_argument("--first", type=int, default=10)
    m.set_defaults(fn=compare)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
