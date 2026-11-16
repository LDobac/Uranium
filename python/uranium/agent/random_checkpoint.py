from pathlib import Path


def add_parser(sub) -> None:
    k = sub.add_parser("random-checkpoint", help="a randomly initialised checkpoint (smoke tests before a BC checkpoint)")
    k.add_argument("--out", required=True)
    k.add_argument("--core", choices=["lstm", "gtrxl"], default="gtrxl")
    k.add_argument("--win-open", action="store_true", help="include the win_open head")
    k.add_argument("--seed", type=int, default=0)
    k.set_defaults(fn=main)


def main(a) -> int:
    import torch

    from uranium import ckpt
    from uranium.model import UraniumNet
    from uranium.model.config import new_net_config

    torch.manual_seed(a.seed)
    cfg = new_net_config(a.core, a.win_open)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    net = UraniumNet(cfg)
    ckpt.save_model(out, net)
    ckpt.save_meta(out, cfg, {"random_init": True, "seed": a.seed}, "f32")
    print(f"{out.name}@{ckpt.policy_version(out)[:12]}")
    return 0
