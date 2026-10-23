"""`python -m uranium <command>` (001-R9: reads `.env` first; process variables win)."""

import argparse
import json
import sys

from uranium import env


def feed_digest(a: argparse.Namespace) -> int:
    """Same output as `uranium-train feed-digest`, hashed here from the `_core` numpy arrays."""
    from uranium import _core
    from uranium.data import group_digest, groups

    cfg = json.loads(_core.bc_config(a.config, a.stage)[0])
    feeder = _core.Feeder(a.config, a.shards, stage=a.stage, prefetch=0, pack=a.pack, synthetic_maps=a.synthetic_maps, check_maps=False)
    for k in range(a.batches):
        b = feeder.next()
        if b is None:
            break
        digests = [group_digest(g) for g in groups(b, a.group, cfg["seq_len"], cfg["pad_entities"], a.max_select)]
        print(json.dumps({"batch": k, "valid": b.valid, "value_valid": b.value_valid, "groups": digests, "state": json.loads(b.state_json())}))
    return 0


def main(argv: list[str] | None = None) -> int:
    env.load()
    p = argparse.ArgumentParser(prog="python -m uranium", description="Uranium training and inference (PyTorch)")
    sub = p.add_subparsers(dest="cmd", required=True)
    fd = sub.add_parser("feed-digest", help="AMB-000-4 check: SHA-256 per lane group of the first BC batches")
    fd.add_argument("--config", default="config/bc.toml")
    fd.add_argument("--shards", action="append", required=True)
    fd.add_argument("--stage", default="s1", choices=["s1", "s2"])
    fd.add_argument("--batches", type=int, default=3)
    fd.add_argument("--group", type=int, default=1)
    fd.add_argument("--max-select", type=int, default=12)
    fd.add_argument("--pack", action="store_true")
    fd.add_argument("--synthetic-maps", action="store_true")
    fd.set_defaults(fn=feed_digest)
    from uranium.bc import eval as bc_eval
    from uranium.bc import profile as bc_profile
    from uranium.bc import run as bc_run

    bc_run.add_parser(sub)
    bc_eval.add_parser(sub)
    bc_profile.add_parser(sub)
    from uranium.agent import serve as agent_serve

    agent_serve.add_parsers(sub)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
