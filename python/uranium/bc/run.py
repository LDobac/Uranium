import argparse
import functools

EXIT_BUSY = 75
EXIT_YIELDED = 76


def add_parser(sub) -> None:
    p = sub.add_parser("bc", help="behaviour cloning over replay shards")
    p.add_argument("--dataset", action="append", default=[], help="dataset under <URANIUM_SHARED_DIR>/datasets/ with a manifest (pulled or verified)")
    p.add_argument("--shards", action="append", default=[], help="shard folder read as is (samples, tests)")
    out = p.add_mutually_exclusive_group(required=True)
    out.add_argument("--out", help="local checkpoint folder (step_<n>/, last/)")
    out.add_argument("--run", help="run on the shared drive: lock, push, resume from LATEST on any training host")
    p.add_argument("--take-over", action="store_true", help="take the run lock from another host now; only after checking it stopped")
    p.add_argument("--take-over-if", help=argparse.SUPPRESS)
    p.add_argument("--config", default="config/bc.toml")
    p.add_argument("--backend", choices=["cuda", "rocm", "cpu"], help="default: this host's GPU, else cpu")
    p.add_argument("--core", choices=["lstm", "gtrxl"], default="gtrxl")
    p.add_argument("--s2", action="store_true", help="S2 fine-tuning: learning rate x s2_lr_scale; start from --init")
    p.add_argument("--stage", choices=["s1", "s2"], help="game set: default s2 with --s2, else s1; --stage s2 without --s2 = ablation A")
    p.add_argument("--steps", type=int, help="stop after this update (smoke runs); the schedule keeps the stage length")
    p.add_argument("--resume", help="local mode: continue a checkpoint folder (weights, optimizer, step, data position)")
    p.add_argument("--init", help="weights only (S1 -> S2): a checkpoint folder or a finished run folder (DONE)")
    p.add_argument("--maps-dir", help=argparse.SUPPRESS)
    p.add_argument("--synthetic-maps", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(fn=main)


def stage_of(a) -> str:
    if a.s2:
        return "s2"
    return "ablation_a" if a.stage == "s2" else "s1"


def main(a) -> int:
    from uranium import _core

    _core.install_stop_handler("bc")
    if (a.take_over or a.take_over_if) and not a.run:
        raise SystemExit("--take-over needs --run")
    if a.resume and (a.init or a.run):
        raise SystemExit("--resume is for --out without --init (a run resumes from LATEST)")

    from uranium.bc.trainer import BcTrainer
    from uranium.device import configure, resolve
    from uranium.model.config import model_net

    backend, device, precision = resolve(a.backend)
    configure(backend, train=True)
    variant, sizes = model_net(a.config)
    if variant.any():
        print(f"[bc] [model] {variant} (032 §2.7)", flush=True)
    if sizes:
        print(f"[bc] [net] {sizes} (032-R28)", flush=True)
    make = functools.partial(_make, BcTrainer, a.core, device, precision, variant, sizes)
    end = _core.run_bc(
        make, a.config, stage=stage_of(a), out=a.out, run=a.run, take_over=a.take_over, take_over_if=a.take_over_if,
        datasets=a.dataset, shards=a.shards, steps=a.steps, resume=a.resume, init=a.init,
        backend=backend, precision=precision, core=a.core, maps_dir=a.maps_dir, synthetic_maps=a.synthetic_maps,
    )
    return {"busy": EXIT_BUSY, "yielded": EXIT_YIELDED}.get(end, 0)


def _make(cls, core, device, precision, variant, sizes, cfg_json, start, dir, entity_chunk):
    return cls.make(cfg_json, start, None if dir is None else str(dir), entity_chunk, core, device, precision, variant, sizes)
