"""`python -m uranium bench` (060-R2): `suite` (kernel / model / inference cases with errors vs
CPU), `bc` (real-data BC updates per lane group size, parity log), `loss` (parity of two `bc`
logs), `rl` (RL learner updates per lane group size and learner, 042) and `rl-parity` (parity of
two `rl` runs). Candidates of an optimisation are compared by running these under each candidate's setting
and comparing the JSON (AMB-000-4 step 2)."""

import json
import sys
from pathlib import Path


def add_parser(sub) -> None:
    p = sub.add_parser("bench", help="bench suite (060-R2)")
    bsub = p.add_subparsers(dest="bench_cmd", required=True)

    s = bsub.add_parser("suite", help="kernel, model and inference cases")
    s.add_argument("--suite", default="kernels,model", help="comma list of kernels, model, infer, variants")
    s.add_argument("--filter", help="regex on case names")
    s.add_argument("--backend", choices=["cuda", "rocm", "cpu"])
    s.add_argument("--precision", choices=["f32", "bf16"], help="default: the backend's training precision (032-R8)")
    s.add_argument("--no-accuracy", action="store_true", help="skip the CPU reference runs")
    s.add_argument("--budget", type=float, default=10.0, help="seconds per timed loop at most")
    s.add_argument("--out", help="markdown report")
    s.add_argument("--json", help="raw results")
    s.set_defaults(fn=suite_main)

    b = bsub.add_parser("bc", help="real-data BC updates per lane group size")
    b.add_argument("--dataset", action="append", default=[])
    b.add_argument("--shards", action="append", default=[])
    b.add_argument("--config", default="config/bc.toml")
    b.add_argument("--stage", default="s1", choices=["s1", "s2", "ablation_a"])
    b.add_argument("--core", choices=["lstm", "gtrxl"], default="gtrxl")
    b.add_argument("--backend", choices=["cuda", "rocm", "cpu"])
    b.add_argument("--precision", choices=["f32", "bf16"], help="default: the backend's training precision (032-R8); f32 on CUDA checks an implementation without bf16 rounding")
    b.add_argument("--groups", default="1", help="comma list of lane group sizes (micro_lanes, 0 = all lanes)")
    b.add_argument("--updates", type=int, default=20, help="measured updates after one warm-up update")
    b.add_argument("--seed", type=int, default=1, help="initial weights")
    b.add_argument("--phases", action="store_true", help="per-phase times (device synchronised at each mark)")
    b.add_argument("--ops", type=int, default=0, help="operators listed from one extra profiled update")
    b.add_argument("--trace", help="Chrome trace of the profiled update")
    b.add_argument("--out", help="markdown report")
    b.add_argument("--json", help="raw results (the parity log for `bench loss`)")
    b.add_argument("--synthetic-maps", action="store_true", help="tests only")
    b.set_defaults(fn=bc_main)

    r = bsub.add_parser("rl", help="RL learner updates on a fixed synthetic batch per lane group size (042, 032-R19)")
    r.add_argument("--checkpoint", required=True, help="BC checkpoint directory (weights; dropout 0 as in RL)")
    r.add_argument("--config", default="config/rl.toml")
    r.add_argument("--algo", choices=["vtrace", "ppo"], default="vtrace")
    r.add_argument("--backend", choices=["cuda", "rocm", "cpu"])
    r.add_argument("--learner", default="new", help="comma list of new, reference (the learner before the 032-R19 changes)")
    r.add_argument("--groups", default="1", help="comma list of lane group sizes (micro_lanes, 0 = all lanes)")
    r.add_argument("--off", default="", help="comma list of the new learner's staging steps left out: trim, pinned, spatial_bf16 (each measured alone; pinned also turns the CUDA read-back into the plain synchronous copy)")
    r.add_argument("--updates", type=int, default=10, help="measured updates after one warm-up update")
    r.add_argument("--lanes", type=int, help="default: the config's lanes")
    r.add_argument("--entities", default="64:192", help="lo:hi entities per synthetic step block")
    r.add_argument("--templates", type=int, default=8, help="distinct step blocks the lanes cycle through (host memory)")
    r.add_argument("--seed", type=int, default=1, help="synthetic batch")
    r.add_argument("--phases", action="store_true", help="per-phase times (device synchronised at each mark)")
    r.add_argument("--ops", type=int, default=0, help="operators listed from one extra profiled update per run")
    r.add_argument("--trace", help="Chrome trace of the profiled update (the last run's)")
    r.add_argument("--out", help="markdown report")
    r.add_argument("--json", help="raw results (the parity log for `bench rl-parity`)")
    r.add_argument("--isolate", action="store_true", help="each learner x group in its own process, so its peak RSS (060-R3) is its own, not the process peak so far")
    r.set_defaults(fn=rl_main)

    rp = bsub.add_parser("rl-parity", help="parity of two `bench rl` runs (042 lane-group equivalence)")
    rp.add_argument("base", help="JSON of `bench rl`")
    rp.add_argument("cand", nargs="?", help="second JSON (default: the base file)")
    rp.add_argument("--base-run", default="reference:1", help="learner:group of the base")
    rp.add_argument("--cand-run", default="new:1", help="learner:group of the candidate")
    rp.set_defaults(fn=rl_parity_main)

    lo = bsub.add_parser("loss", help="parity of two `bench bc` logs (same group size)")
    lo.add_argument("base")
    lo.add_argument("cand")
    lo.add_argument("--group", type=int, help="group size to compare (default: the first of each)")
    lo.add_argument("--window", type=int, default=20)
    lo.set_defaults(fn=loss_main)


def write(path: str | None, text: str) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")


def emit(text: str) -> None:
    """The report on stdout in UTF-8 (a Windows console code page cannot encode the table)."""
    sys.stdout.flush()
    sys.stdout.buffer.write(text.encode("utf-8"))
    sys.stdout.buffer.flush()


def suite_main(a) -> int:
    from uranium.bench.cases import Ctx, run_suites
    from uranium.bench.report import environment, markdown
    from uranium.device import configure, resolve

    backend, device, precision = resolve(a.backend)
    configure(backend)
    precision = a.precision or precision
    ctx = Ctx(device, precision, not a.no_accuracy, a.budget)
    rows = run_suites([s.strip() for s in a.suite.split(",") if s.strip()], ctx, a.filter, log=lambda m: print(m, file=sys.stderr, flush=True))
    env = environment(backend, precision, device)
    cmd = "python -m uranium bench " + " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "python -m uranium bench suite"
    md = markdown(rows, env, cmd)
    write(a.out, md)
    write(a.json, json.dumps({"env": env, "rows": [r.as_dict() for r in rows]}, indent=1))
    emit(md)
    return 0


def bc_main(a) -> int:
    import os

    from uranium.bench.realbc import run_group
    from uranium.bench.report import bc_markdown, environment
    from uranium.device import configure, resolve

    if a.phases:
        os.environ["URANIUM_BC_PROFILE"] = "1"
    import torch

    backend, device, precision = resolve(a.backend)
    precision = a.precision or precision
    configure(backend, train=True)
    env = environment(backend, precision, device) | {"entity_chunk": os.environ.get("URANIUM_BC_ENTITY_CHUNK", "0"), "prefetch": os.environ.get("URANIUM_BC_PREFETCH", "default")}
    cmd = "python -m uranium bench " + " ".join(sys.argv[2:])
    groups = []
    for g in a.groups.split(","):
        try:
            groups.append(run_group(a, int(g), device, precision, log=lambda m: print(m, file=sys.stderr, flush=True)))
        except torch.OutOfMemoryError as e:  # a group size over the device memory is a result (060-R3)
            groups.append({"group": int(g), "updates": [], "median_s": None, "mean_s": None, "p95_s": None, "wait_median_s": None, "peak_gib": None, "peak_rss_gib": None, "failed": str(e).splitlines()[0][:160]})
            torch.cuda.empty_cache()
        # Written after every group: a long sweep that is stopped keeps the groups it finished.
        md = bc_markdown(groups, env, cmd)
        write(a.out, md)
        write(a.json, json.dumps({"env": env, "groups": groups}, indent=1))
    emit(md)
    return 0


def loss_main(a) -> int:
    from uranium.bench.realbc import compare_losses

    pick = lambda d: next(g for g in d["groups"] if a.group is None or g["group"] == a.group)  # noqa: E731
    base, cand = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (a.base, a.cand))
    res = compare_losses(pick(base), pick(cand), a.window)
    for w in res["windows"]:
        print(f"  {w['from']:5d}-{w['to']:5d}  base {w['base']:.5f}  cand {w['cand']:.5f}  rel {w['rel']:.2e}")
    print(json.dumps({k: v for k, v in res.items() if k != "windows"}))
    return 0


def rl_main(a) -> int:
    import os

    import torch

    from uranium import _core, ckpt
    from uranium.bench.report import environment
    from uranium.bench.rl import OFF, SyntheticSegments, markdown, run_learner
    from uranium.device import configure, resolve
    from uranium.rl.learner import MICRO_LANES_ENV
    from uranium.rl.run import _no_dropout

    backend, device, precision = resolve(a.backend)
    off = tuple(x.strip() for x in a.off.split(",") if x.strip())
    bad = [x for x in off if x not in OFF]
    if bad:
        raise SystemExit(f"--off {bad}: expected some of {list(OFF)}")
    # As `python -m uranium rl`: the backend's defaults only, not the BC trainer's process switches.
    configure(backend)
    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
    if os.environ.get(MICRO_LANES_ENV, "").strip():
        log(f"[bench rl] {MICRO_LANES_ENV} is not used here: --groups sets the lane group sizes")
    cfg = json.loads(_core.rl_config(a.config))
    if a.lanes:
        cfg["lanes"] = a.lanes
    lo, hi = (int(x) for x in a.entities.split(":"))
    net = batch = None
    if not a.isolate:  # --isolate: the children load the net and build the batch
        net, _ = ckpt.load_model(Path(a.checkpoint), device, adjust=_no_dropout)
        batch = SyntheticSegments(net, cfg["lanes"], cfg["seq_len"], (lo, hi), a.templates, a.seed)
        net = net.cpu()  # each run copies it to the device
    env = environment(backend, precision, device) | {"checkpoint": Path(a.checkpoint).name, "lanes": cfg["lanes"], "seq_len": cfg["seq_len"], "entities": a.entities}
    cmd = "python -m uranium bench " + " ".join(sys.argv[2:])
    runs: list = []
    md = ""
    for kind in [x.strip() for x in a.learner.split(",") if x.strip()]:
        for g in a.groups.split(","):
            if a.isolate:
                runs.append(_rl_child(a, kind, int(g), log))
                md = markdown(runs, env, cmd)
                write(a.out, md)
                write(a.json, json.dumps({"env": env, "runs": runs}, indent=1))
                continue
            try:
                runs.append(run_learner(kind, int(g), net, cfg, batch, device, precision, a.algo, a.updates, a.phases, a.ops, a.trace, log, off))
            except torch.OutOfMemoryError as e:  # a group size over the device memory is a result (060-R3)
                runs.append({"learner": kind, "group": int(g), "off": list(off), "updates": [], "median_s": None, "mean_s": None, "p95_s": None, "peak_gib": None, "peak_rss_gib": None, "failed": str(e).splitlines()[0][:160]})
                torch.cuda.empty_cache()
            # Written after every run: a long sweep that is stopped keeps the runs it finished.
            md = markdown(runs, env, cmd)
            write(a.out, md)
            write(a.json, json.dumps({"env": env, "runs": runs}, indent=1))
    emit(md)
    return 0


def _rl_child(a, kind: str, group: int, log) -> dict:
    """One `bench rl` run (learner x group) in a child process (`--isolate`): its peak RSS is the
    child's own (the synthetic batch, the nets and the run; 060-R3)."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "run.json"
        argv = [sys.executable, "-m", "uranium", "bench", "rl", "--checkpoint", a.checkpoint, "--config", a.config, "--algo", a.algo, "--learner", kind, "--groups", str(group),
                "--updates", str(a.updates), "--entities", a.entities, "--templates", str(a.templates), "--seed", str(a.seed), "--off", a.off, "--ops", str(a.ops), "--json", str(out)]
        argv += (["--backend", a.backend] if a.backend else []) + (["--lanes", str(a.lanes)] if a.lanes else []) + (["--phases"] if a.phases else []) + (["--trace", a.trace] if a.trace else [])
        log(f"[bench rl] child: {kind} group {group}")
        rc = subprocess.run(argv, stdout=subprocess.DEVNULL).returncode
        if rc != 0 or not out.is_file():
            # As the in-process path records a failed run: with the requested --off list.
            off = [x.strip() for x in a.off.split(",") if x.strip()]
            return {"learner": kind, "group": group, "off": off, "updates": [], "median_s": None, "mean_s": None, "p95_s": None, "peak_gib": None, "peak_rss_gib": None, "failed": f"child exit {rc}"}
        return json.loads(out.read_text(encoding="utf-8"))["runs"][0]


def rl_parity_main(a) -> int:
    from uranium.bench.rl import compare

    def pick(path: str, run: str) -> dict:
        kind, group = run.split(":")
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return next(r for r in d["runs"] if r["learner"] == kind and r["group"] == int(group))

    print(json.dumps(compare(pick(a.base, a.base_run), pick(a.cand or a.base, a.cand_run)), indent=1))
    return 0
