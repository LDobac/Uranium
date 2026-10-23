"""`python -m uranium bench` (060-R2): `suite` (kernel / model / inference cases with errors vs
CPU), `bc` (real-data BC updates per lane group size, parity log) and `loss` (parity of two `bc`
logs). Candidates of an optimisation are compared by running these under each candidate's setting
and comparing the JSON (AMB-000-4 step 2)."""

import json
import sys
from pathlib import Path


def add_parser(sub) -> None:
    p = sub.add_parser("bench", help="bench suite (060-R2)")
    bsub = p.add_subparsers(dest="bench_cmd", required=True)

    s = bsub.add_parser("suite", help="kernel, model and inference cases")
    s.add_argument("--suite", default="kernels,model", help="comma list of kernels, model, infer")
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
    configure(backend)
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
