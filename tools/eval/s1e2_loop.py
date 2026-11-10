"""Per-checkpoint scoring of the second S1 epoch (041-R20 ③; 050-R9 point `s1e2`, "m2-s1e2 체크포인트마다"):
every checkpoint of the run gets val 500 (the protocol's `val500`, Linux ROCm host f32) and T0 stochastic
100 games (`t0_100`), and so does the reference, the S1 final (point `s1e1`, scored first). As Checkpoint 1
did it (reports/m6_bc_eval.md 4: `~/e2val/val_watch.sh`, `~/e2games/games_watch.sh`): each scorer takes the
newest checkpoint not taken yet when it is free, skips the ones in between, always takes the final one
(the run's DONE), never stops the run; val runs have a 2-hour limit (the Checkpoint 1 rule after a hung
run). A failed or timed-out val run is kept as <name>.failed-<k>.* and run again while it is still the
checkpoint to take, up to VAL_ATTEMPTS in all; `val` exits non-zero when the reference or the final one has
no return-code-0 record. An existing output is never overwritten. The cue diagnostics of the point run at the
final checkpoint only (AMB-050-6, `protocol.py diag`). Commands and jobs come from tools/eval/protocol.py
(same refusals).

    python tools/eval/s1e2_loop.py list   --run m2-s1e2 [--reference checkpoints/m2-s1/ckpt-N]
    python tools/eval/s1e2_loop.py val    --run m2-s1e2 [--reference ...] [--watch] [--dry-run]
    python tools/eval/s1e2_loop.py games  --run m2-s1e2 [--reference ...] --hosts tools/eval/hosts.toml [--watch]
                                          [--worker "--priority 2 --backend rocm --local-hostd <addr> --hostd <bin>"] [--dry-run]
    python tools/eval/s1e2_loop.py table  --run m2-s1e2 [--reference ...] [--json out.json] [--md out.md]

Outputs (under URANIUM_SHARED_DIR): val `artifacts/bc_e2_val/val500/<run>__<ckpt>.json` (lineage.toml val500
`output`) with `<...>.protocol.json` (checkpoint sha256, git, backend / precision, 050-R9) and the claim
`<...>.claim`; games `eval/<games root>/<run>__<ckpt>/` (default `eval/<run>-games`), each an eval job made by
`protocol.py job` (protocol.json inside) and played by `python -m uranium eval-worker` (`--worker` runs one
here; else workers are started by hand on the job).
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_common as ec  # noqa: E402
import protocol  # noqa: E402

LINEAGE = "m2"
POINT, REF_POINT = "s1e2", "s1e1"
VAL_METRIC, GAME_METRIC = "val500", "t0_100"
VAL_TIMEOUT_S = 7200  # reports/m6_bc_eval.md 4: "이후 채점마다 2시간 제한을 걸었다"
# val attempts per checkpoint (041-R20 ③ "최종 체크포인트는 반드시 한다"): a failed or timed-out run is kept
# as <name>.failed-<k>.* and run again while the checkpoint is still the one to take, up to this many in
# all (main-session interim decision 2026-11-10: val500 crashed 3 times on one ROCm host before succeeding
# in the lambda scoring, <shared>/eval/m2-lambda-20261110/claims/worker.log).
VAL_ATTEMPTS = 3
CKPT_RE = re.compile(r"^ckpt-(\d+)$")


def checkpoints(shared: Path, run: str) -> list[str]:
    """`checkpoints/<run>/ckpt-N` with weights, by N."""
    d = shared / "checkpoints" / run
    found = []
    for c in d.glob("ckpt-*") if d.is_dir() else []:
        m = CKPT_RE.match(c.name)
        if m and (c / protocol.WEIGHTS).is_file():
            found.append((int(m.group(1)), f"checkpoints/{run}/{c.name}"))
    return [c for _, c in sorted(found)]


def run_done(shared: Path, run: str) -> bool:
    return (shared / "checkpoints" / run / "DONE").is_file()


def output_name(checkpoint: str) -> str:
    return protocol.output_name(checkpoint)


def val_out(lin: dict, shared: Path, checkpoint: str) -> Path:
    run, ck = output_name(checkpoint).split("__", 1)
    return shared / lin["metric"][VAL_METRIC]["output"].replace("<run>", run).replace("<ckpt>", ck)


def game_job(shared: Path, root: str, checkpoint: str) -> Path:
    return shared / root / output_name(checkpoint)


def targets(a, shared: Path) -> list[tuple[str, str]]:
    """(point, checkpoint) of the reference (first) and of the run's checkpoints."""
    out = [(REF_POINT, a.reference.replace("\\", "/"))] if a.reference else []
    return out + [(POINT, c) for c in checkpoints(shared, a.run)]


def required(items: list[tuple[str, str]], done: bool) -> list[str]:
    """The checkpoints that must have a result: the reference, and the final one once the run is done."""
    out = [c for p, c in items if p == REF_POINT]
    run_items = [c for p, c in items if p == POINT]
    return out + (run_items[-1:] if done else [])


def pick(items: list[tuple[str, str]], taken) -> tuple[str, str] | None:
    """041-R20 ③: the reference while it is not taken, else the newest checkpoint not taken (the ones in
    between are skipped; the final one is the newest once the run is done)."""
    if items and items[0][0] == REF_POINT and not taken(items[0][1]):
        return items[0]
    run_items = [x for x in items if x[0] == POINT]
    if run_items and not taken(run_items[-1][1]):
        return run_items[-1]
    return None


def claim(f: Path) -> bool:
    """Atomic claim file (several scoring hosts share the folder)."""
    f.parent.mkdir(parents=True, exist_ok=True)
    try:
        with f.open("x", encoding="utf-8") as fh:
            fh.write(f"{socket.gethostname()} {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}\n")
        return True
    except FileExistsError:
        return False


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- val 500


def val_files(out: Path) -> dict[str, Path]:
    return {"claim": out.with_suffix(".claim"), "record": out.with_suffix(".protocol.json"), "log": out.with_suffix(".log"), "out": out}


def failed_attempts(out: Path) -> int:
    return len(list(out.parent.glob(f"{out.stem}.failed-*.protocol.json")))


def val_record(out: Path) -> dict | None:
    f = val_files(out)["record"]
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


def val_state(out: Path) -> str:
    """done (a run with return code 0), running (claimed, no record yet: here or on another host), failed
    (a record with another return code, not yet set aside), gave_up (VAL_ATTEMPTS failed attempts), open."""
    rec = val_record(out)
    if rec is not None:
        return "done" if rec.get("returncode") == 0 else "failed"
    if out.is_file():
        return "done"  # an output without a record (made by hand): never overwritten (041-R29 ⑨)
    if val_files(out)["claim"].exists():
        return "running"
    return "gave_up" if failed_attempts(out) >= VAL_ATTEMPTS else "open"


def set_aside(out: Path) -> int:
    """A failed attempt's files renamed to <name>.failed-<k>.* (kept, never deleted); returns k."""
    k = failed_attempts(out) + 1
    for kind, f in val_files(out).items():
        if f.exists():
            f.rename(f.with_name(f"{out.stem}.failed-{k}{''.join(f.suffixes)}" if kind != "out" else f"{out.stem}.failed-{k}{out.suffix}"))
    return k


def score_val(lin: dict, shared: Path, pt_id: str, checkpoint: str, dry: bool, timeout_s: int) -> bool:
    pt = protocol.point(lin, pt_id)
    protocol.check_new_measurement(lin, LINEAGE, VAL_METRIC, pt)
    m = protocol.check_metric(lin, pt, VAL_METRIC)
    protocol.check_checkpoint(lin, LINEAGE, pt, checkpoint)
    out = val_out(lin, shared, checkpoint)
    cmd = protocol.bc_eval_args(m, str(protocol.checkpoint_dir(checkpoint)), str(out), "python" if dry else sys.executable)
    if dry:
        print(" ".join(cmd))
        return True
    if out.exists():
        raise protocol.Refused(f"{out} exists (an output is never overwritten, 041-R29 ⑨)")
    files = val_files(out)
    if not claim(files["claim"]):
        return False
    rec = {"protocol": "config/protocol/lineage.toml (050-R9)", "lineage": LINEAGE, "point": pt_id, "metric": VAL_METRIC, "checkpoint": checkpoint,
           "checkpoint_sha256": protocol.checkpoint_sha256(checkpoint), "backend": m["backend"], "precision": protocol.PRECISION[m["backend"]],
           "git": protocol.git_state(), "command": protocol.redact_cmd(["python", *cmd[1:]], {"<shared>": shared}),
           "attempt": failed_attempts(out) + 1, "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    print(f"{now()} start val500 {output_name(checkpoint)} (attempt {rec['attempt']})", flush=True)
    t0 = time.time()
    try:
        with files["log"].open("w", encoding="utf-8") as fh:
            p = subprocess.run(cmd, cwd=protocol.ROOT, stdout=subprocess.DEVNULL, stderr=fh, timeout=timeout_s)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    rec |= {"returncode": rc, "secs": round(time.time() - t0)}
    if rc == 0:
        try:
            protocol.main(["check-bc-eval", str(out), "--metric", VAL_METRIC])
            rec["check"] = "ok"
        except SystemExit as e:  # recorded; the loop goes on (a mismatch is not fixed by running again)
            rec["check"] = str(e.code)
    files["record"].write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"{now()} end val500 {output_name(checkpoint)} rc={rc} secs={rec['secs']}" + (f" check: {rec['check']}" if "check" in rec else ""), flush=True)
    if rc != 0:
        k = set_aside(out)
        print(f"{now()} val500 {output_name(checkpoint)}: attempt kept as .failed-{k} ({VAL_ATTEMPTS - k} more allowed)", flush=True)
    return True


def cmd_val(a) -> None:
    lin = protocol.load_lineage()
    shared = ec.shared_dir()
    out_of = lambda c: val_out(lin, shared, c)  # noqa: E731
    taken = (lambda c: False) if a.dry_run else (lambda c: val_state(out_of(c)) != "open")
    while True:
        done = run_done(shared, a.run)  # read before picking: the final checkpoint is never missed
        items = targets(a, shared)
        if a.dry_run:
            for pt_id, c in items:
                score_val(lin, shared, pt_id, c, True, a.timeout_s)
            return
        nxt = pick(items, taken)
        if nxt is not None:
            score_val(lin, shared, *nxt, False, a.timeout_s)
            continue
        need = {c: val_state(out_of(c)) for c in required(items, done)}
        if a.watch and (not done or "running" in need.values()):
            time.sleep(a.poll_s)
            continue
        bad = {c: s for c, s in need.items() if s != "done" or (val_record(out_of(c)) or {}).get("returncode", 0) != 0}
        print(f"{now()} val500: nothing left" + (" (run done)" if done else ""), flush=True)
        if bad:
            raise SystemExit("val500 without a return-code-0 record: " + ", ".join(f"{c} ({s})" for c, s in bad.items()))
        return


# ---------------------------------------------------------------- T0 100


def make_job(lin: dict, shared: Path, a, pt_id: str, checkpoint: str) -> Path | None:
    """The checkpoint's T0 100 job (protocol.py job); None when another host made it first."""
    job = game_job(shared, a.games_root, checkpoint)
    args = ["job", "--lineage", LINEAGE, "--point", pt_id, "--checkpoint", checkpoint, "--metrics", GAME_METRIC, "--job", job.name,
            "--label", job.name, "--hosts", a.hosts, "--out-root", str(job.parent)]
    if a.allow_dirty:
        args.append("--allow-dirty")
    if a.dry_run:
        args.append("--dry-run")
    try:
        protocol.main(args)
    except FileExistsError:
        print(f"{now()} {job.name}: the job exists (made by another host)", flush=True)
        return None
    return job


def play(job: Path, shared: Path, worker: str | None, poll_s: int) -> None:
    rel = job.relative_to(shared).as_posix()
    if worker is not None:
        cmd = [sys.executable, "-m", "uranium", "eval-worker", "--job", rel, *shlex.split(worker)]
        print(f"{now()} start games {job.name}: {' '.join(['python', *cmd[1:]])}", flush=True)
        subprocess.run(cmd, cwd=protocol.ROOT)
        if not (job / "DONE").is_file():
            print(f"{now()} {job.name}: worker ended before DONE; waiting for other workers", flush=True)
    while not (job / "DONE").is_file():
        time.sleep(poll_s)
    print(f"{now()} end games {job.name}", flush=True)


def cmd_games(a) -> None:
    lin = protocol.load_lineage()
    shared = ec.shared_dir()
    a.games_root = a.games_root or f"eval/{a.run}-games"
    taken = (lambda c: False) if a.dry_run else (lambda c: game_job(shared, a.games_root, c).exists())
    while True:
        done = run_done(shared, a.run)
        items = targets(a, shared)
        if a.dry_run:
            for pt_id, c in items:
                make_job(lin, shared, a, pt_id, c)
            return
        nxt = pick(items, taken)
        if nxt is not None:
            job = make_job(lin, shared, a, *nxt)
            if job is not None:
                play(job, shared, a.worker, a.poll_s)
            continue
        if not a.watch or done:
            print(f"{now()} games: nothing left" + (" (run done)" if done else ""), flush=True)
            return
        time.sleep(a.poll_s)


# ---------------------------------------------------------------- list, table


def cmd_list(a) -> None:
    lin = protocol.load_lineage()
    shared = ec.shared_dir()
    root = a.games_root or f"eval/{a.run}-games"
    print(f"run {a.run}: {'done' if run_done(shared, a.run) else 'training'}")
    for pt_id, c in targets(a, shared):
        v = val_out(lin, shared, c)
        vs = val_state(v) + (f" ({failed_attempts(v)} failed)" if failed_attempts(v) else "")
        j = game_job(shared, root, c)
        gs = "done" if (j / "DONE").is_file() else "job" if j.exists() else "-"
        print(f"{pt_id:5} {c:44} val500 {vs:20} t0_100 {gs}")


VAL_COLS = (("top-1", lambda v: f"{v['action_top1'] * 100:.2f}%"), ("NLL action", lambda v: f"{v['nll']['action']:.3f}"),
            ("delay", lambda v: f"{v['nll']['delay']:.3f}"), ("selected", lambda v: f"{v['nll']['selected']:.3f}"),
            ("target_unit", lambda v: f"{v['nll']['target_unit']:.3f}"), ("target_loc", lambda v: f"{v['nll']['target_loc']:.3f}"),
            ("win_open MSE", lambda v: "-" if v.get("win_open_mse") is None else f"{v['win_open_mse']:.3f}"))
RACES = ("terran", "protoss", "zerg")


def collect(lin: dict, shared: Path, a) -> list[dict]:
    import tables

    cache = ec.cache_dir() / "eval_cmddump"
    root = a.games_root or f"eval/{a.run}-games"
    rows = []
    for pt_id, c in targets(a, shared):
        r = {"point": pt_id, "checkpoint": c, "val500": None, "val500_problem": None, "t0_100": None, "t0_100_problems": None}
        v = val_out(lin, shared, c)
        if v.is_file():
            d = json.loads(v.read_text(encoding="utf-8"))
            m = lin["metric"][VAL_METRIC]
            want = {"split": m["split"], "games": m["games"], "selection_seed": m["seed"]}
            got = {k: d.get(k) for k in want}
            r["val500"] = d
            r["val500_problem"] = None if got == want else f"{got} != pinned {want}"
        j = game_job(shared, root, c)
        if (j / "DONE").is_file():
            res = tables.job_tables(j, cache, lin=lin)
            (t,) = res.values()
            r["t0_100"] = t
            r["t0_100_problems"] = protocol.verify_job(lin, j, LINEAGE, protocol.point(lin, pt_id))
        rows.append(r)
    return rows


def wdl(x: dict) -> str:
    return f"{x['won']} / {x['lost']} / {x['draw']}"


def markdown(rows: list[dict]) -> list[str]:
    L = ["### val 500 (target-map val 500 games, seed 1, Linux ROCm host f32)", "",
         "| checkpoint | " + " | ".join(c for c, _ in VAL_COLS) + " | steps |", "|---|" + "---|" * (len(VAL_COLS) + 1)]
    for r in rows:
        lab = ("reference " if r["point"] == REF_POINT else "") + f"`{r['checkpoint']}`"
        v = r["val500"]
        if v is None:
            L.append(f"| {lab} | not scored |" + " |" * len(VAL_COLS))
            continue
        flag = f" (MISMATCH {r['val500_problem']})" if r["val500_problem"] else ""
        L.append(f"| {lab}{flag} | " + " | ".join(f(v) for _, f in VAL_COLS) + f" | {v.get('steps')} |")
    L += ["", "### T0 stochastic 100 (won / lost / draw; Wilson 95% of the total win rate)", "",
          "| checkpoint | terran | protoss | zerg | total | win rate |", "|---|---|---|---|---|---|"]
    for r in rows:
        lab = ("reference " if r["point"] == REF_POINT else "") + f"`{r['checkpoint']}`"
        t = r["t0_100"]
        if t is None:
            continue
        by = t["table"]["by_race"]
        tot = t["table"]["total"]
        lo, hi = ec.wilson(tot["won"], tot["n"]) if tot["n"] else (0.0, 0.0)
        flag = f" (MISMATCH: {'; '.join(r['t0_100_problems'])})" if r["t0_100_problems"] else ""
        cells = [wdl(by.get(f"stochastic vs {x}", {"won": 0, "lost": 0, "draw": 0})) for x in RACES]
        L.append(f"| {lab}{flag} | " + " | ".join(cells) + f" | {wdl(tot)} | {100 * tot['won'] / max(tot['n'], 1):.0f}% [{100 * lo:.0f}, {100 * hi:.0f}] |")
    return L


def cmd_table(a) -> None:
    lin = protocol.load_lineage()
    rows = collect(lin, ec.shared_dir(), a)
    text = "\n".join(markdown(rows)) + "\n"
    sys.stdout.write(text)
    if a.md:
        Path(a.md).write_text(text, encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
        print("wrote", a.json)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--run", required=True, help="the second-epoch run, e.g. m2-s1e2")
        s.add_argument("--reference", default=None, help="the S1 final (point s1e1), relative to URANIUM_SHARED_DIR")
        s.add_argument("--games-root", default=None, help="default eval/<run>-games")
        return s

    s = common(sub.add_parser("list"))
    s.set_defaults(f=cmd_list)
    s = common(sub.add_parser("val", help="val 500 of the newest unscored checkpoint, here"))
    s.add_argument("--watch", action="store_true", help="keep going until the run is done and its final checkpoint is scored")
    s.add_argument("--poll-s", type=int, default=120)
    s.add_argument("--timeout-s", type=int, default=VAL_TIMEOUT_S)
    s.add_argument("--dry-run", action="store_true", help="print the command of every checkpoint")
    s.set_defaults(f=cmd_val)
    s = common(sub.add_parser("games", help="T0 100 job of the newest checkpoint without one"))
    s.add_argument("--hosts", default=str(protocol.HOSTS))
    s.add_argument("--worker", default=None, help="run `python -m uranium eval-worker --job <job>` here with these arguments")
    s.add_argument("--watch", action="store_true")
    s.add_argument("--poll-s", type=int, default=120)
    s.add_argument("--allow-dirty", action="store_true")
    s.add_argument("--dry-run", action="store_true", help="show the jobs of every checkpoint without writing them")
    s.set_defaults(f=cmd_games)
    s = common(sub.add_parser("table", help="val 500 + T0 100 per checkpoint"))
    s.add_argument("--json", default=None)
    s.add_argument("--md", default=None)
    s.set_defaults(f=cmd_table)
    a = ap.parse_args(argv)
    try:
        a.f(a)
    except protocol.Refused as e:
        raise SystemExit(f"REFUSED: {e}")


if __name__ == "__main__":
    main()
