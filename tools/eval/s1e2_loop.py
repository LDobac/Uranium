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
here; else workers are started by hand on the job). The loop writes the job's `status.txt` (eval-status
output) and `DONE` once eval-status shows every stage played out (nothing in flight, nothing left) in two
readings a poll apart; one host does it (the job's `.merge-claim`).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
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
# a claim without a record is stale (its scorer is gone) this long after the run's time limit
STALE_MARGIN_S = 900
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


def claim(f: Path, timeout_s: int) -> bool:
    """Atomic claim file (several scoring hosts share the folder): host, pid, start, the run's time limit."""
    f.parent.mkdir(parents=True, exist_ok=True)
    try:
        with f.open("x", encoding="utf-8") as fh:
            fh.write(json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "timeout_s": timeout_s,
                                 "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}) + "\n")
        return True
    except FileExistsError:
        return False


def read_claim(f: Path) -> dict:
    """A claim's content ({} for an older plain-text claim)."""
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def pid_alive(pid: int) -> bool:
    """Whether a process of this host with this id runs (no signal is sent)."""
    if os.name == "nt":
        import ctypes

        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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


def claim_stale(out: Path, timeout_s: int) -> str | None:
    """Why a claim without a record is stale (its scorer is gone), else None: older than the run's time limit
    + STALE_MARGIN_S (a host that crashed, rebooted or was stopped), or made on this host by a process that no
    longer runs."""
    f = val_files(out)["claim"]
    try:
        age = time.time() - f.stat().st_mtime
    except FileNotFoundError:
        return None
    c = read_claim(f)
    limit = int(c.get("timeout_s") or timeout_s) + STALE_MARGIN_S
    if age > limit:
        return f"claim of {c.get('host', '?')} older than {limit} s"
    if c.get("host") == socket.gethostname() and c.get("pid") not in (None, os.getpid()) and not pid_alive(c["pid"]):
        return f"claim of this host's process {c['pid']}, which no longer runs"
    return None


def val_state(out: Path, timeout_s: int = VAL_TIMEOUT_S) -> str:
    """done (a run with return code 0, or an output without a record), running (claimed, no record yet: here
    or on another host), stale (a claim whose scorer is gone), failed (a record with another return code,
    not yet set aside), gave_up (VAL_ATTEMPTS failed attempts), open."""
    rec = val_record(out)
    if rec is not None:
        return "done" if rec.get("returncode") == 0 else "failed"
    if out.is_file():
        return "done"  # an output without a record (made by hand): never overwritten (041-R29 ⑨)
    if val_files(out)["claim"].exists():
        return "stale" if claim_stale(out, timeout_s) else "running"
    return "gave_up" if failed_attempts(out) >= VAL_ATTEMPTS else "open"


def set_aside(out: Path) -> int:
    """A failed attempt's files renamed to <name>.failed-<k>.* (kept, never deleted); returns k. Order: the
    output, the log, the record, the claim last, so the state goes failed -> running -> open without a gap."""
    k = failed_attempts(out) + 1
    files = val_files(out)
    for kind in ("out", "log", "record", "claim"):
        f = files[kind]
        if f.exists():
            f.rename(f.with_name(f"{out.stem}.failed-{k}{''.join(f.suffixes)}" if kind != "out" else f"{out.stem}.failed-{k}{out.suffix}"))
    return k


def tidy(out: Path, timeout_s: int) -> None:
    """Before picking: a leftover failed record (a scorer stopped between its record and set_aside) and a stale
    claim (a record "stale claim" is written first; it counts as an attempt) are set aside. Another host doing
    the same at once loses the renames (ignored)."""
    st = val_state(out, timeout_s)
    if st not in ("failed", "stale"):
        return
    try:
        if st == "stale":
            why = claim_stale(out, timeout_s)
            val_files(out)["record"].write_text(json.dumps({"metric": VAL_METRIC, "returncode": "stale claim", "reason": why,
                                                            "claim": read_claim(val_files(out)["claim"]),
                                                            "noticed": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}, indent=1), encoding="utf-8")
        k = set_aside(out)
        print(f"{now()} val500 {out.stem}: {st} attempt kept as .failed-{k}", flush=True)
    except OSError:
        pass


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
    if not claim(files["claim"], timeout_s):
        return False
    rec = {"protocol": "config/protocol/lineage.toml (050-R9)", "lineage": LINEAGE, "point": pt_id, "metric": VAL_METRIC, "checkpoint": checkpoint,
           "attempt": failed_attempts(out) + 1, "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    try:
        rec |= {"checkpoint_sha256": protocol.checkpoint_sha256(checkpoint), "backend": m["backend"], "precision": protocol.PRECISION[m["backend"]],
                "git": protocol.git_state(), "command": protocol.redact_cmd(["python", *cmd[1:]], {"<shared>": shared})}
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
    except BaseException as e:  # Ctrl-C, an error before the record: recorded and set aside, never an orphan claim
        rec["returncode"] = f"error: {e!r}"
        files["record"].write_text(json.dumps(rec, indent=1), encoding="utf-8")
        k = set_aside(out)
        print(f"{now()} val500 {output_name(checkpoint)}: stopped ({e!r}); attempt kept as .failed-{k}", flush=True)
        raise
    print(f"{now()} end val500 {output_name(checkpoint)} rc={rc} secs={rec['secs']}" + (f" check: {rec['check']}" if "check" in rec else ""), flush=True)
    if rc != 0:
        k = set_aside(out)
        print(f"{now()} val500 {output_name(checkpoint)}: attempt kept as .failed-{k} ({VAL_ATTEMPTS - k} more allowed)", flush=True)
    return True


def cmd_val(a) -> None:
    lin = protocol.load_lineage()
    shared = ec.shared_dir()
    out_of = lambda c: val_out(lin, shared, c)  # noqa: E731
    taken = (lambda c: False) if a.dry_run else (lambda c: val_state(out_of(c), a.timeout_s) != "open")
    waiting = None
    while True:
        done = run_done(shared, a.run)  # read before picking: the final checkpoint is never missed
        items = targets(a, shared)
        if a.dry_run:
            for pt_id, c in items:
                score_val(lin, shared, pt_id, c, True, a.timeout_s)
            return
        for _, c in items:
            tidy(out_of(c), a.timeout_s)
        nxt = pick(items, taken)
        if nxt is not None:
            score_val(lin, shared, *nxt, False, a.timeout_s)
            continue
        need = {c: val_state(out_of(c), a.timeout_s) for c in required(items, done)}
        if a.watch and (not done or "running" in need.values()):
            msg = ", ".join(f"{c} running on {read_claim(val_files(out_of(c))['claim']).get('host', '?')}" for c, s in need.items() if s == "running")
            msg = msg or ("the next checkpoint" if not done else "")
            if msg != waiting:
                print(f"{now()} val500: waiting for {msg}", flush=True)
                waiting = msg
            time.sleep(a.poll_s)
            continue
        print(f"{now()} val500: nothing left" + (" (run done)" if done else ""), flush=True)
        for c, s in need.items():
            if s == "done" and val_record(out_of(c)) is None:
                print(f"NOTE val500 {c}: an output without a record (made by hand)", flush=True)
        bad = {c: s for c, s in need.items() if s != "done" or (val_record(out_of(c)) or {}).get("returncode", 0) != 0}
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


STATUS_HEAD = "| stage | games | done | in flight | left | retried records |"


def eval_status(job: Path, write: bool) -> str:
    """`python -m uranium eval-status --job <job>`'s text (020-R15 ⑨); with `write` the merged per-stage
    records and summaries are rewritten too."""
    from uranium import _core

    return _core.eval_status(str(job), write=write)


def status_stages(text: str) -> list[dict]:
    """The per-stage rows of eval-status's table; empty (fail closed) when the table is missing or a row of it
    does not parse."""
    lines = text.splitlines()
    try:
        i = lines.index(STATUS_HEAD)
    except ValueError:
        return []
    out = []
    for ln in lines[i + 2:]:
        if not ln.strip():
            break
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if not ln.startswith("|") or len(cells) != 6 or not all(c.isdigit() for c in cells[1:]):
            return []
        out.append(dict(zip(("stage", "games", "done", "in_flight", "left", "retried"), [cells[0], *map(int, cells[1:])])))
    return out


def status_complete(stages: list[dict]) -> bool:
    """Every stage played out: nothing in flight, nothing left, every game with a final record (given-up games
    included, as when the worker says "lease period ended: done")."""
    return bool(stages) and all(s["in_flight"] == 0 and s["left"] == 0 and s["done"] == s["games"] for s in stages)


def progress(stages: list[dict]) -> str:
    return "; ".join(f"{s['stage']} {s['done']}/{s['games']} done, {s['in_flight']} in flight, {s['left']} left" for s in stages) or "no stage table"


def write_atomic(f: Path, text: str) -> None:
    """A temp file next to it, then a rename (several hosts share the folder: a reader never sees half a file)."""
    tmp = f.with_name(f".{f.name}.tmp-{socket.gethostname()}-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, f)


# The merge (eval-status with write: summaries rewritten) and the DONE write of a job are done by one host: the
# one whose exclusive-create claim file this is. A claim older than MERGE_STALE_S without a DONE (its host
# stopped midway) is set aside (renamed, never deleted) and taken again.
MERGE_CLAIM = ".merge-claim"
MERGE_STALE_S = 600
# jobs with one complete eval-status reading not yet confirmed: str(job) -> time.monotonic() of that reading
_complete_seen: dict[str, float] = {}


def pending(job: Path) -> bool:
    """A complete reading waits for its confirming one (finish)."""
    return str(job) in _complete_seen


def set_aside_merge_claim(job: Path, why: str) -> None:
    f = job / MERGE_CLAIM
    k = len(list(job.glob(f"{MERGE_CLAIM}.*"))) + 1
    try:
        f.rename(job / f"{MERGE_CLAIM}.{why}-{k}")
    except OSError:
        pass  # another host set it aside first


def take_merge_claim(job: Path, quiet: bool) -> bool:
    f = job / MERGE_CLAIM
    if claim(f, MERGE_STALE_S):
        return True
    c = read_claim(f)
    try:
        age = time.time() - f.stat().st_mtime
    except FileNotFoundError:
        return claim(f, MERGE_STALE_S)
    if age <= MERGE_STALE_S or (job / "DONE").is_file():
        if not quiet:
            print(f"{now()} {job.name}: complete; {c.get('host', '?')} writes DONE (merge claim of {round(age)} s)", flush=True)
        return False
    set_aside_merge_claim(job, "stale")
    print(f"{now()} {job.name}: merge claim of {c.get('host', '?')} older than {MERGE_STALE_S} s without DONE, set aside", flush=True)
    return claim(f, MERGE_STALE_S)


def read_status(job: Path, write: bool) -> tuple[str, list[dict], str | None]:
    """eval-status's text, its stage rows and why they do not show a played-out job (None: they do). Every stage
    of job.toml must have a row."""
    text = eval_status(job, write=write)
    stages = status_stages(text)
    n = len(ec.job_stages(job))
    if len(stages) != n:
        return text, stages, f"{len(stages)} stage rows in eval-status, {n} stages in job.toml"
    return text, stages, None if status_complete(stages) else progress(stages)


def finish(job: Path, gap_s: float = 0, quiet: bool = False) -> bool:
    """041-R20 ③ (the step of Checkpoint 1's games_watch.sh after each worker run): whether the job has its
    DONE. Without one, eval-status decides. A played-out job (every stage: nothing in flight, nothing left) is
    confirmed by a second reading at least `gap_s` later (a later call); then this host takes the merge claim,
    runs eval-status with write (the merged summaries), and writes `status.txt` (its text) and `DONE`, each
    atomically. An incomplete job, an unreadable status or a failed step never gets a DONE."""
    done, key = job / "DONE", str(job)
    if done.is_file():
        _complete_seen.pop(key, None)
        return True
    try:
        _, stages, why = read_status(job, write=False)
    except Exception as e:  # a job still being made, the shared drive away: asked again later
        _complete_seen.pop(key, None)
        if not quiet:
            print(f"{now()} {job.name}: eval-status read failed ({e!r}); no DONE", flush=True)
        return False
    if why is not None:
        _complete_seen.pop(key, None)
        if not quiet:
            print(f"{now()} {job.name}: not complete ({why})", flush=True)
        return False
    if key not in _complete_seen:
        _complete_seen[key] = time.monotonic()
        print(f"{now()} {job.name}: complete per eval-status ({progress(stages)}); confirming at the next poll", flush=True)
        return False
    if time.monotonic() - _complete_seen[key] < gap_s or not take_merge_claim(job, quiet):
        return False
    try:
        text, stages, why = read_status(job, write=True)
    except Exception as e:
        set_aside_merge_claim(job, "failed")
        print(f"{now()} {job.name}: eval-status with write failed ({e!r}); no DONE", flush=True)
        return False
    if why is not None:
        set_aside_merge_claim(job, "failed")
        _complete_seen.pop(key, None)
        print(f"{now()} {job.name}: not complete in the merging reading ({why}); no DONE", flush=True)
        return False
    try:
        write_atomic(job / "status.txt", text)
        write_atomic(done, json.dumps({"host": socket.gethostname(), "by": "s1e2_loop.py, eval-status complete",
                                       "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}) + "\n")
    except OSError as e:
        set_aside_merge_claim(job, "failed")
        print(f"{now()} {job.name}: writing status.txt / DONE failed ({e!r}); no DONE", flush=True)
        return False
    _complete_seen.pop(key, None)
    print(f"{now()} {job.name}: complete per eval-status ({progress(stages)}); wrote status.txt and DONE", flush=True)
    return True


def play(job: Path, shared: Path, worker: str | None, poll_s: int) -> None:
    rel = job.relative_to(shared).as_posix()
    if worker is not None:
        cmd = [sys.executable, "-m", "uranium", "eval-worker", "--job", rel, *shlex.split(worker)]
        print(f"{now()} start games {job.name}: {' '.join(['python', *cmd[1:]])}", flush=True)
        rc = subprocess.run(cmd, cwd=protocol.ROOT).returncode
        if not finish(job, poll_s):
            print(f"{now()} {job.name}: worker ended rc={rc}; waiting for DONE", flush=True)
    while not finish(job, poll_s, quiet=True):
        time.sleep(poll_s)
    print(f"{now()} end games {job.name}", flush=True)


def sweep(shared: Path, root: str, poll_s: int) -> list[Path]:
    """finish() on every job folder of the games root without DONE (a job played out by hand-started workers or
    by another host); returns those waiting for their confirming reading."""
    d = shared / root
    jobs = [j for j in sorted(d.iterdir()) if (j / "job.toml").is_file() and not (j / "DONE").is_file()] if d.is_dir() else []
    return [j for j in jobs if not finish(j, poll_s, quiet=True) and pending(j)]


def cmd_games(a) -> None:
    lin = protocol.load_lineage()
    shared = ec.shared_dir()
    a.games_root = a.games_root or f"eval/{a.run}-games"
    joined: set[str] = set()

    def taken(c: str) -> bool:
        # with --worker, an existing job that is not DONE is joined once (another host made it); else taken.
        # A job played out (DONE written by the sweep, or waiting for its confirming reading) is not joined.
        j = game_job(shared, a.games_root, c)
        return j.exists() and (a.worker is None or j.name in joined or (j / "DONE").is_file() or pending(j))

    while True:
        done = run_done(shared, a.run)
        items = targets(a, shared)
        if a.dry_run:
            for pt_id, c in items:
                make_job(lin, shared, a, pt_id, c)
            return
        confirming = sweep(shared, a.games_root, a.poll_s)
        nxt = pick(items, taken)
        if nxt is not None:
            job = game_job(shared, a.games_root, nxt[1])
            if job.exists():
                print(f"{now()} {job.name}: joining the existing job as a worker", flush=True)
            else:
                job = make_job(lin, shared, a, *nxt)
            if job is not None:
                joined.add(job.name)
                play(job, shared, a.worker, a.poll_s)
            continue
        if (not a.watch or done) and not confirming:
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
        if t.get("skipped_lines"):
            flag += f" (skipped {len(t['skipped_lines'])} undecodable result / ledger line(s))"
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
