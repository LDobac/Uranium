"""Shared helpers of the measurement tools in tools/eval (050-R9 measurement protocol, 050-R6 / 050-R7,
docs/measurement_protocol.md).

Moved from the untracked Checkpoint 1 scripts (runs/dagger/test_table.py, t1_table.py, ac1_table.py,
runs/tclone_macro/macro_table.py, runs/bc_ft_teacher/rush_summary.py) with their counting rules kept
as they were, so the same eval job gives the same numbers. Host paths come from `.env` (001-R9); no host
address or personal path is written here (001-R10).
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Frames (24 per game second at Fastest as counted by the session metrics and the old scripts).
FPS = 24
CUT = 11520  # 8:00: the command stream is read up to here (runs/dagger/test_table.py CUT)
RACE_TERRAN = 1
# BWAPI unit type ids in the build command payload.
BUNKER, CC = 125, 106
# Build (0x0c), select (0x09 / 0x0a), right click (0x14), targeted order (0x15).
OP_BUILD, OP_SELECT, OP_SEL_ADD, OP_RCLICK, OP_ORDER = 12, 9, 10, 20, 21
ORDER_MOVE = 6


def load_env() -> None:
    """`.env` into os.environ without overriding set variables (001-R9, same rules as uranium.env)."""
    try:
        from uranium import env
    except ImportError:  # the measurement tables also run without the uranium package
        env = None
    if env is not None:
        env.load()
        return
    d = Path.cwd().resolve()
    for p in [d, *d.parents, ROOT]:
        f = p / ".env"
        if f.is_file():
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, sep, raw = line.removeprefix("export ").partition("=")
                raw = raw.strip()
                if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
                    raw = raw[1:-1]
                if sep and key.strip() and key.strip() not in os.environ:
                    os.environ[key.strip()] = raw
            return


def shared_dir() -> Path:
    load_env()
    v = os.environ.get("URANIUM_SHARED_DIR")
    if not v:
        raise SystemExit("URANIUM_SHARED_DIR is not set (.env, 001-R9)")
    return Path(v)


def cache_dir() -> Path:
    """This host's working cache: URANIUM_CACHE_DIR, else runs/ of this checkout (git-ignored)."""
    load_env()
    return Path(os.environ.get("URANIUM_CACHE_DIR") or ROOT / "runs")


def replay_bin() -> Path:
    """`uranium-replay` (its `cmddump` subcommand): URANIUM_REPLAY_BIN, else target/release of this checkout."""
    load_env()
    v = os.environ.get("URANIUM_REPLAY_BIN")
    if v:
        return Path(v)
    exe = "uranium-replay.exe" if os.name == "nt" else "uranium-replay"
    return ROOT / "target" / "release" / exe


# ---------------------------------------------------------------- eval job files


def job_stages(job: Path) -> list[tuple[int, Path, dict]]:
    """(stage index from 1, plan file, plan) in job.toml order; game ids of stage k are k * 100000 + n."""
    spec = tomllib.loads((job / "job.toml").read_text(encoding="utf-8"))
    out = []
    for k, rel in enumerate(spec["stages"], start=1):
        p = job / rel
        out.append((k, p, tomllib.loads(p.read_text(encoding="utf-8"))))
    return out


_BLANK = b"\x00 \t\r\n\f\v"


class Rows(list):
    """The rows of a job's JSONL files (a plain list of dicts) with `skipped`: the lines that did not decode,
    as {"file", "line" (from 1), "error"}. Lines that are empty once NUL bytes and whitespace are stripped
    are not rows and not counted (050-R9)."""

    def __init__(self, rows=(), skipped=()):
        super().__init__(rows)
        self.skipped: list[dict] = list(skipped)


def read_jsonl_dir(d: Path) -> Rows:
    """Every row of d/*.jsonl, files in name order. A result file on the shared drive can hold write
    corruption (a line of NUL bytes, a truncated last line): a line empty after stripping NULs and
    whitespace is ignored, any other line that does not decode to a JSON object is skipped with a warning
    naming the file and line and counted in `skipped`; every decodable row is kept (050-R9)."""
    out = Rows()
    for f in sorted(d.glob("*.jsonl")):
        for i, line in enumerate(f.read_bytes().splitlines(), start=1):
            s = line.strip(_BLANK)
            if not s:
                continue
            try:
                r = json.loads(s.decode("utf-8"))  # UnicodeDecodeError and JSONDecodeError are ValueErrors
                if not isinstance(r, dict):
                    raise ValueError(f"not a JSON object ({type(r).__name__})")
            except ValueError as e:
                print(f"warning: {f}:{i}: skipped an undecodable line ({e})", file=sys.stderr)
                out.skipped.append({"file": f.name, "line": i, "error": str(e)})
                continue
            out.append(r)
    return out


def read_results(job: Path) -> Rows:
    """Every row of the job's results/*.jsonl, files in name order (a later row of a game wins);
    `skipped` lists the lines that did not decode (read_jsonl_dir)."""
    return read_jsonl_dir(job / "results")


def skipped_lines(results: list[dict], ledger: list[dict]) -> list[dict]:
    """{"file": "results/<name>" | "games/<name>", "line", "error"} of the lines read_results / read_ledger skipped."""
    return [{**s, "file": f"{sub}/{s['file']}"} for sub, rows in (("results", results), ("games", ledger))
            for s in getattr(rows, "skipped", [])]


# Session game ids (crates/uranium-runner/src/eval.rs hostd_id): attempt * 10,000,000 + stage * 100,000 + n.
STAGE_BASE, ATTEMPT_BASE = 100_000, 10_000_000


def game_key(game_id: int) -> int:
    """stage * 100,000 + n of a session / ledger game id (the attempt dropped)."""
    return int(game_id) % ATTEMPT_BASE


def stage_of(game_id: int) -> int:
    return game_key(game_id) // STAGE_BASE


def attempt_of(game_id: int) -> int:
    return int(game_id) // ATTEMPT_BASE


def model_rows(rows: list[dict], stage: int) -> dict[int, dict]:
    """Model-player rows of one stage by plan game number n (runs/dagger/test_table.py stage_rows: the
    row has a `model` object and belongs to the stage). Of several rows of a game the one of the highest
    attempt wins, then the later one in file order.

    Retries (050-R9): a replayed game's row has the attempt in its id. runs/dagger/test_table.py and
    ac1_table.py picked the stage with game_id // 100000 and so dropped those rows; runs/dagger/
    t1_table.py (game_id % 100000, no stage filter), rush_summary.py (% 100000) and the eval worker's
    summary.md counted them. Checkpoint 1 jobs with retried model rows: bc-check-20261105 (6 games) and
    ac1-20261101/baseline/ac1-base-lf0 (1 game); their reported values (rush_summary.py, summary.md)
    already include the retries, as this function does."""
    best: dict[int, tuple[int, dict]] = {}
    for r in rows:
        if r.get("model") and stage_of(r["game_id"]) == stage:
            n, a = game_key(r["game_id"]) % STAGE_BASE, attempt_of(r["game_id"])
            if n not in best or a >= best[n][0]:
                best[n] = (a, r)
    return {n: r for n, (_, r) in best.items()}


def read_ledger(job: Path) -> Rows:
    """Every row of the job's games/*.jsonl (the game ledger), skipping undecodable lines as read_results."""
    return read_jsonl_dir(job / "games")


_EXITS = re.compile(r"launcher exit codes \[([^\]]*)\]")


def opponent_crashes(ledger: list[dict]) -> set[int]:
    """Game ids (attempt dropped) with an Aborted attempt where our launcher (players[0]) exited with
    code 0 and the opponent's (players[1]) did not (`Some(0)`): the opponent crashed during the game.
    Checkpoint 1 counted these games as invalid (050-R9; reports/checkpoint1.md 7.3: BC McRave 1 game,
    AC1 Steamhammer protoss 5 games), whatever the game's later row says (a replayed game or a
    Surrender win). An abort where our side failed too is a normal retry, not an opponent crash."""
    out = set()
    for g in ledger:
        if g.get("outcome") != "Aborted":
            continue
        m = _EXITS.search(g.get("detail") or "")
        if not m:
            continue
        codes = [c.strip() for c in m.group(1).split(",")]
        if len(codes) >= 2 and codes[0] == "Some(0)" and codes[1] != "Some(0)":
            out.add(game_key(g["game_id"]))
    return out


def ledger_backends(ledger: list[dict]) -> dict[str, int]:
    """Finished games per inference backend and host (020-R15 ledger fields), for the record."""
    out: dict[str, int] = {}
    for g in ledger:
        if g.get("outcome") == "Finished":
            k = f"{g.get('backend')}@{g.get('infer_host')}"
            out[k] = out.get(k, 0) + 1
    return out


def plan_seeds(plan: dict) -> list[int]:
    total = sum(m["games"] for m in plan["matchup"])
    if "seeds" in plan:
        return list(plan["seeds"])
    base = plan.get("seed_base", 1000)
    return [base if plan.get("same_seed") else base + n for n in range(1, total + 1)]


def opponent_label(p1: dict) -> str:
    """runs/dagger/test_table.py label: OSS name, else `t0 <build>` (`t0 standard` for no build)."""
    return p1.get("name") or f"t0 {p1.get('build', 'standard')}"


def game_opponents(plan: dict) -> list[str]:
    opp = []
    for m in plan["matchup"]:
        opp += [opponent_label(m["p1"])] * m["games"]
    return opp


def start_condition(seed: int) -> str:
    """050-R6 start condition of a seed: `early` = the opponent's first target is our main (020-R21
    generator, uranium._core.start_slots(seed, 0)[2]), else `late`."""
    from uranium import _core

    return "early" if _core.start_slots(seed, 0)[2] else "late"


# ---------------------------------------------------------------- command streams


def u16(h: str, i: int) -> int:
    return int(h[2 * i:2 * i + 2], 16) + 256 * int(h[2 * i + 2:2 * i + 4], 16)


def analyze(path: Path) -> tuple[dict, int | None]:
    """First build-order frame per unit type and the first single-SCV far move before 3:00 (scout) of
    the Terran player, up to 8:00. Same rule as runs/dagger/test_table.py (runs/adv8/cmds.py)."""
    d = json.loads(path.read_text(encoding="utf-8"))
    tp = [p["id"] for p in d["players"] if p.get("race") == RACE_TERRAN][0]
    first, base, last_sel, scout = {}, None, 0, None
    for f, p, op, h in d["cmds"]:
        if p != tp or f > CUT:
            continue
        if op in (OP_SELECT, OP_SEL_ADD):
            last_sel = int(h[0:2], 16)
        elif op == OP_BUILD:
            x, y, ty = u16(h, 1), u16(h, 3), u16(h, 5)
            first.setdefault(ty, f)
            if base is None:
                base = (x * 32, y * 32)
        elif op in (OP_RCLICK, OP_ORDER):
            x, y, tag = u16(h, 0), u16(h, 1), u16(h, 2)
            far = op == OP_ORDER and int(h[16:18], 16) == ORDER_MOVE or op == OP_RCLICK and tag == 0xFFFF
            if far and base is not None and scout is None and last_sel == 1 and f < 4320 and abs(x - base[0]) + abs(y - base[1]) > 1200:
                scout = f
    return first, scout


def dump_stage(stage_dir: Path, games: list[int], out: Path, binary: Path | None = None) -> dict[int, Path]:
    """`uranium-replay cmddump` of each game's replay into out/g<n>.json (kept: a later run reuses it)."""
    out.mkdir(parents=True, exist_ok=True)
    binary = binary or replay_bin()
    got = {}
    for n in games:
        j = out / f"g{n}.json"
        rep = stage_dir / "replays" / f"g{n}.rep"
        if not j.exists() and rep.exists():
            if not binary.exists():
                raise SystemExit(f"no {binary}: cargo build --release -p uranium-replay (or set URANIUM_REPLAY_BIN)")
            p = subprocess.run([str(binary), "cmddump", str(rep)], capture_output=True, text=True)
            if p.returncode == 0:
                j.write_text(p.stdout, encoding="utf-8")
        if j.exists():
            got[n] = j
    return got


# ---------------------------------------------------------------- statistics (050-R2)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% interval of k / n (050-R2)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher_p(a: int, n1: int, b: int, n2: int) -> float:
    """Two-sided Fisher exact p of a / n1 vs b / n2 (sum of the tables no more likely than the observed)."""
    k = a + b
    n = n1 + n2

    def pmf(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)

    obs = pmf(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= obs * (1 + 1e-9)))


def median(v: list) -> float | None:
    """statistics.median (the old tables' p50)."""
    import statistics

    return statistics.median(v) if v else None
