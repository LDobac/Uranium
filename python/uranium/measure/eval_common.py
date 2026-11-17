from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from uranium import env

ROOT = Path(__file__).resolve().parents[3]
ACTIONS_TOML = ROOT / "crates" / "uranium-env" / "tables" / "actions.toml"
PRECISION = {"cuda": "bf16-mixed", "rocm": "f32", "cpu": "f32"}

FPS = 24
CUT = 11520
RACE_TERRAN = 1
BUNKER, CC = 125, 106
OP_BUILD, OP_SELECT, OP_SEL_ADD, OP_RCLICK, OP_ORDER = 12, 9, 10, 20, 21
ORDER_MOVE = 6


class Refused(Exception):
    pass


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def actions() -> list[dict]:
    return tomllib.loads(ACTIONS_TOML.read_text(encoding="utf-8"))["action"]


def load_env() -> None:
    env.load()


def shared_dir() -> Path:
    load_env()
    v = os.environ.get("URANIUM_SHARED_DIR")
    if not v:
        raise SystemExit("URANIUM_SHARED_DIR is not set (.env, 001-R9)")
    return Path(v)


def cache_dir() -> Path:
    load_env()
    return Path(os.environ.get("URANIUM_CACHE_DIR") or ROOT / "runs")


def replay_bin() -> Path:
    load_env()
    v = os.environ.get("URANIUM_REPLAY_BIN")
    if v:
        return Path(v)
    exe = "uranium-replay.exe" if os.name == "nt" else "uranium-replay"
    return ROOT / "target" / "release" / exe


def job_stages(job: Path) -> list[tuple[int, Path, dict]]:
    spec = tomllib.loads((job / "job.toml").read_text(encoding="utf-8"))
    out = []
    for k, rel in enumerate(spec["stages"], start=1):
        p = job / rel
        out.append((k, p, tomllib.loads(p.read_text(encoding="utf-8"))))
    return out


_BLANK = b"\x00 \t\r\n\f\v"


class Rows(list):

    def __init__(self, rows=(), skipped=()):
        super().__init__(rows)
        self.skipped: list[dict] = list(skipped)


_WARNED: set[tuple[str, int, int, str]] = set()


def read_jsonl_dir(d: Path) -> Rows:
    out = Rows()
    for f in sorted(d.glob("*.jsonl")):
        for i, line in enumerate(f.read_bytes().splitlines(), start=1):
            for j, piece in enumerate(re.split(rb"\x00+", line)):
                s = piece.strip(_BLANK)
                if not s:
                    continue
                try:
                    r = json.loads(s.decode("utf-8"))
                    if not isinstance(r, dict):
                        raise ValueError(f"not a JSON object ({type(r).__name__})")
                except ValueError as e:
                    key = (str(f), i, j, str(e))
                    if key not in _WARNED:
                        _WARNED.add(key)
                        print(f"warning: {f}:{i}: skipped an undecodable line ({e})", file=sys.stderr)
                    out.skipped.append({"file": f.name, "line": i, "error": str(e)})
                    continue
                out.append(r)
    return out


def read_results(job: Path) -> Rows:
    return read_jsonl_dir(job / "results")


def skipped_lines(results: list[dict], ledger: list[dict]) -> list[dict]:
    return [{**s, "file": f"{sub}/{s['file']}"} for sub, rows in (("results", results), ("games", ledger))
            for s in getattr(rows, "skipped", [])]


STAGE_BASE, ATTEMPT_BASE = 100_000, 10_000_000


def game_key(game_id: int) -> int:
    return int(game_id) % ATTEMPT_BASE


def stage_of(game_id: int) -> int:
    return game_key(game_id) // STAGE_BASE


def attempt_of(game_id: int) -> int:
    return int(game_id) // ATTEMPT_BASE


def model_rows(rows: list[dict], stage: int) -> dict[int, dict]:
    best: dict[int, tuple[int, dict]] = {}
    for r in rows:
        if r.get("model") and stage_of(r["game_id"]) == stage:
            n, a = game_key(r["game_id"]) % STAGE_BASE, attempt_of(r["game_id"])
            if n not in best or a >= best[n][0]:
                best[n] = (a, r)
    return {n: r for n, (_, r) in best.items()}


def read_ledger(job: Path) -> Rows:
    return read_jsonl_dir(job / "games")


_EXITS = re.compile(r"launcher exit codes \[([^\]]*)\]")


def opponent_crashes(ledger: list[dict]) -> set[int]:
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
    out: dict[str, int] = {}
    for g in ledger:
        if g.get("outcome") == "Finished":
            k = f"{g.get('backend')}@{g.get('infer_host')}"
            out[k] = out.get(k, 0) + 1
    return out


BRANCH_KEYS = ("act_log", "snapshots", "branches", "verify_state")


def is_branch_plan(plan: dict) -> bool:
    return any(plan.get(k) for k in BRANCH_KEYS)


def plan_seeds(plan: dict) -> list[int]:
    total = sum(m["games"] for m in plan["matchup"])
    if "seeds" in plan:
        return list(plan["seeds"])
    base = plan.get("seed_base", 1000)
    return [base if plan.get("same_seed") else base + n for n in range(1, total + 1)]


def opponent_label(p1: dict) -> str:
    return p1.get("name") or f"t0 {p1.get('build', 'standard')}"


def game_opponents(plan: dict) -> list[str]:
    opp = []
    for m in plan["matchup"]:
        opp += [opponent_label(m["p1"])] * m["games"]
    return opp


def start_condition(seed: int) -> str:
    from uranium import _core

    return "early" if _core.start_slots(seed, 0)[2] else "late"


def u16(h: str, i: int) -> int:
    return int(h[2 * i:2 * i + 2], 16) + 256 * int(h[2 * i + 2:2 * i + 4], 16)


SCOUT_BEFORE = 4320
SCOUT_DIST = 1200


def _scout_old(op: int, h: str) -> tuple[bool, int, int]:
    x, y, tag = u16(h, 0), u16(h, 1), u16(h, 2)
    far = op == OP_ORDER and int(h[16:18], 16) == ORDER_MOVE or op == OP_RCLICK and tag == 0xFFFF
    return far, x, y


def _scout_fixed(op: int, h: str) -> tuple[bool, int, int]:
    x, y, tag = u16(h, 0), u16(h, 2), u16(h, 4)
    far = op == OP_ORDER and int(h[16:18], 16) == ORDER_MOVE or op == OP_RCLICK and tag in (0, 0xFFFF)
    return far, x, y


def analyze_both(path: Path, slot: int = 0) -> tuple[dict, int | None, int | None]:
    d = json.loads(path.read_text(encoding="utf-8"))
    tp = model_player(d, slot)
    if tp is None:
        tp = [p["id"] for p in d["players"] if p.get("race") == RACE_TERRAN][0]
    first, base, last_sel = {}, None, 0
    scout: list[int | None] = [None, None]
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
        elif op in (OP_RCLICK, OP_ORDER) and base is not None and last_sel == 1 and f < SCOUT_BEFORE:
            for i, read in enumerate((_scout_old, _scout_fixed)):
                far, x, y = read(op, h)
                if far and scout[i] is None and abs(x - base[0]) + abs(y - base[1]) > SCOUT_DIST:
                    scout[i] = f
    return first, scout[0], scout[1]


def analyze(path: Path, slot: int = 0) -> tuple[dict, int | None]:
    first, scout, _ = analyze_both(path, slot)
    return first, scout


TEACHER_MODE_CMD_UNTIL = 6 * 60 * FPS
TEACHER_MODE_CMD_MINUTES = 6
TEACHER_MODE_CMDS_PER_MIN = 150
TEACHER_MODE_REFINERY_BEFORE = 4 * 60 * FPS
REFINERY = 110
FACTORY = 113
INF_FRAME = 10**9
ZERG_BUILD_BEFORE = 5 * 60 * FPS
TP_STYLE_CMDS_PER_MIN = 120


def model_player(d: dict, slot: int) -> int | None:
    players = d.get("players", [])
    terran = [p["id"] for p in players if p.get("race") == RACE_TERRAN]
    if len(terran) == 1:
        return terran[0]
    if 0 <= slot < len(players) and players[slot].get("race") == RACE_TERRAN:
        return players[slot]["id"]
    return None


def teacher_mode(path: Path, slot: int) -> dict | None:
    d = json.loads(path.read_text(encoding="utf-8"))
    pid = model_player(d, slot)
    if pid is None:
        return None
    cmds, first = 0, {}
    for f, p, op, h in d["cmds"]:
        if p != pid:
            continue
        if f < TEACHER_MODE_CMD_UNTIL:
            cmds += 1
        if op == OP_BUILD and f < ZERG_BUILD_BEFORE:
            first.setdefault(u16(h, 5), f)
    refinery_4m = first.get(REFINERY, INF_FRAME) < TEACHER_MODE_REFINERY_BEFORE
    refinery_5m, factory_5m = REFINERY in first, FACTORY in first
    zerg_build = cmds < TEACHER_MODE_CMDS_PER_MIN * TEACHER_MODE_CMD_MINUTES and not refinery_5m and not factory_5m
    return {"cmds_0_6m": cmds, "cmds_per_min_0_6m": cmds / TEACHER_MODE_CMD_MINUTES, "refinery_order_4m": refinery_4m,
            "teacher_mode": cmds < TEACHER_MODE_CMDS_PER_MIN * TEACHER_MODE_CMD_MINUTES and not refinery_4m,
            "refinery_order_5m": refinery_5m, "factory_order_5m": factory_5m, "zerg_build": zerg_build,
            "tp_style": cmds < TP_STYLE_CMDS_PER_MIN * TEACHER_MODE_CMD_MINUTES and not zerg_build}


def dump_stage(stage_dir: Path, games: list[int], out: Path, binary: Path | None = None) -> dict[int, Path]:
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


OPS_CACHE_VERSION = 1


def _opsdump(binary: Path, rep: Path, slot: int) -> dict:
    p = subprocess.run([str(binary), "opsdump", "--perspective", str(slot), str(rep)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    if p.returncode != 0 or len(lines) != 1:
        return {"error": (p.stderr.strip().splitlines() or [f"exit {p.returncode}"])[-1]}
    try:
        return json.loads(lines[0])
    except ValueError as e:
        return {"error": f"unreadable opsdump line ({e})"}


def ops_stage(stage_dir: Path, games: dict[int, int], out: Path, binary: Path | None = None, jobs: int = 1) -> tuple[dict[int, dict], dict[int, str]]:
    got: dict[int, dict] = {}
    todo: list[tuple[int, int, Path]] = []
    for n, slot in sorted(games.items()):
        j = out / f"g{n}.ops.json"
        if j.exists():
            d = json.loads(j.read_text(encoding="utf-8"))
            if d.get("perspective") == slot and d.get("ops") is not None and d.get("cache_version") == OPS_CACHE_VERSION:
                got[n] = d
                continue
        rep = stage_dir / "replays" / f"g{n}.rep"
        if rep.exists():
            todo.append((n, slot, rep))
    failed: dict[int, str] = {}
    if todo:
        binary = binary or replay_bin()
        if not binary.exists():
            raise SystemExit(f"no {binary}: cargo build --release -p uranium-replay (or set URANIUM_REPLAY_BIN)")
        from concurrent.futures import ThreadPoolExecutor

        print(f"opsdump: {len(todo)} replay(s) of {stage_dir.name} ({max(1, jobs)} at a time)", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            for (n, slot, _), d in zip(todo, pool.map(lambda t: _opsdump(binary, t[2], t[1]), todo)):
                if d.get("ops") is None:
                    failed[n] = d.get("error") or "no ops"
                    continue
                out.mkdir(parents=True, exist_ok=True)
                (out / f"g{n}.ops.json").write_text(json.dumps({**d, "cache_version": OPS_CACHE_VERSION}) + "\n", encoding="utf-8")
                got[n] = d
    return got, failed


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher_p(a: int, n1: int, b: int, n2: int) -> float:
    k = a + b
    n = n1 + n2

    def pmf(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)

    obs = pmf(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= obs * (1 + 1e-9)))


def median(v: list) -> float | None:
    import statistics

    return statistics.median(v) if v else None
