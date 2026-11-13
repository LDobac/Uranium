"""Shared helpers of the branch-game tools (spec 051): branch_plan.py and branch_collect.py.

The record formats are those of `crates/uranium-env/src/branch.rs` (051-R2 act logs, 051-R10 branch list
lines, 051-R12 `points.jsonl` / `branches.jsonl`); this module only reads and writes them as JSON, with
the same checks where a tool relies on them. Paths in the branch list and the records are relative to
the shared drive (061, 051-R10). No host address or personal path is written here (001-R10).
"""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_common as ec  # noqa: E402

ROOT = ec.ROOT
ACTIONS_TOML = ROOT / "crates" / "uranium-env" / "tables" / "actions.toml"
# 051-R1: plans run to max_frames = H + HORIZON_MARGIN (uranium_env::branch::HORIZON_MARGIN).
HORIZON_MARGIN = 48
FPS = 24
# 032-R8 precision of an inference backend (tools/eval/protocol.py PRECISION).
PRECISION = {"cuda": "bf16-mixed", "rocm": "f32", "cpu": "f32"}
# 051-R11 ④ outcome keys (uranium_env::branch::OUTCOME_METRICS) and the other 050-R7 / 042 session
# metrics of a model session record's `model` object kept in `trunk_outcome`.
OUTCOME_METRICS = ("held_8m", "held_s", "scv_8m", "scv_min_1m", "bunker_s", "barracks_s", "army_s")
SESSION_METRICS = ("scv_8m", "scv_min_1m", "scv_4m", "bunker_s", "barracks_s", "army_s", "scout_main_s", "scout_types_4m", "scout_types_6m")
MAP_KEYS = {"FightingSpirit": "fighting_spirit", "CircuitBreaker": "circuit_breaker"}


class Refused(Exception):
    """An input the tool cannot use as a whole (bad config, missing merged stage files)."""


# ---------------------------------------------------------------- files


def read_jsonl(path: Path) -> list[dict]:
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise Refused(f"{path} line {n}: {e}") from e
    return out


def dumps(row: dict) -> str:
    """One JSON line; NaN / infinity refused (051-R12: they would not read back)."""
    return json.dumps(row, ensure_ascii=False, allow_nan=False)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(dumps(r) + "\n" for r in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shared_root(arg: str | None) -> Path:
    """--root, else URANIUM_SHARED_DIR (061)."""
    return Path(arg) if arg else ec.shared_dir()


def rel_to(root: Path, path: Path) -> str:
    """`path` relative to the shared drive with `/` separators (051-R10: the same games on any worker)."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as e:
        raise Refused(f"{path} is not under the shared drive {root} (051-R10 paths are relative to it)") from e


def on_shared(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else root / p


# ---------------------------------------------------------------- tables and versions


def action_ids() -> dict[str, int]:
    """034 action name -> id from the compiled-in table of uranium-env."""
    t = tomllib.loads(ACTIONS_TOML.read_text(encoding="utf-8"))
    return {a["name"]: int(a["id"]) for a in t["action"]}


def workspace_version() -> str:
    """The Cargo workspace version: the T0 policy version `t0@<ver>` of an agent built from this checkout."""
    t = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    return t["workspace"]["package"]["version"]


def current_schemas() -> dict:
    """{"feature", "action"} schema versions of the built `uranium._core` (030, 034)."""
    from uranium import _core

    d = json.loads(_core.feature_dims())
    return {"feature": int(d["feature_version"]), "action": int(d["action_version"])}


def condition(seed: int) -> str:
    """020-R21 first-target condition of a seed (`hit` / `miss`, 051-R15), the generator in uranium._core."""
    from uranium import _core

    return "hit" if _core.start_slots(int(seed), 0)[2] else "miss"


# ---------------------------------------------------------------- plans


def plan_games(plan: dict) -> list[dict]:
    """Per plan game n (index n - 1): {"seed", "map", "matchup", "p0", "p1"} as `plan::expand` (020-R20 seeds,
    `both` alternating from Fighting Spirit inside a matchup)."""
    seeds = ec.plan_seeds(plan)
    out = []
    for mi, m in enumerate(plan["matchup"]):
        for g in range(m["games"]):
            mp = m["map"] if m["map"] != "both" else ("fighting_spirit" if g % 2 == 0 else "circuit_breaker")
            out.append({"seed": seeds[len(out)], "map": mp, "matchup": mi, "p0": m["p0"], "p1": m["p1"]})
    return out


def is_branch_plan(plan: dict) -> bool:
    """A plan with a 051 key (020-R22): its games are branch-game trunks or branches, never evaluation games (050)."""
    return ec.is_branch_plan(plan)


def toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {toml_value(x)}" for k, x in v.items()) + " }"
    raise Refused(f"cannot write {v!r} as TOML")


# ---------------------------------------------------------------- act logs (051-R2)


class ActLog:
    """A complete act log: observation lines and the end line, with the checks of `ActLog::read` (indices
    from 0, increasing frames, the end line's counts, no line after it). `lenient`: a log without its end
    line (`end` None) or with other counts is read as it is (prefix comparisons only)."""

    def __init__(self, lines: list[dict], end: dict | None):
        self.lines = lines
        self.end = end

    @classmethod
    def read(cls, path: Path, lenient: bool = False) -> "ActLog":
        lines, end = [], None
        for n, text in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not text.strip():
                continue
            if end is not None:
                raise Refused(f"{path} line {n}: line after the end line")
            v = json.loads(text)
            if "end" in v:
                end = v["end"]
                continue
            if v.get("i") != len(lines):
                raise Refused(f"{path} line {n}: observation index {v.get('i')} (expected {len(lines)})")
            if lines and not v["frame"] > lines[-1]["frame"]:
                raise Refused(f"{path} line {n}: frame {v['frame']} not after {lines[-1]['frame']}")
            if (v.get("state") is None) != (v.get("state_sha256") is None):
                raise Refused(f"{path} line {n}: state and state_sha256 go together (051-R3)")
            lines.append(v)
        if end is None:
            if lenient:
                return cls(lines, None)
            raise Refused(f"{path}: act log without its end line (incomplete, 051-R2)")
        decisions = sum(1 for x in lines if x.get("decision") is not None)
        if not lenient and (end.get("obs") != len(lines) or end.get("decisions") != decisions):
            raise Refused(f"{path}: end line counts {end.get('obs')} observations / {end.get('decisions')} decisions, "
                          f"the log holds {len(lines)} / {decisions}")
        return cls(lines, end)


def compare_prefix(trunk: list[dict], branch: list[dict], before_frame: int) -> dict:
    """051-R12 `prefix.opponent`: `uranium_env::branch::compare_prefix` (same index, frame and digest)."""
    same = lambda a, b: a["frame"] == b["frame"] and a["digest"] == b["digest"]  # noqa: E731
    prefix = []
    for x in trunk:
        if x["frame"] >= before_frame:
            break
        prefix.append(x)
    matched = sum(1 for a, b in zip(prefix, branch) if same(a, b))
    first = next((i for i, (a, b) in enumerate(zip(trunk, branch)) if not same(a, b)), None)
    if first is None and len(trunk) != len(branch):
        first = min(len(trunk), len(branch))
    frame = None
    if first is not None:
        x = branch[first] if first < len(branch) else trunk[first] if first < len(trunk) else None
        frame = 0 if x is None else x["frame"]
    return {"obs": len(prefix), "matched": matched, "first_mismatch_frame": frame}

