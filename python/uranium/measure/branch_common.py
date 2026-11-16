from __future__ import annotations

import json
import tomllib
from pathlib import Path

from uranium.measure import eval_common as ec

ROOT = ec.ROOT
HORIZON_MARGIN = 48
FPS = 24
OUTCOME_METRICS = ("held_8m", "held_s", "scv_8m", "scv_min_1m", "bunker_s", "barracks_s", "army_s")
SESSION_METRICS = ("scv_8m", "scv_min_1m", "scv_4m", "bunker_s", "barracks_s", "army_s", "scout_main_s", "scout_types_4m", "scout_types_6m")
MAP_KEYS = {"FightingSpirit": "fighting_spirit", "CircuitBreaker": "circuit_breaker"}


Refused = ec.Refused


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
    return json.dumps(row, ensure_ascii=False, allow_nan=False)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(dumps(r) + "\n" for r in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def shared_root(arg: str | None) -> Path:
    return Path(arg) if arg else ec.shared_dir()


def rel_to(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as e:
        raise Refused(f"{path} is not under the shared drive {root} (051-R10 paths are relative to it)") from e


def job_dir(arg: str | Path, root: Path) -> Path:
    p = Path(arg)
    return p if p.is_absolute() or p.exists() else root / p


def on_shared(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else root / p


def action_ids() -> dict[str, int]:
    return {a["name"]: int(a["id"]) for a in ec.actions()}


def workspace_version() -> str:
    t = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    return t["workspace"]["package"]["version"]


def current_schemas() -> dict:
    from uranium import _core

    d = json.loads(_core.feature_dims())
    return {"feature": int(d["feature_version"]), "action": int(d["action_version"])}


def condition(seed: int) -> str:
    from uranium import _core

    return "hit" if _core.start_slots(int(seed), 0)[2] else "miss"


def plan_games(plan: dict) -> list[dict]:
    seeds = ec.plan_seeds(plan)
    out = []
    for mi, m in enumerate(plan["matchup"]):
        for g in range(m["games"]):
            mp = m["map"] if m["map"] != "both" else ("fighting_spirit" if g % 2 == 0 else "circuit_breaker")
            out.append({"seed": seeds[len(out)], "map": mp, "matchup": mi, "p0": m["p0"], "p1": m["p1"]})
    return out


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


class ActLog:

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
            try:
                v = json.loads(text)
            except json.JSONDecodeError as e:
                if lenient:
                    return cls(lines, None)
                raise Refused(f"{path} line {n}: not JSON ({e}; a log cut short is incomplete, 051-R2)") from e
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

