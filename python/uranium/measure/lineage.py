from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

from uranium.measure import eval_common as ec

ROOT = ec.ROOT
LINEAGE = ROOT / "config" / "protocol" / "lineage.toml"
RECORD = "protocol.json"
PLAN_DEFAULTS = {"seed_base": 1000, "same_seed": False, "max_frames": 86_400, "wall_timeout_s": 1800, "privileged": False,
                 "save_replay": False, "zero_last_failed": False, "surrender": "shadow", "surrender_chat": "gg"}
CKPT, NAME = "@CHECKPOINT@", "@NAME@"


Refused = ec.Refused


def load_lineage(path: Path = LINEAGE) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def model_sides(plan: dict):
    for m in plan["matchup"]:
        for s in ("p0", "p1"):
            if m[s]["kind"] == "model":
                yield m[s]


def normalize(plan: dict) -> dict:
    p = {**PLAN_DEFAULTS, **{k: v for k, v in plan.items() if k not in ("name", "matchup")}}
    if "seeds" in p:
        p.pop("seed_base")
        p.pop("same_seed")
    ms = []
    for m in plan["matchup"]:
        m = {"p1_random_ratio": 0.0, **m}
        sides = {}
        for s in ("p0", "p1"):
            side = dict(m[s])
            if side["kind"] == "model":
                side = {"sample": "argmax", **side, "checkpoint": CKPT}
            elif side["kind"] == "t0":
                side = {"build": "standard", **side}
            sides[s] = side
        ms.append({**m, **sides})
    p["matchup"] = ms
    return p


def conditions_sha256(plan: dict) -> str:
    return hashlib.sha256(json.dumps(normalize(plan), sort_keys=True).encode("utf-8")).hexdigest()


def template(lin: dict, plan_id: str) -> tuple[str, dict]:
    spec = lin["plan"][plan_id]
    text = (ROOT / spec["template"]).read_text(encoding="utf-8")
    return text, tomllib.loads(text)


def check_template(lin: dict, plan_id: str) -> dict:
    spec = lin["plan"][plan_id]
    text, plan = template(lin, plan_id)
    sides = list(model_sides(plan))
    if plan.get("name") != NAME or not sides or any(s.get("checkpoint") != CKPT for s in sides):
        raise Refused(f"template {spec['template']}: name / every model checkpoint (p0 and p1) must be the placeholders")
    h = conditions_sha256(plan)
    if h != spec["conditions_sha256"]:
        raise Refused(f"template {spec['template']} conditions {h[:12]} differ from lineage.toml {spec['conditions_sha256'][:12]}")
    n = normalize(plan)
    pinned = {"zero_last_failed": n["zero_last_failed"], "max_frames": n["max_frames"], "games": sum(m["games"] for m in n["matchup"]),
              "sample": sorted({s["sample"] for s in model_sides(n)})}
    want = {"zero_last_failed": spec["zero_last_failed"], "max_frames": spec["max_frames"], "games": spec["games"], "sample": sorted(spec["sample"])}
    if pinned != want:
        raise Refused(f"template {spec['template']}: {pinned} differs from lineage.toml {want}")
    return plan


def render(lin: dict, plan_id: str, checkpoint: str, name: str) -> str:
    check_template(lin, plan_id)
    text, _ = template(lin, plan_id)
    out = text.replace(f'name = "{NAME}"', f'name = "{name}"', 1).replace(f'"{CKPT}"', f'"{checkpoint}"')
    plan = tomllib.loads(out)
    body = "\n".join(l for l in out.splitlines() if not l.lstrip().startswith("#"))
    if plan["name"] != name or any(s["checkpoint"] != checkpoint for s in model_sides(plan)) or CKPT in body or NAME in body:
        raise Refused(f"template {plan_id}: substitution of the name / checkpoint failed")
    if conditions_sha256(plan) != lin["plan"][plan_id]["conditions_sha256"]:
        raise Refused(f"template {plan_id}: rendered plan changed its conditions")
    return out


def identify(lin: dict, plan: dict) -> str | None:
    if ec.is_branch_plan(plan):
        return None
    h = conditions_sha256(plan)
    for pid, spec in lin["plan"].items():
        if spec["conditions_sha256"] == h:
            return pid
    return None


def read_record(job: Path) -> dict | None:
    f = job / RECORD
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None
