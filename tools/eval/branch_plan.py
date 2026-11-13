"""Branch-game plans (spec 051): the trunk plan, and from the trunk job's results the decision points
(`points.jsonl`), the branch list and the branch plan. Deterministic: the same inputs give the same files.

    python tools/eval/branch_plan.py trunk --config config/branch_b1.toml --out <trunk job>/plans/trunk.toml
    python tools/eval/branch_plan.py branches --trunk-job <dir> --stage <stage> --config config/branch_b1.toml \\
        --out <shared>/eval/branch-b1-<date> [--hosts <hosts file>] [--replacements <plan.toml>]

`trunk` (051-R11 ①, 051-R15, AMB-051-1): per map of `[trunks] maps`, the first `hit` first-target-hit and
`miss` miss seeds (020-R21 generator) of the rush check plan's block against `[trunks] opponent`, in list
order; the model `checkpoint` (stochastic) as players[0]; the 051-R1 ③ settings (zero_last_failed, surrender
off, replays, max_frames = H + 48), `act_log` and the config's `[[snapshots]]` decision-point rules (020-R22).
`--exclude-job <dir>` leaves out the seeds of that job's stages (a replacement trunk stage, see below).

`branches` reads the trunk stages after `python -m uranium eval-status --job <trunk job>` (the merged
`<stage>/games.jsonl` and `results.jsonl`, last attempt of each game) and per trunk game n:
- the act logs `<stage>/acts/g<n>-p0.jsonl` (ours) and `-p1.jsonl` (the T0 opponent), complete (051-R2);
- the model session record's `snapshots` (uranium_env::branch::SnapshotTaken: rules, state, state_sha256, i,
  frame, digest, type_mask, scalar) and its state files `<stage>/states/<state>.state`;
then checks (051-R10): act logs complete; our log line i is the snapshot's (frame, digest, state, sha256)
with a decision; the state file exists with that sha256; the checkpoint's `model.safetensors` sha256 starts
with the act log policy's 12 hex digits; the T0 policy is `t0@<workspace version>` (`--t0-policy`); both
logs' feature / action schemas are those of the built `uranium._core` (`--schemas F,A`). A point that fails
is written with `valid = false` and its `skip_reason`; a rule that never fired, or fired on the decision of
an earlier rule (two points on one decision count once, 051-R11 ①), has no record and is listed in
`branch_plan.json`. A trunk is used when every config point is valid; per (map, condition) the first
`hit` / `miss` usable trunks in rush-plan seed order are used (later ones `cell full`). A short cell is
reported, and `--replacements <plan.toml>` writes a trunk stage with the next seeds of that map and
condition not used in the trunk job (add it to the trunk job's stages, play it, run `branches` again with
every trunk stage).

Outputs under `--out` (the branch job folder = the 051-R12 output folder):
- `points.jsonl`: one uranium_env::branch::PointRecord per point (051-R12);
- `plans/branches_plan.jsonl`: one uranium_env::branch::BranchPlanLine per branch game (051-R10), repeats
  outermost, then the points (map order of the config, trunk order, rule order), then the alternatives
  (config order); an alternative whose t-only `force` action is closed at t is dropped there (051-R11 ②);
- `plans/<name>.toml`: the branch plan (020-R22 `branches`, seeds / maps / opponent of each line's trunk,
  the trunks' checkpoint and T0 side, `act_log`, `--verify-state` for 051-AC6);
- `branch_plan.json`: trunks used and skipped with reasons, cells, dropped alternatives;
- `job.toml` with `--hosts` (020-R15 job of the one stage).
Paths in the records and the branch list are relative to the shared drive (`--root`, default
URANIUM_SHARED_DIR; 051-R10).
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import branch_common as bc  # noqa: E402
import eval_common as ec  # noqa: E402

Refused = bc.Refused
SNAPSHOT_KEYS = {"name", "from_frame", "to_frame", "any_open", "after_rule", "after_s"}
RULE_KEYS = {"op", "action", "from_s", "to_s"}


# ---------------------------------------------------------------- config


def load_config(path: Path) -> dict:
    cfg = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    for k in ("name", "checkpoint", "horizon", "repeats", "trunks", "snapshots", "alternative"):
        if k not in cfg:
            raise Refused(f"{path}: no `{k}`")
    ids = bc.action_ids()
    names = [s["name"] for s in cfg["snapshots"]]
    if len(set(names)) != len(names) or not names:
        raise Refused(f"{path}: snapshot rule names must be unique and present (051-R3)")
    for s in cfg["snapshots"]:
        if set(s) - SNAPSHOT_KEYS or not s["from_frame"] < s["to_frame"]:
            raise Refused(f"{path}: snapshot rule {s.get('name')}: keys {sorted(SNAPSHOT_KEYS)}, non-empty window")
        for a in s.get("any_open", []):
            if a not in ids:
                raise Refused(f"{path}: snapshot rule {s['name']}: unknown action {a} (034)")
        # 051-R3 after_rule / after_s (user decision 2026-11-13): an earlier rule, both keys together.
        if ("after_rule" in s) != ("after_s" in s):
            raise Refused(f"{path}: snapshot rule {s['name']}: after_rule and after_s go together (051-R3)")
        if "after_rule" in s and (s["after_rule"] not in names[:names.index(s["name"])] or int(s["after_s"]) < 0):
            raise Refused(f"{path}: snapshot rule {s['name']}: after_rule {s['after_rule']} must be an earlier rule, after_s >= 0 (051-R3)")
    alts = [a["name"] for a in cfg["alternative"]]
    if len(set(alts)) != len(alts):
        raise Refused(f"{path}: alternative names must be unique")
    for a in cfg["alternative"]:
        a.setdefault("rules", [])
        for r in a["rules"]:
            if set(r) != RULE_KEYS or r["op"] not in ("force", "ban") or r["action"] not in ids:
                raise Refused(f"{path}: alternative {a['name']}: rule {r} (051-R7: op force / ban, a 034 action, from_s, to_s)")
            if not (r["from_s"] < r["to_s"] or r["from_s"] == r["to_s"] == 0):
                raise Refused(f"{path}: alternative {a['name']}: empty window {r['from_s']}..{r['to_s']} s (051-R7)")
            if r["op"] == "ban" and r["action"] == "NoOp":
                raise Refused(f"{path}: alternative {a['name']}: NoOp cannot be banned (051-R7 ③)")
    if int(cfg["horizon"]) <= 0 or int(cfg["repeats"]) < 1:
        raise Refused(f"{path}: horizon > 0 and repeats >= 1")
    return cfg


def max_frames(cfg: dict) -> int:
    return int(cfg["horizon"]) + bc.HORIZON_MARGIN


def rush_blocks(cfg: dict) -> dict[str, list[int]]:
    """map -> the rush check plan's seeds against `[trunks] opponent` on that map, in list order (051-R15)."""
    t = cfg["trunks"]
    plan = tomllib.loads((bc.ROOT / t["rush_plan"]).read_text(encoding="utf-8"))
    out: dict[str, list[int]] = {m: [] for m in t["maps"]}
    for g in bc.plan_games(plan):
        if g["map"] in out and g["p1"] == t["opponent"]:
            out[g["map"]].append(int(g["seed"]))
    for m, seeds in out.items():
        if not seeds:
            raise Refused(f"{t['rush_plan']}: no {t['opponent']} game on {m}")
    return out


def pick_seeds(seeds: list[int], counts: dict[str, int], cond, exclude: set[int] = frozenset()) -> list[int]:
    """The first counts[c] seeds of each condition c in list order, list order kept."""
    left = dict(counts)
    out = []
    for s in seeds:
        if s in exclude:
            continue
        c = cond(s)
        if left.get(c, 0) > 0:
            left[c] -= 1
            out.append(s)
    short = {c: n for c, n in left.items() if n > 0}
    if short:
        raise Refused(f"not enough seeds left: {short} short")
    return out


def plan_head(name: str, cfg: dict, header: list[str]) -> list[str]:
    """051-R1 ③ settings of trunk and branch plans."""
    return [*(f"# {h}" for h in header),
            f"name = {bc.toml_value(name)}",
            "save_replay = true",
            "zero_last_failed = true",
            'surrender = "off"',
            f"max_frames = {max_frames(cfg)}",
            f"wall_timeout_s = {int(cfg.get('wall_timeout_s', 7200))}",
            "act_log = true"]


def seeds_and_matchups(blocks: list[tuple[str, list[int]]], p0: dict, p1: dict) -> tuple[list[str], list[str]]:
    seeds, matchups, n = ["seeds = ["], [], 0
    for m, block in blocks:
        seeds.append(f"  # games {n + 1}..{n + len(block)}: {m}")
        for i in range(0, len(block), 8):
            seeds.append("  " + ", ".join(str(s) for s in block[i:i + 8]) + ",")
        n += len(block)
        matchups += ["[[matchup]]", f"map = {bc.toml_value(m)}", f"games = {len(block)}", f"p0 = {bc.toml_value(p0)}", f"p1 = {bc.toml_value(p1)}"]
    seeds.append("]")
    return seeds, matchups


def snapshot_tables(cfg: dict) -> list[str]:
    out = []
    for s in cfg["snapshots"]:
        out += ["[[snapshots]]", f"name = {bc.toml_value(s['name'])}", f"from_frame = {int(s['from_frame'])}",
                f"to_frame = {int(s['to_frame'])}", f"any_open = {bc.toml_value(list(s.get('any_open', [])))}"]
        if "after_rule" in s:
            out += [f"after_rule = {bc.toml_value(s['after_rule'])}", f"after_s = {int(s['after_s'])}"]
    return out


def render_trunk(name: str, cfg: dict, blocks: list[tuple[str, list[int]]], header: list[str]) -> str:
    p0 = {"kind": "model", "checkpoint": cfg["checkpoint"], "sample": "stochastic"}
    seeds, matchups = seeds_and_matchups(blocks, p0, cfg["trunks"]["opponent"])
    return "\n".join([*plan_head(name, cfg, header), *seeds, *snapshot_tables(cfg), *matchups]) + "\n"


def job_stages(job: Path) -> list:
    """`ec.job_stages`, a missing or unreadable job / stage plan file refused (not a bare OSError)."""
    try:
        return ec.job_stages(job)
    except (OSError, ValueError) as e:
        raise Refused(f"{job}: cannot read the job's stages ({e})") from e


def job_seeds(job: Path) -> set[int]:
    out: set[int] = set()
    if (job / "job.toml").is_file():
        for _, _, plan in job_stages(job):
            out.update(int(s) for s in ec.plan_seeds(plan))
    return out


def trunk_plan(cfg: dict, name: str, cond, exclude: set[int] = frozenset(), counts: dict[str, int] | None = None) -> str:
    """051-R11 ①: per map the first hit / miss seeds of the rush block (list order)."""
    t = cfg["trunks"]
    counts = counts or {"hit": int(t["hit"]), "miss": int(t["miss"])}
    blocks = [(m, pick_seeds(seeds, counts, cond, exclude)) for m, seeds in rush_blocks(cfg).items()]
    header = [f"051 branch-game trunks ({cfg['name']}), generated by tools/eval/branch_plan.py trunk (do not edit by hand).",
              f"Per map the first {counts.get('hit', 0)} first-target-hit and {counts.get('miss', 0)} miss seeds of {t['rush_plan']}",
              f"against {t['opponent']} in list order (051-R11 1, 051-R15, 020-R21); 051-R1 settings, H + 48 frames."]
    return render_trunk(name, cfg, blocks, header)


# ---------------------------------------------------------------- trunk results


def merged(stage_dir: Path, what: str) -> list[dict]:
    p = stage_dir / f"{what}.jsonl"
    if not p.is_file():
        raise Refused(f"{p}: no merged stage file (run `python -m uranium eval-status --job <trunk job>` first)")
    return bc.read_jsonl(p)


def checkpoint_sha(root: Path, checkpoint: str) -> str | None:
    p = bc.on_shared(root, checkpoint) / "model.safetensors"
    return bc.file_sha256(p) if p.is_file() else None


def policy_hex(policy: str) -> str | None:
    """`model:<folder>@<12 hex>:<sample>` -> the 12 hex digits (020-R5)."""
    if not policy.startswith("model:"):
        return None
    ident = policy[len("model:"):].rsplit(":", 1)[0]
    return ident.rsplit("@", 1)[1] if "@" in ident else None


def end_of(rec: dict, horizon: int) -> dict:
    """051-R9 ④ from a session record (uranium_env::branch::BranchEnd)."""
    won = rec.get("won")
    result = "draw" if won is None else ("win" if won else "loss")
    return {"result": result, "end_frame": rec.get("end_frame"), "reason": rec.get("reason"), "before_horizon": (rec.get("end_frame") or 0) < horizon}


def trunk_outcome(rec: dict, horizon: int) -> dict:
    """051-R9 ② ④ of the trunk's model session (control only, 051-R11 ⑤): the 050-R7 / 042 metrics of its
    `model` object, held_8m (scv_8m >= 8) and held_s (scv_min_1m > 4) as tools/eval/tables.py, and `end`."""
    m = rec.get("model") or {}
    out = {"held_8m": (m.get("scv_8m") or 0) >= 8, "held_s": (m.get("scv_min_1m") or 0) > 4}
    out.update({k: m[k] for k in bc.SESSION_METRICS if k in m})
    out["end"] = end_of(rec, horizon)
    return out


class Trunk:
    def __init__(self, stage: str, k: int, n: int, game: dict):
        self.stage, self.k, self.n = stage, k, n
        self.seed, self.map, self.p0, self.p1 = int(game["seed"]), game["map"], game["p0"], game["p1"]
        self.condition: str | None = None
        self.reasons: list[str] = []
        self.points: dict[str, dict] = {}  # rule -> PointRecord (valid or not)
        self.missing: dict[str, str] = {}  # rule -> why it has no record
        self.order = (k, n)
        self.used = False

    @property
    def usable(self) -> bool:
        return not self.reasons and not self.missing and all(p["valid"] for p in self.points.values())


def read_trunks(a, cfg: dict, root: Path, job: Path, expect: dict) -> list[Trunk]:
    rules = [s["name"] for s in cfg["snapshots"]]
    horizon = int(cfg["horizon"])
    stages = {plan["name"]: (k, f, plan) for k, f, plan in job_stages(job)}
    sha_cache: dict[str, str | None] = {}
    out = []
    for stage in a.stage:
        if stage not in stages:
            raise Refused(f"{job}: no stage {stage}")
        k, _, plan = stages[stage]
        want = [{**s, "any_open": list(s.get("any_open", []))} for s in cfg["snapshots"]]
        have = [{**s, "any_open": list(s.get("any_open", []))} for s in plan.get("snapshots", [])]
        if have != want:
            raise Refused(f"stage {stage}: its [[snapshots]] are not the config's (051-R11 1)")
        if int(plan.get("max_frames", 86_400)) != max_frames(cfg):
            raise Refused(f"stage {stage}: max_frames {plan.get('max_frames')} is not H + 48 = {max_frames(cfg)} (051-R1)")
        sd = job / stage
        games = {int(g["game_id"]): g for g in merged(sd, "games")}
        recs: dict[int, list[dict]] = {}
        for r in merged(sd, "results"):
            recs.setdefault(int(r["game_id"]), []).append(r)
        for n, game in enumerate(bc.plan_games(plan), start=1):
            t = Trunk(stage, k, n, game)
            out.append(t)
            t.condition = a.condition_fn(t.seed)
            ckpt = t.p0.get("checkpoint", "")
            if ckpt not in sha_cache:
                sha_cache[ckpt] = checkpoint_sha(root, ckpt)
            read_trunk(t, sd, games.get(n), recs.get(n, []), rules, horizon, root, job, sha_cache[ckpt], expect)
    return out


def read_trunk(t: Trunk, sd: Path, game: dict | None, recs: list[dict], rules: list[str], horizon: int, root: Path,
               job: Path, ckpt_sha: str | None, expect: dict) -> None:
    def miss_all(why: str):
        t.reasons.append(why)
        for r in rules:
            t.missing.setdefault(r, why)

    if game is None:
        return miss_all("not played")
    if game.get("outcome") != "Finished":
        return miss_all(f"game {game.get('outcome')}" + (f" ({game['no_result']})" if game.get("no_result") else ""))
    models = [r for r in recs if r.get("model")]
    others = [r for r in recs if not r.get("model")]
    if len(models) != 1 or len(others) != 1:
        return miss_all(f"{len(models)} model and {len(others)} T0 session records")
    rec = models[0]
    acts = sd / "acts" / f"g{t.n}-p0.jsonl"
    opp_acts = sd / "acts" / f"g{t.n}-p1.jsonl"
    try:
        ours, opp = bc.ActLog.read(acts), bc.ActLog.read(opp_acts)
    except (OSError, Refused, json.JSONDecodeError, KeyError) as e:
        return miss_all(f"act log: {e}")
    # 051-R1 ④ / 051-R10: the same checkpoint, T0 policy version and schemas as the branch plan.
    if ours.end.get("players_index") != 0 or opp.end.get("players_index") != 1:
        t.reasons.append("act logs are not players[0] / players[1]")
    if ckpt_sha is None:
        t.reasons.append(f"checkpoint {t.p0.get('checkpoint')} has no model.safetensors on the shared drive")
    elif policy_hex(ours.end.get("policy", "")) != ckpt_sha[:12]:
        t.reasons.append(f"checkpoint sha256 {ckpt_sha[:12]} is not the trunk's ({ours.end.get('policy')})")
    if opp.end.get("policy") != expect["t0_policy"]:
        t.reasons.append(f"T0 policy {opp.end.get('policy')} is not {expect['t0_policy']}")
    for who, log in (("ours", ours), ("opponent", opp)):
        got = {"feature": log.end.get("feature_schema"), "action": log.end.get("action_schema")}
        if got != expect["schemas"]:
            t.reasons.append(f"{who} schemas {got} are not {expect['schemas']}")
    snaps = rec.get("snapshots") or []
    rel = lambda p: bc.rel_to(root, p)  # noqa: E731
    trunk_ref = {"job": job.name, "stage": t.stage, "plan_game": t.n, "hostd_game": int(game.get("hostd_id", 0)),
                 "replay": rel(sd / "replays" / f"g{t.n}.rep"), "acts": rel(acts), "opponent_acts": rel(opp_acts)}
    backend = game.get("backend") or "?"
    seen: dict[str, str] = {}  # state id -> rule of the point it is
    for rule in rules:
        ent = next((s for s in snaps if rule in (s.get("rules") or [])), None)
        if ent is None:
            t.missing[rule] = "rule did not fire (051-R3)"
            continue
        if ent["state"] in seen:
            t.missing[rule] = f"same decision as {seen[ent['state']]} (counts once, 051-R11 1)"
            continue
        seen[ent["state"]] = rule
        i = int(ent["i"])
        line = ours.lines[i] if 0 <= i < len(ours.lines) else None
        if line is None or line.get("decision") is None:
            t.missing[rule] = f"no decision line {i} in the act log"
            continue
        why = list(t.reasons)
        if (line["frame"], line["digest"], line.get("state"), line.get("state_sha256")) != (ent["frame"], ent["digest"], ent["state"], ent["state_sha256"]):
            why.append(f"act log line {i} is not the snapshot's (frame, digest, state, sha256)")
        sf = sd / "states" / f"{ent['state']}.state"
        if not sf.is_file():
            why.append(f"state file {ent['state']}.state missing")
        elif bc.file_sha256(sf) != ent["state_sha256"]:
            why.append(f"state file {ent['state']}.state sha256 differs")
        t.points[rule] = {
            "point": f"{job.name}/{t.stage}/g{t.n}#{rule}",
            "trunk": trunk_ref,
            "seed": t.seed,
            "map": t.map,
            "opponent": (game.get("opponent") or "").split("@")[0],
            "condition": t.condition,
            "checkpoint_sha256": ckpt_sha or "",
            "policy": ours.end.get("policy", ""),
            "t0_policy": opp.end.get("policy", ""),
            "git": ours.end.get("git", ""),
            "schemas": {"feature": ours.end.get("feature_schema"), "action": ours.end.get("action_schema")},
            "backend": backend,
            "precision": bc.PRECISION.get(backend, "?"),
            "infer_seed": rec.get("infer_seed"),
            "i_t": i,
            "f_t": int(ent["frame"]),
            "digest_t": ent["digest"],
            "type_mask": [int(x) for x in ent["type_mask"]],
            "scalar_t": ent["scalar"],
            "trunk_action": line["decision"],
            "state": bc.rel_to(root, sf),
            "state_sha256": ent["state_sha256"],
            "trunk_outcome": trunk_outcome(rec, horizon),
            "valid": not why,
            "skip_reason": "; ".join(why) or None,
        }


def select(trunks: list[Trunk], cfg: dict, counts: dict[str, int], rush: dict[str, list[int]]) -> dict:
    """Per (map, condition) the first usable trunks in rush-plan seed order (051-R11 1)."""
    pos = {m: {s: i for i, s in enumerate(seeds)} for m, seeds in rush.items()}
    cells = {}
    for m in cfg["trunks"]["maps"]:
        for c in ("hit", "miss"):
            cand = sorted((t for t in trunks if t.map == m and t.condition == c), key=lambda t: (pos[m].get(t.seed, 1 << 30), t.order))
            quota = counts.get(c, 0)
            used = 0
            for t in cand:
                if t.usable and used < quota:
                    t.used = True
                    used += 1
                elif t.usable:
                    for p in t.points.values():
                        p["valid"], p["skip_reason"] = False, "cell full (051-R11 1)"
                else:
                    # A trunk with fewer valid points than rules is replaced: none of its points is branched.
                    for p in t.points.values():
                        if p["valid"]:
                            p["valid"], p["skip_reason"] = False, "trunk replaced (051-R11 1)"
            cells[f"{m}/{c}"] = {"quota": quota, "used": used, "short": quota - used}
    return cells


def dropped_alts(point: dict, cfg: dict, ids: dict[str, int]) -> list[dict]:
    """051-R10 / 051-R11 ②: alternatives whose t-only force action is closed in the t observation's mask."""
    out = []
    for a in cfg["alternative"]:
        closed = [r["action"] for r in a["rules"] if r["op"] == "force" and r["from_s"] == r["to_s"] == 0 and ids[r["action"]] not in point["type_mask"]]
        if closed:
            out.append({"point": point["point"], "alt": a["name"], "closed": closed})
    return out


def build(a) -> dict:
    cfg = load_config(a.config)
    root = bc.shared_root(a.root)
    job = bc.job_dir(a.trunk_job, root)
    out = bc.job_dir(a.out, root)
    counts = {"hit": int(cfg["trunks"]["hit"] if a.hit is None else a.hit), "miss": int(cfg["trunks"]["miss"] if a.miss is None else a.miss)}
    alts = [x for x in cfg["alternative"] if a.alts is None or x["name"] in a.alts.split(",")]
    if a.alts is not None and len(alts) != len(a.alts.split(",")):
        raise Refused(f"--alts {a.alts}: not all in the config ({[x['name'] for x in cfg['alternative']]})")
    repeats = int(cfg["repeats"] if a.repeats is None else a.repeats)
    expect = {"t0_policy": a.t0_policy or f"t0@{bc.workspace_version()}", "schemas": a.schemas_fn()}
    trunks = read_trunks(a, cfg, root, job, expect)
    rush = rush_blocks(cfg)
    cells = select(trunks, cfg, counts, rush)
    rules = [s["name"] for s in cfg["snapshots"]]
    ids = bc.action_ids()
    map_order = {m: i for i, m in enumerate(cfg["trunks"]["maps"])}
    used = sorted((t for t in trunks if t.used), key=lambda t: (map_order[t.map], t.order))
    points = [t.points[r] for t in sorted(trunks, key=lambda t: t.order) for r in rules if r in t.points]
    dropped = [d for t in used for r in rules for d in dropped_alts(t.points[r], {**cfg, "alternative": alts}, ids)]
    drop = {(d["point"], d["alt"]) for d in dropped}
    lines = []
    for rep in range(1, repeats + 1):
        for t in used:
            for r in rules:
                p = t.points[r]
                for alt in alts:
                    if (p["point"], alt["name"]) in drop:
                        continue
                    lines.append({"trunk": p["point"].split("#")[0], "point": p["point"], "acts": p["trunk"]["acts"],
                                  "opponent_acts": p["trunk"]["opponent_acts"], "state": p["state"], "state_sha256": p["state_sha256"],
                                  "i_t": p["i_t"], "f_t": p["f_t"], "seed": p["seed"], "map": p["map"], "opponent": p["opponent"],
                                  "alt": {"name": alt["name"], "rules": [dict(x) for x in alt["rules"]]}, "repeat": rep})
    if not lines:
        raise Refused("no branch game: no usable trunk point")
    sides = {(json.dumps(t.p0, sort_keys=True), json.dumps(t.p1, sort_keys=True)) for t in used}
    if len(sides) != 1:
        raise Refused(f"the used trunks differ in checkpoint or opponent side (051-R1 4): {sorted(sides)}")
    p0, p1 = (json.loads(x) for x in next(iter(sides)))
    blocks: list[tuple[str, list[int]]] = []
    for ln in lines:
        if blocks and blocks[-1][0] == ln["map"]:
            blocks[-1][1].append(ln["seed"])
        else:
            blocks.append((ln["map"], [ln["seed"]]))
    name = a.name or cfg["name"]
    header = [f"051 branch games ({cfg['name']}), generated by tools/eval/branch_plan.py branches (do not edit by hand).",
              f"Trunks: {job.name} stages {', '.join(a.stage)}; {len(used)} trunks, {len(lines)} games, repeats outermost (051-R10).",
              "Game n is line n of the branch list (020-R22 3): its trunk's seed, map and opponent."]
    head = plan_head(name, cfg, header) + ['branches = "branches_plan.jsonl"'] + (["verify_state = true"] if a.verify_state else [])
    seeds, matchups = seeds_and_matchups(blocks, p0, p1)
    text = "\n".join([*head, *seeds, *matchups]) + "\n"
    plans = out / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    (plans / f"{name}.toml").write_bytes(text.encode("utf-8"))
    bc.write_jsonl(plans / "branches_plan.jsonl", lines)
    bc.write_jsonl(out / "points.jsonl", points)
    summary = {
        "config": str(a.config), "trunk_job": job.name, "stages": list(a.stage), "plan": f"plans/{name}.toml",
        "games": len(lines), "repeats": repeats, "alternatives": [x["name"] for x in alts], "cells": cells,
        "trunks": [{"trunk": f"{job.name}/{t.stage}/g{t.n}", "seed": t.seed, "map": t.map, "condition": t.condition, "used": t.used,
                    "reasons": t.reasons, "missing_points": t.missing,
                    "points": {r: {"valid": p["valid"], "skip_reason": p["skip_reason"]} for r, p in t.points.items()}}
                   for t in sorted(trunks, key=lambda t: t.order)],
        "replaced": sum(1 for t in trunks if not t.usable),
        "dropped": dropped,
    }
    short = {k: v["short"] for k, v in cells.items() if v["short"] > 0}
    if short and a.replacements:
        exclude = job_seeds(job)
        blocks = []
        for m, seeds in rush.items():
            need = {c: short.get(f"{m}/{c}", 0) for c in ("hit", "miss")}
            if any(need.values()):
                blocks.append((m, pick_seeds(seeds, need, a.condition_fn, exclude)))
        # A stage name the trunk job does not use yet (a second replacement round gets -r2, -r3, ...).
        taken = {plan["name"] for _, _, plan in job_stages(job)} if (job / "job.toml").is_file() else set()
        rname = a.replacement_name or f"{name}-trunk-r"
        if rname in taken:
            if a.replacement_name:
                raise Refused(f"--replacement-name {rname}: the trunk job already has that stage")
            rname = next(f"{name}-trunk-r{i}" for i in range(2, 1000) if f"{name}-trunk-r{i}" not in taken)
        rheader = [f"051-R11 1 replacement trunks ({cfg['name']}): the next seeds of each short (map, condition) cell",
                   f"not used in {job.name}, generated by tools/eval/branch_plan.py branches --replacements (do not edit by hand)."]
        Path(a.replacements).parent.mkdir(parents=True, exist_ok=True)
        Path(a.replacements).write_bytes(render_trunk(rname, {**cfg, "checkpoint": p0["checkpoint"]}, blocks, rheader).encode("utf-8"))
        summary["replacements"] = {"plan": str(a.replacements), "short": short}
    (out / "branch_plan.json").write_bytes((json.dumps(summary, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    if a.hosts:
        hosts = Path(a.hosts)
        (out / "job.toml").write_bytes(f"stages = [{bc.toml_value(f'plans/{name}.toml')}]\nhosts = {bc.toml_value(hosts.as_posix())}\n".encode("utf-8"))
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trunk", help="write the trunk plan (051-R11 1)")
    t.add_argument("--config", type=Path, default=bc.ROOT / "config" / "branch_b1.toml")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--name", default=None, help="plan (stage) name, default <config name>-trunk")
    t.add_argument("--exclude-job", type=Path, action="append", default=[], help="leave out the seeds of this job's stages")
    t.add_argument("--hit", type=int, default=None, help="first-target-hit trunks per map (default the config's)")
    t.add_argument("--miss", type=int, default=None, help="miss trunks per map (default the config's)")
    b = sub.add_parser("branches", help="points, branch list and branch plan from the trunk job (051-R10)")
    b.add_argument("--trunk-job", required=True, help="trunk job folder (as given if it exists, else under URANIUM_SHARED_DIR)")
    b.add_argument("--stage", action="append", required=True, help="trunk stage name (repeat for replacement stages)")
    b.add_argument("--config", type=Path, default=bc.ROOT / "config" / "branch_b1.toml")
    b.add_argument("--out", required=True, help="branch job folder (051-R12 output folder; as given if it exists or is absolute, else under URANIUM_SHARED_DIR)")
    b.add_argument("--root", default=None, help="shared drive (default URANIUM_SHARED_DIR)")
    b.add_argument("--name", default=None, help="branch plan (stage) name, default the config name")
    b.add_argument("--alts", default=None, help="comma list of alternatives to keep (default all, 051-AC6 uses none,force_bunker)")
    b.add_argument("--repeats", type=int, default=None)
    b.add_argument("--hit", type=int, default=None)
    b.add_argument("--miss", type=int, default=None)
    b.add_argument("--verify-state", action="store_true", help="verify_state = true (051-R6 test option, 051-AC6)")
    b.add_argument("--hosts", default=None, help="also write job.toml with this hosts file (relative to the job folder)")
    b.add_argument("--replacements", default=None, help="write the replacement trunk plan here when a cell is short")
    b.add_argument("--replacement-name", default=None)
    b.add_argument("--t0-policy", default=None, help="expected T0 policy (default t0@<workspace version>)")
    b.add_argument("--schemas", default=None, help="expected feature,action schema versions (default the built uranium._core)")
    a = ap.parse_args(argv)
    a.condition_fn = bc.condition
    try:
        if a.cmd == "trunk":
            cfg = load_config(a.config)
            counts = {"hit": int(cfg["trunks"]["hit"] if a.hit is None else a.hit), "miss": int(cfg["trunks"]["miss"] if a.miss is None else a.miss)}
            exclude = set().union(*(job_seeds(j) for j in a.exclude_job)) if a.exclude_job else set()
            text = trunk_plan(cfg, a.name or f"{cfg['name']}-trunk", a.condition_fn, exclude, counts)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_bytes(text.encode("utf-8"))
            print(f"{a.out}: {sum(counts.values()) * len(cfg['trunks']['maps'])} trunk games")
            return
        if a.schemas:
            f, s = (int(x) for x in a.schemas.split(","))
            a.schemas_fn = lambda: {"feature": f, "action": s}
        else:
            a.schemas_fn = bc.current_schemas
        s = build(a)
    except Refused as e:
        raise SystemExit(f"REFUSED: {e}") from e
    print(f"{a.out}: {s['games']} branch games from {sum(t['used'] for t in s['trunks'])} trunks; cells {s['cells']}")
    if s["dropped"]:
        print(f"dropped alternatives (force action closed at t): {len(s['dropped'])}")


if __name__ == "__main__":
    main()
