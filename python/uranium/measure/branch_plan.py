"""Branch-game plans: the trunk plan, then points, branch list and branch plan from the trunk job"""

from __future__ import annotations

import argparse
import functools
import json
import tomllib
from fractions import Fraction
from pathlib import Path

from uranium.diag.planning.dump import op_actions
from uranium.measure import branch_common as bc
from uranium.measure import eval_common as ec
from uranium.measure import rush_check_plan as rcp

Refused = bc.Refused
SNAPSHOT_KEYS = {"name", "from_frame", "to_frame", "any_open", "after_rule", "after_s", "after_gap", "min_open_ops"}
RULE_KEYS = {"op", "action", "from_s", "to_s"}
MIX_KEYS = {"games", "maps", "opponents"}
B2_SAMPLE_FIRST = 991
F8M = 8 * 60 * 24


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
        if ("after_rule" in s) != ("after_s" in s):
            raise Refused(f"{path}: snapshot rule {s['name']}: after_rule and after_s go together (051-R3)")
        if "after_rule" in s and (s["after_rule"] not in names[:names.index(s["name"])] or not isinstance(s["after_s"], int)
                                  or isinstance(s["after_s"], bool) or not 0 <= s["after_s"] < 2**32 // 24):
            raise Refused(f"{path}: snapshot rule {s['name']}: after_rule {s['after_rule']} must be an earlier rule, after_s an integer >= 0 (051-R3)")
        if s.get("after_gap", False) is not False and (s["after_gap"] is not True or s.get("after_s", 0) < 1):
            raise Refused(f"{path}: snapshot rule {s['name']}: after_gap = true goes with after_rule and after_s >= 1 (051-R3)")
        if "min_open_ops" in s and (not isinstance(s["min_open_ops"], int) or isinstance(s["min_open_ops"], bool) or s["min_open_ops"] < 1):
            raise Refused(f"{path}: snapshot rule {s['name']}: min_open_ops is an integer >= 1 (051-R3)")
    alts = [a["name"] for a in cfg["alternative"]]
    if len(set(alts)) != len(alts):
        raise Refused(f"{path}: alternative names must be unique")
    for a in cfg["alternative"]:
        a.setdefault("rules", [])
        if "policy_rank" in a:
            k = a["policy_rank"]
            if a["rules"] or not isinstance(k, int) or isinstance(k, bool) or k < 1:
                raise Refused(f"{path}: alternative {a['name']}: policy_rank is an integer >= 1 without rules (051-R16 3)")
        for r in a["rules"]:
            if set(r) != RULE_KEYS or r["op"] not in ("force", "ban") or r["action"] not in ids:
                raise Refused(f"{path}: alternative {a['name']}: rule {r} (051-R7: op force / ban, a 034 action, from_s, to_s)")
            if not (r["from_s"] < r["to_s"] or r["from_s"] == r["to_s"] == 0):
                raise Refused(f"{path}: alternative {a['name']}: empty window {r['from_s']}..{r['to_s']} s (051-R7)")
            if r["op"] == "ban" and r["action"] == "NoOp":
                raise Refused(f"{path}: alternative {a['name']}: NoOp cannot be banned (051-R7 ③)")
    if int(cfg["horizon"]) <= 0 or int(cfg["repeats"]) < 1:
        raise Refused(f"{path}: horizon > 0 and repeats >= 1")
    if is_mix(cfg):
        t = cfg["trunks"]
        if set(t) != MIX_KEYS or int(t["games"]) < 1 or not t["maps"] or not t["opponents"]:
            raise Refused(f"{path}: [trunks] of an opponent mix has exactly {sorted(MIX_KEYS)}, games >= 1 (051-R16)")
        for o in t["opponents"]:
            if (set(o) != {"share", "p1"} or o["p1"].get("kind") != "t0" or "race" not in o["p1"] or set(o["p1"]) - {"kind", "race", "build"}
                    or not int(o["share"]) > 0):
                raise Refused(f"{path}: [trunks] opponent {o}: share > 0 and a T0 p1 (051-R1 1)")
    return cfg


def is_mix(cfg: dict) -> bool:
    return "opponents" in cfg["trunks"]


def split(n: int, weights: list[int]) -> list[int]:
    total = sum(weights)
    ideal = [Fraction(n * w, total) for w in weights]
    base = [int(x) for x in ideal]
    for i in sorted(range(len(weights)), key=lambda i: (-(ideal[i] - base[i]), i))[:n - sum(base)]:
        base[i] += 1
    return base


def mix_cells(cfg: dict, games: int | None = None) -> list[tuple[str, dict, int]]:
    t = cfg["trunks"]
    games = int(t["games"] if games is None else games)
    maps = list(t["maps"])
    out, per_map = [], dict.fromkeys(maps, 0)
    for o, n in zip(t["opponents"], split(games, [int(o["share"]) for o in t["opponents"]])):
        half = dict.fromkeys(maps, n // len(maps))
        for m in sorted(maps, key=lambda m: (per_map[m], maps.index(m)))[:n % len(maps)]:
            half[m] += 1
        for m in maps:
            per_map[m] += half[m]
            if half[m]:
                out.append((m, o["p1"], half[m]))
    return out


def cell_key(m: str, p1: dict) -> str:
    return f"{m}/{p1['race']}" + (f"-{p1['build']}" if p1.get("build") else "")


def b2_seeds(first: int, n: int, exclude: set[int] = frozenset(), end: int | None = None) -> list[int]:
    out, k = [], first
    while len(out) < n:
        if k not in rcp.BRANCH_B2_BLOCK or (end is not None and k >= end):
            raise Refused(f"the B2 seed block {rcp.BRANCH_B2_BLOCK[0]}..{rcp.BRANCH_B2_BLOCK[-1]} is used up from {first} (051-R16)")
        s = rcp.branch_b2_seed(k)
        if s not in exclude:
            out.append(s)
        k += 1
    others = rcp.taken_seeds(bc.ROOT / "config" / "plans", ()) | rcp.reserved_seeds(skip_branch_b2=True)
    if others & set(out):
        raise Refused("a B2 seed is another plan's or another reserved block's (051-R16)")
    return out


def mix_trunk_plan(cfg: dict, name: str, first: int = 1, games: int | None = None) -> str:
    cells = mix_cells(cfg, games)
    seeds = b2_seeds(first, sum(n for *_, n in cells))
    blocks, k = [], 0
    for m, p1, n in cells:
        blocks.append((m, p1, seeds[k:k + n]))
        k += n
    return render_trunk(name, cfg, blocks)


def max_frames(cfg: dict) -> int:
    return int(cfg["horizon"]) + bc.HORIZON_MARGIN


def rush_blocks(cfg: dict) -> dict[str, list[int]]:
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


def plan_head(name: str, cfg: dict) -> list[str]:
    return [f"name = {bc.toml_value(name)}",
            "save_replay = true",
            "zero_last_failed = true",
            'surrender = "off"',
            f"max_frames = {max_frames(cfg)}",
            f"wall_timeout_s = {int(cfg.get('wall_timeout_s', 7200))}",
            "act_log = true"]


def seeds_and_matchups(blocks: list[tuple[str, dict, list[int]]], p0: dict) -> tuple[list[str], list[str]]:
    seeds, matchups = ["seeds = ["], []
    for m, p1, block in blocks:
        for i in range(0, len(block), 8):
            seeds.append("  " + ", ".join(str(s) for s in block[i:i + 8]) + ",")
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
        if s.get("after_gap"):
            out.append("after_gap = true")
        if "min_open_ops" in s:
            out.append(f"min_open_ops = {int(s['min_open_ops'])}")
    return out


def render_trunk(name: str, cfg: dict, blocks: list[tuple[str, dict, list[int]]]) -> str:
    p0 = {"kind": "model", "checkpoint": cfg["checkpoint"], "sample": "stochastic"}
    seeds, matchups = seeds_and_matchups(blocks, p0)
    return "\n".join([*plan_head(name, cfg), *seeds, *snapshot_tables(cfg), *matchups]) + "\n"


def job_stages(job: Path) -> list:
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


def excluded_seeds(jobs: list[Path], root: str | Path | None) -> set[int]:
    out: set[int] = set()
    for j in jobs:
        p = Path(j)
        job = p if p.is_absolute() or p.exists() else bc.shared_root(root) / p
        if not (job / "job.toml").is_file():
            raise Refused(f"--exclude-job {j}: no job.toml in {job}")
        out |= job_seeds(job)
    return out


def trunk_plan(cfg: dict, name: str, cond, exclude: set[int] = frozenset(), counts: dict[str, int] | None = None) -> str:
    t = cfg["trunks"]
    counts = counts or {"hit": int(t["hit"]), "miss": int(t["miss"])}
    blocks = [(m, t["opponent"], pick_seeds(seeds, counts, cond, exclude)) for m, seeds in rush_blocks(cfg).items()]
    return render_trunk(name, cfg, blocks)


def merged(stage_dir: Path, what: str) -> list[dict]:
    p = stage_dir / f"{what}.jsonl"
    if not p.is_file():
        raise Refused(f"{p}: no merged stage file (run `python -m uranium eval-status --job <trunk job>` first)")
    return bc.read_jsonl(p)


def checkpoint_sha(root: Path, checkpoint: str) -> str | None:
    p = bc.on_shared(root, checkpoint) / "model.safetensors"
    return ec.file_sha256(p) if p.is_file() else None


def policy_hex(policy: str) -> str | None:
    if not policy.startswith("model:"):
        return None
    ident = policy[len("model:"):].rsplit(":", 1)[0]
    return ident.rsplit("@", 1)[1] if "@" in ident else None


def end_of(rec: dict, horizon: int) -> dict:
    won = rec.get("won")
    result = "draw" if won is None else ("win" if won else "loss")
    return {"result": result, "end_frame": rec.get("end_frame"), "reason": rec.get("reason"), "before_horizon": (rec.get("end_frame") or 0) < horizon}


def trunk_outcome(rec: dict, horizon: int) -> dict:
    m = rec.get("model") or {}
    won_early = rec.get("won") is True and isinstance(rec.get("end_frame"), int) and rec["end_frame"] < F8M
    out = {"held_8m": (m.get("scv_8m") or 0) >= 8 or won_early, "held_s": (m.get("scv_min_1m") or 0) > 4}
    out.update({k: m[k] for k in bc.SESSION_METRICS if k in m})
    out["end"] = end_of(rec, horizon)
    return out


class Trunk:
    def __init__(self, stage: str, k: int, n: int, game: dict):
        self.stage, self.k, self.n = stage, k, n
        self.seed, self.map, self.p0, self.p1 = int(game["seed"]), game["map"], game["p0"], game["p1"]
        self.condition: str | None = None
        self.reasons: list[str] = []
        self.points: dict[str, dict] = {}
        self.missing: dict[str, str] = {}
        self.order = (k, n)
        self.used = False

    @property
    def usable(self) -> bool:
        return not self.reasons and not self.missing and all(p["valid"] for p in self.points.values())


def norm_rule(s: dict) -> dict:
    return {**s, "any_open": list(s.get("any_open", [])), "after_gap": bool(s.get("after_gap", False))}


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
        want = [norm_rule(s) for s in cfg["snapshots"]]
        have = [norm_rule(s) for s in plan.get("snapshots", [])]
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
            t.condition = None if is_mix(cfg) else a.condition_fn(t.seed)
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
    seen: dict[str, str] = {}
    for rule in rules:
        ent = next((s for s in snaps if rule in (s.get("rules") or [])), None)
        if ent is None:
            t.missing[rule] = "rule did not fire (051-R3)"
            continue
        if ent["state"] in seen:
            t.missing[rule] = f"same decision as {seen[ent['state']]} (counts once: a rule without after_rule, 051-R3, 051-R11 1)"
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
        elif ec.file_sha256(sf) != ent["state_sha256"]:
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
            "precision": ec.PRECISION.get(backend, "?"),
            "infer_seed": rec.get("infer_seed"),
            "i_t": i,
            "f_t": int(ent["frame"]),
            "digest_t": ent["digest"],
            "type_mask": [int(x) for x in ent["type_mask"]],
            "scalar_t": ent["scalar"],
            "trunk_action": line["decision"],
            "policy_top": line.get("policy_top"),
            "state": bc.rel_to(root, sf),
            "state_sha256": ent["state_sha256"],
            "trunk_outcome": trunk_outcome(rec, horizon),
            "valid": not why,
            "skip_reason": "; ".join(why) or None,
        }


def select(trunks: list[Trunk], cfg: dict, counts: dict[str, int], rush: dict[str, list[int]]) -> dict:
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
                    for p in t.points.values():
                        if p["valid"]:
                            p["valid"], p["skip_reason"] = False, "trunk replaced (051-R11 1)"
            cells[f"{m}/{c}"] = {"quota": quota, "used": used, "short": quota - used}
    return cells


def select_mix(trunks: list[Trunk], cfg: dict, games: int | None = None) -> dict:
    quota: dict[str, int] = {}
    for m, p1, n in mix_cells(cfg, games):
        quota[cell_key(m, p1)] = quota.get(cell_key(m, p1), 0) + n
    used: dict[str, int] = dict.fromkeys(quota, 0)
    for t in sorted(trunks, key=lambda t: t.order):
        c = cell_key(t.map, t.p1)
        if t.usable and used.get(c, 0) < quota.get(c, 0):
            t.used = True
            used[c] += 1
            continue
        why = "cell full (051-R16)" if t.usable else "trunk replaced (051-R16)"
        for p in t.points.values():
            if p["valid"]:
                p["valid"], p["skip_reason"] = False, why
    return {c: {"quota": q, "used": used[c], "short": q - used[c]} for c, q in quota.items()}


@functools.cache
def op_set() -> frozenset[int]:
    return frozenset(op_actions())


def ranked_ops(point: dict) -> list[int]:
    ops = op_set()
    return [int(a) for a, _ in point.get("policy_top") or [] if int(a) in ops]


def point_alts(point: dict, alts: list[dict], names: dict[int, str]) -> list[dict]:
    out = []
    for a in alts:
        if "policy_rank" not in a:
            out.append({"name": a["name"], "rules": [dict(x) for x in a["rules"]]})
            continue
        ranked, k = ranked_ops(point), a["policy_rank"]
        rules = [{"op": "force", "action": names[ranked[k - 1]], "from_s": 0, "to_s": 0}] if len(ranked) >= k else []
        out.append({"name": a["name"], "rules": rules, "policy_rank": k})
    return out


def dropped_alts(point: dict, alts: list[dict], ids: dict[str, int]) -> list[dict]:
    out = []
    for a in alts:
        if "policy_rank" in a and not a["rules"]:
            out.append({"point": point["point"], "alt": a["name"], "closed": [],
                        "why": f"policy_top has fewer than {a['policy_rank']} open op actions (051-R16 3)"})
            continue
        closed = [r["action"] for r in a["rules"] if r["op"] == "force" and r["from_s"] == r["to_s"] == 0 and ids[r["action"]] not in point["type_mask"]]
        if closed:
            out.append({"point": point["point"], "alt": a["name"], "closed": closed})
    return out


def build(a) -> dict:
    cfg = load_config(a.config)
    root = bc.shared_root(a.root)
    job = bc.job_dir(a.trunk_job, root)
    out = bc.job_dir(a.out, root)
    counts = {} if is_mix(cfg) else {"hit": int(cfg["trunks"]["hit"] if a.hit is None else a.hit), "miss": int(cfg["trunks"]["miss"] if a.miss is None else a.miss)}
    alts = [x for x in cfg["alternative"] if a.alts is None or x["name"] in a.alts.split(",")]
    if a.alts is not None and len(alts) != len(a.alts.split(",")):
        raise Refused(f"--alts {a.alts}: not all in the config ({[x['name'] for x in cfg['alternative']]})")
    repeats = int(cfg["repeats"] if a.repeats is None else a.repeats)
    expect = {"t0_policy": a.t0_policy or f"t0@{bc.workspace_version()}", "schemas": a.schemas_fn()}
    trunks = read_trunks(a, cfg, root, job, expect)
    mix = is_mix(cfg)
    if any("policy_rank" in x for x in alts):
        for t in trunks:
            for p in t.points.values():
                if p["policy_top"] is None:
                    p["valid"] = False
                    p["skip_reason"] = "; ".join(filter(None, [p["skip_reason"], "no policy_top on the act log line (051-R2)"]))
    rush = None if mix else rush_blocks(cfg)
    cells = select_mix(trunks, cfg, a.games) if mix else select(trunks, cfg, counts, rush)
    rules = [s["name"] for s in cfg["snapshots"]]
    ids = bc.action_ids()
    names = {v: k for k, v in ids.items()}
    map_order = {m: i for i, m in enumerate(cfg["trunks"]["maps"])}
    used = sorted((t for t in trunks if t.used), key=lambda t: t.order if mix else (map_order[t.map], t.order))
    points = [t.points[r] for t in sorted(trunks, key=lambda t: t.order) for r in rules if r in t.points]
    per_point = {t.points[r]["point"]: point_alts(t.points[r], alts, names) for t in used for r in rules}
    dropped = [d for t in used for r in rules for d in dropped_alts(t.points[r], per_point[t.points[r]["point"]], ids)]
    drop = {(d["point"], d["alt"]) for d in dropped}
    lines = []
    for rep in range(1, repeats + 1):
        for t in used:
            for r in rules:
                p = t.points[r]
                for alt in per_point[p["point"]]:
                    if (p["point"], alt["name"]) in drop:
                        continue
                    lines.append({"trunk": p["point"].split("#")[0], "point": p["point"], "acts": p["trunk"]["acts"],
                                  "opponent_acts": p["trunk"]["opponent_acts"], "state": p["state"], "state_sha256": p["state_sha256"],
                                  "i_t": p["i_t"], "f_t": p["f_t"], "seed": p["seed"], "map": p["map"], "opponent": p["opponent"],
                                  "alt": {"name": alt["name"], "rules": alt["rules"]}, "repeat": rep})
    if not lines:
        raise Refused("no branch game: no usable trunk point")
    p0s = {json.dumps(t.p0, sort_keys=True) for t in used}
    if len(p0s) != 1:
        raise Refused(f"the used trunks differ in the model side (051-R1 4): {sorted(p0s)}")
    if not mix and len({json.dumps(t.p1, sort_keys=True) for t in used}) != 1:
        raise Refused("the used trunks differ in the opponent side (051-R1 4)")
    p0 = json.loads(next(iter(p0s)))
    side = {t.points[r]["point"]: t.p1 for t in used for r in rules}
    blocks: list[tuple[str, dict, list[int]]] = []
    for ln in lines:
        p1 = side[ln["point"]]
        if blocks and blocks[-1][0] == ln["map"] and blocks[-1][1] == p1:
            blocks[-1][2].append(ln["seed"])
        else:
            blocks.append((ln["map"], p1, [ln["seed"]]))
    name = a.name or cfg["name"]
    head = plan_head(name, cfg) + ['branches = "branches_plan.jsonl"'] + (["verify_state = true"] if a.verify_state else [])
    seeds, matchups = seeds_and_matchups(blocks, p0)
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
        if mix:
            fresh = iter(b2_seeds(rcp.BRANCH_B2_BLOCK[0], sum(short.values()), exclude, B2_SAMPLE_FIRST))
            for m, p1, _ in mix_cells(cfg, a.games):
                c = cell_key(m, p1)
                if short.get(c):
                    blocks.append((m, p1, [next(fresh) for _ in range(short.pop(c))]))
            short = {k: v["short"] for k, v in cells.items() if v["short"] > 0}
        else:
            for m, seeds in rush.items():
                need = {c: short.get(f"{m}/{c}", 0) for c in ("hit", "miss")}
                if any(need.values()):
                    blocks.append((m, cfg["trunks"]["opponent"], pick_seeds(seeds, need, a.condition_fn, exclude)))
        taken = {plan["name"] for _, _, plan in job_stages(job)} if (job / "job.toml").is_file() else set()
        rname = a.replacement_name or f"{name}-trunk-r"
        if rname in taken:
            if a.replacement_name:
                raise Refused(f"--replacement-name {rname}: the trunk job already has that stage")
            rname = next(f"{name}-trunk-r{i}" for i in range(2, 1000) if f"{name}-trunk-r{i}" not in taken)
        Path(a.replacements).parent.mkdir(parents=True, exist_ok=True)
        Path(a.replacements).write_bytes(render_trunk(rname, {**cfg, "checkpoint": p0["checkpoint"]}, blocks).encode("utf-8"))
        summary["replacements"] = {"plan": str(a.replacements), "short": short}
    (out / "branch_plan.json").write_bytes((json.dumps(summary, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    if a.hosts:
        hosts = Path(a.hosts)
        (out / "job.toml").write_bytes(f"stages = [{bc.toml_value(f'plans/{name}.toml')}]\nhosts = {bc.toml_value(hosts.as_posix())}\n".encode("utf-8"))
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trunk", help="write the trunk plan")
    t.add_argument("--config", type=Path, default=bc.ROOT / "config" / "branch_b1.toml")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--name", default=None, help="plan (stage) name, default <config name>-trunk")
    t.add_argument("--exclude-job", type=Path, action="append", default=[], help="leave out the seeds of this job's stages (as given if it exists, else under URANIUM_SHARED_DIR)")
    t.add_argument("--root", default=None, help="shared drive for --exclude-job (default URANIUM_SHARED_DIR)")
    t.add_argument("--hit", type=int, default=None, help="first-target-hit trunks per map (default the config's)")
    t.add_argument("--miss", type=int, default=None, help="miss trunks per map (default the config's)")
    t.add_argument("--games", type=int, default=None, help="trunk games of an opponent-mix config (default the config's; 051-R16)")
    t.add_argument("--first", type=int, default=1, help="first game number in the B2 seed block of an opponent-mix config (051-R16)")
    b = sub.add_parser("branches", help="points, branch list and branch plan from the trunk job")
    b.add_argument("--trunk-job", required=True, help="trunk job folder (as given if it exists, else under URANIUM_SHARED_DIR)")
    b.add_argument("--stage", action="append", required=True, help="trunk stage name (repeat for replacement stages)")
    b.add_argument("--config", type=Path, default=bc.ROOT / "config" / "branch_b1.toml")
    b.add_argument("--out", required=True, help="branch job folder (as given if it exists or is absolute, else under URANIUM_SHARED_DIR)")
    b.add_argument("--root", default=None, help="shared drive (default URANIUM_SHARED_DIR)")
    b.add_argument("--name", default=None, help="branch plan (stage) name, default the config name")
    b.add_argument("--alts", default=None, help="comma list of alternatives to keep (default all; e.g. none,force_bunker)")
    b.add_argument("--repeats", type=int, default=None)
    b.add_argument("--hit", type=int, default=None)
    b.add_argument("--miss", type=int, default=None)
    b.add_argument("--games", type=int, default=None, help="trunk games of an opponent-mix config the cells are filled to (default the config's; 051-R16)")
    b.add_argument("--verify-state", action="store_true", help="set verify_state = true (test option)")
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
            if is_mix(cfg):
                if a.exclude_job or a.hit is not None or a.miss is not None:
                    raise Refused("--exclude-job, --hit and --miss are for the B1 rush config; an opponent-mix config takes --games and --first")
                text = mix_trunk_plan(cfg, a.name or f"{cfg['name']}-trunk", a.first, a.games)
                n = sum(x for *_, x in mix_cells(cfg, a.games))
            else:
                counts = {"hit": int(cfg["trunks"]["hit"] if a.hit is None else a.hit), "miss": int(cfg["trunks"]["miss"] if a.miss is None else a.miss)}
                exclude = excluded_seeds(a.exclude_job, a.root) if a.exclude_job else set()
                text = trunk_plan(cfg, a.name or f"{cfg['name']}-trunk", a.condition_fn, exclude, counts)
                n = sum(counts.values()) * len(cfg["trunks"]["maps"])
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_bytes(text.encode("utf-8"))
            print(f"{a.out}: {n} trunk games")
            return
        if a.schemas:
            f, s = (int(x) for x in a.schemas.split(","))
            a.schemas_fn = lambda: {"feature": f, "action": s}
        else:
            a.schemas_fn = bc.current_schemas
        if a.cmd == "branches" and is_mix(load_config(a.config)) and (a.hit is not None or a.miss is not None):
            raise Refused("--hit and --miss are for the B1 rush config; an opponent-mix config takes --games")
        s = build(a)
    except Refused as e:
        raise SystemExit(f"REFUSED: {e}") from e
    print(f"{a.out}: {s['games']} branch games from {sum(t['used'] for t in s['trunks'])} trunks; cells {s['cells']}")
    if s["dropped"]:
        print(f"dropped alternatives (051-R10, 051-R16 3; reasons in branch_plan.json): {len(s['dropped'])}")
