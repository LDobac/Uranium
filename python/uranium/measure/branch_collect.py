"""Collect branch-game results: one branches.jsonl line per played branch game"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from uranium.diag.planning import labels as pl
from uranium.diag.planning.dump import own_columns, own_state
from uranium.measure import branch_common as bc
from uranium.measure import eval_common as ec

Refused = bc.Refused
OUTCOME_FIELDS = ("state_check", "intervention", "samples", "metrics", "dilution", "end")


def own_at(samples: dict, name: str) -> dict | None:
    s = (samples or {}).get(name)
    if not isinstance(s, dict) or s.get("scalar") is None:
        return None
    o = own_state(np.asarray(s["scalar"], np.float32)[None])[0]
    return {c: float(v) for c, v in zip(own_columns()[0], o)}


def outcome_extras(model: dict, aborted: bool, samples: dict) -> dict:
    return {"labels": dict(pl.NONE) if aborted or not model else pl.bot_heads(model, "model"),
            "ops": (model.get("model") or {}).get("ops"),
            "own_horizon": own_at(samples, "horizon")}


def stage_of(job: Path, name: str | None) -> tuple[int, Path, dict]:
    stages = [s for s in ec.job_stages(job) if ec.is_branch_plan(s[2]) and s[2].get("branches")]
    if name is not None:
        stages = [s for s in stages if s[2]["name"] == name]
    if len(stages) != 1:
        raise Refused(f"{job}: {len(stages)} branch stages{' named ' + name if name else ''} (pass --stage)")
    return stages[0]


def opponent_prefix(root: Path, line: dict, branch_log: Path) -> dict | None:
    trunk_log = bc.on_shared(root, line["opponent_acts"])
    if not trunk_log.is_file() or not branch_log.is_file():
        return None
    trunk = bc.ActLog.read(trunk_log, lenient=True).lines
    branch = bc.ActLog.read(branch_log, lenient=True).lines
    return bc.compare_prefix(trunk, branch, int(line["f_t"]))


def collect(job: Path, stage: str | None, root: Path) -> dict:
    k, f, plan = stage_of(job, stage)
    name = plan["name"]
    sd = job / name
    lines = bc.read_jsonl(f.parent / plan["branches"])
    if len(lines) != sum(m["games"] for m in plan["matchup"]):
        raise Refused(f"{plan['branches']}: {len(lines)} lines for {sum(m['games'] for m in plan['matchup'])} games (020-R22 3)")
    points = {p["point"]: p for p in bc.read_jsonl(job / "points.jsonl")}
    games_p, results_p = sd / "games.jsonl", sd / "results.jsonl"
    if not games_p.is_file() or not results_p.is_file():
        raise Refused(f"{sd}: no merged stage files (run `python -m uranium eval-status --job {job}` first)")
    games = {int(g["game_id"]): g for g in bc.read_jsonl(games_p)}
    recs: dict[int, list[dict]] = {}
    for r in bc.read_jsonl(results_p):
        recs.setdefault(int(r["game_id"]), []).append(r)
    rows, missing = [], []
    no_end: dict[str, list[int]] = {"ours": [], "opponent": []}
    no_opp_prefix: list[int] = []
    for n, line in enumerate(lines, start=1):
        game = games.get(n)
        if game is None:
            missing.append({"game": n, "why": "not played"})
            continue
        outcome = next((r["branch"] for r in recs.get(n, []) if r.get("model") and r.get("branch")), None)
        model = next((r for r in recs.get(n, []) if r.get("model")), {})
        if outcome is not None and outcome.get("end") is None:
            outcome = None
        if outcome is None:
            why = game.get("no_result") or ("given up" if game.get("gave_up") else None) or f"no branch outcome ({game.get('outcome')})"
            missing.append({"game": n, "why": why})
            continue
        point = points.get(line["point"])
        if point is None:
            raise Refused(f"game {n}: point {line['point']} is not in points.jsonl")
        hostd = int(game.get("hostd_id", 0))
        replay = sd / "replays" / f"g{n}.rep"
        backend = game.get("backend") or "?"
        ours_p, opp_p = sd / "acts" / f"g{n}-p0.jsonl", sd / "acts" / f"g{n}-p1.jsonl"
        ours = bc.ActLog.read(ours_p, lenient=True).end if ours_p.is_file() else None
        theirs = bc.ActLog.read(opp_p, lenient=True).end if opp_p.is_file() else None
        git = (ours or {}).get("git")
        if ours is None:
            no_end["ours"].append(n)
        if theirs is None:
            no_end["opponent"].append(n)
        opp = opponent_prefix(root, line, opp_p)
        if opp is None:
            no_opp_prefix.append(n)
        valid, invalid = bool(outcome.get("valid")), outcome.get("invalid")
        got = {"policy": (ours or {}).get("policy"), "t0_policy": (theirs or {}).get("policy")}
        want = {"policy": point["policy"], "t0_policy": point["t0_policy"]}
        diff = {k: {"branch": got[k], "trunk": want[k]} for k in got if got[k] is not None and got[k] != want[k]}
        if valid and diff:
            valid, invalid = False, {"reason": "missing_input", "obs": None, "frame": None, "policy_mismatch": diff}
        first = None if opp is None else opp["first_mismatch_frame"]
        if valid and opp is not None and (opp["matched"] < opp["obs"] or (first is not None and first <= int(line["f_t"]))):
            valid, invalid = False, {"reason": "digest", "obs": None, "frame": first, "side": "opponent"}
        row = {
            "point": line["point"],
            "alt": line["alt"]["name"],
            "rules": line["alt"].get("rules", []),
            "repeat": line["repeat"],
            "game": {"job": job.name, "stage": name, "plan_game": n, "hostd_game": hostd, "attempt": ec.attempt_of(hostd),
                     "replay": bc.rel_to(root, replay) if replay.is_file() else None, "host": game.get("host", ""),
                     "backend": backend, "precision": ec.PRECISION.get(backend, "?"), "infer_seed": model.get("infer_seed")},
            "checkpoint_sha256": point["checkpoint_sha256"],
            "git": git or "unknown",
            "valid": valid,
            "invalid": invalid,
            "prefix": {"obs": outcome.get("prefix_obs", 0), "matched": outcome.get("prefix_matched", 0), "opponent": opp},
        }
        for key in OUTCOME_FIELDS:
            row[key] = outcome.get(key)
        row["samples"] = row["samples"] or {}
        row["metrics"] = row["metrics"] or {}
        row["intervention"] = row["intervention"] or []
        row["dilution"] = row["dilution"] or []
        row.update(outcome_extras(model, game.get("outcome") == "Aborted", row["samples"]))
        rows.append(row)
    bc.write_jsonl(job / "branches.jsonl", rows)
    by_reason: dict[str, int] = {}
    by_alt: dict[str, dict] = {}
    for r in rows:
        a = by_alt.setdefault(r["alt"], {"games": 0, "valid": 0})
        a["games"] += 1
        a["valid"] += r["valid"]
        if not r["valid"]:
            why = (r["invalid"] or {}).get("reason", "?")
            by_reason[why] = by_reason.get(why, 0) + 1
    summary = {"job": job.name, "stage": name, "planned": len(lines), "records": len(rows),
               "valid": sum(r["valid"] for r in rows), "invalid_by_reason": by_reason, "by_alt": by_alt, "missing": missing,
               "act_log_incomplete": no_end, "opponent_prefix_unchecked": no_opp_prefix}
    (job / "branch_collect.json").write_bytes((json.dumps(summary, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--job", required=True, type=Path, help="branch job folder (as given if it exists, else under URANIUM_SHARED_DIR)")
    ap.add_argument("--stage", default=None, help="branch stage name (default: the job's only branch stage)")
    ap.add_argument("--root", default=None, help="shared drive (default URANIUM_SHARED_DIR)")
    a = ap.parse_args(argv)
    try:
        root = bc.shared_root(a.root)
        a.job = bc.job_dir(a.job, root)
        s = collect(a.job, a.stage, root)
    except Refused as e:
        raise SystemExit(f"REFUSED: {e}") from e
    print(f"{a.job / 'branches.jsonl'}: {s['records']} of {s['planned']} games, {s['valid']} valid; "
          f"invalid {s['invalid_by_reason']}; missing {len(s['missing'])}")
