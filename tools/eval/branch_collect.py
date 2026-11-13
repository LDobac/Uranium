"""Branch-game results (spec 051): one `branches.jsonl` line per played branch game (051-R12), from the
branch job after `python -m uranium eval-status --job <branch job>` (the merged `<stage>/games.jsonl` and
`results.jsonl`, last attempt of each game).

    python tools/eval/branch_collect.py --job <shared>/eval/branch-b1-<date> [--stage <stage>] [--root <shared>]

Per plan game n with line n of the stage's branch list (020-R22 `branches`, 051-R10): the model session
record's `branch` object (uranium_env::branch::BranchOutcome, 051-R9) gives `valid` / `invalid`, the prefix
counts, `state_check`, `intervention`, `samples`, `metrics`, `dilution` and `end` as they are; the line gives
`point`, `alt`, `rules`, `repeat`; the ledger gives `game` (job, stage, plan game, hostd id, attempt, replay,
game host, inference backend and precision; `infer_seed` from the session record, 051-R8); `points.jsonl`
the point's `checkpoint_sha256`; our branch act log's end line the `git` commit.

`prefix.opponent` compares the opponent's act log of the branch game (`<stage>/acts/g<n>-p1.jsonl`) with
the trunk's (`opponent_acts` of the line) before f_t (051-R12, uranium_env::branch::compare_prefix; null when
either log is missing). A branch the session called valid whose opponent prefix differs from the trunk's
(fewer matched than compared observations, or the first difference at or before f_t) is marked invalid here
(`invalid = {"reason": "digest", "obs": null, "frame": <first different frame>, "side": "opponent"}`): the
game did not reproduce the trunk up to t (051-R5 ①, interim decision of the tool). A valid branch whose own
act log end line names another model policy (checkpoint id) than the point's `policy`, or whose opponent's
names another T0 policy than the point's `t0_policy`, is invalid too (051-R1 ④:
`{"reason": "missing_input", ..., "policy_mismatch": {key: {"branch", "trunk"}}}`, the U1 reasons having no
version reason).

Games without a finished branch outcome (not played, Aborted, given up, session lost, or an outcome with
`end = null`: the session never saw GameEnd) get no line; they are counted
in `branch_collect.json` with the invalid branches by reason, the counts per alternative, the games whose
act logs are missing or cut short (`act_log_incomplete`: those policies were not checked) and the games
whose opponent prefix could not be compared (`opponent_prefix_unchecked`). Branch games
are never part of the evaluation tables (050 "분기 게임(051)"; tools/eval/tables.py leaves the stage out).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import branch_common as bc  # noqa: E402
import eval_common as ec  # noqa: E402

Refused = bc.Refused
# Fields of BranchOutcome copied into the record as they are (051-R9, 051-R12).
OUTCOME_FIELDS = ("state_check", "intervention", "samples", "metrics", "dilution", "end")


def stage_of(job: Path, name: str | None) -> tuple[int, Path, dict]:
    stages = [s for s in ec.job_stages(job) if bc.is_branch_plan(s[2]) and s[2].get("branches")]
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
    no_end: dict[str, list[int]] = {"ours": [], "opponent": []}  # branch act logs missing or cut short
    no_opp_prefix: list[int] = []  # opponent prefix not compared (a log missing)
    for n, line in enumerate(lines, start=1):
        game = games.get(n)
        if game is None:
            missing.append({"game": n, "why": "not played"})
            continue
        outcome = next((r["branch"] for r in recs.get(n, []) if r.get("model") and r.get("branch")), None)
        model = next((r for r in recs.get(n, []) if r.get("model")), {})
        if outcome is not None and outcome.get("end") is None:
            # A session that never saw GameEnd (aborted game) leaves an unfinished outcome: no result.
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
        # 051-R1 ④: the branch game ran the trunk's checkpoint and T0 policy version (act-log end lines).
        got = {"policy": (ours or {}).get("policy"), "t0_policy": (theirs or {}).get("policy")}
        want = {"policy": point["policy"], "t0_policy": point["t0_policy"]}
        diff = {k: {"branch": got[k], "trunk": want[k]} for k in got if got[k] is not None and got[k] != want[k]}
        if valid and diff:
            valid, invalid = False, {"reason": "missing_input", "obs": None, "frame": None, "policy_mismatch": diff}
        # 051-R5 ①: the opponent saw the trunk's observations up to t (its first difference after f_t).
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
                     "backend": backend, "precision": bc.PRECISION.get(backend, "?"), "infer_seed": model.get("infer_seed")},
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
    ap.add_argument("--job", required=True, type=Path, help="branch job folder (051-R12 output folder)")
    ap.add_argument("--stage", default=None, help="branch stage name (default: the job's only branch stage)")
    ap.add_argument("--root", default=None, help="shared drive (default URANIUM_SHARED_DIR)")
    a = ap.parse_args(argv)
    try:
        s = collect(a.job, a.stage, bc.shared_root(a.root))
    except Refused as e:
        raise SystemExit(f"REFUSED: {e}") from e
    print(f"{a.job / 'branches.jsonl'}: {s['records']} of {s['planned']} games, {s['valid']} valid; "
          f"invalid {s['invalid_by_reason']}; missing {len(s['missing'])}")


if __name__ == "__main__":
    main()
