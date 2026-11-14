"""044-R6 round verdict of a distillation round (spec/044 044-R6 / 044-R7, AMB-044-7 Resolved 2026-11-14;
docs/measurement_protocol.md "증류 라운드 판정"). It only reports: whatever the verdict, the round's results go to
the user and the next step is the user's (044-R6, as 050-R10).

  pass  = the whole 050-R10 gate against m2-s2e2 (tools/eval/regression_gate.py: checks 1, 2, 3a, 3b all PASS)
          + TvZ kept (041-R29 ④ qualification: TvZ 32 >= 16 wins, late held_8m >= 24 / 64)
          + held sum (early + late held_8m of the rush 128) >= S's held sum - 6
  d2    = (round d1 only) not passed, but T0 Protoss wins (t0_tp) >= S's + 4 and TvZ kept
  stop  = TvZ not kept, or T0 Protoss wins < S's + 4; and a d2 that does not pass (d2 is the last round)

The teacher round t1 (044-R8, user decision 2026-11-14: one round after the distillation rounds, judged by the 044-R6
rules) is `--round t1`: pass or stop only, no "d2" (t1 is the last round; 메인 임시 결정 2026-11-14, 사용자 확인 대기).
Its jobs are of lineage point t1.

Only these three verdicts (user decision 2026-11-14): a gate check that is MISSING or a cell without a value (games
without a row, no val output) counts as failed. To keep the never-overwritten verdict file right, the tool refuses to
run (no verdict, no file) while an input job is unfinished (no DONE) or has undecodable result / ledger lines. The
baseline job must be of point s2e2 and S's two jobs of one selection candidate point (r1 / r2 / r3 / s2e2) and checkpoint.

S = the 041-R29 ④ selection (the Checkpoint 1 counterpart model); its rush 128 and t0_tp jobs give the held sum and
the Protoss wins the round is compared with. Boundaries are inclusive (">= 16", ">= 24", ">= S - 6", ">= S + 4").
The teacher-mode share (050-R10, TvT / TvP) of the round's and S's t0_tp games is recorded, not judged.

    python tools/eval/distill_verdict.py --round d1 \\
        --baseline-val <m2-s2e2 val500 json> --candidate-val <round val500 json> \\
        --baseline-games <m2-s2e2 t0_tp job> --candidate-games <round t0_tp job> \\
        --candidate-tvz <round job with the TvZ 32 stage> --candidate-rush <round job with the rush 128 stage> \\
        --s-games <S's t0_tp job> --s-rush <S's rush 128 job> \\
        [--verdict <shared>/eval/m2-lineage/d1_first_test.verdict] [--json out.json] [--cache DIR]

Rush and TvZ counts come from the session rows (held_8m = scv_8m >= 8, 050-R7; opponent-crash games invalid), as
tools/eval/tables.py. A verdict file is never overwritten (041-R29 ⑨).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_common as ec  # noqa: E402
import protocol  # noqa: E402
import regression_gate as rg  # noqa: E402
import tables  # noqa: E402

# AMB-044-7 Resolved 2026-11-14 (044-R6): the thresholds.
TVZ_WINS = 16          # TvZ 32 >= 16 wins (041-R29 ④ qualification)
LATE_HELD = 24         # late (missed-first-target) held_8m >= 24 / 64
HELD_SUM_SLACK = 6     # held sum (early + late held_8m) >= S's - 6
D2_PROTOSS_GAIN = 4    # d2: T0 Protoss wins >= S's + 4
# 044-R8: t1 = the teacher round (pass or stop only; 메인 임시 결정 2026-11-14, 사용자 확인 대기).
ROUNDS = ("d1", "d2", "t1")
RUSH_PLANS = ("rush_check",)
TVZ_PLANS = ("tvz_bc", "tvz_rl")
PASS, D2, STOP = "pass", "d2", "stop"
# 041-R29 ④: S is r1, r2 or r3 (r3 runs, user decision 2026-11-14), or the BC baseline when no candidate qualifies.
S_POINTS = ("r1", "r2", "r3", "s2e2")
BASELINE_POINT = "s2e2"


def ge(x: int | None, y: int | None) -> bool:
    """x >= y; a missing value fails (user decision 2026-11-14)."""
    return x is not None and y is not None and x >= y


def one_stage(job: Path, lin: dict, plan_ids: tuple[str, ...]) -> tuple[int, Path, dict]:
    stages = [(k, f, p) for k, f, p in ec.job_stages(job) if protocol.identify(lin, p) in plan_ids]
    if len(stages) != 1:
        raise SystemExit(f"044-R6: {job} has {len(stages)} stages of the protocol plan {' / '.join(plan_ids)} (want 1)")
    return stages[0]


def stage_rows(job: Path, k: int, plan: dict) -> tuple[dict[int, dict], set[int], int, int]:
    """(model rows by game n, opponent-crash game keys, planned games with neither a row nor an invalid mark,
    undecodable job lines)."""
    results, ledger = ec.read_results(job), ec.read_ledger(job)
    rows = ec.model_rows(results, k)
    invalid = ec.opponent_crashes(ledger)
    planned = sum(m["games"] for m in plan["matchup"])
    missing = sum(1 for n in range(1, planned + 1) if n not in rows and k * ec.STAGE_BASE + n not in invalid)
    return rows, invalid, missing, len(ec.skipped_lines(results, ledger))


def rush_counts(job: Path, lin: dict) -> dict:
    """held_8m by start condition of the job's rush 128 stage (tables.rush_table, session rows); `complete` = every
    planned game has a row or an invalid mark and no job line was undecodable."""
    k, _, plan = one_stage(job, lin, RUSH_PLANS)
    rows, invalid, missing, skipped = stage_rows(job, k, plan)
    t = tables.rush_table(k, plan, rows, invalid, {})
    return {"job": job.name, "early": t["early"]["held_8m"], "late": t["late"]["held_8m"], "n_early": t["early"]["n"],
            "n_late": t["late"]["n"], "invalid": t["early"]["invalid"] + t["late"]["invalid"], "missing": missing,
            "skipped_lines": skipped, "complete": missing == 0 and skipped == 0}


def tvz_counts(job: Path, lin: dict) -> dict:
    k, _, plan = one_stage(job, lin, TVZ_PLANS)
    rows, invalid, missing, skipped = stage_rows(job, k, plan)
    t = tables.winrate_table(k, plan, rows, invalid)["total"]
    return {"job": job.name, "won": t["won"], "n": t["n"], "invalid": t["invalid"], "missing": missing,
            "skipped_lines": skipped, "complete": missing == 0 and skipped == 0}


def t0_teacher_mode(job: Path, cache: Path, lin: dict) -> dict:
    """050-R10 teacher-mode share of the job's t0_tp / t0_tp_rl stage (record only)."""
    k, _, plan = one_stage(job, lin, rg.T0_TP_PLANS)
    rows, invalid, _, _ = stage_rows(job, k, plan)
    return tables.stage_teacher_mode(job / plan["name"], k, plan, rows, invalid, cache / job.name / plan["name"]).get("teacher_mode", {})


def protoss_wins(c: dict | None) -> int | None:
    """T0 Protoss wins of t0_tp counts (regression_gate.t0_tp_counts); unknown for an unfinished job or one with
    undecodable lines."""
    if not c or c.get("skipped_lines") or not c.get("protoss") or c["protoss"].get("missing"):
        return None
    return c["protoss"]["won"]


def decide(rnd: str, gate_overall: str | None, tvz: dict | None, rush: dict | None, s_rush: dict | None,
           protoss: int | None, s_protoss: int | None) -> tuple[str, dict]:
    """The 044-R6 verdict and its conditions. A gate that is not PASS (FAIL or MISSING) and a count of a job with
    games missing count as failed (user decision 2026-11-14)."""
    if rnd not in ROUNDS:
        raise SystemExit(f"044-R6: round {rnd!r} is not one of {', '.join(ROUNDS)} (d2 is the last round; t1 the teacher round, 044-R8)")
    ok = lambda c: c is not None and c["complete"]  # noqa: E731
    held = rush["early"] + rush["late"] if ok(rush) else None
    s_held = s_rush["early"] + s_rush["late"] if ok(s_rush) else None
    cond = {
        "gate": gate_overall == rg.PASS,
        "tvz_wins": ok(tvz) and ge(tvz["won"], TVZ_WINS),
        "late_held": ok(rush) and ge(rush["late"], LATE_HELD),
        "held_sum": ge(held, None if s_held is None else s_held - HELD_SUM_SLACK),
        "protoss_gain": ge(protoss, None if s_protoss is None else s_protoss + D2_PROTOSS_GAIN),
    }
    cond["tvz_kept"] = cond["tvz_wins"] and cond["late_held"]
    if cond["gate"] and cond["tvz_kept"] and cond["held_sum"]:
        return PASS, cond
    if rnd == "d1" and cond["tvz_kept"] and cond["protoss_gain"]:
        return D2, cond
    return STOP, cond


def _cnt(x, n) -> str:
    return "-" if x is None else f"{x}/{n}"


def _yn(c: bool) -> str:
    return "yes" if c else "NO"


def _tm(t: dict) -> str:
    if not t:
        return "no Terran / Protoss games"
    if "error" in t:
        return f"not counted ({t['error']})"
    return ", ".join(f"Tv{r[0].upper()} {v['teacher']}/{v['n']}" for r, v in t.items())


def report_lines(rnd: str, word: str, cond: dict, g: dict, tvz: dict, rush: dict, s_rush: dict, protoss, s_protoss,
                 tm: dict, s_tm: dict) -> list[str]:
    held = rush["early"] + rush["late"]
    s_held = s_rush["early"] + s_rush["late"]
    return [
        f"044-R6 verdict of round {rnd}: {word}",
        f"050-R10 gate vs m2-s2e2: {g['overall']} (" + ", ".join(f"{c['check']} {c['race']} {c['metric']} {c['result']}" for c in g["checks"] if c["result"] != rg.PASS) + ")"
        if g["overall"] != rg.PASS else "050-R10 gate vs m2-s2e2: PASS (1, 2, 3a, 3b)",
        f"TvZ 32: {tvz['won']}/{tvz['n']} won (need >= {TVZ_WINS}): {_yn(cond['tvz_wins'])}; missing {tvz['missing']}, invalid {tvz['invalid']}",
        f"late held_8m: {rush['late']}/{rush['n_late']} (need >= {LATE_HELD}): {_yn(cond['late_held'])}; early {rush['early']}/{rush['n_early']}; missing {rush['missing']}",
        f"held sum (early + late held_8m): {held} vs S {s_held} - {HELD_SUM_SLACK} = {s_held - HELD_SUM_SLACK}: {_yn(cond['held_sum'])}",
        f"T0 Protoss wins: {protoss if protoss is not None else '-'} vs S {s_protoss if s_protoss is not None else '-'} "
        f"(d2 needs >= S + {D2_PROTOSS_GAIN}): {_yn(cond['protoss_gain'])}",
        f"teacher mode (050-R10, record only): round {_tm(tm)}; S {_tm(s_tm)}",
        "044-R6: report the results to the user; the next step is the user's (no automatic stop)",
    ]


def candidate_records(jobs: list[Path], rnd: str) -> dict:
    """protocol.json of the round's jobs: every one of point `rnd` and one checkpoint (a job without the record is
    noted, e.g. one rendered by hand)."""
    recs = {j.name: protocol.read_record(j) for j in dict.fromkeys(jobs)}
    for name, r in recs.items():
        if r is not None and r.get("point") != rnd:
            raise SystemExit(f"044-R6: job {name} is of point {r.get('point')!r}, not round {rnd}")
    shas = {r.get("checkpoint_sha256") for r in recs.values() if r is not None}
    if len(shas) > 1:
        raise SystemExit(f"044-R6: the round's jobs are of different checkpoints ({', '.join(sorted(map(str, shas)))})")
    first = next((r for r in recs.values() if r is not None), None)
    return {"checkpoint": first and first.get("checkpoint"), "checkpoint_sha256": first and first.get("checkpoint_sha256"),
            "no_protocol_json": sorted(n for n, r in recs.items() if r is None)}


def reference_records(baseline: Path, s_jobs: list[Path]) -> dict:
    """The baseline job is of point s2e2 (050-R10: m2-s2e2) and S's jobs of one selection candidate point (041-R29 ④)
    and one checkpoint; each needs its lineage-2 protocol.json."""
    def rec(j: Path) -> dict:
        r = protocol.read_record(j)
        if r is None or r.get("lineage") != "m2":
            raise SystemExit(f"044-R6: {j.name} has no lineage-2 {protocol.RECORD} (the baseline and S jobs are protocol jobs)")
        return r
    b = rec(baseline)
    if b.get("point") != BASELINE_POINT:
        raise SystemExit(f"044-R6: baseline job {baseline.name} is of point {b.get('point')!r}, not {BASELINE_POINT} (050-R10 baseline m2-s2e2)")
    s = [rec(j) for j in s_jobs]
    pts, shas = {r.get("point") for r in s}, {r.get("checkpoint_sha256") for r in s}
    if len(pts) != 1 or not pts <= set(S_POINTS) or len(shas) != 1:
        raise SystemExit(f"044-R6: S's jobs must be of one selection candidate point ({' / '.join(S_POINTS)}) and one checkpoint; "
                         f"got points {sorted(map(str, pts))}, {len(shas)} checkpoint(s)")
    return {"baseline_checkpoint": b.get("checkpoint"), "s_point": pts.pop(), "s_checkpoint": s[0].get("checkpoint")}


def check_finished(jobs: list[Path]) -> None:
    """Refuse (no verdict, no file) while an input job is unfinished or has undecodable lines (user decision
    2026-11-14: the verdict file is never overwritten, so it is written from finished jobs only)."""
    problems = []
    for j in dict.fromkeys(jobs):
        if not (j / "DONE").is_file():
            problems.append(f"{j.name}: not finished (no DONE)")
        bad = ec.skipped_lines(ec.read_results(j), ec.read_ledger(j))
        if bad:
            problems.append(f"{j.name}: {len(bad)} undecodable line(s) (" + ", ".join(f"{x['file']}:{x['line']}" for x in bad) + ")")
    if problems:
        raise SystemExit("044-R6: no verdict while an input job is unfinished or damaged:\n  " + "\n  ".join(problems))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--round", required=True, choices=ROUNDS)
    ap.add_argument("--baseline-val", help="bc-eval JSON of m2-s2e2 (val500, by_enemy_race, --bunker-before 5760)")
    ap.add_argument("--candidate-val", help="bc-eval JSON of the round's checkpoint, same selection")
    ap.add_argument("--baseline-games", type=Path, required=True, help="m2-s2e2's eval job with the t0_tp stage")
    ap.add_argument("--candidate-games", type=Path, required=True, help="the round's eval job with the t0_tp stage")
    ap.add_argument("--candidate-tvz", type=Path, required=True, help="the round's eval job with the TvZ 32 stage")
    ap.add_argument("--candidate-rush", type=Path, required=True, help="the round's eval job with the rush 128 stage")
    ap.add_argument("--s-games", type=Path, required=True, help="S's eval job with the t0_tp stage")
    ap.add_argument("--s-rush", type=Path, required=True, help="S's eval job with the rush 128 stage")
    ap.add_argument("--cache", type=Path, default=None, help="command stream cache (default <URANIUM_CACHE_DIR or runs>/eval_cmddump)")
    ap.add_argument("--verdict", type=Path, default=None, help="write the verdict file (first line the verdict); never overwritten")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.verdict is not None and a.verdict.exists():
        raise SystemExit(f"{a.verdict} exists: a verdict is not overwritten (041-R29 ⑨)")
    if a.verdict is not None:
        # A forgotten or mistyped val path would fail gate check 2 / 3a for good in a never-overwritten file.
        for flag, v in (("--baseline-val", a.baseline_val), ("--candidate-val", a.candidate_val)):
            if v is None or not Path(v).is_file():
                raise SystemExit(f"044-R6: --verdict needs {flag} (an existing bc-eval JSON); got {v!r}")
    ec.load_env()
    cache = a.cache or ec.cache_dir() / "eval_cmddump"
    lin = protocol.load_lineage()
    check_finished([a.baseline_games, a.candidate_games, a.candidate_tvz, a.candidate_rush, a.s_games, a.s_rush])
    rec = candidate_records([a.candidate_games, a.candidate_tvz, a.candidate_rush], a.round)
    ref = reference_records(a.baseline_games, [a.s_games, a.s_rush])
    bg = rg.t0_tp_counts(a.baseline_games, cache, lin)
    cg = rg.t0_tp_counts(a.candidate_games, cache, lin)
    sg = rg.t0_tp_counts(a.s_games, cache, lin)
    g = rg.gate(rg._load(a.baseline_val), rg._load(a.candidate_val), bg, cg)
    tvz, rush, s_rush = tvz_counts(a.candidate_tvz, lin), rush_counts(a.candidate_rush, lin), rush_counts(a.s_rush, lin)
    protoss, s_protoss = protoss_wins(cg), protoss_wins(sg)
    word, cond = decide(a.round, g["overall"], tvz, rush, s_rush, protoss, s_protoss)
    tm, s_tm = t0_teacher_mode(a.candidate_games, cache, lin), t0_teacher_mode(a.s_games, cache, lin)
    lines = report_lines(a.round, word, cond, g, tvz, rush, s_rush, protoss, s_protoss, tm, s_tm)
    rg.print_gate(g)
    print()
    print("\n".join(lines))
    out = {"round": a.round, "verdict": word, "conditions": cond, "gate": g, "tvz": tvz, "rush": rush, "s_rush": s_rush,
           "protoss_wins": protoss, "s_protoss_wins": s_protoss, "teacher_mode": tm, "s_teacher_mode": s_tm, "candidate": rec, "reference": ref,
           "inputs": {k: None if v is None else Path(v).name for k, v in (("baseline_val", a.baseline_val), ("candidate_val", a.candidate_val),
                      ("baseline_games", a.baseline_games), ("candidate_games", a.candidate_games), ("candidate_tvz", a.candidate_tvz),
                      ("candidate_rush", a.candidate_rush), ("s_games", a.s_games), ("s_rush", a.s_rush))}}
    if a.verdict is not None:
        stamp = datetime.datetime.now().astimezone().isoformat(timespec="minutes")
        head = f"# {stamp}, 044-R6 round {a.round}, checkpoint {rec['checkpoint'] or '?'} (sha256 {str(rec['checkpoint_sha256'])[:12]})"
        jobs = (f"# jobs: t0_tp {a.candidate_games.name}, tvz {a.candidate_tvz.name}, rush {a.candidate_rush.name}; "
                f"S ({ref['s_point']}) {a.s_games.name}, {a.s_rush.name}; baseline {a.baseline_games.name}")
        a.verdict.parent.mkdir(parents=True, exist_ok=True)
        with a.verdict.open("x", encoding="utf-8") as fh:
            fh.write("\n".join([word, head, jobs] + [f"# {x}" for x in lines[1:]]) + "\n")
        print("wrote", a.verdict)
    if a.json:
        a.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("wrote", a.json)
    return out


if __name__ == "__main__":
    main()
