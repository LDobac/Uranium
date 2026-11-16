"""044-R6 round verdict of a distillation round (docs/spec/2026_11_14_08_18_044-self-game-distillation.md 044-R6 / 044-R7, AMB-044-7 Resolved 2026-11-14;"""

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

TVZ_WINS = 16
LATE_HELD = 24
HELD_SUM_SLACK = 6
D2_PROTOSS_GAIN = 4
ROUNDS = ("d1", "d2", "t1")
RUSH_PLANS = ("rush_check",)
TVZ_PLANS = ("tvz_bc", "tvz_rl")
PASS, D2, STOP = "pass", "d2", "stop"
NLL_JUDGED_ROUNDS = ("d1", "d2")
NLL_WARN = 0.03
RECORD, WARN = "RECORD", "WARN"
TEACHER_MODE_MAX = 3
D_HELD_SUM = 60
D_T0_WINS = 20
D1_CHECKPOINT = "checkpoints/m2-distill-d1/ckpt-2000"
T1_S_POINT = "r2"
S_POINTS = ("r1", "r2", "r3", "s2e2")
BASELINE_POINT = "s2e2"


def ge(x: int | None, y: int | None) -> bool:
    return x is not None and y is not None and x >= y


def one_stage(job: Path, lin: dict, plan_ids: tuple[str, ...]) -> tuple[int, Path, dict]:
    stages = [(k, f, p) for k, f, p in ec.job_stages(job) if protocol.identify(lin, p) in plan_ids]
    if len(stages) != 1:
        raise SystemExit(f"044-R6: {job} has {len(stages)} stages of the protocol plan {' / '.join(plan_ids)} (want 1)")
    return stages[0]


def stage_rows(job: Path, k: int, plan: dict) -> tuple[dict[int, dict], set[int], int, int]:
    results, ledger = ec.read_results(job), ec.read_ledger(job)
    rows = ec.model_rows(results, k)
    invalid = ec.opponent_crashes(ledger)
    planned = sum(m["games"] for m in plan["matchup"])
    missing = sum(1 for n in range(1, planned + 1) if n not in rows and k * ec.STAGE_BASE + n not in invalid)
    return rows, invalid, missing, len(ec.skipped_lines(results, ledger))


def rush_counts(job: Path, lin: dict) -> dict:
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
    k, _, plan = one_stage(job, lin, rg.T0_TP_PLANS)
    rows, invalid, _, _ = stage_rows(job, k, plan)
    return tables.stage_teacher_mode(job / plan["name"], k, plan, rows, invalid, cache / job.name / plan["name"]).get("teacher_mode", {})


def nll_record_only(g: dict) -> dict:
    checks = []
    for c in g["checks"]:
        if c["check"] == "2" and c["metric"] == rg.NLL_HEAD:
            b, v = c["baseline"], c["candidate"]
            warn = b is not None and v is not None and v - b > NLL_WARN + rg.EPS
            c = {**c, "threshold": f"record only (WARN > baseline + {NLL_WARN})", "result": WARN if warn else RECORD}
        checks.append(c)
    res = {c["result"] for c in checks if c["result"] not in (RECORD, WARN)}
    overall = rg.FAIL if rg.FAIL in res else (rg.MISSING if rg.MISSING in res else rg.PASS)
    return {**g, "overall": overall, "checks": checks,
            "note": g["note"] + " 044-R6 (2026-11-15, t1 on): check 2 nll.action recorded only."}


def teacher_cells(tm: dict, c: dict | None) -> dict:
    out = {}
    for race in tables.TEACHER_MODE_RACES:
        v = None if "error" in tm else tm.get(race)
        cc = (c or {}).get(race)
        why = ("command streams not counted" if "error" in tm else "no games" if not v else
               f"{v['no_stream']} game(s) without a command stream" if v["no_stream"] else
               "no games" if not v["n"] else
               "no t0_tp counts" if not cc else
               f"{cc['missing']} game(s) without a result" if cc.get("missing") else
               "undecodable job lines" if c.get("skipped_lines") else "")
        teacher, n = (v["teacher"], v["n"]) if v else (None, 0)
        out[race] = {"teacher": teacher, "n": n, "ok": not why and teacher <= TEACHER_MODE_MAX, "missing": why}
    return out


def stream_gaps(tm: dict) -> None:
    if "error" in tm:
        raise SystemExit(f"044-R6: no verdict while the round's t0_tp command streams are not counted ({tm['error']})")
    gaps = [f"Tv{r[0].upper()} {v['no_stream']}" for r, v in tm.items() if v.get("no_stream")]
    if gaps:
        raise SystemExit("044-R6: no verdict while valid T0 games of the round have no command stream (" + ", ".join(gaps) +
                         "); run uranium-replay cmddump (or fix the replays) and run again")


def t0_wins(c: dict | None) -> int | None:
    rs = [(c or {}).get(r) for r in rg.RACES]
    if not c or c.get("skipped_lines") or not all(rs) or any(r.get("missing") for r in rs):
        return None
    return sum(r["won"] for r in rs)


def d_choice(word: str, tvz: dict | None, rush: dict | None, wins: int | None, tcells: dict, checkpoint: str | None) -> dict:
    ok = lambda c: c is not None and c["complete"]  # noqa: E731
    held = rush["early"] + rush["late"] if ok(rush) else None
    cond = {
        "tvz_wins": {"value": tvz["won"] if ok(tvz) else None, "need": f">= {TVZ_WINS}", "ok": ok(tvz) and ge(tvz["won"], TVZ_WINS)},
        "late_held": {"value": rush["late"] if ok(rush) else None, "need": f">= {LATE_HELD}", "ok": ok(rush) and ge(rush["late"], LATE_HELD)},
        "held_sum": {"value": held, "need": f">= {D_HELD_SUM}", "ok": ge(held, D_HELD_SUM)},
        "t0_wins": {"value": wins, "need": f">= {D_T0_WINS}", "ok": ge(wins, D_T0_WINS)},
    }
    for race in tables.TEACHER_MODE_RACES:
        t = (tcells or {}).get(race) or {"teacher": None, "ok": False}
        cond[f"teacher_mode_{race}"] = {"value": t["teacher"], "need": f"<= {TEACHER_MODE_MAX}", "ok": t["ok"] is True}
    t1 = word == PASS or all(x["ok"] for x in cond.values())
    return {"d": "t1" if t1 else "d1", "checkpoint": (checkpoint or "?") if t1 else D1_CHECKPOINT,
            "by": "verdict pass" if word == PASS else ("044-R7 conditions met" if t1 else "044-R7 conditions not met"),
            "conditions": cond}


def d_choice_lines(d: dict) -> list[str]:
    c = d["conditions"]
    v = lambda k: "-" if c[k]["value"] is None else c[k]["value"]  # noqa: E731
    return [
        f"044-R7 RL start: D = {d['d']} ({d['checkpoint']}; {d['by']}); no stop",
        f"044-R7 conditions: TvZ {v('tvz_wins')}/32 >= {TVZ_WINS}: {_yn(c['tvz_wins']['ok'])}; late held_8m {v('late_held')}/64 >= {LATE_HELD}: "
        f"{_yn(c['late_held']['ok'])}; held sum {v('held_sum')} >= {D_HELD_SUM}: {_yn(c['held_sum']['ok'])}; "
        f"T0 Terran + Protoss wins {v('t0_wins')}/64 >= {D_T0_WINS}: {_yn(c['t0_wins']['ok'])}; "
        f"teacher mode TvT {v('teacher_mode_terran')}, TvP {v('teacher_mode_protoss')} <= {TEACHER_MODE_MAX}/32: "
        f"{_yn(c['teacher_mode_terran']['ok'])}, {_yn(c['teacher_mode_protoss']['ok'])}",
    ]


def protoss_wins(c: dict | None) -> int | None:
    if not c or c.get("skipped_lines") or not c.get("protoss") or c["protoss"].get("missing"):
        return None
    return c["protoss"]["won"]


def decide(rnd: str, gate_overall: str | None, tvz: dict | None, rush: dict | None, s_rush: dict | None,
           protoss: int | None, s_protoss: int | None, tcells: dict | None = None) -> tuple[str, dict]:
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
    if rnd not in NLL_JUDGED_ROUNDS:
        cond["teacher_mode"] = tcells is not None and all((tcells.get(r) or {}).get("ok") is True for r in tables.TEACHER_MODE_RACES)
    if cond["gate"] and cond["tvz_kept"] and cond["held_sum"] and cond.get("teacher_mode", True):
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


def _tcell(t: dict) -> str:
    return f"{t['teacher']}/{t['n']}" if not t["missing"] else f"- ({t['missing']})"


def report_lines(rnd: str, word: str, cond: dict, g: dict, tvz: dict, rush: dict, s_rush: dict, protoss, s_protoss,
                 tm: dict, s_tm: dict, tcells: dict | None = None, d: dict | None = None) -> list[str]:
    held = rush["early"] + rush["late"]
    s_held = s_rush["early"] + s_rush["late"]
    if rnd not in NLL_JUDGED_ROUNDS:
        nll = [c for c in g["checks"] if c["check"] == "2" and c["metric"] == rg.NLL_HEAD]
        bad = [c for c in g["checks"] if c["result"] not in (rg.PASS, RECORD, WARN)]
        return [
            f"044-R6 verdict of round {rnd}: {word}",
            f"050-R10 gate vs m2-s2e2 (2 nll.action record only): {g['overall']} ("
            + ", ".join(f"{c['check']} {c['race']} {c['metric']} {c['result']}" for c in bad) + ")"
            if bad else "050-R10 gate vs m2-s2e2: PASS (1, 2 top-1, 3a, 3b; 2 nll.action record only)",
            f"050-R10 2 nll.action (record only, WARN > baseline + {NLL_WARN}): "
            + ", ".join(f"{c['race']} {rg._fmt(c['baseline'])} -> {rg._fmt(c['candidate'])} {c['result']}" for c in nll),
            f"TvZ 32: {tvz['won']}/{tvz['n']} won (need >= {TVZ_WINS}): {_yn(cond['tvz_wins'])}; missing {tvz['missing']}, invalid {tvz['invalid']}",
            f"late held_8m: {rush['late']}/{rush['n_late']} (need >= {LATE_HELD}): {_yn(cond['late_held'])}; early {rush['early']}/{rush['n_early']}; missing {rush['missing']}",
            f"held sum (early + late held_8m): {held} vs S {s_held} - {HELD_SUM_SLACK} = {s_held - HELD_SUM_SLACK}: {_yn(cond['held_sum'])}",
            f"teacher mode (050-R10, need <= {TEACHER_MODE_MAX} each): TvT {_tcell(tcells['terran'])}, TvP {_tcell(tcells['protoss'])}: "
            f"{_yn(cond['teacher_mode'])}; S {_tm(s_tm)} (recorded)",
            f"T0 Protoss wins (recorded): {protoss if protoss is not None else '-'} vs S {s_protoss if s_protoss is not None else '-'}",
        ] + d_choice_lines(d) + ["044-R6 / 044-R7: report the results to the user; RL starts from D (no stop)"]
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
        for flag, v in (("--baseline-val", a.baseline_val), ("--candidate-val", a.candidate_val)):
            if v is None or not Path(v).is_file():
                raise SystemExit(f"044-R6: --verdict needs {flag} (an existing bc-eval JSON); got {v!r}")
    ec.load_env()
    cache = a.cache or ec.cache_dir() / "eval_cmddump"
    lin = protocol.load_lineage()
    check_finished([a.baseline_games, a.candidate_games, a.candidate_tvz, a.candidate_rush, a.s_games, a.s_rush])
    rec = candidate_records([a.candidate_games, a.candidate_tvz, a.candidate_rush], a.round)
    ref = reference_records(a.baseline_games, [a.s_games, a.s_rush])
    later = a.round not in NLL_JUDGED_ROUNDS
    if later and ref["s_point"] != T1_S_POINT:
        raise SystemExit(f"044-R8: round {a.round}'s S is {T1_S_POINT} (m2-dagger-r2/ckpt-1408); S's jobs are of point {ref['s_point']!r}")
    if later and a.verdict is not None and not rec["checkpoint"]:
        raise SystemExit(f"044-R7: --verdict for round {a.round} needs the round's checkpoint (D) from a round job's {protocol.RECORD}; "
                         f"none has one ({', '.join(rec['no_protocol_json'])})")
    bg = rg.t0_tp_counts(a.baseline_games, cache, lin)
    cg = rg.t0_tp_counts(a.candidate_games, cache, lin)
    sg = rg.t0_tp_counts(a.s_games, cache, lin)
    g = rg.gate(rg._load(a.baseline_val), rg._load(a.candidate_val), bg, cg)
    if later:
        g = nll_record_only(g)
    tvz, rush, s_rush = tvz_counts(a.candidate_tvz, lin), rush_counts(a.candidate_rush, lin), rush_counts(a.s_rush, lin)
    protoss, s_protoss = protoss_wins(cg), protoss_wins(sg)
    tm, s_tm = t0_teacher_mode(a.candidate_games, cache, lin), t0_teacher_mode(a.s_games, cache, lin)
    tcells = teacher_cells(tm, cg) if later else None
    if later and a.verdict is not None:
        stream_gaps(tm)
    word, cond = decide(a.round, g["overall"], tvz, rush, s_rush, protoss, s_protoss, tcells)
    d = d_choice(word, tvz, rush, t0_wins(cg), tcells, rec["checkpoint"]) if later else None
    lines = report_lines(a.round, word, cond, g, tvz, rush, s_rush, protoss, s_protoss, tm, s_tm, tcells, d)
    rg.print_gate(g)
    print()
    print("\n".join(lines))
    out = {"round": a.round, "verdict": word, "conditions": cond, "gate": g, "tvz": tvz, "rush": rush, "s_rush": s_rush,
           "protoss_wins": protoss, "s_protoss_wins": s_protoss, "teacher_mode": tm, "s_teacher_mode": s_tm, "candidate": rec, "reference": ref,
           "inputs": {k: None if v is None else Path(v).name for k, v in (("baseline_val", a.baseline_val), ("candidate_val", a.candidate_val),
                      ("baseline_games", a.baseline_games), ("candidate_games", a.candidate_games), ("candidate_tvz", a.candidate_tvz),
                      ("candidate_rush", a.candidate_rush), ("s_games", a.s_games), ("s_rush", a.s_rush))}}
    if later:
        out["teacher_cells"], out["d_choice"] = tcells, d
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
