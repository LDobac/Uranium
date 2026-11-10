"""050-R10 Terran / Protoss regression gate (050-AC5, AMB-050-3 Resolved 2026-11-09; docs/measurement_protocol.md
"회귀 게이트"): a lineage-2 point (mix epoch, DAgger round, the Checkpoint 1 counterpart) against the lineage-2 BC
baseline `m2-s2e2`. It only reports: a failed check goes to the user with the point's results, nothing is
stopped (050-R10).

Checks (thresholds of AMB-050-3 = reports/jepa_survey.md 7.5 main criterion; Terran and Protoss opponents):
  1  T0 standard Terran / Protoss 32 games each (protocol plan `t0_tp`): candidate win rate >= the baseline's
     win rate on the same games (wins / valid games; draws are not wins; opponent-crash games invalid).
  2  val500 per enemy race (bc-eval `by_enemy_race`, protocol metric val500_race): action top-1 >= baseline
     - 0.5 %p and action-type NLL (`nll.action`, user confirmed 2026-11-10) <= baseline + 0.01.
  3  bunker before 4:00 does not increase:
     3a human TvT / TvP val games: the mean teacher-forced p(Build_Bunker) over the scored steps before 4:00
        (bc-eval --bunker-before 5760 -> `bunker_before.by_enemy_race`) <= the baseline's;
     3b the T0 Terran / Protoss games of check 1: the share of games with bunker_order_4m (first Build_Bunker
        command < 4:00, command stream) and with bunker_unit_4m (session bunker_s < 240) <= the baseline's.
Boundaries are inclusive ("within", "at least", "does not increase"). Every check also prints the numbers;
game counts carry Wilson 95% intervals and the two-sided Fisher p (050-R2): 32 games are about +-12 %p of
noise per condition, so a small difference is reported for the user to judge (AMB-050-3 note).
A check whose inputs are not given, or whose T0 job has unfinished games, is MISSING (e.g. a mix epoch
without the T0 games). 3a is a step mean (main-session interim decision 2026-11-10).

    python tools/eval/regression_gate.py --baseline-val <bc-eval json> --candidate-val <bc-eval json> \\
        --baseline-games <eval job dir> --candidate-games <eval job dir> [--json out.json] [--cache DIR]

The job dirs are eval jobs (protocol.py job) with a stage whose plan is the protocol plan `t0_tp`; the T0
bunker command streams come from `uranium-replay cmddump` like tables.py (cached under --cache).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_common as ec  # noqa: E402
import protocol  # noqa: E402
import tables  # noqa: E402

# 050-R10 / AMB-050-3: the gate's races, thresholds and the 4:00 boundary.
RACES = ("terran", "protoss")
TOP1_DROP = 0.005   # (2) per-race val top-1 within -0.5 %p of the baseline
NLL_RISE = 0.01     # (2) per-race val NLL within +0.01 of the baseline
NLL_HEAD = "nll.action"  # (2) the action-type head NLL (user confirmed 2026-11-10, AMB-050-3)
BUNKER_BEFORE = 4 * 60 * ec.FPS  # 3: before 4:00 = frame < 5,760
# Float slack of the inclusive boundaries (the inputs are rounded JSON floats).
EPS = 1e-9
PASS, FAIL, MISSING = "PASS", "FAIL", "MISSING"


def _check(check: str, race: str, metric: str, base, cand, threshold: str, result: str, note: str = "") -> dict:
    return {"check": check, "race": race, "metric": metric, "baseline": base, "candidate": cand, "threshold": threshold, "result": result, "note": note}


def _same_selection(base: dict, cand: dict) -> None:
    keys = ("split", "games", "selection_seed")
    if any(base.get(k) != cand.get(k) for k in keys):
        raise SystemExit(f"050-R10: baseline and candidate bc-eval differ in {', '.join(k for k in keys if base.get(k) != cand.get(k))} "
                         f"({ {k: base.get(k) for k in keys} } vs { {k: cand.get(k) for k in keys} })")


def judge_val(base: dict | None, cand: dict | None) -> list[dict]:
    """(2): per-race top-1 and action NLL of two bc-eval reports."""
    out = []
    if base is None or cand is None:
        return [_check("2", r, m, None, None, t, MISSING, "no bc-eval output") for r in RACES
                for m, t in (("action_top1", f">= baseline - {TOP1_DROP * 100:.1f} %p"), (NLL_HEAD, f"<= baseline + {NLL_RISE}"))]
    _same_selection(base, cand)
    for r in RACES:
        b, c = (base.get("by_enemy_race") or {}).get(r), (cand.get("by_enemy_race") or {}).get(r)
        if not b or not c or b.get("action_top1") is None or c.get("action_top1") is None:
            out.append(_check("2", r, "action_top1", None, None, f">= baseline - {TOP1_DROP * 100:.1f} %p", MISSING, "no by_enemy_race steps"))
            out.append(_check("2", r, NLL_HEAD, None, None, f"<= baseline + {NLL_RISE}", MISSING, "no by_enemy_race steps"))
            continue
        bt, ct = b["action_top1"], c["action_top1"]
        out.append(_check("2", r, "action_top1", bt, ct, f">= baseline - {TOP1_DROP * 100:.1f} %p",
                          PASS if ct - bt >= -TOP1_DROP - EPS else FAIL, f"{(ct - bt) * 100:+.2f} %p"))
        bn, cn = b["nll"]["action"], c["nll"]["action"]
        out.append(_check("2", r, NLL_HEAD, bn, cn, f"<= baseline + {NLL_RISE}", PASS if cn - bn <= NLL_RISE + EPS else FAIL, f"{cn - bn:+.4f}"))
    return out


def judge_tf_bunker(base: dict | None, cand: dict | None) -> list[dict]:
    """(3a): mean p(Build_Bunker) before 4:00 on the human val games against Terran / Protoss."""
    thr = "<= baseline"
    bb, cb = (base or {}).get("bunker_before"), (cand or {}).get("bunker_before")
    if not bb or not cb:
        return [_check("3a", r, "p_bunker_before_4m", None, None, thr, MISSING, "bc-eval without --bunker-before 5760") for r in RACES]
    _same_selection(base, cand)
    for x in (bb, cb):
        if x.get("before_frame") != BUNKER_BEFORE:
            raise SystemExit(f"050-R10 3: bunker_before.before_frame {x.get('before_frame')} != {BUNKER_BEFORE} (4:00)")
    out = []
    for r in RACES:
        b, c = bb["by_enemy_race"].get(r, {}), cb["by_enemy_race"].get(r, {})
        if b.get("mean_p") is None or c.get("mean_p") is None:
            out.append(_check("3a", r, "p_bunker_before_4m", b.get("mean_p"), c.get("mean_p"), thr, MISSING, "no step before 4:00"))
            continue
        out.append(_check("3a", r, "p_bunker_before_4m", b["mean_p"], c["mean_p"], thr, PASS if c["mean_p"] <= b["mean_p"] + EPS else FAIL,
                          f"{c['mean_p'] - b['mean_p']:+.5f} ({b['steps']} / {c['steps']} steps)"))
    return out


def _rate_note(bk: int, bn: int, ck: int, cn: int) -> str:
    lo_b, hi_b = ec.wilson(bk, bn)
    lo_c, hi_c = ec.wilson(ck, cn)
    return (f"{bk}/{bn} [{lo_b * 100:.0f}, {hi_b * 100:.0f}] vs {ck}/{cn} [{lo_c * 100:.0f}, {hi_c * 100:.0f}], "
            f"Fisher p {ec.fisher_p(bk, bn, ck, cn):.3f}")


def judge_games(base: dict | None, cand: dict | None) -> list[dict]:
    """(1) and (3b) from the per-race T0 counts {race: {n, won, bunker_order_4m, bunker_unit_4m, dumped, missing}}.
    A job with planned games that have neither a result row nor an invalid mark (`missing`, an unfinished job)
    is not judged."""
    out = []
    for r in RACES:
        b, c = (base or {}).get(r), (cand or {}).get(r)
        for chk, metric, thr in (("1", "won", ">= baseline"), ("3b", "bunker_order_4m", "<= baseline"), ("3b", "bunker_unit_4m", "<= baseline")):
            if not b or not c or not b["n"] or not c["n"]:
                out.append(_check(chk, r, metric, None, None, thr, MISSING, "no T0 t0_tp games"))
                continue
            if b.get("missing") or c.get("missing"):
                out.append(_check(chk, r, metric, [b[metric], b["n"]], [c[metric], c["n"]], thr, MISSING,
                                  f"unfinished job: games without a result {b.get('missing', 0)} (baseline) / {c.get('missing', 0)} (candidate)"))
                continue
            if metric == "bunker_order_4m" and (b["dumped"] < b["n"] or c["dumped"] < c["n"]):
                out.append(_check(chk, r, metric, [b[metric], b["n"]], [c[metric], c["n"]], thr, MISSING,
                                  f"command streams missing (dumped {b['dumped']}/{b['n']} vs {c['dumped']}/{c['n']})"))
                continue
            bk, bn, ck, cn = b[metric], b["n"], c[metric], c["n"]
            # Rates compared exactly: ck / cn vs bk / bn.
            ok = ck * bn >= bk * cn if metric == "won" else ck * bn <= bk * cn
            out.append(_check(chk, r, metric, [bk, bn], [ck, cn], thr, PASS if ok else FAIL, _rate_note(bk, bn, ck, cn)))
    return out


def t0_tp_counts(job: Path, cache: Path, lin: dict | None = None) -> dict:
    """Per opponent race of the job's `t0_tp` stage: valid games, wins, games with a Build_Bunker command
    before 4:00 (bunker_order_4m), games whose bunker unit appeared before 240 s (bunker_unit_4m), games
    with a command stream (dumped), planned games (games) and planned games with neither a result row nor
    an invalid mark (missing: the job has not finished them). Opponent-crash games are invalid (tables.py)."""
    lin = lin or protocol.load_lineage()
    stages = [(k, f, p) for k, f, p in ec.job_stages(job) if protocol.identify(lin, p) == "t0_tp"]
    if len(stages) != 1:
        raise SystemExit(f"050-R10: {job} has {len(stages)} stages of the protocol plan t0_tp (want 1)")
    k, _, plan = stages[0]
    rows = ec.model_rows(ec.read_results(job), k)
    invalid = ec.opponent_crashes(ec.read_ledger(job))
    dumps = ec.dump_stage(job / plan["name"], sorted(rows), cache / job.name / plan["name"])
    out: dict[str, dict] = {}
    for m, a, b in tables._blocks(plan):
        race = m["p1"]["race"]
        c = out.setdefault(race, {"n": 0, "won": 0, "bunker_order_4m": 0, "bunker_unit_4m": 0, "dumped": 0, "invalid": 0, "games": 0, "missing": 0})
        for n in range(a, b + 1):
            c["games"] += 1
            if k * ec.STAGE_BASE + n in invalid:
                c["invalid"] += 1
                continue
            if n not in rows:
                c["missing"] += 1
                continue
            r = rows[n]
            c["n"] += 1
            c["won"] += r.get("won") is True
            bs = r["model"].get("bunker_s")
            c["bunker_unit_4m"] += bs is not None and bs < 240
            if n in dumps:
                c["dumped"] += 1
                first = ec.analyze(dumps[n])[0]
                c["bunker_order_4m"] += first.get(ec.BUNKER, tables.INF) < BUNKER_BEFORE
    return out


def gate(base_val: dict | None, cand_val: dict | None, base_games: dict | None, cand_games: dict | None) -> dict:
    games = judge_games(base_games, cand_games)
    checks = [c for c in games if c["check"] == "1"] + judge_val(base_val, cand_val) + judge_tf_bunker(base_val, cand_val) + \
        [c for c in games if c["check"] == "3b"]
    res = {c["result"] for c in checks}
    overall = FAIL if FAIL in res else (MISSING if MISSING in res else PASS)
    return {"overall": overall, "checks": checks,
            "note": "050-R10: report to the user with the point's results; training is not stopped. 32-game counts carry about +-12 %p noise."}


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, list):
        return f"{v[0]}/{v[1]}"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def print_gate(g: dict) -> None:
    print("check | race | metric | baseline | candidate | threshold | result | note")
    for c in g["checks"]:
        print(f"{c['check']} | {c['race']} | {c['metric']} | {_fmt(c['baseline'])} | {_fmt(c['candidate'])} | {c['threshold']} | {c['result']} | {c['note']}")
    print(f"overall: {g['overall']}  ({g['note']})")


def _load(p: str | None) -> dict | None:
    return None if p is None else json.loads(Path(p).read_text(encoding="utf-8"))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline-val", help="bc-eval JSON of the baseline (val500, by_enemy_race; with --bunker-before 5760 for 3a)")
    ap.add_argument("--candidate-val", help="bc-eval JSON of the candidate, same selection")
    ap.add_argument("--baseline-games", type=Path, help="eval job of the baseline with a t0_tp stage")
    ap.add_argument("--candidate-games", type=Path, help="eval job of the candidate with a t0_tp stage")
    ap.add_argument("--cache", type=Path, default=None, help="command stream cache (default <URANIUM_CACHE_DIR or runs>/eval_cmddump)")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    ec.load_env()
    cache = a.cache or ec.cache_dir() / "eval_cmddump"
    lin = protocol.load_lineage()
    bg = t0_tp_counts(a.baseline_games, cache, lin) if a.baseline_games else None
    cg = t0_tp_counts(a.candidate_games, cache, lin) if a.candidate_games else None
    g = gate(_load(a.baseline_val), _load(a.candidate_val), bg, cg)
    g["inputs"] = {"baseline_val": a.baseline_val and Path(a.baseline_val).name, "candidate_val": a.candidate_val and Path(a.candidate_val).name,
                   "baseline_games": None if a.baseline_games is None else a.baseline_games.name,
                   "candidate_games": None if a.candidate_games is None else a.candidate_games.name,
                   "baseline_t0_tp": bg, "candidate_t0_tp": cg}
    print_gate(g)
    if a.json:
        a.json.write_text(json.dumps(g, indent=1), encoding="utf-8")
        print("wrote", a.json)
    return g


if __name__ == "__main__":
    main()
