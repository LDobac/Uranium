"""Result tables of eval jobs (050-R6 / 050-R7 / 050-R8, docs/measurement_protocol.md): every stage of each
job, by the protocol plan its conditions match (config/protocol/lineage.toml):

  rush     050-R6 128-game rush check by start condition: held_8m (scv_8m >= 8), held_s (scv_min_1m > 4,
           session held), scout before 3:00 (command stream), bunker_order_3m / _4m (first Build_Bunker
           command), bunker_unit_3m / _4m (bunker_s), scv_4m p50, wins
  macro    050-R8 macro 40 by opponent: bunker_order_4m, bunker_unit_4m, CC order by 5:00, scv_8m >= 16, W/D/L
  t1       T1 300 by bot and race: W/D/L, surrender wins, minutes p50
  winrate  T0 / TvZ / AC1: W/D/L per matchup and opponent race, Wilson 95%

Opponent-crash games (an Aborted attempt whose opponent launcher did not exit with 0, eval_common.
opponent_crashes) are invalid in every table. Both held and both bunker definitions are always printed.
Command streams come from `uranium-replay cmddump` (cached as <cache>/<job>/<stage>/g<n>.json).

    python tools/eval/tables.py <job dir>... [--json out.json] [--cache DIR] [--kind <stage name>=<kind>]

Moved from runs/dagger/test_table.py, t1_table.py, ac1_table.py, runs/tclone_macro/macro_table.py and
runs/bc_ft_teacher/rush_summary.py with the same counting rules.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_common as ec  # noqa: E402
import protocol  # noqa: E402

INF = 10**9


def rush_table(stage: int, plan: dict, rows: dict[int, dict], invalid: set[int], dumps: dict[int, Path]) -> dict:
    seeds = ec.plan_seeds(plan)
    cond = {n: ec.start_condition(seeds[n - 1]) for n in rows}
    out = {}
    for c in ("early", "late"):
        g = [(n, r) for n, r in sorted(rows.items()) if cond[n] == c and stage * 100000 + n not in invalid]
        an = {n: ec.analyze(dumps[n]) if n in dumps else (None, None) for n, _ in g}
        m = [r["model"] for _, r in g]
        first = [an[n][0] for n, _ in g]
        bunker_s = [x.get("bunker_s") for x in m]
        out[c] = {
            "n": len(g),
            "held_8m": sum(1 for x in m if (x.get("scv_8m") or 0) >= 8),
            "held_s": sum(1 for x in m if (x.get("scv_min_1m") or 0) > 4),
            "scout_3m": sum(1 for n, _ in g if an[n][1] is not None),
            "bunker_order_3m": sum(1 for f in first if f and f.get(ec.BUNKER, INF) < 3 * 60 * ec.FPS),
            "bunker_order_4m": sum(1 for f in first if f and f.get(ec.BUNKER, INF) < 4 * 60 * ec.FPS),
            "bunker_unit_3m": sum(1 for b in bunker_s if b is not None and b < 180),
            "bunker_unit_4m": sum(1 for b in bunker_s if b is not None and b < 240),
            # test_table.py (a game over before 4:00 counts 0) and runs/mix/g3.py (such games left out, JSON only).
            "scv_4m_p50": ec.median([x.get("scv_4m") or 0 for x in m]),
            "scv_4m_p50_known": ec.median([x["scv_4m"] for x in m if x.get("scv_4m") is not None]),
            "won": sum(1 for _, r in g if r.get("won") is True),
            "dumped": sum(1 for f in first if f is not None),
            "invalid": sum(1 for n in rows if cond[n] == c and stage * 100000 + n in invalid),
        }
    return out


def macro_table(stage: int, plan: dict, rows: dict[int, dict], invalid: set[int], dumps: dict[int, Path]) -> dict:
    opp = ec.game_opponents(plan)
    out = {}
    for o in dict.fromkeys(opp):
        g = [(n, r) for n, r in sorted(rows.items()) if opp[n - 1] == o and stage * 100000 + n not in invalid]
        first = [ec.analyze(dumps[n])[0] if n in dumps else None for n, _ in g]
        m = [r["model"] for _, r in g]
        w = sum(1 for _, r in g if r["won"])
        d = sum(1 for _, r in g if r["won"] is None)
        out[o] = {
            "n": len(g),
            "bunker_order_4m": sum(1 for f in first if f and f.get(ec.BUNKER, INF) < 4 * 60 * ec.FPS),
            "bunker_unit_4m": sum(1 for x in m if x.get("bunker_s") is not None and x["bunker_s"] < 240),
            "cc_order_5m": sum(1 for f in first if f and f.get(ec.CC, INF) <= 5 * 60 * ec.FPS),
            "scv_8m_16": sum(1 for x in m if (x.get("scv_8m") or 0) >= 16),
            "won": w, "draw": d, "lost": len(g) - w - d,
            "dumped": sum(1 for f in first if f is not None),
        }
    tot = {k: sum(v[k] for v in out.values()) for k in ("n", "bunker_order_4m", "bunker_unit_4m", "cc_order_5m", "scv_8m_16", "won", "draw", "lost", "dumped")}
    out["total"] = tot
    return out


def _blocks(plan: dict):
    n0 = 0
    for m in plan["matchup"]:
        yield m, n0 + 1, n0 + m["games"]
        n0 += m["games"]


def t1_table(stage: int, plan: dict, rows: dict[int, dict], invalid: set[int]) -> dict:
    """runs/dagger/t1_table.py per block, minus opponent-crash games (ac1_table.py, checkpoint1.md 7.3)."""
    blocks, per_bot, tot = [], {}, {"n": 0, "won": 0, "draw": 0, "lost": 0, "surrender_wins": 0, "invalid": 0, "missing": 0}
    for m, a, b in _blocks(plan):
        bad = [n for n in range(a, b + 1) if stage * 100000 + n in invalid]
        g = [rows[n] for n in range(a, b + 1) if n in rows and n not in bad]
        w = sum(1 for r in g if r["won"])
        d = sum(1 for r in g if r["won"] is None)
        s = sum(1 for r in g if r["won"] and r["reason"] == "Surrender")
        mins = sorted(r["end_frame"] / 1440 for r in g)
        row = {"bot": m["p1"].get("name") or ec.opponent_label(m["p1"]), "race": m["p1"]["race"], "n": len(g), "won": w, "draw": d,
               "lost": len(g) - w - d, "surrender_wins": s, "minutes_p50": mins[len(mins) // 2] if mins else None,
               "invalid": len(bad), "missing": sum(1 for n in range(a, b + 1) if n not in rows and n not in bad)}
        blocks.append(row)
        for c in (tot, per_bot.setdefault(row["bot"], {"n": 0, "won": 0, "draw": 0, "lost": 0, "surrender_wins": 0, "invalid": 0, "missing": 0})):
            for k in c:
                c[k] += row[k]
    return {"blocks": blocks, "bots": per_bot, "total": tot}


def winrate_table(stage: int, plan: dict, rows: dict[int, dict], invalid: set[int]) -> dict:
    blocks, by_race, tot = [], {}, {"n": 0, "won": 0, "draw": 0, "lost": 0, "invalid": 0, "missing": 0}
    for m, a, b in _blocks(plan):
        bad = [n for n in range(a, b + 1) if stage * 100000 + n in invalid]
        g = [rows[n] for n in range(a, b + 1) if n in rows and n not in bad]
        w = sum(1 for r in g if r["won"])
        d = sum(1 for r in g if r["won"] is None)
        row = {"opponent": ec.opponent_label(m["p1"]), "race": m["p1"]["race"], "map": m["map"], "sample": m["p0"].get("sample", "argmax"),
               "n": len(g), "won": w, "draw": d, "lost": len(g) - w - d, "invalid": len(bad),
               "missing": sum(1 for n in range(a, b + 1) if n not in rows and n not in bad)}
        blocks.append(row)
        key = f"{row['sample']} vs {row['race']}"
        for c in (tot, by_race.setdefault(key, {"n": 0, "won": 0, "draw": 0, "lost": 0, "invalid": 0, "missing": 0})):
            for k in c:
                c[k] += row[k]
    return {"blocks": blocks, "by_race": by_race, "total": tot}


def stage_kind(lin: dict, plan: dict, override: dict[str, str]) -> tuple[str, str | None]:
    if plan["name"] in override:
        return override[plan["name"]], None
    pid = protocol.identify(lin, plan)
    if pid is not None:
        return lin["plan"][pid]["kind"], pid
    return "winrate", None


def job_tables(job: Path, cache: Path, override: dict[str, str] | None = None, lin: dict | None = None) -> dict:
    lin = lin or protocol.load_lineage()
    rows_all = ec.read_results(job)
    ledger = ec.read_ledger(job)
    invalid = ec.opponent_crashes(ledger)
    out = {}
    for k, f, plan in ec.job_stages(job):
        kind, pid = stage_kind(lin, plan, override or {})
        rows = ec.model_rows(rows_all, k)
        res = {"job": job.name, "stage": k, "plan_file": f.name, "plan": pid, "kind": kind,
               "conditions_sha256": protocol.conditions_sha256(plan), "plan_sha256": protocol.file_sha256(f),
               "games_planned": sum(m["games"] for m in plan["matchup"]), "rows": len(rows),
               "invalid_games": sorted(g for g in invalid if ec.stage_of(g) == k),
               "checkpoints": sorted({r["model"].get("checkpoint") for r in rows.values()}),
               "zero_last_failed": sorted({r["model"].get("zero_last_failed") for r in rows.values()}),
               "sample": sorted({r["model"].get("sample") for r in rows.values()}),
               "inference": {b: {"games": c, "precision": protocol.PRECISION.get(b.split("@")[0], "?")} for b, c in ec.ledger_backends([g for g in ledger if ec.stage_of(g["game_id"]) == k]).items()}}
        if kind in ("rush", "macro"):
            dumps = ec.dump_stage(job / plan["name"], sorted(rows), cache / job.name / plan["name"])
            res["table"] = rush_table(k, plan, rows, invalid, dumps) if kind == "rush" else macro_table(k, plan, rows, invalid, dumps)
            if kind == "rush":
                res["counts"] = {c: {key: [v[key], v["n"]] for key in ("held_8m", "held_s", "scout_3m", "bunker_order_3m", "bunker_order_4m", "bunker_unit_3m", "bunker_unit_4m", "won")} for c, v in res["table"].items()}
            else:
                res["counts"] = {o: {key: [v[key], v["n"]] for key in ("bunker_order_4m", "bunker_unit_4m", "cc_order_5m", "scv_8m_16", "won")} for o, v in res["table"].items()}
        elif kind == "t1":
            res["table"] = t1_table(k, plan, rows, invalid)
            res["counts"] = {b: {"won": [v["won"], v["n"]]} for b, v in {**res["table"]["bots"], "total": res["table"]["total"]}.items()}
        else:
            res["table"] = winrate_table(k, plan, rows, invalid)
            res["counts"] = {r: {"won": [v["won"], v["n"]]} for r, v in {**res["table"]["by_race"], "total": res["table"]["total"]}.items()}
        out[f"{job.name}/{plan['name']}"] = res
    return out


def _pct(k: int, n: int) -> str:
    if n == 0:
        return "-"
    lo, hi = ec.wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def print_tables(res: dict) -> None:
    for key, s in res.items():
        t = s["table"]
        print(f"## {key}: {s['kind']} (plan {s['plan'] or 'not a protocol plan'}), {s['rows']} / {s['games_planned']} games with a model row, "
              f"invalid {len(s['invalid_games'])}, checkpoint {', '.join(map(str, s['checkpoints']))}, zero_last_failed {s['zero_last_failed']}")
        if s["kind"] == "rush":
            print("cond | n | held_8m (scv_8m>=8) | held_s (scv_min_1m>4) | scout<3:00 | bunker_order<3:00 | bunker_order<4:00 | bunker_unit<3:00 | bunker_unit<4:00 | scv_4m p50 | won | dumped | invalid")
            for c, v in t.items():
                s4 = "-" if v["scv_4m_p50"] is None else f"{v['scv_4m_p50']:.0f}"
                print(f"{c} | {v['n']} | {v['held_8m']} | {v['held_s']} | {v['scout_3m']} | {v['bunker_order_3m']} | {v['bunker_order_4m']} | {v['bunker_unit_3m']} | {v['bunker_unit_4m']} | {s4} | {v['won']} | {v['dumped']} | {v['invalid']}")
        elif s["kind"] == "macro":
            print("opponent | n | bunker_order<4:00 | bunker_unit<4:00 | CC order<=5:00 | scv_8m>=16 | won/draw/lost | dumped")
            for o, v in t.items():
                print(f"{o} | {v['n']} | {v['bunker_order_4m']} | {v['bunker_unit_4m']} | {v['cc_order_5m']} | {v['scv_8m_16']} | {v['won']}/{v['draw']}/{v['lost']} | {v['dumped']}")
        elif s["kind"] == "t1":
            print("bot | race | n valid | won | draw | lost | surrender-wins | minutes p50 | invalid | missing")
            for v in t["blocks"]:
                mp = "-" if v["minutes_p50"] is None else f"{v['minutes_p50']:.1f}"
                print(f"{v['bot']} | {v['race']} | {v['n']} | {v['won']} | {v['draw']} | {v['lost']} | {v['surrender_wins']} | {mp} | {v['invalid']} | {v['missing']}")
            for b, v in t["bots"].items():
                print(f"  {b}: {v['won']}/{v['n']} won, {v['draw']} draws, invalid {v['invalid']}")
            v = t["total"]
            print(f"total: {_pct(v['won'], v['n'])}, {v['draw']} draws, invalid {v['invalid']}, missing {v['missing']}")
        else:
            print("opponent | race | map | sample | n | won | draw | lost | invalid | missing")
            for v in t["blocks"]:
                print(f"{v['opponent']} | {v['race']} | {v['map']} | {v['sample']} | {v['n']} | {v['won']} | {v['draw']} | {v['lost']} | {v['invalid']} | {v['missing']}")
            for r, v in t["by_race"].items():
                print(f"  {r}: {_pct(v['won'], v['n'])}, {v['draw']} draws")
            v = t["total"]
            print(f"total: {_pct(v['won'], v['n'])}, {v['draw']} draws, invalid {v['invalid']}, missing {v['missing']}")
        if s["inference"]:
            print("inference: " + ", ".join(f"{b} {x['games']} ({x['precision']})" for b, x in s["inference"].items()))
        print()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jobs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None, help="write every stage's numbers (input of protocol.py compare)")
    ap.add_argument("--cache", type=Path, default=None, help="command stream cache (default <URANIUM_CACHE_DIR or runs>/eval_cmddump)")
    ap.add_argument("--kind", action="append", default=[], help="<stage name>=<rush|macro|t1|winrate> for a plan outside the protocol")
    a = ap.parse_args(argv)
    ec.load_env()
    cache = a.cache or Path(os.environ.get("URANIUM_CACHE_DIR") or ec.ROOT / "runs") / "eval_cmddump"
    override = dict(k.split("=", 1) for k in a.kind)
    lin = protocol.load_lineage()
    res = {}
    for j in a.jobs:
        res.update(job_tables(j, cache, override, lin))
    print_tables(res)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("wrote", a.json)


if __name__ == "__main__":
    main()
