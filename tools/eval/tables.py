"""Result tables of eval jobs (050-R9 / 050-AC5 measurement protocol, 050-R6 / 050-R7 / 050-R8,
docs/measurement_protocol.md): every stage of each job, by the protocol plan its conditions match
(config/protocol/lineage.toml; a plan outside the protocol is refused unless --kind names its table):

  rush     050-R6 128-game rush check by start condition: held_8m (scv_8m >= 8), held_s (scv_min_1m > 4,
           session held), scout before 3:00 (command stream; scout_3m = the old reading the DAgger thresholds
           use, scout_3m_fixed = the payload read right, 050-R9 2026-11-14), bunker_order_3m / _4m (first Build_Bunker
           command), bunker_unit_3m / _4m (bunker_s), scv_4m p50, wins; JSON also held_8m_winpass
           (rush_summary.py: a win before 8:00 passes) and scv_4m_p50_known (g3.py)
  macro    050-R8 macro 40 by opponent: bunker_order_4m, bunker_unit_4m, CC order by 5:00, scv_8m >= 16
           (a game over before 8:00 fails, as the Checkpoint 1 model tables), W/D/L; JSON also
           scv_8m_16_winpass (annihilation win before 8:00 passes, the teacher-table rule)
  t1       T1 300 by bot and race: W/D/L, surrender wins, minutes p50
  winrate  T0 / TvZ / AC1: W/D/L per matchup and opponent race, Wilson 95%

Every stage also gets the 052-R8 tables (`ops`, `rush_combat`: by matchup, medians / means, rush entry and
SCV pull / bunker / focus, split by held_s and held_8m) from `uranium-replay opsdump` replay values (052-R7, cached
as <cache>/<job>/<stage>/g<n>.ops.json; --ops live reads the session rows' own values, --ops off skips them).
Every t1 / winrate stage with Terran or Protoss opponents also gets the 050-R10 teacher-mode share (2026-11-14, 044-R6,
record only): per opponent race, the games with fewer than 150 commands per minute over 0-6 min and no refinery
build order before 4:00 (eval_common.teacher_mode), W/D/L and losses before 10:00 per mode.

Opponent-crash games (an Aborted attempt where our launcher exited with 0 and the opponent's did not,
eval_common.opponent_crashes) are invalid in every table. Both held and both bunker definitions are always
printed. A job with protocol.json also shows its checkpoint sha256, git commit, uncommitted-file flag and
host platforms.
Command streams come from `uranium-replay cmddump` (cached as <cache>/<job>/<stage>/g<n>.json).

    python tools/eval/tables.py <job dir>... [--json out.json] [--cache DIR] [--kind <stage name>=<kind>]
                                [--ops replay|live|off] [--ops-jobs N]

Moved from runs/dagger/test_table.py, t1_table.py, ac1_table.py, runs/tclone_macro/macro_table.py and
runs/bc_ft_teacher/rush_summary.py with the same counting rules.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
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
        an = {n: ec.analyze_both(dumps[n], r.get("slot", 0)) if n in dumps else (None, None, None) for n, r in g}
        m = [r["model"] for _, r in g]
        first = [an[n][0] for n, _ in g]
        bunker_s = [x.get("bunker_s") for x in m]
        out[c] = {
            "n": len(g),
            # 050-R7: scv_8m >= 8, a game over before 8:00 fails (runs/dagger/test_table.py, the DAgger tests and
            # the final gate). held_8m_winpass (JSON only): such a game passes when won (runs/bc_ft_teacher/
            # rush_summary.py, the earlier gates).
            "held_8m": sum(1 for x in m if (x.get("scv_8m") or 0) >= 8),
            "held_8m_winpass": sum(1 for x, (_, r) in zip(m, g) if (x["scv_8m"] >= 8 if x.get("scv_8m") is not None else bool(r.get("won")))),
            "held_s": sum(1 for x in m if (x.get("scv_min_1m") or 0) > 4),
            # 050-R9 (2026-11-14 개정, 사용자 결정 2026-11-14): scout_3m keeps the old payload reading (single-unit
            # Move orders only, one byte off) that 041-R22 ④ / 041-R29 ④ thresholds were set on; scout_3m_fixed
            # reads 0x14 / 0x15 as stored and also counts ground right clicks.
            "scout_3m": sum(1 for n, _ in g if an[n][1] is not None),
            "scout_3m_fixed": sum(1 for n, _ in g if an[n][2] is not None),
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
        first = [ec.analyze(dumps[n], r.get("slot", 0))[0] if n in dumps else None for n, r in g]
        m = [r["model"] for _, r in g]
        w = sum(1 for _, r in g if r["won"])
        d = sum(1 for _, r in g if r["won"] is None)
        out[o] = {
            "n": len(g),
            "bunker_order_4m": sum(1 for f in first if f and f.get(ec.BUNKER, INF) < 4 * 60 * ec.FPS),
            "bunker_unit_4m": sum(1 for x in m if x.get("bunker_s") is not None and x["bunker_s"] < 240),
            "cc_order_5m": sum(1 for f in first if f and f.get(ec.CC, INF) <= 5 * 60 * ec.FPS),
            # As the Checkpoint 1 model tables (runs/dagger/test_table.py): a game over before 8:00 fails. The
            # interim rule "an annihilation win before 8:00 passes" (reports/checkpoint1.md 9.2) was applied to
            # the teacher session tables (summary.md T0 Terran sessions) only; scv_8m_16_winpass (JSON) has it.
            "scv_8m_16": sum(1 for x in m if (x.get("scv_8m") or 0) >= 16),
            "scv_8m_16_winpass": sum(1 for x, (_, r) in zip(m, g) if (x["scv_8m"] >= 16 if x.get("scv_8m") is not None else r.get("won") is True)),
            "won": w, "draw": d, "lost": len(g) - w - d,
            "dumped": sum(1 for f in first if f is not None),
        }
    tot = {k: sum(v[k] for v in out.values()) for k in ("n", "bunker_order_4m", "bunker_unit_4m", "cc_order_5m", "scv_8m_16", "scv_8m_16_winpass", "won", "draw", "lost", "dumped")}
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


# 050-R10 / 044-R6 (2026-11-14): the teacher-mode share is recorded against these races (TvT and TvP apart); a loss
# before 10:00 is an early loss (the cause analysis' "10분 안 패").
TEACHER_MODE_RACES = ("terran", "protoss")
EARLY_LOSS = 10 * 60 * ec.FPS


def teacher_mode_games(plan: dict) -> list[int]:
    """Plan game numbers whose opponent is Terran or Protoss (the plan's p1 race)."""
    return [n for m, a, b in _blocks(plan) if m["p1"].get("race") in TEACHER_MODE_RACES for n in range(a, b + 1)]


def _mode_counts() -> dict:
    return {"n": 0, "won": 0, "draw": 0, "lost": 0, "lost_10m": 0}


def teacher_mode_table(stage: int, plan: dict, rows: dict[int, dict], invalid: set[int], dumps: dict[int, Path]) -> dict:
    """050-R10 teacher-mode share per opponent race (the plan's p1 race, Terran and Protoss only; map blocks of a race
    together) over the valid games with a model row: `n` = games with a command stream, `teacher` = teacher-mode
    games (eval_common.teacher_mode), `no_stream` = games without one (no replay, or the model's player not found),
    and W/D/L with losses before 10:00 (`lost_10m`, end_frame < 14,400) for the teacher-mode and the other games."""
    out: dict = {}
    for m, a, b in _blocks(plan):
        race = m["p1"].get("race")
        if race not in TEACHER_MODE_RACES:
            continue
        c = out.setdefault(race, {"n": 0, "teacher": 0, "no_stream": 0, "teacher_mode": _mode_counts(), "other": _mode_counts()})
        for n in range(a, b + 1):
            if n not in rows or stage * ec.STAGE_BASE + n in invalid:
                continue
            r = rows[n]
            tm = ec.teacher_mode(dumps[n], r.get("slot", 0)) if n in dumps else None
            if tm is None:
                c["no_stream"] += 1
                continue
            c["n"] += 1
            c["teacher"] += tm["teacher_mode"]
            g = c["teacher_mode" if tm["teacher_mode"] else "other"]
            g["n"] += 1
            if r.get("won") is True:
                g["won"] += 1
            elif r.get("won") is None:
                g["draw"] += 1
            else:
                g["lost"] += 1
                g["lost_10m"] += (r.get("end_frame") if r.get("end_frame") is not None else INF) < EARLY_LOSS
    return out


def stage_teacher_mode(stage_dir: Path, stage: int, plan: dict, rows: dict[int, dict], invalid: set[int], cache: Path,
                       games: dict[int, int] | None = None) -> dict:
    """The stage's `teacher_mode` table (empty without Terran / Protoss games). Command streams from `uranium-replay
    cmddump` (cached as <cache>/g<n>.json; `games` maps a row's n to its replay's game number when they differ). A
    host without the binary gets an `error` entry instead (the stage's other tables do not need it)."""
    want = [n for n in teacher_mode_games(plan) if n in rows and stage * ec.STAGE_BASE + n not in invalid]
    if not want:
        return {}
    src = games or {n: n for n in want}
    try:
        got = ec.dump_stage(stage_dir, sorted(src[n] for n in want), cache)
    except SystemExit as e:
        return {"teacher_mode": {"error": str(e)}}
    dumps = {n: got[src[n]] for n in want if src[n] in got}
    return {"teacher_mode": teacher_mode_table(stage, plan, rows, invalid, dumps)}


def add_teacher_mode_counts(res: dict) -> None:
    """`counts` rows `teacher_mode vs <race>` (input of protocol.py compare: the share with Fisher p)."""
    for race, v in (res.get("teacher_mode") or {}).items():
        if race in TEACHER_MODE_RACES:
            res["counts"][f"teacher_mode vs {race}"] = {"teacher_mode": [v["teacher"], v["n"]]}


def print_teacher_mode(t: dict) -> None:
    print("teacher mode (050-R10: < 150 commands/min over 0-6 min and no refinery order before 4:00) | vs | games | "
          "teacher-mode games | teacher-mode W/D/L, lost < 10:00 | other W/D/L, lost < 10:00 | no command stream")
    if "error" in t:
        print(f"  not counted: {t['error']}")
        return
    for race, v in t.items():
        a, o = v["teacher_mode"], v["other"]
        print(f"  | {race} | {v['n']} | {_pct(v['teacher'], v['n'])} | {a['won']}/{a['draw']}/{a['lost']}, {a['lost_10m']} | "
              f"{o['won']}/{o['draw']}/{o['lost']}, {o['lost_10m']} | {v['no_stream']}")


# 052-R8: the ops keys (052 Interfaces order) and those that are time integrals (also divided by the window length).
OPS_KEYS = ("end_s", "mined", "gas", "inc_4m", "inc_6m", "inc_8m", "inc_10m", "inc_scv_4m", "inc_scv_6m", "inc_scv_8m", "inc_scv_10m",
            "mline_bldg_n", "mline_bldg_s", "cc_lift_s", "stranded_scv_s", "idle_scv_s", "oversat_scv_s", "sblock_s", "sblock_n",
            "sblock_first_s", "float_min_rs", "float_gas_rs")
OPS_INTEGRALS = ("mline_bldg_s", "cc_lift_s", "stranded_scv_s", "idle_scv_s", "oversat_scv_s", "sblock_s", "float_min_rs", "float_gas_rs")
OPS_SOURCES = ("replay", "live", "off")


def _matchup(r: dict) -> str:
    """Our race against the race the enemy actually played (session row), e.g. TvZ."""
    return f"{(r.get('self_race') or 'Terran')[:1]}v{(r.get('enemy_race') or 'Unknown')[:1]}"


def _summary(v: list) -> dict:
    v = [x for x in v if x is not None]
    return {"n": len(v), "median": ec.median(v), "mean": sum(v) / len(v) if v else None}


def _rush_stats(items: list[tuple[dict, dict]]) -> dict:
    """052-R8 rush_combat cells over (row, value) pairs: entries; among them pull_s present (rate, median), scv_lost /
    kills medians, bunker_load_s present rate, focus_hhi median (games without a value left out)."""
    rc = [v["rush_combat"] for _, v in items if v.get("rush_combat")]
    k = len(rc)

    def present(key: str) -> list:
        return [x[key] for x in rc if x.get(key) is not None]

    pull, load = present("pull_s"), present("bunker_load_s")
    return {"n": len(items), "entered": k, "entered_rate": k / len(items) if items else None,
            "pull_n": len(pull), "pull_rate": len(pull) / k if k else None, "pull_s_p50": ec.median(pull),
            "scv_lost_p50": ec.median(present("scv_lost")), "kills_p50": ec.median(present("kills")),
            "bunker_load_n": len(load), "bunker_load_rate": len(load) / k if k else None,
            "focus_hhi_p50": ec.median(present("focus_hhi"))}


def ops_tables(rows: dict[int, dict], values: dict[int, dict]) -> tuple[dict, dict]:
    """052-R8 tables of one stage's valid model rows (by game n) and their 052 values ({n: {"ops", "rush_combat"}},
    replay or live): `ops` by matchup (and `all`): games with values, games without (`missing`), each key's
    n / median / mean (empty values left out), integrals also `<key>_per_s` = mean of value / end_s; `rush_combat`
    by matchup, each over all games and split by held_s (scv_min_1m > 4) and held_8m (scv_8m >= 8, 050-R7)."""
    groups: dict[str, list[int]] = {"all": sorted(rows)}
    for n in sorted(rows):
        groups.setdefault(_matchup(rows[n]), []).append(n)
    ops, rush = {}, {}
    for g, ns in groups.items():
        have = [n for n in ns if n in values and values[n].get("ops") is not None]
        o = [values[n]["ops"] for n in have]
        t = {"n": len(have), "missing": len(ns) - len(have)}
        for key in OPS_KEYS:
            t[key] = _summary([x.get(key) for x in o])
        for key in OPS_INTEGRALS:
            t[f"{key}_per_s"] = _summary([x[key] / x["end_s"] for x in o if x.get(key) is not None and x.get("end_s")])
        ops[g] = t
        items = [(rows[n], values[n]) for n in have]
        m = lambda r: r.get("model") or {}  # noqa: E731
        held_s = lambda r: (m(r).get("scv_min_1m") or 0) > 4  # noqa: E731
        held_8m = lambda r: (m(r).get("scv_8m") or 0) >= 8  # noqa: E731
        rush[g] = {"all": _rush_stats(items),
                   "held_s": _rush_stats([x for x in items if held_s(x[0])]), "not_held_s": _rush_stats([x for x in items if not held_s(x[0])]),
                   "held_8m": _rush_stats([x for x in items if held_8m(x[0])]), "not_held_8m": _rush_stats([x for x in items if not held_8m(x[0])])}
    return ops, rush


def stage_ops(stage_dir: Path, rows: dict[int, dict], cache: Path, source: str, jobs: int = 1, games: dict[int, int] | None = None) -> dict:
    """052-R7 / R8: the stage's ops / rush_combat tables from `source`: `replay` = `uranium-replay opsdump` of the
    stage's replays from the model's slot (cached as <cache>/g<n>.ops.json; `games` maps a row's n to its replay's
    game number when they differ, e.g. a subset of a larger job), `live` = the session rows' own values. Path
    comparisons (050-R9) use replay values on both sides."""
    if source not in OPS_SOURCES:
        raise UnknownPlan(f"--ops {source}: one of {', '.join(OPS_SOURCES)}")
    if source == "off":
        return {}
    failed: dict[int, str] = {}
    if source == "replay":
        src = games or {n: n for n in rows}
        got, bad = ec.ops_stage(stage_dir, {src[n]: rows[n].get("slot", 0) for n in rows}, cache, jobs=jobs)
        back = {g: n for n, g in src.items()}
        values = {back[g]: d for g, d in got.items()}
        failed = {back[g]: e for g, e in bad.items()}
    else:
        values = {n: {"ops": r["model"].get("ops"), "rush_combat": r["model"].get("rush_combat")} for n, r in rows.items() if r["model"].get("ops") is not None}
    ops, rush = ops_tables(rows, values)
    return {"ops_source": source, "ops_failed": {str(n): e for n, e in sorted(failed.items())}, "ops": ops, "rush_combat": rush}


KINDS = ("rush", "macro", "t1", "winrate")


class UnknownPlan(Exception):
    pass


def stage_kind(lin: dict, plan: dict, override: dict[str, str]) -> tuple[str, str | None]:
    """The protocol plan a stage's conditions match (050-R9); a plan outside the protocol needs --kind."""
    if ec.is_branch_plan(plan):
        # 050 "분기 게임(051)": branch-game games are never tabulated, --kind included.
        raise UnknownPlan(f"stage {plan['name']}: a branch-game plan (020-R22 keys, 051) is not an evaluation table (050)")
    pid = protocol.identify(lin, plan)
    if plan["name"] in override:
        if override[plan["name"]] not in KINDS:
            raise UnknownPlan(f"--kind {plan['name']}={override[plan['name']]}: one of {', '.join(KINDS)}")
        return override[plan["name"]], pid
    if pid is not None:
        return lin["plan"][pid]["kind"], pid
    raise UnknownPlan(f"stage {plan['name']}: its conditions match no protocol plan (config/protocol/lineage.toml); "
                      f"pass --kind {plan['name']}=<{'|'.join(KINDS)}> to tabulate it anyway")


def game_keys(plan: dict) -> dict[tuple, int]:
    """(seed, map, opponent, sampling) -> plan game number n."""
    seeds = ec.plan_seeds(plan)
    out, n = {}, 0
    for m in plan["matchup"]:
        p1 = json.dumps(m["p1"], sort_keys=True)
        for _ in range(m["games"]):
            n += 1
            out[(seeds[n - 1], m["map"], p1, m["p0"].get("sample", "argmax"))] = n
    return out


def subset_tables(job: Path, plan_id: str, lin: dict | None = None, cache: Path | None = None, ops: str = "replay", ops_jobs: int = 1) -> dict:
    """050-R9: a protocol plan's games found inside a larger Checkpoint 1 job by (seed, map, opponent,
    sampling), e.g. the T0 Terran / Protoss 32 + 32 (t0_tp) inside the BC baseline's T0 300, or the TvZ 32
    inside a T0 100. The source stage must have the plan's zero_last_failed and max_frames. With a `cache`, the
    052-R8 tables of those games too (`stage_ops`; the source stage's replays)."""
    lin = lin or protocol.load_lineage()
    target = tomllib.loads(protocol.render(lin, plan_id, "checkpoints/subset/ckpt-0", plan_id))
    want = game_keys(target)
    rows_all = ec.read_results(job)
    ledger = ec.read_ledger(job)
    invalid = ec.opponent_crashes(ledger)
    skipped = ec.skipped_lines(rows_all, ledger)  # job-wide, as in job_tables
    for k, f, plan in ec.job_stages(job):
        if ec.is_branch_plan(plan):  # 050 "분기 게임(051)": never a source of an evaluation table
            continue
        have = game_keys(plan)
        if not set(want) <= set(have):
            continue
        for key in ("zero_last_failed", "max_frames"):
            if protocol.normalize(plan)[key] != protocol.normalize(target)[key]:
                raise protocol.Refused(f"{job.name} stage {plan['name']}: {key} differs from plan {plan_id}")
        src = ec.model_rows(rows_all, k)
        rows = {m: src[have[key]] for key, m in want.items() if have[key] in src}
        bad = {ec.STAGE_BASE + m for key, m in want.items() if k * ec.STAGE_BASE + have[key] in invalid}
        res = {"job": job.name, "stage": k, "plan_file": f.name, "plan": plan_id, "kind": "winrate", "subset_of": plan["name"],
               "protocol": None, "conditions_sha256": lin["plan"][plan_id]["conditions_sha256"], "plan_sha256": protocol.file_sha256(f),
               "games_planned": len(want), "rows": len(rows), "invalid_games": sorted(bad),
               "skipped_lines": skipped,
               "checkpoints": sorted({r["model"].get("checkpoint") for r in rows.values()}),
               "zero_last_failed": sorted({r["model"].get("zero_last_failed") for r in rows.values()}),
               "sample": sorted({r["model"].get("sample") for r in rows.values()}), "inference": {}}
        res["table"] = winrate_table(1, target, rows, bad)
        res["counts"] = {r: {"won": [v["won"], v["n"]]} for r, v in {**res["table"]["by_race"], "total": res["table"]["total"]}.items()}
        if cache is not None:
            valid = {m: r for m, r in rows.items() if ec.STAGE_BASE + m not in bad}
            res.update(stage_ops(job / plan["name"], valid, cache / job.name / plan["name"], ops, ops_jobs, {m: have[key] for key, m in want.items() if m in valid}))
            res.update(stage_teacher_mode(job / plan["name"], 1, target, rows, bad, cache / job.name / plan["name"],
                                          {m: have[key] for key, m in want.items() if m in rows}))
            add_teacher_mode_counts(res)
        return {f"{job.name}/{plan['name']}[{plan_id}]": res}
    raise protocol.Refused(f"{job.name}: no stage holds every game of plan {plan_id}")


def job_tables(job: Path, cache: Path, override: dict[str, str] | None = None, lin: dict | None = None,
               stages: set[str] | None = None, ops: str = "replay", ops_jobs: int = 1) -> dict:
    lin = lin or protocol.load_lineage()
    rows_all = ec.read_results(job)
    ledger = ec.read_ledger(job)
    invalid = ec.opponent_crashes(ledger)
    # Job-wide (050-R9): an undecodable line has no readable game id, so every stage carries the job's list.
    skipped = ec.skipped_lines(rows_all, ledger)
    rec = protocol.read_record(job)
    out = {}
    for k, f, plan in ec.job_stages(job):
        if stages is not None and plan["name"] not in stages:
            continue
        if ec.is_branch_plan(plan):
            # 050 "분기 게임(051)": branch-game stages stay out of every table (051 records only).
            print(f"note: {job.name}/{plan['name']}: branch-game stage (051) left out of the tables", file=sys.stderr)
            continue
        kind, pid = stage_kind(lin, plan, override or {})
        rows = ec.model_rows(rows_all, k)
        res = {"job": job.name, "stage": k, "plan_file": f.name, "plan": pid, "kind": kind,
               "protocol": None if rec is None else {
                   "lineage": rec.get("lineage"), "point": rec.get("point"), "checkpoint_sha256": rec.get("checkpoint_sha256"),
                   "git_commit": rec.get("git", {}).get("commit"), "protocol_files_dirty": rec.get("git", {}).get("protocol_files_dirty"),
                   "platforms": sorted({h.get("platform") or "?" for h in rec.get("hosts", [])}), "t1_platforms": rec.get("t1_platforms")},
               "conditions_sha256": protocol.conditions_sha256(plan), "plan_sha256": protocol.file_sha256(f),
               "games_planned": sum(m["games"] for m in plan["matchup"]), "rows": len(rows),
               "invalid_games": sorted(g for g in invalid if ec.stage_of(g) == k),
               "skipped_lines": skipped,
               "checkpoints": sorted({r["model"].get("checkpoint") for r in rows.values()}),
               "zero_last_failed": sorted({r["model"].get("zero_last_failed") for r in rows.values()}),
               "sample": sorted({r["model"].get("sample") for r in rows.values()}),
               "inference": {b: {"games": c, "precision": protocol.PRECISION.get(b.split("@")[0], "?")} for b, c in ec.ledger_backends([g for g in ledger if ec.stage_of(g["game_id"]) == k]).items()}}
        if kind in ("rush", "macro"):
            dumps = ec.dump_stage(job / plan["name"], sorted(rows), cache / job.name / plan["name"])
            res["table"] = rush_table(k, plan, rows, invalid, dumps) if kind == "rush" else macro_table(k, plan, rows, invalid, dumps)
            if kind == "rush":
                res["counts"] = {c: {key: [v[key], v["n"]] for key in ("held_8m", "held_s", "scout_3m", "scout_3m_fixed", "bunker_order_3m", "bunker_order_4m", "bunker_unit_3m", "bunker_unit_4m", "won")} for c, v in res["table"].items()}
            else:
                res["counts"] = {o: {key: [v[key], v["n"]] for key in ("bunker_order_4m", "bunker_unit_4m", "cc_order_5m", "scv_8m_16", "won")} for o, v in res["table"].items()}
        elif kind == "t1":
            res["table"] = t1_table(k, plan, rows, invalid)
            res["counts"] = {b: {"won": [v["won"], v["n"]]} for b, v in {**res["table"]["bots"], "total": res["table"]["total"]}.items()}
        else:
            res["table"] = winrate_table(k, plan, rows, invalid)
            res["counts"] = {r: {"won": [v["won"], v["n"]]} for r, v in {**res["table"]["by_race"], "total": res["table"]["total"]}.items()}
        if kind in ("t1", "winrate"):
            # 050-R10 / 044-R6: the teacher-mode share against Terran / Protoss opponents (record only).
            res.update(stage_teacher_mode(job / plan["name"], k, plan, rows, invalid, cache / job.name / plan["name"]))
            add_teacher_mode_counts(res)
        # 052-R8: every game stage (opponent-crash games left out, as in the other tables).
        valid = {n: r for n, r in rows.items() if k * ec.STAGE_BASE + n not in invalid}
        res.update(stage_ops(job / plan["name"], valid, cache / job.name / plan["name"], ops, ops_jobs))
        out[f"{job.name}/{plan['name']}"] = res
    return out


def _pct(k: int, n: int) -> str:
    if n == 0:
        return "-"
    lo, hi = ec.wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def _num(x) -> str:
    return "-" if x is None else f"{x:.3g}" if isinstance(x, float) else str(x)


def print_ops(s: dict) -> None:
    """052-R8: medians (means) of the ops keys and the rush_combat cells per matchup."""
    failed = s.get("ops_failed") or {}
    print(f"052 ops ({s['ops_source']} values" + (f", {len(failed)} replay(s) failed: games {', '.join(failed)}" if failed else "") + ")")
    keys = ["mined", "gas", "inc_4m", "inc_6m", "inc_8m", "inc_10m", "inc_scv_4m", "inc_scv_6m", "inc_scv_8m", "inc_scv_10m",
            "mline_bldg_n", "sblock_n", "sblock_first_s"] + [k for k in OPS_INTEGRALS] + [f"{k}_per_s" for k in OPS_INTEGRALS]
    print("matchup | n | missing | " + " | ".join(f"{k} p50 (mean)" for k in keys))
    for g, t in s["ops"].items():
        print(f"{g} | {t['n']} | {t['missing']} | " + " | ".join(f"{_num(t[k]['median'])} ({_num(t[k]['mean'])})" for k in keys))
    print("matchup | split | n | entered | entered % | pull_s present % | pull_s p50 | scv_lost p50 | kills p50 | bunker_load_s present % | focus_hhi p50")
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    for g, t in s["rush_combat"].items():
        for split, v in t.items():
            print(f"{g} | {split} | {v['n']} | {v['entered']} | {pct(v['entered_rate'])} | {pct(v['pull_rate'])} | {_num(v['pull_s_p50'])} | "
                  f"{_num(v['scv_lost_p50'])} | {_num(v['kills_p50'])} | {pct(v['bunker_load_rate'])} | {_num(v['focus_hhi_p50'])}")


def print_tables(res: dict) -> None:
    for key, s in res.items():
        t = s["table"]
        print(f"## {key}: {s['kind']} (plan {s['plan'] or 'not a protocol plan'}), {s['rows']} / {s['games_planned']} games with a model row, "
              f"invalid {len(s['invalid_games'])}, checkpoint {', '.join(map(str, s['checkpoints']))}, zero_last_failed {s['zero_last_failed']}")
        p = s.get("protocol")
        if p is None:
            print("protocol.json: none (a job made before the protocol, or by hand)")
        else:
            dirty = "UNCOMMITTED protocol files" if p["protocol_files_dirty"] else "clean"
            print(f"protocol.json: {p['lineage']} / {p['point']}, checkpoint sha256 {p['checkpoint_sha256']}, git {str(p['git_commit'])[:12]} ({dirty}), "
                  f"host platforms {', '.join(p['platforms'])}" + (f", T1 on {', '.join(p['t1_platforms'])}" if p.get("t1_platforms") else ""))
        sk = s.get("skipped_lines") or []
        if sk:
            print(f"SKIPPED {len(sk)} undecodable line(s) of the job (counted as no row): " + ", ".join(f"{x['file']}:{x['line']}" for x in sk))
        if s["kind"] == "rush":
            print("cond | n | held_8m (scv_8m>=8) | held_s (scv_min_1m>4) | scout<3:00 (old) | scout<3:00 fixed | bunker_order<3:00 | bunker_order<4:00 | bunker_unit<3:00 | bunker_unit<4:00 | scv_4m p50 | won | dumped | invalid")
            for c, v in t.items():
                s4 = "-" if v["scv_4m_p50"] is None else f"{v['scv_4m_p50']:.0f}"
                print(f"{c} | {v['n']} | {v['held_8m']} | {v['held_s']} | {v['scout_3m']} | {v['scout_3m_fixed']} | {v['bunker_order_3m']} | {v['bunker_order_4m']} | {v['bunker_unit_3m']} | {v['bunker_unit_4m']} | {s4} | {v['won']} | {v['dumped']} | {v['invalid']}")
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
        if s.get("teacher_mode"):
            print_teacher_mode(s["teacher_mode"])
        if s["inference"]:
            print("inference: " + ", ".join(f"{b} {x['games']} ({x['precision']})" for b, x in s["inference"].items()))
        if s.get("ops"):
            print_ops(s)
        print()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jobs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None, help="write every stage's numbers (input of protocol.py compare)")
    ap.add_argument("--cache", type=Path, default=None, help="command stream cache (default <URANIUM_CACHE_DIR or runs>/eval_cmddump)")
    ap.add_argument("--kind", action="append", default=[], help="<stage name>=<rush|macro|t1|winrate> for a plan outside the protocol")
    ap.add_argument("--ops", choices=OPS_SOURCES, default="replay", help="052-R8 tables from opsdump replay values (default), the session rows' live values, or none")
    ap.add_argument("--ops-jobs", type=int, default=1, help="parallel opsdump processes (OpenBW playbacks: not on a training host)")
    a = ap.parse_args(argv)
    ec.load_env()
    cache = a.cache or ec.cache_dir() / "eval_cmddump"
    override = dict(k.split("=", 1) for k in a.kind)
    lin = protocol.load_lineage()
    res = {}
    try:
        for j in a.jobs:
            res.update(job_tables(j, cache, override, lin, ops=a.ops, ops_jobs=a.ops_jobs))
    except UnknownPlan as e:
        raise SystemExit(f"REFUSED: {e}")
    print_tables(res)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("wrote", a.json)


if __name__ == "__main__":
    main()
