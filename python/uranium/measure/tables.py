"""Result tables of eval jobs (rush, macro, T1, win rate)"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path

from uranium.measure import eval_common as ec
from uranium.measure import lineage

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
            "held_8m": sum(1 for x in m if (x.get("scv_8m") or 0) >= 8),
            "held_8m_winpass": sum(1 for x, (_, r) in zip(m, g) if (x["scv_8m"] >= 8 if x.get("scv_8m") is not None else bool(r.get("won")))),
            "held_s": sum(1 for x in m if (x.get("scv_min_1m") or 0) > 4),
            "scout_3m": sum(1 for n, _ in g if an[n][1] is not None),
            "scout_3m_fixed": sum(1 for n, _ in g if an[n][2] is not None),
            "bunker_order_3m": sum(1 for f in first if f and f.get(ec.BUNKER, INF) < 3 * 60 * ec.FPS),
            "bunker_order_4m": sum(1 for f in first if f and f.get(ec.BUNKER, INF) < 4 * 60 * ec.FPS),
            "bunker_unit_3m": sum(1 for b in bunker_s if b is not None and b < 180),
            "bunker_unit_4m": sum(1 for b in bunker_s if b is not None and b < 240),
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


TEACHER_MODE_RACES = ("terran", "protoss")
EARLY_LOSS = 10 * 60 * ec.FPS


def teacher_mode_games(plan: dict) -> list[int]:
    return [n for m, a, b in _blocks(plan) if m["p1"].get("race") in TEACHER_MODE_RACES for n in range(a, b + 1)]


STYLES = ("zerg_build", "tp_style", "human")


def _mode_counts() -> dict:
    return {"n": 0, "won": 0, "draw": 0, "lost": 0, "lost_10m": 0}


def _tally(g: dict, r: dict) -> None:
    g["n"] += 1
    if r.get("won") is True:
        g["won"] += 1
    elif r.get("won") is None:
        g["draw"] += 1
    else:
        g["lost"] += 1
        g["lost_10m"] += (r.get("end_frame") if r.get("end_frame") is not None else INF) < EARLY_LOSS


def game_style(tm: dict) -> str:
    return "zerg_build" if tm["zerg_build"] else "tp_style" if tm["tp_style"] else "human"


def teacher_mode_table(stage: int, plan: dict, rows: dict[int, dict], invalid: set[int], dumps: dict[int, Path]) -> dict:
    out: dict = {}
    for m, a, b in _blocks(plan):
        race = m["p1"].get("race")
        if race not in TEACHER_MODE_RACES:
            continue
        c = out.setdefault(race, {"n": 0, "teacher": 0, "no_stream": 0, "teacher_mode": _mode_counts(), "other": _mode_counts(),
                                  "zerg_build": 0, "tp_style": 0, "style": {k: _mode_counts() for k in STYLES}})
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
            c["zerg_build"] += tm["zerg_build"]
            c["tp_style"] += tm["tp_style"]
            _tally(c["teacher_mode" if tm["teacher_mode"] else "other"], r)
            _tally(c["style"][game_style(tm)], r)
    return out


def stage_teacher_mode(stage_dir: Path, stage: int, plan: dict, rows: dict[int, dict], invalid: set[int], cache: Path,
                       games: dict[int, int] | None = None) -> dict:
    want = [n for n in teacher_mode_games(plan) if n in rows and stage * ec.STAGE_BASE + n not in invalid]
    if not want:
        return {}
    src = games if games is not None else {n: n for n in want}
    try:
        got = ec.dump_stage(stage_dir, sorted(src[n] for n in want), cache)
    except SystemExit as e:
        return {"teacher_mode": {"error": str(e)}}
    dumps = {n: got[src[n]] for n in want if src[n] in got}
    return {"teacher_mode": teacher_mode_table(stage, plan, rows, invalid, dumps)}


def add_teacher_mode_counts(res: dict) -> None:
    for race, v in (res.get("teacher_mode") or {}).items():
        if race in TEACHER_MODE_RACES:
            res["counts"][f"teacher_mode vs {race}"] = {"teacher_mode": [v["teacher"], v["n"]]}
            for k in ("zerg_build", "tp_style"):
                res["counts"][f"{k} vs {race}"] = {k: [v[k], v["n"]]}


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
    print("game style (050-R10: zerg_build = < 150 commands/min over 0-6 min and no refinery or factory order before 5:00; "
          "tp_style = < 120 commands/min and not zerg_build; else human) | vs | games | zerg_build games | tp_style games | "
          "zerg_build W/D/L, lost < 10:00 | tp_style W/D/L, lost < 10:00 | human W/D/L, lost < 10:00")
    for race, v in t.items():
        w = " | ".join(f"{g['won']}/{g['draw']}/{g['lost']}, {g['lost_10m']}" for g in (v["style"][k] for k in STYLES))
        print(f"  | {race} | {v['n']} | {_pct(v['zerg_build'], v['n'])} | {_pct(v['tp_style'], v['n'])} | {w}")


OPS_KEYS = ("end_s", "mined", "gas", "inc_4m", "inc_6m", "inc_8m", "inc_10m", "inc_scv_4m", "inc_scv_6m", "inc_scv_8m", "inc_scv_10m",
            "mline_bldg_n", "mline_bldg_s", "cc_lift_s", "stranded_scv_s", "idle_scv_s", "oversat_scv_s", "sblock_s", "sblock_n",
            "sblock_first_s", "float_min_rs", "float_gas_rs")
OPS_INTEGRALS = ("mline_bldg_s", "cc_lift_s", "stranded_scv_s", "idle_scv_s", "oversat_scv_s", "sblock_s", "float_min_rs", "float_gas_rs")
OPS_SOURCES = ("replay", "live", "off")


def _matchup(r: dict) -> str:
    return f"{(r.get('self_race') or 'Terran')[:1]}v{(r.get('enemy_race') or 'Unknown')[:1]}"


def _summary(v: list) -> dict:
    v = [x for x in v if x is not None]
    return {"n": len(v), "median": ec.median(v), "mean": sum(v) / len(v) if v else None}


def _rush_stats(items: list[tuple[dict, dict]]) -> dict:
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
    if ec.is_branch_plan(plan):
        raise UnknownPlan(f"stage {plan['name']}: a branch-game plan (020-R22 keys, 051) is not an evaluation table (050)")
    pid = lineage.identify(lin, plan)
    if plan["name"] in override:
        if override[plan["name"]] not in KINDS:
            raise UnknownPlan(f"--kind {plan['name']}={override[plan['name']]}: one of {', '.join(KINDS)}")
        return override[plan["name"]], pid
    if pid is not None:
        return lin["plan"][pid]["kind"], pid
    raise UnknownPlan(f"stage {plan['name']}: its conditions match no protocol plan (config/protocol/lineage.toml); "
                      f"pass --kind {plan['name']}=<{'|'.join(KINDS)}> to tabulate it anyway")


def game_keys(plan: dict) -> dict[tuple, int]:
    seeds = ec.plan_seeds(plan)
    out, n = {}, 0
    for m in plan["matchup"]:
        p1 = json.dumps(m["p1"], sort_keys=True)
        for _ in range(m["games"]):
            n += 1
            out[(seeds[n - 1], m["map"], p1, m["p0"].get("sample", "argmax"))] = n
    return out


def subset_tables(job: Path, plan_id: str, lin: dict | None = None, cache: Path | None = None, ops: str = "replay", ops_jobs: int = 1) -> dict:
    lin = lin or lineage.load_lineage()
    target = tomllib.loads(lineage.render(lin, plan_id, "checkpoints/subset/ckpt-0", plan_id))
    want = game_keys(target)
    rows_all = ec.read_results(job)
    ledger = ec.read_ledger(job)
    invalid = ec.opponent_crashes(ledger)
    skipped = ec.skipped_lines(rows_all, ledger)
    for k, f, plan in ec.job_stages(job):
        if ec.is_branch_plan(plan):
            continue
        have = game_keys(plan)
        if not set(want) <= set(have):
            continue
        for key in ("zero_last_failed", "max_frames"):
            if lineage.normalize(plan)[key] != lineage.normalize(target)[key]:
                raise lineage.Refused(f"{job.name} stage {plan['name']}: {key} differs from plan {plan_id}")
        src = ec.model_rows(rows_all, k)
        rows = {m: src[have[key]] for key, m in want.items() if have[key] in src}
        bad = {ec.STAGE_BASE + m for key, m in want.items() if k * ec.STAGE_BASE + have[key] in invalid}
        res = {"job": job.name, "stage": k, "plan_file": f.name, "plan": plan_id, "kind": "winrate", "subset_of": plan["name"],
               "protocol": None, "conditions_sha256": lin["plan"][plan_id]["conditions_sha256"], "plan_sha256": ec.file_sha256(f),
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
    raise lineage.Refused(f"{job.name}: no stage holds every game of plan {plan_id}")


def job_tables(job: Path, cache: Path, override: dict[str, str] | None = None, lin: dict | None = None,
               stages: set[str] | None = None, ops: str = "replay", ops_jobs: int = 1) -> dict:
    lin = lin or lineage.load_lineage()
    rows_all = ec.read_results(job)
    ledger = ec.read_ledger(job)
    invalid = ec.opponent_crashes(ledger)
    skipped = ec.skipped_lines(rows_all, ledger)
    rec = lineage.read_record(job)
    out = {}
    for k, f, plan in ec.job_stages(job):
        if stages is not None and plan["name"] not in stages:
            continue
        if ec.is_branch_plan(plan):
            print(f"note: {job.name}/{plan['name']}: branch-game stage (051) left out of the tables", file=sys.stderr)
            continue
        kind, pid = stage_kind(lin, plan, override or {})
        rows = ec.model_rows(rows_all, k)
        res = {"job": job.name, "stage": k, "plan_file": f.name, "plan": pid, "kind": kind,
               "protocol": None if rec is None else {
                   "lineage": rec.get("lineage"), "point": rec.get("point"), "checkpoint_sha256": rec.get("checkpoint_sha256"),
                   "git_commit": rec.get("git", {}).get("commit"), "protocol_files_dirty": rec.get("git", {}).get("protocol_files_dirty"),
                   "platforms": sorted({h.get("platform") or "?" for h in rec.get("hosts", [])}), "t1_platforms": rec.get("t1_platforms")},
               "conditions_sha256": lineage.conditions_sha256(plan), "plan_sha256": ec.file_sha256(f),
               "games_planned": sum(m["games"] for m in plan["matchup"]), "rows": len(rows),
               "invalid_games": sorted(g for g in invalid if ec.stage_of(g) == k),
               "skipped_lines": skipped,
               "checkpoints": sorted({r["model"].get("checkpoint") for r in rows.values()}),
               "zero_last_failed": sorted({r["model"].get("zero_last_failed") for r in rows.values()}),
               "sample": sorted({r["model"].get("sample") for r in rows.values()}),
               "inference": {b: {"games": c, "precision": ec.PRECISION.get(b.split("@")[0], "?")} for b, c in ec.ledger_backends([g for g in ledger if ec.stage_of(g["game_id"]) == k]).items()}}
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
            res.update(stage_teacher_mode(job / plan["name"], k, plan, rows, invalid, cache / job.name / plan["name"]))
            add_teacher_mode_counts(res)
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
    ap.add_argument("--ops", choices=OPS_SOURCES, default="replay", help="ops tables from opsdump replay values (default), the session rows' live values, or none")
    ap.add_argument("--ops-jobs", type=int, default=1, help="parallel opsdump processes (OpenBW playbacks: not on a training host)")
    a = ap.parse_args(argv)
    ec.load_env()
    cache = a.cache or ec.cache_dir() / "eval_cmddump"
    override = dict(k.split("=", 1) for k in a.kind)
    lin = lineage.load_lineage()
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
