"""Tables of the 050-R9 diagnostics (cue_diag, dead_channels; config/protocol/lineage.toml, docs/
measurement_protocol.md) from the output folders `protocol.py diag --run` writes: one row per checkpoint.

- cue probes, per cue hold-out set (teacher: Zerg; human: Terran / Protoss) and race: rows / games, ridge R2
  (AUC for a class task) with the game-bootstrap 95% CI at the reported stages for each reported task
  (`report.stages`, `report.tasks`), the h_t MLP(64) value, the nuisance-only ridge, the gain over the
  nuisance at h_t; a skipped task shows its reason. The enemy max-seen production-building count
  (`count|focus_rows`) has values on the human set; on the Zerg teacher rows it is constant (skipped).
- intervention (teacher set, all classes): games / edited, mean delta [CI] of every reported effect and metric
- dead channels (041-R25 training-log records, `spatial_log.json`, tools/eval/spatial_logs.py): alarms, worst
  dead layer share, max const S8 / s_pool, min positive share. Checkpoint 1 lineage: not measured (no
  training-log statistics).

Every folder is checked first (`protocol.py check-diag`); a folder that does not conform is refused unless
--no-check (then marked). Moved from the probe-table part of the 041-R24 lambda-run summary
(<shared>/eval/m2-lambda-20261110/scripts/summary.py) with the same cells.

    python tools/eval/diag_table.py <label>=<output folder>... [--metrics cue_diag,dead_channels] [--json out.json] [--md out.md] [--no-check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol  # noqa: E402


def _load(f: Path) -> dict | None:
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


def ci(m: dict | None, pt: str = "pt", lo: str = "lo", hi: str = "hi", d: int = 3, sign: bool = False) -> str:
    if not isinstance(m, dict) or m.get(pt) is None:
        return "-"
    s = f"{m[pt]:+.{d}f}" if sign else f"{m[pt]:.{d}f}"
    return f"{s} [{m[lo]:.{d}f}, {m[hi]:.{d}f}]" if m.get(lo) is not None else s


def cue_rows(lin: dict, name: str, pr: dict) -> dict:
    """{race: {task: cells}} of a `diag probes` output of one cue set for its reported tasks / stages."""
    s = protocol.cue_set(lin, name)
    rep = s["report"]
    out = {}
    for race in s["probes"]["races"]:
        tasks = pr.get("races", {}).get(race, {}).get("tasks", {})
        rr = {}
        for t in rep["tasks"]:
            x = tasks.get(t, {})
            if "metrics" not in x:
                rr[t] = {"skipped": x.get("skipped", "not run"), "rows": x.get("rows")}
                continue
            m = x["metrics"]
            rr[t] = {"metric": x["metric"], "rows": x["rows"], "games": x["games"],
                     "stages": {st: {k: m.get(f"{st}|ridge", {}).get(k) for k in ("pt", "lo", "hi")} for st in rep["stages"]},
                     "gain": {st: {k: m.get(f"{st}+nuis|ridge", {}).get(k) for k in ("gain", "gain_lo", "gain_hi")} for st in rep["stages"]},
                     "h_mlp": m.get("h|mlp", {}).get("pt"), "nuisance": {k: m.get("nuis|ridge", {}).get(k) for k in ("pt", "lo", "hi")}}
        out[race] = rr
    return out


def intervention_rows(lin: dict, name: str, iv: dict) -> dict:
    s = protocol.cue_set(lin, name)
    rep = s["report"]
    out = {}
    for race in s["intervene"]["races"]:
        r = iv.get("races", {}).get(race)
        if r is None:
            continue
        allc = r["by_class"]["all"]
        out[race] = {"games": r["games"], "edited": r["edited"], "edited_unf": r.get("edited_unf"), "count_edited": r.get("count_edited"),
                     "effects": {e: {m: allc.get(e, {}).get(m) for m in rep["effect_metrics"]} for e in rep["effects"]}}
    return out


def folder_table(lin: dict, out: Path, metrics: list[str], check: bool = True) -> dict:
    problems = [p for m in metrics for p in protocol.check_diag(lin, m, out)] if check else None
    if problems:
        raise protocol.Refused(f"{out}: " + "; ".join(problems))
    rec = _load(out / protocol.DIAG_RECORD)
    res = {"folder": str(out), "checked": check, "protocol": None if rec is None else {k: rec.get(k) for k in ("lineage", "point", "checkpoint", "checkpoint_sha256")}
           | {"git_commit": (rec.get("git") or {}).get("commit"), "protocol_files_dirty": (rec.get("git") or {}).get("protocol_files_dirty")}}
    if "cue_diag" in metrics:
        res["cue"], res["intervention"] = {}, {}
        for name in protocol.diag_metric(lin, "cue_diag")["sets"]:
            pr = _load(out / name / "probes.json")
            res["cue"][name] = None if pr is None else cue_rows(lin, name, pr)
            if "intervene" in protocol.cue_set(lin, name):
                iv = _load(out / name / "intervene.json")
                res["intervention"][name] = None if iv is None else intervention_rows(lin, name, iv)
    if "dead_channels" in metrics:
        res["train_log"] = _load(out / protocol.TRAIN_LOG)
    return res


def markdown(lin: dict, res: dict[str, dict]) -> list[str]:
    L: list[str] = []
    for name in protocol.diag_metric(lin, "cue_diag")["sets"]:
        s = protocol.cue_set(lin, name)
        rep = s["report"]
        cue = {k: v["cue"][name] for k, v in res.items() if (v.get("cue") or {}).get(name)}
        for race in s["probes"]["races"] if cue else []:
            for t in rep["tasks"]:
                L += ["", f"### cue probe, {name} hold-out, {race} `{t}` (ridge R2 / AUC, game bootstrap 95% CI)", ""]
                head = ["checkpoint", "rows / games"] + list(rep["stages"]) + ["h MLP(64)", "nuisance", "gain at h"]
                L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                for lab, r in cue.items():
                    x = r.get(race, {}).get(t, {})
                    if "stages" not in x:
                        L.append(f"| {lab} | skipped: {x.get('skipped')} |" + " |" * (len(head) - 2))
                        continue
                    cells = [ci(x["stages"][st]) for st in rep["stages"]]
                    mlp = "-" if x["h_mlp"] is None else f"{x['h_mlp']:.3f}"
                    L.append(f"| {lab} | {x['rows']} / {x['games']} | " + " | ".join(cells) + f" | {mlp} | {ci(x['nuisance'])} | "
                             f"{ci(x['gain'].get('h'), 'gain', 'gain_lo', 'gain_hi', sign=True)} |")
        iv = {k: v["intervention"][name] for k, v in res.items() if (v.get("intervention") or {}).get(name)}
        for race in s.get("intervene", {}).get("races", []) if iv else []:
            L += ["", f"### counterfactual intervention, {name} hold-out, {race} (all classes, mean delta [95% CI])", ""]
            head = ["checkpoint", "games / edited"] + [f"{e} {m}" for e in rep["effects"] for m in rep["effect_metrics"]]
            L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            for lab, r in iv.items():
                x = r.get(race)
                if x is None:
                    continue
                cells = [ci(x["effects"][e][m], "mean", sign=True, d=4) if (x["effects"][e][m] or {}).get("n") else "-" for e in rep["effects"] for m in rep["effect_metrics"]]
                L.append(f"| {lab} | {x['games']} / {x['edited']} | " + " | ".join(cells) + " |")
    tl = {k: v for k, v in res.items() if "train_log" in v}
    if tl:
        L += ["", "### dead channels (041-R25 training-log records)", "",
              "| checkpoint | records | alarms (lines / records / crossing) | worst dead layer | max const S8 | max const s_pool | min pos |", "|---|---|---|---|---|---|---|"]
        for lab, r in tl.items():
            s = r["train_log"]
            if s is None:
                L.append(f"| {lab} | not measured (no training-log statistics) ||||||")
            elif "dead" not in s:
                L.append(f"| {lab} | none from update {s.get('from_update')} ||||||")
            else:
                w = s["worst_dead_layer"]
                L.append(f"| {lab} | {s['sp_records_from']} | {len(s['alarm_lines'])} / {len(s['alarm_records'])} / {len(s['crossing_records'])} | "
                         f"{w} {s['worst_dead_share'] * 100:.1f}% ({s['dead'][w]['max']:g} @ {s['dead'][w]['at']}) | {s['const']['sp_const_s8']['max']:g} | "
                         f"{s['const']['sp_const_pool']['max']:g} | {s['min_pos_any']:.3f} |")
    return L


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folders", nargs="+", help="<label>=<output folder>")
    ap.add_argument("--metrics", default="cue_diag,dead_channels", help="a Checkpoint 1 folder has cue_diag only")
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args(argv)
    lin = protocol.load_lineage()
    metrics = [m.strip() for m in a.metrics.split(",") if m.strip()]
    res = {}
    try:
        for x in a.folders:
            lab, _, f = x.partition("=")
            if not f:
                lab, f = Path(x).name, x
            res[lab] = folder_table(lin, Path(f), metrics, check=not a.no_check)
    except protocol.Refused as e:
        raise SystemExit(f"REFUSED: {e}")
    text = "\n".join(markdown(lin, res)).lstrip("\n") + "\n"
    if a.no_check:
        text = "(not checked against the protocol: --no-check)\n\n" + text
    sys.stdout.write(text)
    if a.md:
        Path(a.md).write_text(text, encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("wrote", a.json)


if __name__ == "__main__":
    main()
