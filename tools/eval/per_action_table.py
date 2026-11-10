"""Per-action teacher hold-out table (050-R9 protocol metric `per_action`, 041-R21): one column per
tools/eval/per_action.py JSON. Rows: top-1 of the key actions (with tile accuracy for positional actions and
target accuracy for Attack_Unit), the overall step-weighted top-1, steps / games, and the column-to-column
change of Build_Bunker top-1 (the bc-mix extension rule input). Moved from runs/mix/table.py with the same
cells. A column scored with the 041-R27 rush / macro split (per_action.py `r27`) adds the Build_Bunker top-1 on
rush-opponent games (the extension rule input from model 2 on, AMB-041-16) with its column-to-column change,
and the macro-opponent games' mean p(Build_Bunker) before 4:00 (recorded only); other columns show "-".

    python tools/eval/per_action_table.py BC=<json> mix-e1=<json> ... [--json out.json]

--json writes {label: {"file", "steps", "games", "overall_top1", "actions": {action: per_action row}, "r27"}}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = ["Build_Bunker", "Move_Pos", "Train_Marine", "Attack_Unit", "Repair_Unit", "NoOp", "Train_SCV", "Attack_Pos"]


def load(p: Path) -> dict:
    d = json.loads(p.read_text(encoding="utf-8"))
    tot = sum(r["n"] for r in d["per_action"])
    return {"file": str(p), "steps": d.get("steps"), "games": d.get("games"),
            "overall_top1": sum(r["top1"] * r["n"] for r in d["per_action"]) / tot,
            "actions": {r["action"]: r for r in d["per_action"]}, "r27": d.get("r27")}


def cell(r: dict | None, action: str) -> str:
    if r is None:
        return "-"
    c = f"{r['top1'] * 100:.1f}"
    if r["loc_n"]:
        c += f" (tile {r['loc_exact'] * 100:.1f})"
    if action == "Attack_Unit" and r["tu_n"]:
        c += f" (tgt {r['tu_top1'] * 100:.0f})"
    return c


def table(data: dict[str, dict]) -> list[str]:
    cols = list(data)
    out = ["action | " + " | ".join(cols)]
    for a in ROWS:
        out.append(f"{a} | " + " | ".join(cell(data[k]["actions"].get(a), a) for k in cols))
    out.append("overall top-1 | " + " | ".join(f"{data[k]['overall_top1'] * 100:.1f}" for k in cols))
    out.append("steps / games | " + " | ".join(f"{data[k]['steps']} / {data[k]['games']}" for k in cols))
    bk = [data[k]["actions"]["Build_Bunker"]["top1"] * 100 for k in cols if "Build_Bunker" in data[k]["actions"]]
    if len(bk) >= 2:
        out.append(f"bunker top-1 per column {[round(b, 1) for b in bk]}, last change {bk[-1] - bk[-2]:+.1f} %p")
    r27 = [data[k].get("r27") for k in cols]
    if any(r27):
        def rush(r):
            return "-" if not r or r["rush_bunker_top1"] is None else f"{r['rush_bunker_top1'] * 100:.1f}"

        def macro(r):
            return "-" if not r or r["macro_bunker_p_before_4m"] is None else f"{r['macro_bunker_p_before_4m']:.4f}"
        out.append("Build_Bunker top-1, rush-opponent games (041-R27) | " + " | ".join(rush(r) for r in r27))
        out.append("rush games / bunker steps | " + " | ".join("-" if not r else f"{r['rush_games']} / {r['rush_bunker_steps']}" for r in r27))
        out.append("macro mean p(Build_Bunker) < 4:00 (record) | " + " | ".join(macro(r) for r in r27))
        out.append("macro games / steps < 4:00 | " + " | ".join("-" if not r else f"{r['macro_games']} / {r['macro_steps_before_4m']}" for r in r27))
        rb = [r["rush_bunker_top1"] * 100 for r in r27 if r and r["rush_bunker_top1"] is not None]
        if len(rb) >= 2:
            out.append(f"041-R27 rush bunker top-1 per column {[round(b, 1) for b in rb]}, last change {rb[-1] - rb[-2]:+.1f} %p")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("columns", nargs="+", help="<label>=<per_action json>")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    data = {}
    for c in a.columns:
        label, sep, path = c.partition("=")
        if not sep:
            raise SystemExit(f"column {c!r}: expected <label>=<json>")
        data[label] = load(Path(path))
    print("\n".join(table(data)))
    if a.json:
        a.json.write_text(json.dumps(data, indent=1), encoding="utf-8")
        print("wrote", a.json)


if __name__ == "__main__":
    main()
