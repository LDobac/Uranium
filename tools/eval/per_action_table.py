"""Per-action teacher hold-out table (050-R9 protocol metric `per_action`, 041-R21): one column per
tools/eval/per_action.py JSON. Rows: top-1 of the key actions (with tile accuracy for positional actions and
target accuracy for Attack_Unit), the overall step-weighted top-1, steps / games, and the column-to-column
change of Build_Bunker top-1 (the bc-mix extension rule input). Moved from runs/mix/table.py with the same
cells.

    python tools/eval/per_action_table.py BC=<json> mix-e1=<json> ... [--json out.json]

--json writes {label: {"file", "steps", "games", "overall_top1", "actions": {action: per_action row}}}.
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
            "actions": {r["action"]: r for r in d["per_action"]}}


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
