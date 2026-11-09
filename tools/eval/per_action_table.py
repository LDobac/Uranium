"""Per-action teacher hold-out table (041-R21, protocol metric `per_action`): one column per per_action.py
JSON, top-1 of the key actions (tile accuracy for positional actions, target accuracy for Attack_Unit),
the overall step-weighted top-1 and the epoch-to-epoch change of Build_Bunker top-1 (the bc-mix extension
rule input). Moved from runs/mix/table.py with the same cells.

    python tools/eval/per_action_table.py BC=<json> mix-e1=<json> ... [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = ["Build_Bunker", "Move_Pos", "Train_Marine", "Attack_Unit", "Repair_Unit", "NoOp", "Train_SCV", "Attack_Pos"]


def load(p: Path) -> dict:
    d = json.loads(p.read_text(encoding="utf-8"))
    by = {r["action"]: r for r in d["per_action"]}
    tot = sum(r["n"] for r in d["per_action"])
    by["_overall"] = sum(r["top1"] * r["n"] for r in d["per_action"]) / tot
    by["_steps"], by["_games"] = d.get("steps"), d.get("games")
    return by


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
        out.append(f"{a} | " + " | ".join(cell(data[k].get(a), a) for k in cols))
    out.append("overall top-1 | " + " | ".join(f"{data[k]['_overall'] * 100:.1f}" for k in cols))
    out.append("steps / games | " + " | ".join(f"{data[k]['_steps']} / {data[k]['_games']}" for k in cols))
    bk = [data[k]["Build_Bunker"]["top1"] * 100 for k in cols if "Build_Bunker" in data[k]]
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
        label, _, path = c.partition("=")
        data[label] = load(Path(path))
    print("\n".join(table(data)))
    if a.json:
        a.json.write_text(json.dumps({k: {r: v for r, v in d.items()} for k, d in data.items()}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
