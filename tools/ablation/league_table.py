"""032-R20 ③: the league table of the architecture comparison from `python -m uranium run` outputs.

    python tools/ablation/league_table.py --run <out dir> <plan.toml> [--run ...]... [--out table.md]

Each `--run` is an output folder of model-vs-model games (games.jsonl, results.jsonl) with the
plan it played; several runs are joined (the league and the second core ablation's head-to-head
for the LSTM-GTrXL pair). Each game's two model sessions give their policy id and whether they
won, so the in-game slot (not always the plan's p0/p1) does not matter. A policy id is named by
the run folder of the plan checkpoint it plays as p1 (games.jsonl `opponent`; with the slot swap
every model is p1 somewhere). Per pair: wins, losses, draws, win rate over all
games with a Wilson 95% interval (as reports/m6_core_ablation.md 4), the decisive games' share, and
games without a result (wall-time limit). Then a score matrix (win 1, draw 0.5) and a Bradley-Terry
ranking with draws as half wins and a bootstrap 95% interval over games.
"""

import argparse
import json
import math
import random
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

Z = 1.959964


def wilson(k: float, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def name_of(player: dict) -> str:
    return Path(player["checkpoint"]).parent.name if player.get("kind") == "model" else player.get("kind", "?")


def load(out: Path, plan_path: Path) -> tuple[list[tuple[str, str, int | None]], int]:
    """(player a, player b, 0 if a won / 1 if b won / None for a draw) per game with both
    sessions' results, and the number of games without."""
    plan = tomllib.loads(plan_path.read_text(encoding="utf-8"))
    p1_of = [name_of(m["p1"]) for m in plan["matchup"]]
    started = read_jsonl(out / "games.jsonl")
    names: dict[str, str] = {}
    for g in started:
        n = names.setdefault(g["opponent"], p1_of[g["matchup"]])
        if n != p1_of[g["matchup"]]:
            raise SystemExit(f"{out}: policy {g['opponent']} plays as {n} and {p1_of[g['matchup']]}")
    run_name = lambda s: names[s["policy"]]  # noqa: E731
    started = {g["game_id"] for g in started}
    sessions: dict[int, list] = defaultdict(list)
    for r in read_jsonl(out / "results.jsonl"):
        sessions[r["game_id"]].append(r)
    games, missing = [], 0
    for gid in sorted(started | set(sessions)):
        ss = sessions.get(gid, [])
        if len(ss) != 2:
            missing += 1
            continue
        a, b = ss
        if a["won"] is None and b["won"] is None:
            w = None
        elif a["won"] != b["won"] and None not in (a["won"], b["won"]):
            w = 0 if a["won"] else 1
        else:
            raise SystemExit(f"{out}: game {gid}: the two sessions disagree ({a['won']}, {b['won']})")
        games.append((run_name(a), run_name(b), w))
    return games, missing


def bradley_terry(games, players, iters=500) -> dict[str, float]:
    """Strengths by the MM algorithm, draws as half a win to each side; log-strengths centred at 0
    and shown on the Elo scale (400 log10)."""
    wins = defaultdict(float)
    n = defaultdict(int)
    for a, b, w in games:
        n[(a, b)] += 1
        n[(b, a)] += 1
        if w is None:
            wins[a] += 0.5
            wins[b] += 0.5
        else:
            wins[a if w == 0 else b] += 1.0
    p = {x: 1.0 for x in players}
    for _ in range(iters):
        new = {}
        for i in players:
            den = sum(n[(i, j)] / (p[i] + p[j]) for j in players if j != i and n[(i, j)])
            new[i] = max(wins[i], 1e-3) / den if den else p[i]
        g = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        p = {k: v / g for k, v in new.items()}
    return {k: 400 * math.log10(v) for k, v in p.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, action="append", metavar=("OUT", "PLAN"), required=True)
    ap.add_argument("--out", help="write the markdown here (default stdout)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    games, missing = [], 0
    for out, plan in a.run:
        g, m = load(Path(out), Path(plan))
        games += g
        missing += m
    players = sorted({x for g in games for x in g[:2]})
    pair = defaultdict(lambda: [0, 0, 0])  # (a, b) with a < b: a wins, b wins, draws
    for p0, p1, w in games:
        a_, b_ = sorted((p0, p1))
        rec = pair[(a_, b_)]
        if w is None:
            rec[2] += 1
        else:
            rec[0 if (p0 if w == 0 else p1) == a_ else 1] += 1
    lines = [f"games with a result: {len(games)}, without (wall-time limit or no session result): {missing}", ""]
    lines += ["| A | B | A wins | B wins | draws | A win rate (Wilson 95%) | B win rate | A share of decisive |", "|---|---|---|---|---|---|---|---|"]
    for (x, y), (wa, wb, d) in sorted(pair.items()):
        n = wa + wb + d
        la, ha = wilson(wa, n)
        lb, hb = wilson(wb, n)
        ld, hd = wilson(wa, wa + wb)
        dec = f"{wa / (wa + wb):.0%} [{ld:.0%}, {hd:.0%}]" if wa + wb else "-"
        lines.append(f"| {x} | {y} | {wa} | {wb} | {d} | {wa / n:.0%} [{la:.0%}, {ha:.0%}] | {wb / n:.0%} [{lb:.0%}, {hb:.0%}] | {dec} |")
    lines += ["", "Score of the row against the column (win 1, draw 0.5):", "", "| | " + " | ".join(players) + " |", "|---" * (len(players) + 1) + "|"]
    for r in players:
        cells = []
        for c in players:
            if r == c:
                cells.append("-")
                continue
            x, y = sorted((r, c))
            wa, wb, d = pair.get((x, y), (0, 0, 0))
            n = wa + wb + d
            mine = wa if r == x else wb
            cells.append(f"{(mine + 0.5 * d) / n:.2f}" if n else "")
        lines.append(f"| {r} | " + " | ".join(cells) + " |")
    bt = bradley_terry(games, players)
    rng = random.Random(20261025)
    by_pair = defaultdict(list)
    for g in games:
        by_pair[tuple(sorted(g[:2]))].append(g)
    samples = defaultdict(list)
    for _ in range(a.bootstrap):
        resampled = [rng.choice(gs) for gs in by_pair.values() for _ in gs]
        for k, v in bradley_terry(resampled, players, 200).items():
            samples[k].append(v)
    lines += ["", "Bradley-Terry (Elo scale, draws half a win, mean 0; bootstrap 95% over games within each pair):", "", "| rank | model | rating | 95% |", "|---|---|---|---|"]
    for i, k in enumerate(sorted(players, key=lambda x: -bt[x]), 1):
        s = sorted(samples[k])
        lo, hi = s[int(0.025 * len(s))], s[min(len(s) - 1, int(0.975 * len(s)))]
        lines.append(f"| {i} | {k} | {bt[k]:+.0f} | [{lo:+.0f}, {hi:+.0f}] |")
    text = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
