"""032-R20 ③: the league table of the architecture comparison from `python -m uranium run` outputs.

    python tools/ablation/league_table.py --run <out dir> <plan.toml> [--run ...]... [--out table.md]

Each `--run` is an output folder of model-vs-model games (games.jsonl, results.jsonl) with the
plan it played; several runs are joined (the league and the second core ablation's head-to-head
for the LSTM-GTrXL pair). The winner is read from the winning session's policy id, so the
in-game slot (not always the plan's p0/p1) does not matter; a policy id is named by the run folder
of its plan checkpoint (games.jsonl `opponent` is p1's id, the game's other session is p0).
Games are counted as the runner's summary.md counts them (`load`); the others are listed by
reason. Per pair: wins, losses, draws, win rate over all counted games with a Wilson 95% interval
(as reports/m6_core_ablation.md 4) and the decisive games' share. Then a score matrix (win 1,
draw 0.5) and a Bradley-Terry ranking with draws as half wins and a bootstrap 95% interval over
games.
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


# As the runner's own summary (crates/uranium-runner/src/report.rs, 020-R13).
SHORT_FRAMES = 300


def load(out: Path, plan_path: Path) -> tuple[list[tuple[str, str, int | None]], dict]:
    """(player a, player b, 0 if a won / 1 if b won / None for a draw) per counted game, and the
    games not counted by reason. A game counts when a session reported a real end (no error, not
    a disconnect), whatever hostd's outcome (a result written just before the wall-time limit
    counts, as in reports/m6_core_ablation.md 4); the deciding session is the one that ended last
    (020-R13). Not counted: no such session (e.g. cut at the wall-time limit), a mutual leave
    (both claim the win, a drop) and games under 300 frames (suspect), the rules of the runner's
    summary.md."""
    plan = tomllib.loads(plan_path.read_text(encoding="utf-8"))
    sides = [(name_of(m["p0"]), name_of(m["p1"])) for m in plan["matchup"]]
    kinds = [(m["p0"].get("kind"), m["p1"].get("kind")) for m in plan["matchup"]]
    started = read_jsonl(out / "games.jsonl")
    sessions: dict[int, list] = defaultdict(list)
    for r in read_jsonl(out / "results.jsonl"):
        sessions[r["game_id"]].append(r)
    # Model policy id -> run name: p1's id is in games.jsonl, the game's other model session is p0.
    names: dict[str, str] = {}

    def name(policy: str, run: str) -> None:
        if names.setdefault(policy, run) != run:
            raise SystemExit(f"{out}: policy {policy} plays as {names[policy]} and {run}")

    for g in started:
        (p0, p1), (k0, k1) = sides[g["matchup"]], kinds[g["matchup"]]
        if k1 == "model":
            name(g["opponent"], p1)
        for s in sessions.get(g["game_id"], []):
            if k0 == "model" and str(s.get("policy")).startswith("model:") and s["policy"] != g["opponent"]:
                name(s["policy"], p0)
    games, skipped = [], defaultdict(int)
    for g in started:
        ss = sessions.get(g["game_id"], [])
        usable = [s for s in ss if s.get("error") is None and s.get("reason") != "Disconnected"]
        if not usable:
            skipped[f"no result (outcome {g['outcome']})"] += 1
            continue
        if len(ss) == 2 and all(s.get("reason") == "Surrender" and s.get("winner") is not None and s["winner"] == s["slot"] for s in ss):
            skipped["mutual leave (dropped)"] += 1
            continue
        decide = usable[0]
        for s in usable[1:]:
            if (s.get("end_frame") or 0) > (decide.get("end_frame") or 0):
                decide = s
        if (decide.get("end_frame") or 0) < SHORT_FRAMES:
            skipped[f"under {SHORT_FRAMES} frames (suspect)"] += 1
            continue
        a, b = sides[g["matchup"]]
        w = decide.get("winner")
        if w is None:
            games.append((a, b, None))
            continue
        winner = next((s.get("policy") for s in ss if s["slot"] == w), None)
        non_model = [x for x, k in zip((a, b), kinds[g["matchup"]]) if k != "model"]
        if winner in names:
            winner_run = names[winner]
        elif winner is None or not str(winner).startswith("model:"):
            # No session for the winning slot, or a scripted player's: the winner is the side
            # whose session is not a known model loser.
            loser = next((names[s["policy"]] for s in ss if s["slot"] != w and s.get("policy") in names), None)
            if loser is not None:
                winner_run = b if loser == a else a
            elif len(non_model) == 1:
                winner_run = non_model[0]
            else:
                raise SystemExit(f"{out}: game {g['game_id']}: cannot tell the winner's side")
        else:
            raise SystemExit(f"{out}: game {g['game_id']}: policy {winner} has no run name")
        games.append((a, b, 0 if winner_run == a else 1))
    return games, dict(skipped)


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
    games, skipped = [], defaultdict(int)
    for out, plan in a.run:
        g, s = load(Path(out), Path(plan))
        games += g
        for k, v in s.items():
            skipped[k] += v
    players = sorted({x for g in games for x in g[:2]})
    pair = defaultdict(lambda: [0, 0, 0])  # (a, b) with a < b: a wins, b wins, draws
    for p0, p1, w in games:
        a_, b_ = sorted((p0, p1))
        rec = pair[(a_, b_)]
        if w is None:
            rec[2] += 1
        else:
            rec[0 if (p0 if w == 0 else p1) == a_ else 1] += 1
    left_out = ", ".join(f"{k} {v}" for k, v in sorted(skipped.items())) or "none"
    lines = [f"games counted: {len(games)}; not counted: {left_out}", ""]
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
