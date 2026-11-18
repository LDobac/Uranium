"""Writes a 054 exploration game plan: F stochastic against T0 and T1 bots, three explore points per game, replay, memory and act logs on"""

import argparse
import tomllib
from fractions import Fraction
from pathlib import Path

from uranium.measure import rush_check_plan as rcp

ROOT = Path(__file__).resolve().parents[3]
MAPS = rcp.MAPS
MAX_FRAMES = 14400
RACES = (("zerg", 50), ("protoss", 30), ("terran", 20))
T0_BUILDS = {"zerg": (None, "zerg_rush"), "protoss": (None,), "terran": (None,)}
T1_BOTS = {
    "zerg": ("ualbertabot", "steamhammer", "mcrave", "zzzkbot"),
    "protoss": ("ualbertabot", "steamhammer", "mcrave", "stardust"),
    "terran": ("ualbertabot", "steamhammer", "mcrave", "iron"),
}
EXPLORE = (("points", "3"), ("window_s", "[120, 540]"), ("gap_s", "60"), ("k", "4"), ("epsilon", "0.5"), ("min_open", "2"))
EXPLORE_SEED_KEY = rcp.EXPLORE_SEED_KEY
EXPLORE_BLOCK = rcp.EXPLORE_BLOCK
PILOT_GAMES = 1000


def t0_side(race: str, build: str | None) -> str:
    return f'{{ kind = "t0", race = "{race}"' + (f', build = "{build}"' if build else "") + " }"


def t1_side(race: str, bot: str) -> str:
    return f'{{ kind = "oss", name = "{bot}", race = "{race}" }}'


def leaves() -> list[tuple[str, str, str, Fraction]]:
    total = sum(w for _, w in RACES)
    out = []
    for race, w in RACES:
        share = Fraction(w, total) / 2
        for kind, sides in (("t0", [t0_side(race, b) for b in T0_BUILDS[race]]), ("t1", [t1_side(race, b) for b in T1_BOTS[race]])):
            for side in sides:
                for m in MAPS:
                    out.append((race, kind, side, m, share / len(sides) / len(MAPS)))
    assert sum(x[-1] for x in out) == 1
    return out


def split(n: int, weights: list) -> list[int]:
    total = sum(weights)
    ideal = [Fraction(n) * w / total for w in weights]
    base = [int(x) for x in ideal]
    for i in sorted(range(len(weights)), key=lambda i: (-(ideal[i] - base[i]), i))[:n - sum(base)]:
        base[i] += 1
    return base


def blocks(games: int) -> list[tuple[str, str, str, str, int]]:
    assert games > 0, "--games is a game count"
    out, per_map = [], dict.fromkeys(MAPS, 0)
    for (race, _), n_race in zip(RACES, split(games, [w for _, w in RACES])):
        kinds = (("t0", [t0_side(race, b) for b in T0_BUILDS[race]]), ("t1", [t1_side(race, b) for b in T1_BOTS[race]]))
        for (kind, sides), n_kind in zip(kinds, split(n_race, [1, 1])):
            for side, n in zip(sides, split(n_kind, [1] * len(sides))):
                half = dict.fromkeys(MAPS, n // len(MAPS))
                if n % len(MAPS):
                    half[min(MAPS, key=lambda m: (per_map[m], MAPS.index(m)))] += 1
                for m in MAPS:
                    per_map[m] += half[m]
                    if half[m]:
                        out.append((race, kind, side, m, half[m]))
    assert sum(b[-1] for b in out) == games
    return out


def explore_seeds(first: int, games: int) -> list[int]:
    assert first >= EXPLORE_BLOCK[0] and first + games - 1 <= EXPLORE_BLOCK[-1], \
        f"--first and --games must stay in the explore seed block {EXPLORE_BLOCK[0]}..{EXPLORE_BLOCK[-1]}"
    return [rcp.splitmix64(EXPLORE_SEED_KEY + k) >> 32 for k in range(first, first + games)]


def render(checkpoint: str, name: str, games: int, first: int, zero_last_failed: bool, surrender: str) -> tuple[str, list[int]]:
    bs = blocks(games)
    seeds = explore_seeds(first, games)
    assert len(set(seeds)) == games, "explore seeds repeat"
    p0 = f'{{ kind = "model", checkpoint = "{checkpoint}", sample = "stochastic" }}'
    lines = [
        f'name = "{name}"',
        "save_replay = true",
        "memory_log = true",
        "act_log = true",
        "wall_timeout_s = 7200",
        f"zero_last_failed = {'true' if zero_last_failed else 'false'}",
        f'surrender = "{surrender}"',
        f"max_frames = {MAX_FRAMES}",
        "seeds = [",
    ]
    k = 0
    for *_, n in bs:
        block = seeds[k:k + n]
        for i in range(0, n, 8):
            lines.append("  " + ", ".join(str(s) for s in block[i:i + 8]) + ",")
        k += n
    assert k == games
    lines.append("]")
    lines.append("[explore]")
    lines += [f"{key} = {v}" for key, v in EXPLORE]
    for _, _, side, m, n in bs:
        lines += ["[[matchup]]", f'map = "{m}"', f"games = {n}", f"p0 = {p0}", f"p1 = {side}"]
    return "\n".join(lines) + "\n", seeds


def write(out_dir: Path, checkpoint: str, name: str, games: int, first: int, zero_last_failed: bool, surrender: str, out: Path | None = None) -> Path:
    own = out_dir / f"{name.replace('-', '_')}.toml"
    out = out or own
    text, seeds = render(checkpoint, name, games, first, zero_last_failed, surrender)
    taken = rcp.taken_seeds(out_dir, (out, own)) | rcp.reserved_seeds(skip_explore=True)
    assert not taken & set(seeds), "a seed of this plan is another plan's (botgen, check, dagger, distill, 022 teacher, protocol or a reserved block)"
    for f in {out.resolve(), own.resolve()}:
        if f.exists():
            assert rcp.plan_seeds(f) == set(seeds) and tomllib.loads(f.read_text(encoding="utf-8")).get("seeds") == seeds, \
                f"{f} exists with other seeds: refusing to overwrite it (games of it may be played); use another --first / --out"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(text.encode("utf-8"))
    by = {}
    for race, kind, _, m, n in blocks(games):
        by[(race, kind)] = by.get((race, kind), 0) + n
    parts = ", ".join(f"{r} {k} {n}" for (r, k), n in by.items())
    print(f"{out}: {games} games (seed block {first}..{first + games - 1} of {EXPLORE_BLOCK[-1]}), {parts}, {len(taken)} other seeds disjoint")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="the F checkpoint (054-R1; relative = under URANIUM_SHARED_DIR)")
    ap.add_argument("--zero-last-failed", required=True, choices=("true", "false"), help="F's 050 evaluation setting (054-R1)")
    ap.add_argument("--surrender", required=True, choices=("off", "shadow", "on"), help="F's 050 evaluation setting (054-R1)")
    ap.add_argument("--games", type=int, default=PILOT_GAMES, help="game count (054-R6 pilot 1,000)")
    ap.add_argument("--first", type=int, default=1, help=f"first game number in the explore seed block {EXPLORE_BLOCK[0]}..{EXPLORE_BLOCK[-1]} (a later plan starts after the earlier ones)")
    ap.add_argument("--name", default="explore-pilot", help="plan name (the evaluation stage name)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "config" / "plans", help="output folder, also scanned for seed overlap")
    a = ap.parse_args(argv)
    write(a.out_dir, a.checkpoint, a.name, a.games, a.first, a.zero_last_failed == "true", a.surrender, a.out)
