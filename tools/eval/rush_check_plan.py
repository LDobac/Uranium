"""050-R6 (AMB-042-19): writes the condition-controlled rush check plan, by default the tracked
config/plans/rush5_check.toml. T0 `zerg_rush` and `ualbertabot12poollings` x Fighting Spirit /
Circuit Breaker x 32 games = 128 games; the seeds are the 020-R21 generator's 050-R6 list
(`uranium._core.rush_check_seeds`: blocks in matchup order, candidates from 45,001, per block 4
games of each of the 4 first-target-hit slot pairs and 2 of each of the 8 miss pairs, 020-R20
`seeds`). Every checkpoint is checked with the same list: only the model changes.

    python tools/eval/rush_check_plan.py                    # rewrites config/plans/rush5_check.toml
    python tools/eval/rush_check_plan.py --checkpoint checkpoints/rush-s5/ckpt-N/agents/main \\
        --name rush-s5-c1 --out <job>/plans/01-rush.toml   # the plan of one check (same seeds)
    python tools/eval/rush_check_plan.py --teacher          # rewrites config/plans/teacher_check.toml

`--teacher` (050-R8, AMB-021-4): the same 128 games and seeds with the 021-R8 teacher
`{ kind = "t0", race = "terran", build = "terran_react" }` as players[0] instead of a model (no GPU,
`uranium-env run`); the condition of a game is the same 020-R21 generator's.

`--botgen` (040-R14 / 050-R8, AMB-021-4, user delegation 2026-11-06): rewrites the two tracked
teacher botgen plans config/plans/teacher_botgen_rush.toml (4 rush opponents x 2 maps x 200 =
1,600) and teacher_botgen_macro.toml (5 macro opponents x 2 maps x 40 = 400): the teacher against
each opponent, 10 min, replays kept. Seeds (020-R20 `seeds`): a hash of the plan game number
(splitmix64 of 60,000 + n, rush n = 1..1,600 then macro n = 1,601..2,000), so the start slots
follow the natural distribution (neighbouring `seed_base + n` seeds share start slots, 020-R20)
and no seed is a check plan's (asserted).

    python tools/eval/rush_check_plan.py --botgen [--out-dir config/plans]

`--botgen --first-game G --name <plan name>` (user delegation 2026-11-07, 9th advisory
reports/m6_rl_stall_advice9.md section 2 item 4): writes one more macro-opponent plan (the same 5 x 2
maps x 40 = 400 games) whose botgen game numbers are G..G+399 (G > 2,000, after the two plans
above), e.g. `--first-game 2001 --name teacher-botgen-macro2` -> config/plans/teacher_botgen_macro2.toml.
Asserted: no seed repeats, and none is a seed of any other plan in --out-dir (the two botgen plans,
the check plans' `seeds` lists and `seed_base + 1..games` ranges).

    python tools/eval/rush_check_plan.py --botgen --first-game 2001 --name teacher-botgen-macro2

`--dagger --round R [--half] [--checkpoint C]` (041-R22, user delegation 2026-11-07, 10th advisory
reports/m6_rl_stall_advice10.md section 2 item 2): writes config/plans/dagger_r<R>.toml, the model
(default checkpoints/bc-mix-e10/ckpt-1053, stochastic) against the 4 botgen rush opponents x 2 maps
x 40 and the 5 macro opponents x 2 maps x 28 = 600 games, 10 min, replays kept (input of the dagger
relabel). `--half` (11th advisory reports/m6_rl_stall_advice11.md section 2, the round-2 plan):
rush x 20 and macro x 14 = 300 games. Seeds: splitmix64(70,000 + n) >> 32 with a round stride of
600 in both modes, n = (R - 1) * 600 + 1..(R - 1) * 600 + games (full R * 600, half
(R - 1) * 600 + 300), so a half round uses the first half of its full range; asserted disjoint from
the botgen and check seeds and every other plan in --out-dir.

    python tools/eval/rush_check_plan.py --dagger --round 1
    python tools/eval/rush_check_plan.py --dagger --round 2 --half --checkpoint checkpoints/bc-dagger-r1/ckpt-N
"""

import argparse
import tomllib
from pathlib import Path

from uranium import _core

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "config" / "plans" / "rush5_check.toml"
TEACHER_OUT = ROOT / "config" / "plans" / "teacher_check.toml"
# 050-R8: the 021-R8 teacher as players[0].
TEACHER = '{ kind = "t0", race = "terran", build = "terran_react" }'
# 050-R8 (7th advisory F): teacher games stop at 10:00 (the data is cut at 8:00).
TEACHER_MAX_FRAMES = 14400
# 042 rush-s5 c0: the AC1 checkpoint is one of the two c0 models (the other is rush-s4 at 3.0M).
CHECKPOINT = "checkpoints/ac1-vtrace-s1/ckpt-3133569/agents/main"
# Block order of the 050-R6 seed selection (= matchup order).
OPPONENTS = (
    ("T0 zerg_rush", '{ kind = "t0", race = "zerg", build = "zerg_rush" }'),
    ("ualbertabot12poollings", '{ kind = "oss", name = "ualbertabot12poollings", race = "zerg" }'),
)
MAPS = ("fighting_spirit", "circuit_breaker")
GAMES = 32
FIRST_SEED = 45001

# 040-R14 / 050-R8 teacher botgen (AMB-021-4, user delegation 2026-11-06, 6th advisory section 2).
BOTGEN_RUSH = (
    ("T0 zerg_rush", '{ kind = "t0", race = "zerg", build = "zerg_rush" }'),
    ("ualbertabot12poollings", '{ kind = "oss", name = "ualbertabot12poollings", race = "zerg" }'),
    ("ualbertabot9pool10lings", '{ kind = "oss", name = "ualbertabot9pool10lings", race = "zerg" }'),
    ("ualbertabot9poolhydra", '{ kind = "oss", name = "ualbertabot9poolhydra", race = "zerg" }'),
)
BOTGEN_MACRO = (
    ("ualbertabot2hatchhydra", '{ kind = "oss", name = "ualbertabot2hatchhydra", race = "zerg" }'),
    ("ualbertabot3hatchmuta", '{ kind = "oss", name = "ualbertabot3hatchmuta", race = "zerg" }'),
    ("steamhammer2hatchmuta", '{ kind = "oss", name = "steamhammer2hatchmuta", race = "zerg" }'),
    ("T0 zerg_muta", '{ kind = "t0", race = "zerg", build = "zerg_muta" }'),
    ("T0 zerg standard", '{ kind = "t0", race = "zerg" }'),
)
BOTGEN_RUSH_GAMES = 200
BOTGEN_MACRO_GAMES = 40
# Hash key of the botgen seeds (far from the check seeds 45,001.., 46,001.., 47,001..).
BOTGEN_SEED_KEY = 60000
# seed_base of the teacher macro / T0 zerg check plans (their seeds are seed_base + n).
CHECK_SEED_RANGES = ((46000, 40), (47000, 10))


def splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def botgen_seeds(first: int, count: int) -> list[int]:
    """Seeds of botgen games first..first+count-1 (plan game numbers over both plans, from 1)."""
    return [splitmix64(BOTGEN_SEED_KEY + n) >> 32 for n in range(first, first + count)]


BOTGEN_HEADER = (
    "# seeds (020-R20): splitmix64(60,000 + n) >> 32 of the botgen game number n (rush plan 1..1,600,",
    "# macro plan 1,601..2,000): natural start slots, none of the check plans' seeds.",
)


def render_botgen(name: str, opponents, games: int, seeds: list[int], what: str,
                  delegation: str = "user delegation 2026-11-06", seed_note=BOTGEN_HEADER, args: str = "--botgen") -> str:
    lines = [
        f"# 040-R14 / 050-R8 / 021-R8 (AMB-021-4, {delegation}): teacher botgen, {what}.",
        f"# Generated by tools/eval/rush_check_plan.py {args} (do not edit by hand). The teacher",
        "# t0-terran-react players[0] against each opponent x Fighting Spirit / Circuit Breaker,",
        "# 10 min (max_frames 14,400; the data is cut at 8:00 by uranium-replay build --until-frame",
        "# 11520, 040-R21), replays kept; scripted players and OSS bots only (uranium-env run).",
        *seed_note,
        f'name = "{name}"',
        "save_replay = true",
        "wall_timeout_s = 7200",
        f"max_frames = {TEACHER_MAX_FRAMES}",
        "seeds = [",
    ]
    k = 0
    for opp, _ in opponents:
        for m in MAPS:
            block = seeds[k:k + games]
            lines.append(f"  # games {k + 1}..{k + games}: {opp}, {m}")
            for i in range(0, games, 8):
                lines.append("  " + ", ".join(str(s) for s in block[i:i + 8]) + ",")
            k += games
    assert k == len(seeds)
    lines.append("]")
    for _, p1 in opponents:
        for m in MAPS:
            lines += ["[[matchup]]", f'map = "{m}"', f"games = {games}", f"p0 = {TEACHER}", f"p1 = {p1}"]
    return "\n".join(lines) + "\n"


def write_botgen(out_dir: Path) -> None:
    n_rush = len(BOTGEN_RUSH) * len(MAPS) * BOTGEN_RUSH_GAMES
    n_macro = len(BOTGEN_MACRO) * len(MAPS) * BOTGEN_MACRO_GAMES
    rush = botgen_seeds(1, n_rush)
    macro = botgen_seeds(n_rush + 1, n_macro)
    taken = set(_core.rush_check_seeds(len(OPPONENTS) * len(MAPS), FIRST_SEED))
    for base, count in CHECK_SEED_RANGES:
        taken.update(range(base + 1, base + count + 1))
    assert len(set(rush + macro)) == n_rush + n_macro, "botgen seeds repeat"
    assert not taken & set(rush + macro), "a botgen seed is a check plan seed"
    for name, opps, games, seeds, what in (
        ("teacher-botgen-rush", BOTGEN_RUSH, BOTGEN_RUSH_GAMES, rush, "rush opponents (4 x 2 maps x 200 = 1,600)"),
        ("teacher-botgen-macro", BOTGEN_MACRO, BOTGEN_MACRO_GAMES, macro, "macro opponents (5 x 2 maps x 40 = 400)"),
    ):
        out = out_dir / (name.replace("-", "_") + ".toml")
        out.write_bytes(render_botgen(name, opps, games, seeds, what).encode("utf-8"))
        hits = sum(_core.start_slots(s, 0)[2] for s in seeds)
        print(f"{out}: {len(seeds)} games, first-target hit {hits} ({hits / len(seeds):.1%}, natural 1/3)")


def plan_seeds(path: Path) -> set[int]:
    """Every seed a plan file uses: its `seeds` list, else `seed_base + 1..total games` (020-R20)."""
    p = tomllib.loads(path.read_text(encoding="utf-8"))
    if "seeds" in p:
        return set(p["seeds"])
    if "seed_base" in p:
        total = sum(m.get("games", 0) for m in p.get("matchup", []))
        return set(range(p["seed_base"] + 1, p["seed_base"] + total + 1))
    return set()


def write_botgen_extra(out_dir: Path, first: int, name: str) -> None:
    """One more macro-opponent botgen plan, botgen game numbers first..first+399 (9th advisory 2.4)."""
    n_prev = (len(BOTGEN_RUSH) * BOTGEN_RUSH_GAMES + len(BOTGEN_MACRO) * BOTGEN_MACRO_GAMES) * len(MAPS)
    assert first > n_prev, f"--first-game must be after the first two botgen plans (> {n_prev})"
    n = len(BOTGEN_MACRO) * len(MAPS) * BOTGEN_MACRO_GAMES
    seeds = botgen_seeds(first, n)
    out = out_dir / (name.replace("-", "_") + ".toml")
    assert len(set(seeds)) == n, "botgen seeds repeat"
    taken = set(botgen_seeds(1, n_prev)) | set(_core.rush_check_seeds(len(OPPONENTS) * len(MAPS), FIRST_SEED))
    for base, count in CHECK_SEED_RANGES:
        taken.update(range(base + 1, base + count + 1))
    for other in sorted(out_dir.glob("*.toml")):
        if other.resolve() != out.resolve():
            taken |= plan_seeds(other)
    assert not taken & set(seeds), "a seed of this plan is another plan's (botgen or check)"
    last = first + n - 1
    note = (
        f"# seeds (020-R20): splitmix64(60,000 + n) >> 32 of the botgen game number n = {first:,}..{last:,}",
        "# (after teacher_botgen_rush 1..1,600 and teacher_botgen_macro 1,601..2,000): natural start",
        "# slots, none of another plan's seeds (botgen or check, asserted by the generator).",
    )
    text = render_botgen(
        name, BOTGEN_MACRO, BOTGEN_MACRO_GAMES, seeds, f"macro opponents again (5 x 2 maps x 40 = {n})",
        delegation="user delegation 2026-11-07, 9th advisory 2.4", seed_note=note,
        args=f"--botgen --first-game {first} --name {name}",
    )
    out.write_bytes(text.encode("utf-8"))
    hits = sum(_core.start_slots(s, 0)[2] for s in seeds)
    print(f"{out}: {n} games, first-target hit {hits} ({hits / n:.1%}, natural 1/3), {len(taken)} other seeds disjoint")


# DAgger model games (user delegation 2026-11-07, 10th advisory reports/m6_rl_stall_advice10.md
# section 2 item 2): the latest model, stochastic, against the botgen opponents; the games only feed
# the dagger relabel (040 source `dagger`), so they are cut like the teacher games (10 min).
DAGGER_CHECKPOINT = "checkpoints/bc-mix-e10/ckpt-1053"
DAGGER_RUSH_GAMES = 40
DAGGER_MACRO_GAMES = 28
# --half (11th advisory reports/m6_rl_stall_advice11.md section 2): the round-2 plan, 300 games.
DAGGER_HALF_RUSH_GAMES = 20
DAGGER_HALF_MACRO_GAMES = 14
# Hash key of the DAgger seeds: game number n of round r is (r - 1) * DAGGER_ROUND_STRIDE + i,
# i = 1..games of the plan (600 full, 300 half); the stride stays 600 so rounds never overlap.
DAGGER_SEED_KEY = 70000
DAGGER_ROUND_STRIDE = 600


def write_dagger(out_dir: Path, rnd: int, checkpoint: str, out: Path | None, half: bool = False) -> None:
    """One DAgger round plan (041-R22): rush 4 x 2 maps x 40 + macro 5 x 2 maps x 28 = 600 model
    games, or with `half` rush x 20 + macro x 14 = 300 (11th advisory section 2)."""
    assert rnd >= 1, "--round starts at 1"
    if half:
        rush_games, macro_games = DAGGER_HALF_RUSH_GAMES, DAGGER_HALF_MACRO_GAMES
    else:
        rush_games, macro_games = DAGGER_RUSH_GAMES, DAGGER_MACRO_GAMES
    blocks = [(o, m, rush_games) for o in BOTGEN_RUSH for m in MAPS]
    blocks += [(o, m, macro_games) for o in BOTGEN_MACRO for m in MAPS]
    n = sum(g for _, _, g in blocks)
    assert n <= DAGGER_ROUND_STRIDE, "a round plan must fit in its seed stride"
    # The stride is fixed (not n): a half round 2 is n = 601..900, never round 1's 1..600.
    first = (rnd - 1) * DAGGER_ROUND_STRIDE + 1
    last = first + n - 1
    seeds = [splitmix64(DAGGER_SEED_KEY + k) >> 32 for k in range(first, last + 1)]
    name = f"dagger-r{rnd}"
    own = out_dir / f"dagger_r{rnd}.toml"
    out = out or own
    assert len(set(seeds)) == n, "dagger seeds repeat"
    n_prev = (len(BOTGEN_RUSH) * BOTGEN_RUSH_GAMES + len(BOTGEN_MACRO) * BOTGEN_MACRO_GAMES) * len(MAPS)
    # + 400: teacher_botgen_macro2 (botgen game numbers 2,001..2,400).
    taken = set(botgen_seeds(1, n_prev + 400)) | set(_core.rush_check_seeds(len(OPPONENTS) * len(MAPS), FIRST_SEED))
    for base, count in CHECK_SEED_RANGES:
        taken.update(range(base + 1, base + count + 1))
    for other in sorted(out_dir.glob("*.toml")):
        if other.resolve() not in (out.resolve(), own.resolve()):
            taken |= plan_seeds(other)
    assert not taken & set(seeds), "a seed of this plan is another plan's (botgen, check or dagger)"
    p0 = f'{{ kind = "model", checkpoint = "{checkpoint}", sample = "stochastic" }}'
    # Full mode keeps the header of the tracked config/plans/dagger_r1.toml byte for byte.
    flag, span = (" --half", "(r - 1) * 600 + 300, the first half of the round's 600") if half else ("", "r * 600")
    lines = [
        "# 040 source dagger (user delegation 2026-11-07, 10th advisory reports/m6_rl_stall_advice10.md",
        f"# section 2 item 2): DAgger round {rnd} model games, generated by tools/eval/rush_check_plan.py",
        f"# --dagger --round {rnd}{flag} (do not edit by hand). The model players[0] (stochastic) against the",
        f"# 4 botgen rush opponents x Fighting Spirit / Circuit Breaker x {rush_games} and the 5 macro opponents x 2",
        f"# maps x {macro_games} = {n} games, 10 min (max_frames 14,400; the relabel build cuts at 8:00), replays kept.",
        *(("# Half size (041-R22, 11th advisory reports/m6_rl_stall_advice11.md section 2: the round-2 plan).",) if half else ()),
        f"# seeds (020-R20): splitmix64(70,000 + n) >> 32, n = {first:,}..{last:,} (round r: n = (r - 1) * 600 + 1..{span}):",
        "# natural start slots, none of another plan's seeds (botgen, check, dagger; asserted by the generator).",
        f'name = "{name}"',
        "save_replay = true",
        "wall_timeout_s = 7200",
        "zero_last_failed = false",
        f"max_frames = {TEACHER_MAX_FRAMES}",
        "seeds = [",
    ]
    k = 0
    for (opp, _), m, games in blocks:
        block = seeds[k:k + games]
        lines.append(f"  # games {k + 1}..{k + games}: {opp}, {m}")
        for i in range(0, games, 8):
            lines.append("  " + ", ".join(str(s) for s in block[i:i + 8]) + ",")
        k += games
    lines.append("]")
    for (_, p1), m, games in blocks:
        lines += ["[[matchup]]", f'map = "{m}"', f"games = {games}", f"p0 = {p0}", f"p1 = {p1}"]
    out.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    hits = sum(_core.start_slots(s, 0)[2] for s in seeds)
    print(f"{out}: {n} games, first-target hit {hits} ({hits / n:.1%}, natural 1/3), {len(taken)} other seeds disjoint")


def render(checkpoint: str | None, name: str) -> str:
    """`checkpoint` None = the 021-R8 teacher plays players[0] (050-R8)."""
    blocks = len(OPPONENTS) * len(MAPS)
    seeds = _core.rush_check_seeds(blocks, FIRST_SEED)
    assert len(seeds) == blocks * GAMES
    if checkpoint is None:
        lines = [
            "# 050-R8 / 021-R8 (AMB-021-4): the scouting-reactive teacher on the 050-R6 condition-controlled",
            "# rush check, generated by tools/eval/rush_check_plan.py --teacher (do not edit by hand).",
            "# T0 zerg_rush and ualbertabot12poollings x Fighting Spirit / Circuit Breaker x 32 games, the",
            "# teacher t0-terran-react players[0], 10 min (max_frames 14,400: the teacher data is cut at",
            "# 8:00, 7th advisory F), replays kept; scripted players only (uranium-env run).",
            "# Held / 8:00 SCV from the teacher session's results.jsonl `scripted` row (050-R7 keys).",
            "# seeds (020-R20): the same list as config/plans/rush5_check.toml (020-R21 generator, 45,001 on).",
            f'name = "{name}"',
            "save_replay = true",
            "wall_timeout_s = 7200",
            f"max_frames = {TEACHER_MAX_FRAMES}",
            "seeds = [",
        ]
        p0 = TEACHER
    else:
        lines = [
            "# 050-R6 / 042 rush-s5 (AMB-042-19): condition-controlled rush check, generated by",
            "# tools/eval/rush_check_plan.py (do not edit by hand; a check changes only the model).",
            "# T0 zerg_rush and ualbertabot12poollings x Fighting Spirit / Circuit Breaker x 32 games, the",
            "# model players[0] (stochastic, real last_failed), 60 min, plan surrender default, replays kept;",
            "# no rush mini-game / concede / scouting reward / model-game cap (eval games).",
            "# seeds (020-R20): the 020-R21 generator's 050-R6 list, blocks in matchup order from 45,001;",
            "# per block 16 first-target-hit games (4 per hit pair) and 16 miss games (2 per miss pair).",
            f'name = "{name}"',
            "save_replay = true",
            "wall_timeout_s = 7200",
            "zero_last_failed = false",
            "seeds = [",
        ]
        p0 = f'{{ kind = "model", checkpoint = "{checkpoint}", sample = "stochastic" }}'
    k = 0
    for opp, _ in OPPONENTS:
        for m in MAPS:
            block = seeds[k:k + GAMES]
            lines.append(f"  # games {k + 1}..{k + GAMES}: {opp}, {m}")
            for i in range(0, GAMES, 8):
                lines.append("  " + ", ".join(str(s) for s in block[i:i + 8]) + ",")
            k += GAMES
    lines.append("]")
    for _, p1 in OPPONENTS:
        for m in MAPS:
            lines += [
                "[[matchup]]",
                f'map = "{m}"',
                f"games = {GAMES}",
                f"p0 = {p0}",
                f"p1 = {p1}",
            ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=None, help="model checkpoint (relative = under URANIUM_SHARED_DIR; default the AC1 one, --dagger bc-mix-e10)")
    ap.add_argument("--teacher", action="store_true", help="the 021-R8 teacher plays players[0] instead of a model (050-R8)")
    ap.add_argument("--name", default=None, help="plan name (the evaluation stage name; default rush5-check / teacher-check)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--botgen", action="store_true", help="write the two teacher botgen plans (040-R14 / 050-R8)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "config" / "plans", help="--botgen / --dagger output folder (and the seed disjointness scan)")
    ap.add_argument("--first-game", type=int, default=None,
                    help="--botgen: write one more macro plan from this botgen game number (> 2,000) instead")
    ap.add_argument("--dagger", action="store_true", help="write a DAgger round plan (10th advisory 2.2)")
    ap.add_argument("--round", type=int, default=1, help="--dagger: round number (seeds per round disjoint)")
    ap.add_argument("--half", action="store_true", help="--dagger: half-size round plan, 300 games (11th advisory 2)")
    a = ap.parse_args()
    assert a.dagger or not a.half, "--half needs --dagger"
    if a.dagger:
        write_dagger(a.out_dir, a.round, a.checkpoint or DAGGER_CHECKPOINT, a.out, a.half)
        raise SystemExit(0)
    if a.first_game is not None:
        assert a.botgen and a.name, "--first-game needs --botgen and --name"
        write_botgen_extra(a.out_dir, a.first_game, a.name)
        raise SystemExit(0)
    if a.botgen:
        write_botgen(a.out_dir)
        raise SystemExit(0)
    name = a.name or ("teacher-check" if a.teacher else "rush5-check")
    out = a.out or (TEACHER_OUT if a.teacher else OUT)
    out.write_bytes(render(None if a.teacher else (a.checkpoint or CHECKPOINT), name).encode("utf-8"))
    print(f"{out}: {len(OPPONENTS) * len(MAPS)} matchups x {GAMES} games")
