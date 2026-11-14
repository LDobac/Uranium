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

`--distill --round R [--half] --checkpoint C` (044-R1, AMB-044-3 / AMB-044-4, user decision 2026-11-14):
writes config/plans/distill_m2_d<R>.toml, the self-game distillation games of round d<R>: the model C
(d1 = the 041-R29 ④ selection S, d2 = the d1 model; no default) stochastic against Terran and Protoss
opponents, T0 standard Protoss 2 maps x 120 + T0 standard Terran 2 maps x 60 + T1 Protoss (UAlbertaBot P,
Stardust) 2 bots x 2 maps x 30 + T1 Terran (UAlbertaBot T, Iron) 2 bots x 2 maps x 30 = 600 games (Protoss :
Terran = 3 : 2), races fixed (no Random), no E-Regression scenario builds, no McRave; `--half` (round d2)
halves every block = 300 games. 10 min, replays kept, decision logs always on (`memory_log`, 041-R28 ①;
`--memory-log` is accepted and changes nothing). Seeds: splitmix64(80,000 + n) >> 32 with a round stride of
600 as --dagger (n = (R - 1) * 600 + 1..), asserted disjoint from the botgen, check and dagger seeds and from
every plan under --out-dir including its subfolders (the 050-R9 protocol templates config/plans/protocol/,
in particular t0_tp's T0 standard Terran / Protoss 32 games: T0 is deterministic, the same seed is the same
game, 021-AC3 / 020-AC2).

    python tools/eval/rush_check_plan.py --distill --round 1 --checkpoint checkpoints/<S run>/ckpt-N
    python tools/eval/rush_check_plan.py --distill --round 2 --half --checkpoint checkpoints/m2-distill-d1/ckpt-N

`--distill --round R --extra N --part K --checkpoint C` (044-R5 ③ top-up, user decision 2026-11-14): when
`uranium-train mix-fractions --share` reports the round's distill steps below their share, N more games of
round d<R> with the round's model and the same 044-R1 composition (every block x N / 600; N a multiple of 20)
in config/plans/distill_m2_d<R>_x<K>.toml (plan name `distill-m2-d<R>-x<K>`). Seeds: game numbers
n = 100,000 + (R - 1) * 10,000 + (K - 1) * 1,000 + 1.. (at most 1,000 per part), asserted disjoint from every
round block n = 1..6,000 (generated or not) and, like a round plan, from every plan under --out-dir (the
round plans and earlier parts included). R is 1..10 and K 1..10, so every (R, K) block is its own; K is
required, a new one for each top-up, and an existing plan file is never overwritten with other seeds.

    python tools/eval/rush_check_plan.py --distill --round 1 --extra 120 --part 1 --checkpoint checkpoints/<S run>/ckpt-N

`--teacher-tp` (022-R11 / 022-AC2, AMB-022-1 Resolved 2026-11-14): rewrites config/plans/teacher_tp_check.toml, the
check of the 022 TvT / TvP teacher `{ kind = "t0", race = "terran", build = "terran_tp" }` (players[0]) against T0
standard Protoss and Terran x 2 maps x 32 (pass >= 44/64 each) and the T1 bots UAlbertaBot P, Stardust, UAlbertaBot T,
Iron x 2 maps x 10 (recorded only) = 208 games, 60 min, replays kept. `--botgen-tp` (022-R12 / 044-R8, AMB-022-7):
rewrites config/plans/teacher_botgen_tp.toml, the teacher's botgen games: T0 standard Protoss and Terran x 2 maps x
150 + the same four T1 bots x 2 maps x 25 = 800 games, 10 min (build --src botgen --until-frame 14400 --time-limit
14400). Both: races fixed (no Random, 022-R2), seeds splitmix64(1,000,000 + n) >> 32 (check n = 1..208, botgen
n = 1,001..1,800), asserted disjoint from each other, from every plan under --out-dir and its subfolders (050-R9
protocol templates included), from the botgen / check seeds, from the DAgger and distillation round blocks not
generated yet (rounds 1..10, distillation top-ups) and from the teacher's development seeds 77,001..77,264.

    python tools/eval/rush_check_plan.py --teacher-tp
    python tools/eval/rush_check_plan.py --botgen-tp

`--teacher-round --round R [--half] --checkpoint C` (044-R8 / 022-R12, AMB-022-7): from the round after the 022
teacher passes 022-AC2, writes config/plans/dagger_m2_t<R>.toml (plan name `dagger-m2-t<R>`), the round's TvT / TvP
self games that the 022 teacher's shadow relabels (`uranium-replay build --src dagger --relabel terran_tp --games
<dir>/games.jsonl`, shards dagger_m2/t<R>): the model C (the round's start model; no default) stochastic with the
044-R1 composition of a round (as --distill: 600 games, `--half` 300), races fixed (no Random), 10 min, replays kept,
decision logs always on (`memory_log`, 041-R28; `--memory-log` is accepted and changes nothing), the last failed
command shown (`zero_last_failed = false`, 041-R22 DAgger games). Seeds: splitmix64(2,000,000 + n) >> 32 with a round
stride of 600 (n = (R - 1) * 600 + 1..), asserted disjoint from every plan under --out-dir and its subfolders, the
botgen / check seeds, the DAgger / distillation round blocks and top-ups and the other teacher round blocks not
generated yet (rounds 1..10) and the 022 teacher's development seeds; the 022 plans keep clear of the teacher round
blocks too.

    python tools/eval/rush_check_plan.py --teacher-round --round 2 --checkpoint checkpoints/m2-distill-d1/ckpt-N
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
    """Every seed a plan file uses (020-R20, crates/uranium-runner plan.rs `seed_of`): its `seeds` list,
    else `seed_base` (default 1,000) with `same_seed`, else `seed_base + 1..total games`."""
    p = tomllib.loads(path.read_text(encoding="utf-8"))
    if "seeds" in p:
        return set(p["seeds"])
    base = p.get("seed_base", 1000)
    if p.get("same_seed", False):
        return {base}
    total = sum(m.get("games", 0) for m in p.get("matchup", []))
    return set(range(base + 1, base + total + 1))


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


def write_dagger(out_dir: Path, rnd: int, checkpoint: str, out: Path | None, half: bool = False, memory_log: bool = False) -> None:
    """One DAgger round plan (041-R22): rush 4 x 2 maps x 40 + macro 5 x 2 maps x 28 = 600 model
    games, or with `half` rush x 20 + macro x 14 = 300 (11th advisory section 2). `memory_log`:
    the model writes its decision logs for the 040-R22 `memory_action` relabel (041-R28, model 2
    path); off, the plan is the Checkpoint 1 path's byte for byte."""
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
        *(("# 041-R28: decision logs memory/g<id>.jsonl beside replays/ (uranium-replay build --memory-log).", "memory_log = true") if memory_log else ()),
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


# 044-R1 (AMB-044-3 Resolved 2026-11-14): self-game distillation opponents, (label, p1, games per map of a
# full round). Races fixed (no Random); T0 standard builds only (no E-Regression scenario builds); T1 bots as
# the 050-R9 T1 plans name them, without McRave (slow in lockstep, AMB-042-13).
DISTILL_OPPONENTS = (
    ("T0 protoss", '{ kind = "t0", race = "protoss" }', 120),
    ("T0 terran", '{ kind = "t0", race = "terran" }', 60),
    ("ualbertabot protoss", '{ kind = "oss", name = "ualbertabot", race = "protoss" }', 30),
    ("stardust", '{ kind = "oss", name = "stardust", race = "protoss" }', 30),
    ("ualbertabot terran", '{ kind = "oss", name = "ualbertabot", race = "terran" }', 30),
    ("iron", '{ kind = "oss", name = "iron", race = "terran" }', 30),
)
# Hash key of the distillation seeds (botgen 60,000.., dagger 70,000..): game number n of round d<R> is
# (R - 1) * DISTILL_ROUND_STRIDE + i, i = 1..games (600 full, 300 half), as the DAgger rounds.
DISTILL_SEED_KEY = 80000
DISTILL_ROUND_STRIDE = 600
# 044-R5 ③ top-up (user decision 2026-11-14): more games of round d<R> when its distill steps miss their 20%
# share. Part K of round R uses game numbers n = DISTILL_EXTRA_BASE + (R - 1) * DISTILL_EXTRA_ROUND
# + (K - 1) * DISTILL_EXTRA_PART + i, i = 1..N: far above every round block (n <= R * 600), one block per
# (round, part). N is a multiple of DISTILL_EXTRA_UNIT so every 044-R1 block keeps its ratio exactly (the
# smallest block, 30 of 600, is 1/20).
DISTILL_EXTRA_BASE = 100000
DISTILL_EXTRA_ROUND = 10000
DISTILL_EXTRA_PART = 1000
DISTILL_EXTRA_UNIT = 20
# Round game numbers 1 .. this x 600 are asserted disjoint from a top-up plan's seeds even before those
# rounds are generated (044-R1 has at most 2 rounds; 10 leaves room).
DISTILL_ROUNDS_RESERVED = 10
# Top-up parts per round: the parts of round R end before round R + 1's top-up block begins.
DISTILL_EXTRA_PARTS = DISTILL_EXTRA_ROUND // DISTILL_EXTRA_PART
# Every allowed (round, part) has its own block, and all of them lie above every round block:
# rounds 1..DISTILL_ROUNDS_RESERVED use n <= DISTILL_ROUNDS_RESERVED x 600 < DISTILL_EXTRA_BASE.
assert DISTILL_ROUNDS_RESERVED * DISTILL_ROUND_STRIDE < DISTILL_EXTRA_BASE


def distill_blocks(half: bool, extra: int = 0) -> list[tuple[str, str, str, int]]:
    """(opponent label, p1, map, games) in plan order: each opponent x Fighting Spirit / Circuit Breaker.
    `extra` > 0 (044-R5 ③ top-up): `extra` games in the 044-R1 ratio (each block x extra / 600)."""
    if extra:
        assert extra % DISTILL_EXTRA_UNIT == 0 and 0 < extra <= DISTILL_EXTRA_PART, \
            f"--extra must be a multiple of {DISTILL_EXTRA_UNIT} in {DISTILL_EXTRA_UNIT}..{DISTILL_EXTRA_PART} (the 044-R1 ratio exactly); round it up"
        return [(o, p1, m, g * extra // DISTILL_ROUND_STRIDE) for o, p1, g in DISTILL_OPPONENTS for m in MAPS]
    return [(o, p1, m, g // 2 if half else g) for o, p1, g in DISTILL_OPPONENTS for m in MAPS]


def distill_first(rnd: int, extra: int = 0, part: int = 1) -> int:
    """The first game number n of round d<rnd> (044-R1), or of its top-up part `part` (044-R5 ③)."""
    assert 1 <= rnd <= DISTILL_ROUNDS_RESERVED, f"--round must be in 1..{DISTILL_ROUNDS_RESERVED} (the reserved round seed blocks)"
    if extra:
        assert 1 <= part <= DISTILL_EXTRA_PARTS, f"--part must be in 1..{DISTILL_EXTRA_PARTS} (a round's top-up seed blocks)"
        return DISTILL_EXTRA_BASE + (rnd - 1) * DISTILL_EXTRA_ROUND + (part - 1) * DISTILL_EXTRA_PART + 1
    return (rnd - 1) * DISTILL_ROUND_STRIDE + 1


def distill_seeds(rnd: int, games: int, extra: int = 0, part: int = 1) -> tuple[int, list[int]]:
    """(first game number n, seeds) of round d<rnd> (044-R1) or of its top-up part (044-R5 ③)."""
    first = distill_first(rnd, extra, part)
    return first, [splitmix64(DISTILL_SEED_KEY + k) >> 32 for k in range(first, first + games)]


def taken_seeds(out_dir: Path, skip: tuple[Path, ...]) -> set[int]:
    """Seeds the distillation plan must not use (044-R1): the botgen game numbers 1..2,400 (rush, macro,
    macro2), the 050-R6 rush check list, the teacher check ranges, and every plan under `out_dir` and its
    subfolders (config/plans/protocol: the 050-R9 templates) except `skip`."""
    n_prev = (len(BOTGEN_RUSH) * BOTGEN_RUSH_GAMES + len(BOTGEN_MACRO) * BOTGEN_MACRO_GAMES) * len(MAPS)
    taken = set(botgen_seeds(1, n_prev + 400)) | set(_core.rush_check_seeds(len(OPPONENTS) * len(MAPS), FIRST_SEED))
    for base, count in CHECK_SEED_RANGES:
        taken.update(range(base + 1, base + count + 1))
    skip = tuple(p.resolve() for p in skip)
    for other in sorted(out_dir.rglob("*.toml")):
        if other.resolve() not in skip:
            taken |= plan_seeds(other)
    return taken


def render_distill(rnd: int, checkpoint: str, name: str, half: bool, extra: int = 0, part: int = 1) -> tuple[str, list[int]]:
    """The text and seeds of the round d<rnd> distillation plan (044-R1), or with `extra` > 0 its top-up part
    `part` (044-R5 ③, user decision 2026-11-14): `extra` more games of the same composition."""
    assert 1 <= rnd <= DISTILL_ROUNDS_RESERVED, f"--round must be in 1..{DISTILL_ROUNDS_RESERVED} (the reserved round seed blocks)"
    assert not (half and extra), "--half and --extra are separate plans"
    blocks = distill_blocks(half, extra)
    n = sum(g for *_, g in blocks)
    assert n <= (DISTILL_EXTRA_PART if extra else DISTILL_ROUND_STRIDE), "a plan must fit in its seed stride"
    assert not extra or n == extra
    first, seeds = distill_seeds(rnd, n, extra, part)
    last = first + n - 1
    assert len(set(seeds)) == n, "distill seeds repeat"
    if extra:
        # Every round block, generated or not yet, stays disjoint (by number, and asserted by seed value).
        rounds = {splitmix64(DISTILL_SEED_KEY + k) >> 32 for k in range(1, DISTILL_ROUNDS_RESERVED * DISTILL_ROUND_STRIDE + 1)}
        assert not rounds & set(seeds), "a top-up seed is a round seed"
    p0 = f'{{ kind = "model", checkpoint = "{checkpoint}", sample = "stochastic" }}'
    t0p, t0t = (sum(g for o, _, _, g in blocks if o == k) for k in ("T0 protoss", "T0 terran"))
    t1 = n - t0p - t0t
    span = "(r - 1) * 600 + 300, the first half of the round's 600" if half else "r * 600"
    if extra:
        what = [
            f"# 044-R5 ③ top-up (user decision 2026-11-14): {n} more self-game distillation games of round d{rnd} (part {part}),",
            f"# generated by tools/eval/rush_check_plan.py --distill --round {rnd} --extra {n} --part {part} (do not edit by hand):",
            "# the round's distill steps miss their 20% share, so more games of the same 044-R1 composition are played",
            "# with the round's model and built / labelled into the same round's distill_pre / distill_m2 folders.",
        ]
        seed_note = (f"# seeds (020-R20): splitmix64(80,000 + n) >> 32, n = {first:,}..{last:,} (top-up part k of round r:"
                     " n = 100,000 + (r - 1) * 10,000 + (k - 1) * 1,000 + 1..; no round seed n = 1..6,000):")
    else:
        what = [
            "# 044-R1 / 041-R31 (AMB-044-3 / AMB-044-4, user decision 2026-11-14): self-game distillation round",
            f"# d{rnd} games, generated by tools/eval/rush_check_plan.py --distill --round {rnd}{' --half' if half else ''} (do not edit by hand).",
        ]
        seed_note = f"# seeds (020-R20): splitmix64(80,000 + n) >> 32, n = {first:,}..{last:,} (round r: n = (r - 1) * 600 + 1..{span}):"
    lines = [
        *what,
        f"# The model players[0] (stochastic) against T0 standard Protoss ({t0p}) and Terran ({t0t}) and the T1",
        f"# Protoss (UAlbertaBot P, Stardust) and Terran (UAlbertaBot T, Iron) bots ({t1}) x Fighting Spirit / Circuit",
        f"# Breaker = {n} games{' (half size, round d2)' if half else ''}, races fixed (no Random), 10 min (max_frames 14,400; build",
        "# --src distill --until-frame 14400 --time-limit 14400, 044-R2), replays kept. T1 bots: the 050-R9 T1 rule",
        "# (NDEBUG builds, config/protocol/t1_bots.toml; opponent-crash games are invalid).",
        seed_note,
        "# natural start slots, none of another plan's seeds (botgen, check, dagger, 050-R9 protocol; asserted).",
        f'name = "{name}"',
        "save_replay = true",
        "# 041-R28 / 044-R2: decision logs memory/g<id>.jsonl beside replays/ (uranium-replay build --memory-log).",
        "memory_log = true",
        "wall_timeout_s = 7200",
        "zero_last_failed = false",
        f"max_frames = {TEACHER_MAX_FRAMES}",
        "seeds = [",
    ]
    k = 0
    for opp, _, m, games in blocks:
        block = seeds[k:k + games]
        lines.append(f"  # games {k + 1}..{k + games}: {opp}, {m}")
        for i in range(0, games, 8):
            lines.append("  " + ", ".join(str(s) for s in block[i:i + 8]) + ",")
        k += games
    lines.append("]")
    for _, p1, m, games in blocks:
        lines += ["[[matchup]]", f'map = "{m}"', f"games = {games}", f"p0 = {p0}", f"p1 = {p1}"]
    return "\n".join(lines) + "\n", seeds


def write_distill(out_dir: Path, rnd: int, checkpoint: str, out: Path | None, half: bool = False,
                  name: str | None = None, extra: int = 0, part: int = 1) -> Path:
    """One distillation round plan (044-R1): 600 games, or with `half` (round d2) 300; with `extra` (044-R5 ③
    top-up) its part `part`, `extra` more games: config/plans/distill_m2_d<R>_x<K>.toml."""
    stem = f"distill_m2_d{rnd}_x{part}" if extra else f"distill_m2_d{rnd}"
    own = out_dir / f"{stem}.toml"
    out = out or own
    text, seeds = render_distill(rnd, checkpoint, name or stem.replace("_", "-"), half, extra, part)
    taken = taken_seeds(out_dir, (out, own))
    assert not taken & set(seeds), "a seed of this plan is another plan's (botgen, check, dagger or protocol)"
    # Rewriting its own file is fine only with the same seeds: a top-up part rewritten with another N (or a
    # round with --half toggled) would replay seeds of games already played (044-R5 ③); use another --part.
    for f in {out.resolve(), own.resolve()}:
        if f.exists():
            assert plan_seeds(f) == set(seeds), f"{f} exists with other seeds: refusing to overwrite it (games of it may be played); use another --part / --out"
    out.write_bytes(text.encode("utf-8"))
    hits = sum(_core.start_slots(s, 0)[2] for s in seeds)
    print(f"{out}: {len(seeds)} games, first-target hit {hits} ({hits / len(seeds):.1%}, natural 1/3), {len(taken)} other seeds disjoint")
    return out


# 022 TvT / TvP teacher `terran_tp` (021-R9): the 022-R11 check plan and the 022-R12 botgen plan (AMB-022-1 /
# AMB-022-7 Resolved 2026-11-14). The teacher players[0]; opponents with fixed races (022-R2: no Random games in
# the data), T0 standard builds only, the T1 Terran / Protoss bots of 022-R11 / R12 (the 044-R1 four).
TP_TEACHER = '{ kind = "t0", race = "terran", build = "terran_tp" }'
TP_T0 = (
    ("T0 protoss", '{ kind = "t0", race = "protoss" }'),
    ("T0 terran", '{ kind = "t0", race = "terran" }'),
)
TP_T1 = (
    ("ualbertabot protoss", '{ kind = "oss", name = "ualbertabot", race = "protoss" }'),
    ("stardust", '{ kind = "oss", name = "stardust", race = "protoss" }'),
    ("ualbertabot terran", '{ kind = "oss", name = "ualbertabot", race = "terran" }'),
    ("iron", '{ kind = "oss", name = "iron", race = "terran" }'),
)
# 022-R11: T0 standard Protoss / Terran 2 maps x 32 (pass: >= 44 / 64 each), T1 4 bots x 2 maps x 10 (recorded).
TP_CHECK_T0_GAMES = 32
TP_CHECK_T1_GAMES = 10
# 022-AC2: 60-minute cap.
TP_CHECK_MAX_FRAMES = 86400
# 022-R12: T0 standard Protoss / Terran 2 maps x 150 (600) + T1 4 bots x 2 maps x 25 (200) = 800, 10-minute cap.
TP_BOTGEN_T0_GAMES = 150
TP_BOTGEN_T1_GAMES = 25
# Hash key of the 022 teacher seeds, far above every other key's game numbers (botgen 60,000 + n, dagger
# 70,000 + n, distill 80,000 + n with n <= 200,000): the check plan is game numbers 1..208, the botgen plan
# 1,001..1,800.
TP_SEED_KEY = 1_000_000
TP_CHECK_FIRST = 1
TP_BOTGEN_FIRST = 1001
# 022-R11: the teacher's development runs used seeds 77,001..77,264; the check plan must not tune on them.
TP_DEV_SEEDS = range(77001, 77265)
# Seed blocks of plans not generated yet that the 022 plans keep clear of: DAgger rounds 1..10 (70,000 + n,
# n <= 6,000), distillation rounds 1..10 (80,000 + n, n <= 6,000) and their top-up parts (n = 100,001..200,000).
RESERVED_ROUNDS = 10


def tp_blocks(check: bool) -> list[tuple[str, str, str, int]]:
    """(opponent label, p1, map, games) of the 022-R11 check plan (`check`) or the 022-R12 botgen plan, in plan
    order: T0 Protoss, T0 Terran, then the T1 bots, each x Fighting Spirit / Circuit Breaker."""
    t0, t1 = (TP_CHECK_T0_GAMES, TP_CHECK_T1_GAMES) if check else (TP_BOTGEN_T0_GAMES, TP_BOTGEN_T1_GAMES)
    return [(o, p1, m, g) for opps, g in ((TP_T0, t0), (TP_T1, t1)) for o, p1 in opps for m in MAPS]


def tp_seeds(check: bool) -> tuple[int, list[int]]:
    """(first game number, seeds) of the 022 check plan (`check`) or botgen plan: splitmix64(1,000,000 + n) >> 32."""
    first = TP_CHECK_FIRST if check else TP_BOTGEN_FIRST
    n = sum(g for *_, g in tp_blocks(check))
    return first, [splitmix64(TP_SEED_KEY + k) >> 32 for k in range(first, first + n)]


def reserved_seeds(skip_teacher_round: int | None = None) -> set[int]:
    """Seeds of plans other generators will write (DAgger and distillation rounds, distillation top-ups, the 044-R8
    teacher rounds) and the 022 teacher's development seeds: the 022 plans do not take them either.
    `skip_teacher_round`: leave out that teacher round's own block (its generator's own seeds)."""
    n = RESERVED_ROUNDS * DAGGER_ROUND_STRIDE
    s = {splitmix64(DAGGER_SEED_KEY + k) >> 32 for k in range(1, n + 1)}
    s |= {splitmix64(DISTILL_SEED_KEY + k) >> 32 for k in range(1, DISTILL_ROUNDS_RESERVED * DISTILL_ROUND_STRIDE + 1)}
    top = range(DISTILL_EXTRA_BASE + 1, DISTILL_EXTRA_BASE + DISTILL_ROUNDS_RESERVED * DISTILL_EXTRA_ROUND + 1)
    s |= {splitmix64(DISTILL_SEED_KEY + k) >> 32 for k in top}
    for r in range(1, RESERVED_ROUNDS + 1):
        if r != skip_teacher_round:
            first = teacher_round_first(r)
            s |= {splitmix64(TEACHER_ROUND_SEED_KEY + k) >> 32 for k in range(first, first + TEACHER_ROUND_STRIDE)}
    return s | set(TP_DEV_SEEDS)


def render_tp(check: bool, name: str) -> tuple[str, list[int]]:
    """The text and seeds of the 022-R11 teacher check plan (`check`) or the 022-R12 teacher botgen plan."""
    blocks = tp_blocks(check)
    first, seeds = tp_seeds(check)
    n = len(seeds)
    last = first + n - 1
    assert len(set(seeds)) == n, "022 teacher seeds repeat"
    other_first, other = tp_seeds(not check)
    assert not set(seeds) & set(other), "the 022 check and botgen plans share a seed"
    t0 = sum(g for o, *_, g in blocks if o.startswith("T0 "))
    if check:
        what = [
            "# 022-R11 / 022-AC2 (AMB-022-1 Resolved 2026-11-14): check of the TvT / TvP teacher t0-terran-tp, generated by",
            "# tools/eval/rush_check_plan.py --teacher-tp (do not edit by hand). The teacher players[0] against T0 standard",
            f"# Protoss and Terran x Fighting Spirit / Circuit Breaker x {TP_CHECK_T0_GAMES} ({t0}) and the T1 Terran / Protoss bots",
            f"# (UAlbertaBot P, Stardust, UAlbertaBot T, Iron) x 2 maps x {TP_CHECK_T1_GAMES} ({n - t0}) = {n} games, races fixed, 60 min, replays kept;",
            "# scripted players and OSS bots only (uranium-env run). Results: the T0 Terran session rows' scripted.tp (022-R10).",
            "# Pass (measured once on this plan; development runs use other seeds): T0 Protoss >= 44/64 and T0 Terran >= 44/64",
            "# wins, no crash. 022-AC4: refinery order <= 2:00 and factory order <= 3:30 in >= 95% of the games. T1 games are",
            "# recorded only: a bot whose win rate is below the BC baseline's T1 300 rate against the same bot is reported.",
            "# T1 bots: the 050-R9 T1 rule (NDEBUG builds, config/protocol/t1_bots.toml; opponent-crash games are invalid).",
        ]
        max_frames = TP_CHECK_MAX_FRAMES
    else:
        what = [
            "# 022-R12 / 044-R8 / 040-R14 (AMB-022-7 Resolved 2026-11-14): teacher botgen games of the TvT / TvP teacher",
            "# t0-terran-tp, generated by tools/eval/rush_check_plan.py --botgen-tp (do not edit by hand); played only after",
            "# the 022-AC2 check passes. The teacher players[0] against T0 standard Protoss and Terran x Fighting Spirit /",
            f"# Circuit Breaker x {TP_BOTGEN_T0_GAMES} ({t0}) and the T1 Terran / Protoss bots (UAlbertaBot P, Stardust, UAlbertaBot T,",
            f"# Iron) x 2 maps x {TP_BOTGEN_T1_GAMES} ({n - t0}) = {n} games, races fixed (no Random games in the data, 022-R2), 10 min",
            "# (max_frames 14,400), replays kept; build: uranium-replay build <dir>/replays --src botgen --games",
            "# <dir>/games.jsonl --until-frame 14400 --time-limit 14400 --out <shared>/datasets/replays/shards/botgen/teacher-tp-<date>",
            "# (--games: the teacher's view only, lobby races from games.jsonl; spec 040). The Zerg : Protoss : Terran",
            "# = 50 : 30 : 20 race split inside the teacher + dagger share is set by the training mix, not by this plan.",
            "# T1 bots: the 050-R9 T1 rule (NDEBUG builds, config/protocol/t1_bots.toml; opponent-crash games are invalid).",
        ]
        max_frames = TEACHER_MAX_FRAMES
    lines = [
        *what,
        f"# seeds (020-R20): splitmix64(1,000,000 + n) >> 32, n = {first:,}..{last:,} (check plan 1..208, botgen plan 1,001..1,800):",
        "# natural start slots, none of another plan's seeds (botgen, check, dagger, distill, 050-R9 protocol, the",
        "# teacher's development seeds 77,001..77,264; asserted by the generator).",
        f'name = "{name}"',
        "save_replay = true",
        "wall_timeout_s = 7200",
        f"max_frames = {max_frames}",
        "seeds = [",
    ]
    k = 0
    for opp, _, m, games in blocks:
        block = seeds[k:k + games]
        lines.append(f"  # games {k + 1}..{k + games}: {opp}, {m}")
        for i in range(0, games, 8):
            lines.append("  " + ", ".join(str(s) for s in block[i:i + 8]) + ",")
        k += games
    lines.append("]")
    for _, p1, m, games in blocks:
        lines += ["[[matchup]]", f'map = "{m}"', f"games = {games}", f"p0 = {TP_TEACHER}", f"p1 = {p1}"]
    return "\n".join(lines) + "\n", seeds


def write_tp(out_dir: Path, check: bool, out: Path | None = None) -> Path:
    """Writes the 022-R11 check plan config/plans/teacher_tp_check.toml (`check`) or the 022-R12 botgen plan
    config/plans/teacher_botgen_tp.toml; asserts its seeds are disjoint from every other plan under `out_dir` (and
    subfolders), the botgen / check / reserved round blocks and the teacher's development seeds."""
    stem, name = ("teacher_tp_check", "teacher-tp-check") if check else ("teacher_botgen_tp", "teacher-botgen-tp")
    own = out_dir / f"{stem}.toml"
    out = out or own
    text, seeds = render_tp(check, name)
    taken = taken_seeds(out_dir, (out, own)) | reserved_seeds()
    assert not taken & set(seeds), "a seed of this plan is another plan's (botgen, check, dagger, distill, protocol or development)"
    out.write_bytes(text.encode("utf-8"))
    hits = sum(_core.start_slots(s, 0)[2] for s in seeds)
    print(f"{out}: {len(seeds)} games, first-target hit {hits} ({hits / len(seeds):.1%}, natural 1/3), {len(taken)} other seeds disjoint")
    return out


# 044-R8 / 022-R12 (AMB-022-7 Resolved 2026-11-14): from the round after the 022 teacher passes, the round's
# TvT / TvP self games are labelled by the 022 teacher's shadow (`--relabel terran_tp --games`, source `dagger`,
# shards dagger_m2/t<R>) instead of the baseline labeller. The games are the round's 044-R1 self games (the
# DISTILL_OPPONENTS composition, `--half` the d2 size), the 041-R22 ① DAgger form: the round's model stochastic,
# decision logs for the 041-R28 memory_action relabel, last failed command shown (zero_last_failed = false).
# Hash key of their seeds, far from every other key's game numbers (botgen 60,000 + n, dagger 70,000 + n,
# distill 80,000 + n with n <= 200,000, 022 teacher 1,000,000 + n with n <= 1,800): game number n of teacher
# round t<R> is (R - 1) * TEACHER_ROUND_STRIDE + i, i = 1..games (600 full, 300 half), as the DAgger rounds.
TEACHER_ROUND_SEED_KEY = 2_000_000
TEACHER_ROUND_STRIDE = 600


def teacher_round_first(rnd: int) -> int:
    """The first game number n of teacher round t<rnd> (044-R8)."""
    assert 1 <= rnd <= RESERVED_ROUNDS, f"--round must be in 1..{RESERVED_ROUNDS} (the reserved round seed blocks)"
    return (rnd - 1) * TEACHER_ROUND_STRIDE + 1


def render_teacher_round(rnd: int, checkpoint: str, name: str, half: bool) -> tuple[str, list[int]]:
    """The text and seeds of the teacher round t<rnd> self-game plan (044-R8): the 044-R1 composition."""
    blocks = distill_blocks(half)
    n = sum(g for *_, g in blocks)
    assert n <= TEACHER_ROUND_STRIDE, "a round plan must fit in its seed stride"
    first = teacher_round_first(rnd)
    last = first + n - 1
    seeds = [splitmix64(TEACHER_ROUND_SEED_KEY + k) >> 32 for k in range(first, last + 1)]
    assert len(set(seeds)) == n, "teacher round seeds repeat"
    p0 = f'{{ kind = "model", checkpoint = "{checkpoint}", sample = "stochastic" }}'
    t0p, t0t = (sum(g for o, _, _, g in blocks if o == k) for k in ("T0 protoss", "T0 terran"))
    t1 = n - t0p - t0t
    span = "(r - 1) * 600 + 300, the first half of the round's 600" if half else "r * 600"
    lines = [
        f"# 044-R8 / 022-R12 (AMB-022-7 Resolved 2026-11-14): teacher round t{rnd} self games, generated by",
        f"# tools/eval/rush_check_plan.py --teacher-round --round {rnd}{' --half' if half else ''} (do not edit by hand). The round after the 022",
        "# teacher passes 022-AC2: the round's TvT / TvP self games are labelled by the 022 teacher's shadow, not the",
        "# baseline labeller (041-R22 ② form, source dagger): uranium-replay build <dir>/replays --src dagger",
        "# --relabel terran_tp --games <dir>/games.jsonl --memory-log <dir>/memory --until-frame 11520 --time-limit 14400",
        f"# --out <shared>/datasets/replays/shards/dagger_m2/t{rnd} (docs/commands.md: the model's view only, lobby races from",
        "# games.jsonl). Composition as the 044-R1 self games of a round:",
        f"# the model players[0] (stochastic) against T0 standard Protoss ({t0p}) and Terran ({t0t}) and the T1",
        f"# Protoss (UAlbertaBot P, Stardust) and Terran (UAlbertaBot T, Iron) bots ({t1}) x Fighting Spirit / Circuit",
        f"# Breaker = {n} games{' (half size)' if half else ''}, races fixed (no Random), 10 min (max_frames 14,400), replays kept. T1 bots:",
        "# the 050-R9 T1 rule (NDEBUG builds, config/protocol/t1_bots.toml; opponent-crash games are invalid).",
        f"# seeds (020-R20): splitmix64(2,000,000 + n) >> 32, n = {first:,}..{last:,} (round r: n = (r - 1) * 600 + 1..{span}):",
        "# natural start slots, none of another plan's seeds (botgen, check, dagger, distill, 022 teacher, 050-R9 protocol,",
        "# the DAgger / distillation / teacher round blocks not generated yet, the 022 development seeds; asserted).",
        f'name = "{name}"',
        "save_replay = true",
        "# 041-R28: decision logs memory/g<id>.jsonl beside replays/ (uranium-replay build --memory-log).",
        "memory_log = true",
        "wall_timeout_s = 7200",
        "zero_last_failed = false",
        f"max_frames = {TEACHER_MAX_FRAMES}",
        "seeds = [",
    ]
    k = 0
    for opp, _, m, games in blocks:
        block = seeds[k:k + games]
        lines.append(f"  # games {k + 1}..{k + games}: {opp}, {m}")
        for i in range(0, games, 8):
            lines.append("  " + ", ".join(str(s) for s in block[i:i + 8]) + ",")
        k += games
    lines.append("]")
    for _, p1, m, games in blocks:
        lines += ["[[matchup]]", f'map = "{m}"', f"games = {games}", f"p0 = {p0}", f"p1 = {p1}"]
    return "\n".join(lines) + "\n", seeds


def write_teacher_round(out_dir: Path, rnd: int, checkpoint: str, out: Path | None = None, half: bool = False,
                        name: str | None = None) -> Path:
    """The teacher round t<rnd> self-game plan (044-R8) config/plans/dagger_m2_t<rnd>.toml: 600 games, `half` 300.
    Asserts its seeds are disjoint from every plan under `out_dir` (and subfolders), the botgen / check blocks, the
    reserved DAgger / distillation / other teacher round blocks and the 022 development seeds; an existing plan
    file is rewritten only with the same seeds."""
    stem = f"dagger_m2_t{rnd}"
    own = out_dir / f"{stem}.toml"
    out = out or own
    text, seeds = render_teacher_round(rnd, checkpoint, name or stem.replace("_", "-"), half)
    taken = taken_seeds(out_dir, (out, own)) | reserved_seeds(skip_teacher_round=rnd)
    assert not taken & set(seeds), "a seed of this plan is another plan's (botgen, check, dagger, distill, 022 teacher, protocol or a reserved round)"
    for f in {out.resolve(), own.resolve()}:
        if f.exists():
            assert plan_seeds(f) == set(seeds), f"{f} exists with other seeds: refusing to overwrite it (games of it may be played); use another --out"
    out.write_bytes(text.encode("utf-8"))
    hits = sum(_core.start_slots(s, 0)[2] for s in seeds)
    print(f"{out}: {len(seeds)} games, first-target hit {hits} ({hits / len(seeds):.1%}, natural 1/3), {len(taken)} other seeds disjoint")
    return out


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
    ap.add_argument("--out-dir", type=Path, default=ROOT / "config" / "plans", help="--botgen / --dagger / --distill output folder (and the seed disjointness scan; --distill also scans subfolders)")
    ap.add_argument("--first-game", type=int, default=None,
                    help="--botgen: write one more macro plan from this botgen game number (> 2,000) instead")
    ap.add_argument("--dagger", action="store_true", help="write a DAgger round plan (10th advisory 2.2)")
    ap.add_argument("--distill", action="store_true", help="write a self-game distillation round plan (044-R1; needs --checkpoint)")
    ap.add_argument("--round", type=int, default=1, help="--dagger / --distill / --teacher-round: round number (seeds per round disjoint)")
    ap.add_argument("--half", action="store_true", help="--dagger / --distill / --teacher-round: half-size round plan, 300 games (11th advisory 2, 044-R1 d2)")
    ap.add_argument("--extra", type=int, default=0,
                    help="--distill: a top-up plan of N more games of round --round (044-R5 (3), user decision 2026-11-14; N a multiple of 20)")
    ap.add_argument("--part", type=int, default=None,
                    help="--distill --extra (required there): top-up part number 1..10 of the round, a new one per top-up (its own seed block)")
    ap.add_argument("--memory-log", action="store_true",
                    help="--dagger: the model writes decision logs for the memory_action relabel (041-R28); --distill / --teacher-round: always on")
    ap.add_argument("--teacher-tp", action="store_true", help="write the 022-R11 check plan of the 022 teacher terran_tp (config/plans/teacher_tp_check.toml)")
    ap.add_argument("--botgen-tp", action="store_true", help="write the 022-R12 botgen plan of the 022 teacher terran_tp (config/plans/teacher_botgen_tp.toml)")
    ap.add_argument("--teacher-round", action="store_true",
                    help="write a 044-R8 teacher round self-game plan config/plans/dagger_m2_t<R>.toml (the 044-R1 composition; needs --checkpoint)")
    a = ap.parse_args()
    if a.teacher_tp or a.botgen_tp:
        # 022-R11 / R12: fixed composition and seeds; only --out-dir (and --out) apply.
        assert not (a.teacher_tp and a.botgen_tp), "--teacher-tp and --botgen-tp are separate plans"
        assert not (a.teacher or a.botgen or a.dagger or a.distill or a.teacher_round or a.half or a.extra or a.part or a.memory_log
                    or a.first_game is not None or a.checkpoint or a.name), "--teacher-tp / --botgen-tp take only --out-dir / --out"
        write_tp(a.out_dir, a.teacher_tp, a.out)
        raise SystemExit(0)
    if a.teacher_round:
        # 044-R8: the round's model, never a default; the 044-R1 composition (--half the d2 size), no top-up.
        assert not (a.teacher or a.botgen or a.dagger or a.distill or a.extra or a.part is not None or a.first_game is not None),             "--teacher-round takes only --round / --half / --memory-log / --checkpoint / --name / --out / --out-dir"
        assert a.checkpoint, "--teacher-round needs --checkpoint (044-R8: the model of the round)"
        write_teacher_round(a.out_dir, a.round, a.checkpoint, a.out, a.half, a.name)
        raise SystemExit(0)
    assert not (a.dagger and a.distill), "--dagger and --distill are separate plans"
    assert a.dagger or a.distill or not a.half, "--half needs --dagger or --distill"
    assert a.dagger or a.distill or not a.memory_log, "--memory-log needs --dagger or --distill"
    assert a.distill or not (a.extra or a.part is not None), "--extra / --part need --distill"
    assert a.extra or a.part is None, "--part needs --extra"
    # 044-R5 ③: every top-up names its part, so a second top-up cannot reuse the first one's seeds.
    assert not a.extra or a.part is not None, "--extra needs --part K (1 for the round's first top-up, 2 for the next, ...)"
    if a.distill:
        # 044-R1: the model is S (d1) or the d1 model (d2), never a default.
        assert a.checkpoint, "--distill needs --checkpoint (044-R1: d1 = the 041-R29 (4) selection S, d2 = the d1 model)"
        write_distill(a.out_dir, a.round, a.checkpoint, a.out, a.half, a.name, a.extra, a.part or 1)
        raise SystemExit(0)
    if a.dagger:
        write_dagger(a.out_dir, a.round, a.checkpoint or DAGGER_CHECKPOINT, a.out, a.half, a.memory_log)
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
