"""Writes the rush check plan and the teacher, botgen, DAgger and distillation game plans"""

import argparse
import tomllib
from pathlib import Path

from uranium import _core

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "config" / "plans" / "rush5_check.toml"
TEACHER_OUT = ROOT / "config" / "plans" / "teacher_check.toml"
TEACHER = '{ kind = "t0", race = "terran", build = "terran_react" }'
TEACHER_MAX_FRAMES = 14400
CHECKPOINT = "checkpoints/ac1-vtrace-s1/ckpt-3133569/agents/main"
OPPONENTS = (
    ("T0 zerg_rush", '{ kind = "t0", race = "zerg", build = "zerg_rush" }'),
    ("ualbertabot12poollings", '{ kind = "oss", name = "ualbertabot12poollings", race = "zerg" }'),
)
MAPS = ("fighting_spirit", "circuit_breaker")
GAMES = 32
FIRST_SEED = 45001

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
BOTGEN_SEED_KEY = 60000
CHECK_SEED_RANGES = ((46000, 40), (47000, 10))


def splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def botgen_seeds(first: int, count: int) -> list[int]:
    return [splitmix64(BOTGEN_SEED_KEY + n) >> 32 for n in range(first, first + count)]


def render_botgen(name: str, opponents, games: int, seeds: list[int]) -> str:
    lines = [
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
    for name, opps, games, seeds in (
        ("teacher-botgen-rush", BOTGEN_RUSH, BOTGEN_RUSH_GAMES, rush),
        ("teacher-botgen-macro", BOTGEN_MACRO, BOTGEN_MACRO_GAMES, macro),
    ):
        out = out_dir / (name.replace("-", "_") + ".toml")
        out.write_bytes(render_botgen(name, opps, games, seeds).encode("utf-8"))
        hits = sum(_core.start_slots(s, 0)[2] for s in seeds)
        print(f"{out}: {len(seeds)} games, first-target hit {hits} ({hits / len(seeds):.1%}, natural 1/3)")


def plan_seeds(path: Path) -> set[int]:
    p = tomllib.loads(path.read_text(encoding="utf-8"))
    if "seeds" in p:
        return set(p["seeds"])
    base = p.get("seed_base", 1000)
    if p.get("same_seed", False):
        return {base}
    total = sum(m.get("games", 0) for m in p.get("matchup", []))
    return set(range(base + 1, base + total + 1))


def write_botgen_extra(out_dir: Path, first: int, name: str) -> None:
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
    text = render_botgen(name, BOTGEN_MACRO, BOTGEN_MACRO_GAMES, seeds)
    out.write_bytes(text.encode("utf-8"))
    hits = sum(_core.start_slots(s, 0)[2] for s in seeds)
    print(f"{out}: {n} games, first-target hit {hits} ({hits / n:.1%}, natural 1/3), {len(taken)} other seeds disjoint")


DAGGER_CHECKPOINT = "checkpoints/bc-mix-e10/ckpt-1053"
DAGGER_RUSH_GAMES = 40
DAGGER_MACRO_GAMES = 28
DAGGER_HALF_RUSH_GAMES = 20
DAGGER_HALF_MACRO_GAMES = 14
DAGGER_SEED_KEY = 70000
DAGGER_ROUND_STRIDE = 600


def write_dagger(out_dir: Path, rnd: int, checkpoint: str, out: Path | None, half: bool = False, memory_log: bool = False) -> None:
    assert rnd >= 1, "--round starts at 1"
    if half:
        rush_games, macro_games = DAGGER_HALF_RUSH_GAMES, DAGGER_HALF_MACRO_GAMES
    else:
        rush_games, macro_games = DAGGER_RUSH_GAMES, DAGGER_MACRO_GAMES
    blocks = [(o, m, rush_games) for o in BOTGEN_RUSH for m in MAPS]
    blocks += [(o, m, macro_games) for o in BOTGEN_MACRO for m in MAPS]
    n = sum(g for _, _, g in blocks)
    assert n <= DAGGER_ROUND_STRIDE, "a round plan must fit in its seed stride"
    first = (rnd - 1) * DAGGER_ROUND_STRIDE + 1
    last = first + n - 1
    seeds = [splitmix64(DAGGER_SEED_KEY + k) >> 32 for k in range(first, last + 1)]
    name = f"dagger-r{rnd}"
    own = out_dir / f"dagger_r{rnd}.toml"
    out = out or own
    assert len(set(seeds)) == n, "dagger seeds repeat"
    n_prev = (len(BOTGEN_RUSH) * BOTGEN_RUSH_GAMES + len(BOTGEN_MACRO) * BOTGEN_MACRO_GAMES) * len(MAPS)
    taken = set(botgen_seeds(1, n_prev + 400)) | set(_core.rush_check_seeds(len(OPPONENTS) * len(MAPS), FIRST_SEED))
    for base, count in CHECK_SEED_RANGES:
        taken.update(range(base + 1, base + count + 1))
    for other in sorted(out_dir.glob("*.toml")):
        if other.resolve() not in (out.resolve(), own.resolve()):
            taken |= plan_seeds(other)
    assert not taken & set(seeds), "a seed of this plan is another plan's (botgen, check or dagger)"
    p0 = f'{{ kind = "model", checkpoint = "{checkpoint}", sample = "stochastic" }}'
    lines = [
        f'name = "{name}"',
        "save_replay = true",
        *(("memory_log = true",) if memory_log else ()),
        "wall_timeout_s = 7200",
        "zero_last_failed = false",
        f"max_frames = {TEACHER_MAX_FRAMES}",
        "seeds = [",
    ]
    k = 0
    for (opp, _), m, games in blocks:
        block = seeds[k:k + games]
        for i in range(0, games, 8):
            lines.append("  " + ", ".join(str(s) for s in block[i:i + 8]) + ",")
        k += games
    lines.append("]")
    for (_, p1), m, games in blocks:
        lines += ["[[matchup]]", f'map = "{m}"', f"games = {games}", f"p0 = {p0}", f"p1 = {p1}"]
    out.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    hits = sum(_core.start_slots(s, 0)[2] for s in seeds)
    print(f"{out}: {n} games, first-target hit {hits} ({hits / n:.1%}, natural 1/3), {len(taken)} other seeds disjoint")


DISTILL_OPPONENTS = (
    ("T0 protoss", '{ kind = "t0", race = "protoss" }', 120),
    ("T0 terran", '{ kind = "t0", race = "terran" }', 60),
    ("ualbertabot protoss", '{ kind = "oss", name = "ualbertabot", race = "protoss" }', 30),
    ("stardust", '{ kind = "oss", name = "stardust", race = "protoss" }', 30),
    ("ualbertabot terran", '{ kind = "oss", name = "ualbertabot", race = "terran" }', 30),
    ("iron", '{ kind = "oss", name = "iron", race = "terran" }', 30),
)
DISTILL_SEED_KEY = 80000
DISTILL_ROUND_STRIDE = 600
DISTILL_EXTRA_BASE = 100000
DISTILL_EXTRA_ROUND = 10000
DISTILL_EXTRA_PART = 1000
DISTILL_EXTRA_UNIT = 20
DISTILL_ROUNDS_RESERVED = 10
DISTILL_EXTRA_PARTS = DISTILL_EXTRA_ROUND // DISTILL_EXTRA_PART
assert DISTILL_ROUNDS_RESERVED * DISTILL_ROUND_STRIDE < DISTILL_EXTRA_BASE


def distill_blocks(half: bool, extra: int = 0) -> list[tuple[str, str, str, int]]:
    if extra:
        assert extra % DISTILL_EXTRA_UNIT == 0 and 0 < extra <= DISTILL_EXTRA_PART, \
            f"--extra must be a multiple of {DISTILL_EXTRA_UNIT} in {DISTILL_EXTRA_UNIT}..{DISTILL_EXTRA_PART} (the 044-R1 ratio exactly); round it up"
        return [(o, p1, m, g * extra // DISTILL_ROUND_STRIDE) for o, p1, g in DISTILL_OPPONENTS for m in MAPS]
    return [(o, p1, m, g // 2 if half else g) for o, p1, g in DISTILL_OPPONENTS for m in MAPS]


def distill_first(rnd: int, extra: int = 0, part: int = 1) -> int:
    assert 1 <= rnd <= DISTILL_ROUNDS_RESERVED, f"--round must be in 1..{DISTILL_ROUNDS_RESERVED} (the reserved round seed blocks)"
    if extra:
        assert 1 <= part <= DISTILL_EXTRA_PARTS, f"--part must be in 1..{DISTILL_EXTRA_PARTS} (a round's top-up seed blocks)"
        return DISTILL_EXTRA_BASE + (rnd - 1) * DISTILL_EXTRA_ROUND + (part - 1) * DISTILL_EXTRA_PART + 1
    return (rnd - 1) * DISTILL_ROUND_STRIDE + 1


def distill_seeds(rnd: int, games: int, extra: int = 0, part: int = 1) -> tuple[int, list[int]]:
    first = distill_first(rnd, extra, part)
    return first, [splitmix64(DISTILL_SEED_KEY + k) >> 32 for k in range(first, first + games)]


def taken_seeds(out_dir: Path, skip: tuple[Path, ...]) -> set[int]:
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
    assert 1 <= rnd <= DISTILL_ROUNDS_RESERVED, f"--round must be in 1..{DISTILL_ROUNDS_RESERVED} (the reserved round seed blocks)"
    assert not (half and extra), "--half and --extra are separate plans"
    blocks = distill_blocks(half, extra)
    n = sum(g for *_, g in blocks)
    assert n <= (DISTILL_EXTRA_PART if extra else DISTILL_ROUND_STRIDE), "a plan must fit in its seed stride"
    assert not extra or n == extra
    _, seeds = distill_seeds(rnd, n, extra, part)
    assert len(set(seeds)) == n, "distill seeds repeat"
    if extra:
        rounds = {splitmix64(DISTILL_SEED_KEY + k) >> 32 for k in range(1, DISTILL_ROUNDS_RESERVED * DISTILL_ROUND_STRIDE + 1)}
        assert not rounds & set(seeds), "a top-up seed is a round seed"
    p0 = f'{{ kind = "model", checkpoint = "{checkpoint}", sample = "stochastic" }}'
    lines = [
        f'name = "{name}"',
        "save_replay = true",
        "memory_log = true",
        "wall_timeout_s = 7200",
        "zero_last_failed = false",
        f"max_frames = {TEACHER_MAX_FRAMES}",
        "seeds = [",
    ]
    k = 0
    for opp, _, m, games in blocks:
        block = seeds[k:k + games]
        for i in range(0, games, 8):
            lines.append("  " + ", ".join(str(s) for s in block[i:i + 8]) + ",")
        k += games
    lines.append("]")
    for _, p1, m, games in blocks:
        lines += ["[[matchup]]", f'map = "{m}"', f"games = {games}", f"p0 = {p0}", f"p1 = {p1}"]
    return "\n".join(lines) + "\n", seeds


def write_distill(out_dir: Path, rnd: int, checkpoint: str, out: Path | None, half: bool = False,
                  name: str | None = None, extra: int = 0, part: int = 1) -> Path:
    stem = f"distill_m2_d{rnd}_x{part}" if extra else f"distill_m2_d{rnd}"
    own = out_dir / f"{stem}.toml"
    out = out or own
    text, seeds = render_distill(rnd, checkpoint, name or stem.replace("_", "-"), half, extra, part)
    taken = taken_seeds(out_dir, (out, own))
    assert not taken & set(seeds), "a seed of this plan is another plan's (botgen, check, dagger or protocol)"
    for f in {out.resolve(), own.resolve()}:
        if f.exists():
            assert plan_seeds(f) == set(seeds), f"{f} exists with other seeds: refusing to overwrite it (games of it may be played); use another --part / --out"
    out.write_bytes(text.encode("utf-8"))
    hits = sum(_core.start_slots(s, 0)[2] for s in seeds)
    print(f"{out}: {len(seeds)} games, first-target hit {hits} ({hits / len(seeds):.1%}, natural 1/3), {len(taken)} other seeds disjoint")
    return out


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
TP_CHECK_T0_GAMES = 32
TP_CHECK_T1_GAMES = 10
TP_CHECK_MAX_FRAMES = 86400
TP_BOTGEN_T0_GAMES = 150
TP_BOTGEN_T1_GAMES = 25
TP_SEED_KEY = 1_000_000
TP_CHECK_FIRST = 1
TP_BOTGEN_FIRST = 1001
TP_DEV_SEEDS = range(77001, 77265)
TP_DEV_TUNE = range(5001, 5133)


def tp_dev_seeds() -> set[int]:
    return set(TP_DEV_SEEDS) | {splitmix64(TP_SEED_KEY + n) >> 32 for n in TP_DEV_TUNE}
RESERVED_ROUNDS = 10
TP_EXTRA_BASE = 10000
TP_EXTRA_PART = 1000
TP_EXTRA_PARTS = 10
TP_EXTRA_RACE_UNIT = 16
TP_EXTRA_MAX = TP_EXTRA_PART // TP_EXTRA_RACE_UNIT * TP_EXTRA_RACE_UNIT
TP_EXTRA_RACES = {"P": "protoss", "T": "terran"}


def tp_race(p1: str) -> str:
    return next(k for k, race in TP_EXTRA_RACES.items() if f'race = "{race}"' in p1)


def tp_blocks(check: bool, extra: dict[str, int] | None = None) -> list[tuple[str, str, str, int]]:
    t0, t1 = (TP_CHECK_T0_GAMES, TP_CHECK_T1_GAMES) if check else (TP_BOTGEN_T0_GAMES, TP_BOTGEN_T1_GAMES)
    blocks = [(o, p1, m, g) for opps, g in ((TP_T0, t0), (TP_T1, t1)) for o, p1 in opps for m in MAPS]
    if extra:
        assert not check, "a top-up is of the botgen plan"
        assert set(extra) <= set(TP_EXTRA_RACES), f"top-up races are {sorted(TP_EXTRA_RACES)}"
        for r, n in extra.items():
            assert n >= 0 and n % TP_EXTRA_RACE_UNIT == 0, \
                f"--extra-{r.lower()} must be a multiple of {TP_EXTRA_RACE_UNIT} (the race's 022-R12 ratio exactly); round it up"
        total = sum(extra.values())
        assert 0 < total <= TP_EXTRA_MAX, f"--extra-p + --extra-t must be in {TP_EXTRA_RACE_UNIT}..{TP_EXTRA_MAX} (one part's seed block); split it over --part runs"
        full = {r: sum(g for _, p1, _, g in blocks if tp_race(p1) == r) for r in TP_EXTRA_RACES}
        blocks = [(o, p1, m, g * extra.get(tp_race(p1), 0) // full[tp_race(p1)]) for o, p1, m, g in blocks]
        blocks = [b for b in blocks if b[3] > 0]
        assert sum(b[3] for b in blocks) == total
    return blocks


def tp_first(check: bool, extra: dict[str, int] | None = None, part: int = 1) -> int:
    if extra:
        assert 1 <= part <= TP_EXTRA_PARTS, f"--part must be in 1..{TP_EXTRA_PARTS} (the botgen top-up seed blocks)"
        return TP_EXTRA_BASE + (part - 1) * TP_EXTRA_PART + 1
    return TP_CHECK_FIRST if check else TP_BOTGEN_FIRST


def tp_seeds(check: bool, extra: dict[str, int] | None = None, part: int = 1) -> tuple[int, list[int]]:
    first = tp_first(check, extra, part)
    n = sum(g for *_, g in tp_blocks(check, extra))
    return first, [splitmix64(TP_SEED_KEY + k) >> 32 for k in range(first, first + n)]


def tp_extra_block(part: int) -> set[int]:
    first = tp_first(False, {"P": TP_EXTRA_RACE_UNIT}, part)
    return {splitmix64(TP_SEED_KEY + k) >> 32 for k in range(first, first + TP_EXTRA_PART)}


def reserved_seeds(skip_teacher_round: bool = False, skip_tp_extra_part: int | None = None, skip_teacher_round_z: bool = False) -> set[int]:
    n = RESERVED_ROUNDS * DAGGER_ROUND_STRIDE
    s = {splitmix64(DAGGER_SEED_KEY + k) >> 32 for k in range(1, n + 1)}
    s |= {splitmix64(DISTILL_SEED_KEY + k) >> 32 for k in range(1, DISTILL_ROUNDS_RESERVED * DISTILL_ROUND_STRIDE + 1)}
    top = range(DISTILL_EXTRA_BASE + 1, DISTILL_EXTRA_BASE + DISTILL_ROUNDS_RESERVED * DISTILL_EXTRA_ROUND + 1)
    s |= {splitmix64(DISTILL_SEED_KEY + k) >> 32 for k in top}
    if not skip_teacher_round:
        s |= {splitmix64(TEACHER_ROUND_SEED_KEY + k) >> 32 for k in TEACHER_ROUND_BLOCK}
    if not skip_teacher_round_z:
        s |= teacher_round_z_block()
    for part in range(1, TP_EXTRA_PARTS + 1):
        if part != skip_tp_extra_part:
            s |= tp_extra_block(part)
    return s | tp_dev_seeds()


def render_tp(check: bool, name: str, extra: dict[str, int] | None = None, part: int = 1) -> tuple[str, list[int]]:
    blocks = tp_blocks(check, extra)
    _, seeds = tp_seeds(check, extra, part)
    n = len(seeds)
    assert len(set(seeds)) == n, "022 teacher seeds repeat"
    for c in (True, False):
        if extra or c != check:
            assert not set(seeds) & set(tp_seeds(c)[1]), "the 022 check, botgen and top-up plans share a seed"
    max_frames = TP_CHECK_MAX_FRAMES if check else TEACHER_MAX_FRAMES
    lines = [
        f'name = "{name}"',
        "save_replay = true",
        "wall_timeout_s = 7200",
        f"max_frames = {max_frames}",
        "seeds = [",
    ]
    k = 0
    for opp, _, m, games in blocks:
        block = seeds[k:k + games]
        for i in range(0, games, 8):
            lines.append("  " + ", ".join(str(s) for s in block[i:i + 8]) + ",")
        k += games
    lines.append("]")
    for _, p1, m, games in blocks:
        lines += ["[[matchup]]", f'map = "{m}"', f"games = {games}", f"p0 = {TP_TEACHER}", f"p1 = {p1}"]
    return "\n".join(lines) + "\n", seeds


def write_tp(out_dir: Path, check: bool, out: Path | None = None, extra: dict[str, int] | None = None, part: int = 1) -> Path:
    if extra:
        stem = f"teacher_botgen_tp_x{part}"
    else:
        stem = "teacher_tp_check" if check else "teacher_botgen_tp"
    own = out_dir / f"{stem}.toml"
    out = out or own
    text, seeds = render_tp(check, stem.replace("_", "-"), extra, part)
    taken = taken_seeds(out_dir, (out, own)) | reserved_seeds(skip_tp_extra_part=part if extra else None)
    assert not taken & set(seeds), "a seed of this plan is another plan's (botgen, check, dagger, distill, protocol or development)"
    if extra:
        want = tomllib.loads(text)
        for f in {out.resolve(), own.resolve()}:
            if f.exists():
                have = tomllib.loads(f.read_text(encoding="utf-8"))
                assert plan_seeds(f) == set(seeds) and have.get("seeds") == want["seeds"], \
                    f"{f} exists with other seeds: refusing to overwrite it (games of it may be played); use another --part / --out"
                assert have.get("matchup") == want["matchup"], \
                    f"{f} exists with other matchups (another --extra-p / --extra-t split): refusing to overwrite it (games of it may be played); use another --part / --out"
    out.write_bytes(text.encode("utf-8"))
    hits = sum(_core.start_slots(s, 0)[2] for s in seeds)
    print(f"{out}: {len(seeds)} games, first-target hit {hits} ({hits / len(seeds):.1%}, natural 1/3), {len(taken)} other seeds disjoint")
    return out


TEACHER_ROUND_SEED_KEY = 3_000_000
TEACHER_ROUND_BLOCK = range(1, 601)
TEACHER_ROUND_FOLLOW_UP = (
    "uranium-replay build <dir>/replays --src dagger --relabel terran_tp --games <dir>/games.jsonl"
    " --memory-log <dir>/memory --until-frame 14400 --time-limit 14400"
    " --out <shared>/datasets/replays/shards/dagger_m2/t1"
)
TEACHER_ROUND_GAMES = 300
TEACHER_ROUNDS = ("t1", "t2")
TEACHER_ROUND_Z_ROUNDS = ("t2",)
TEACHER_ROUND_Z_SEED_KEY = 5_000_000
TEACHER_ROUND_Z_BLOCK = range(1, 3001)
TEACHER_ROUND_Z_HALF = (len(BOTGEN_RUSH) * DAGGER_HALF_RUSH_GAMES + len(BOTGEN_MACRO) * DAGGER_HALF_MACRO_GAMES) * len(MAPS)
TEACHER_ROUND_Z_UNIT = TEACHER_ROUND_Z_HALF // 2
TEACHER_ROUND_Z_FOLLOW_UP = (
    "uranium-replay build <dir>/replays --src dagger --relabel terran_react"
    " --memory-log <dir>/memory --until-frame 14400 --time-limit 14400"
    " --out <shared>/datasets/replays/shards/dagger_m2/{rnd}z"
)


def teacher_round_follow_up(rnd: str = "t1", zerg: bool = False) -> str:
    assert rnd in TEACHER_ROUNDS, f"--round of --teacher-round is one of {', '.join(TEACHER_ROUNDS)}"
    if zerg:
        return TEACHER_ROUND_Z_FOLLOW_UP.format(rnd=rnd)
    return TEACHER_ROUND_FOLLOW_UP.replace("dagger_m2/t1", f"dagger_m2/{rnd}")


def teacher_round_seeds(rnd: str) -> list[int]:
    assert rnd in TEACHER_ROUNDS, f"--round of --teacher-round is one of {', '.join(TEACHER_ROUNDS)}"
    first = TEACHER_ROUND_BLOCK[0] + TEACHER_ROUNDS.index(rnd) * TEACHER_ROUND_GAMES
    last = first + TEACHER_ROUND_GAMES - 1
    assert last <= TEACHER_ROUND_BLOCK[-1], "the round plan must fit in its seed block"
    return [splitmix64(TEACHER_ROUND_SEED_KEY + k) >> 32 for k in range(first, last + 1)]


def render_teacher_round(checkpoint: str, name: str, rnd: str = "t1") -> tuple[str, list[int]]:
    blocks = distill_blocks(half=True)
    n = sum(g for *_, g in blocks)
    assert n == TEACHER_ROUND_GAMES, "a teacher round is the 044-R1 composition halved"
    seeds = teacher_round_seeds(rnd)
    assert len(set(seeds)) == n, "teacher round seeds repeat"
    return render_model_games(checkpoint, name, blocks, seeds), seeds


def teacher_round_z_blocks(games: int) -> list[tuple[str, str, str, int]]:
    assert games > 0 and games % TEACHER_ROUND_Z_UNIT == 0 and games <= len(TEACHER_ROUND_Z_BLOCK), (
        f"--zerg must be a multiple of {TEACHER_ROUND_Z_UNIT} (the DAgger round composition exactly) "
        f"in {TEACHER_ROUND_Z_UNIT}..{len(TEACHER_ROUND_Z_BLOCK)} (its seed block); round it up")
    blocks = [(o, p1, m, DAGGER_HALF_RUSH_GAMES * games // TEACHER_ROUND_Z_HALF) for o, p1 in BOTGEN_RUSH for m in MAPS]
    blocks += [(o, p1, m, DAGGER_HALF_MACRO_GAMES * games // TEACHER_ROUND_Z_HALF) for o, p1 in BOTGEN_MACRO for m in MAPS]
    assert sum(b[3] for b in blocks) == games
    return blocks


def teacher_round_z_block() -> set[int]:
    return {splitmix64(TEACHER_ROUND_Z_SEED_KEY + k) >> 32 for k in TEACHER_ROUND_Z_BLOCK}


def render_teacher_round_z(checkpoint: str, name: str, games: int, rnd: str = "t2") -> tuple[str, list[int]]:
    assert rnd in TEACHER_ROUND_Z_ROUNDS, f"--zerg goes with --round {' / '.join(TEACHER_ROUND_Z_ROUNDS)} (its own seed block)"
    blocks = teacher_round_z_blocks(games)
    first = TEACHER_ROUND_Z_BLOCK[0]
    seeds = [splitmix64(TEACHER_ROUND_Z_SEED_KEY + k) >> 32 for k in range(first, first + games)]
    assert len(set(seeds)) == games, "teacher round Zerg seeds repeat"
    return render_model_games(checkpoint, name, blocks, seeds), seeds


def render_model_games(checkpoint: str, name: str, blocks, seeds: list[int]) -> str:
    p0 = f'{{ kind = "model", checkpoint = "{checkpoint}", sample = "stochastic" }}'
    lines = [
        f'name = "{name}"',
        "save_replay = true",
        "memory_log = true",
        "wall_timeout_s = 7200",
        "zero_last_failed = false",
        f"max_frames = {TEACHER_MAX_FRAMES}",
        "seeds = [",
    ]
    k = 0
    for opp, _, m, games in blocks:
        block = seeds[k:k + games]
        for i in range(0, games, 8):
            lines.append("  " + ", ".join(str(s) for s in block[i:i + 8]) + ",")
        k += games
    assert k == len(seeds)
    lines.append("]")
    for _, p1, m, games in blocks:
        lines += ["[[matchup]]", f'map = "{m}"', f"games = {games}", f"p0 = {p0}", f"p1 = {p1}"]
    return "\n".join(lines) + "\n"


def write_teacher_round(out_dir: Path, checkpoint: str, out: Path | None = None, name: str | None = None,
                        rnd: str = "t1", zerg: int = 0) -> Path:
    stem = f"dagger_m2_{rnd}_z" if zerg else f"dagger_m2_{rnd}"
    own = out_dir / f"{stem}.toml"
    out = out or own
    if zerg:
        text, seeds = render_teacher_round_z(checkpoint, name or stem.replace("_", "-"), zerg, rnd)
        taken = taken_seeds(out_dir, (out, own)) | reserved_seeds(skip_teacher_round_z=True)
    else:
        text, seeds = render_teacher_round(checkpoint, name or stem.replace("_", "-"), rnd)
        taken = taken_seeds(out_dir, (out, own)) | reserved_seeds(skip_teacher_round=True)
    assert not taken & set(seeds), "a seed of this plan is another plan's (botgen, check, dagger, distill, 022 teacher, protocol or a reserved block)"
    for f in {out.resolve(), own.resolve()}:
        if f.exists():
            assert plan_seeds(f) == set(seeds), f"{f} exists with other seeds: refusing to overwrite it (games of it may be played); use another --out"
    out.write_bytes(text.encode("utf-8"))
    hits = sum(_core.start_slots(s, 0)[2] for s in seeds)
    print(f"{out}: {len(seeds)} games, first-target hit {hits} ({hits / len(seeds):.1%}, natural 1/3), {len(taken)} other seeds disjoint")
    return out


def render(checkpoint: str | None, name: str) -> str:
    blocks = len(OPPONENTS) * len(MAPS)
    seeds = _core.rush_check_seeds(blocks, FIRST_SEED)
    assert len(seeds) == blocks * GAMES
    if checkpoint is None:
        lines = [
            f'name = "{name}"',
            "save_replay = true",
            "wall_timeout_s = 7200",
            f"max_frames = {TEACHER_MAX_FRAMES}",
            "seeds = [",
        ]
        p0 = TEACHER
    else:
        lines = [
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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=f"--teacher-round follow-up (044-R8):\n  {TEACHER_ROUND_FOLLOW_UP}\n"
                                        f"--teacher-round --round t2 follow-up:\n  {teacher_round_follow_up('t2')}\n"
                                        f"--teacher-round --round t2 --zerg N follow-up (021-R8 teacher, 022-R2):\n  {teacher_round_follow_up('t2', zerg=True)}")
    ap.add_argument("--checkpoint", default=None, help="model checkpoint (relative = under URANIUM_SHARED_DIR; default ac1-vtrace-s1, --dagger bc-mix-e10)")
    ap.add_argument("--teacher", action="store_true", help="the scripted teacher plays players[0] instead of a model")
    ap.add_argument("--name", default=None, help="plan name (the evaluation stage name; default rush5-check / teacher-check)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--botgen", action="store_true", help="write the two teacher botgen plans")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "config" / "plans", help="--botgen / --dagger / --distill output folder, also scanned for seed overlap")
    ap.add_argument("--first-game", type=int, default=None,
                    help="--botgen: write one more macro plan from this botgen game number (> 2,000) instead")
    ap.add_argument("--dagger", action="store_true", help="write a DAgger round plan")
    ap.add_argument("--distill", action="store_true", help="write a self-game distillation round plan (needs --checkpoint)")
    ap.add_argument("--round", default=None,
                    help="--dagger / --distill: round number, default 1 (seeds per round disjoint); --teacher-round: t1 (default) or t2")
    ap.add_argument("--half", action="store_true", help="--dagger / --distill: half-size round plan, 300 games")
    ap.add_argument("--extra-p", type=int, default=0,
                    help="--botgen-tp: N more teacher botgen games against Protoss (a multiple of 16)")
    ap.add_argument("--extra-t", type=int, default=0,
                    help="--botgen-tp: M more teacher botgen games against Terran (multiple of 16; N + M <= 992 per --part)")
    ap.add_argument("--extra", type=int, default=0,
                    help="--distill: a top-up plan of N more games of round --round (a multiple of 20)")
    ap.add_argument("--part", type=int, default=None,
                    help="top-up part 1..10, a new one per top-up (own seed block); required with --extra, --extra-p/-t")
    ap.add_argument("--memory-log", action="store_true",
                    help="--dagger: write decision logs for the memory_action relabel (always on: --distill, --teacher-round)")
    ap.add_argument("--teacher-tp", action="store_true", help="write the check plan of the terran_tp teacher (config/plans/teacher_tp_check.toml)")
    ap.add_argument("--botgen-tp", action="store_true", help="write the botgen plan of the terran_tp teacher (config/plans/teacher_botgen_tp.toml)")
    ap.add_argument("--teacher-round", action="store_true",
                    help="write the teacher round self-game plan config/plans/dagger_m2_<round>.toml (needs --checkpoint)")
    ap.add_argument("--zerg", type=int, default=0,
                    help=f"--teacher-round --round t2: write its Zerg DAgger plan dagger_m2_t2_z.toml of N games instead (a multiple of {TEACHER_ROUND_Z_UNIT})")
    a = ap.parse_args(argv)
    if a.teacher_tp or a.botgen_tp:
        assert not (a.teacher_tp and a.botgen_tp), "--teacher-tp and --botgen-tp are separate plans"
        assert not (a.teacher or a.botgen or a.dagger or a.distill or a.teacher_round or a.half or a.memory_log or a.round is not None
                    or a.first_game is not None or a.checkpoint or a.name or a.extra or a.zerg), \
            "--teacher-tp / --botgen-tp take only --out-dir / --out (--botgen-tp also --extra-p / --extra-t / --part)"
        topup = bool(a.extra_p or a.extra_t)
        assert a.botgen_tp or not (topup or a.part is not None), "--extra-p / --extra-t / --part are for --botgen-tp, not --teacher-tp"
        assert topup == (a.part is not None), \
            "--botgen-tp --extra-p N / --extra-t M needs --part K (1 for the first top-up, 2 for the next, ...) and --part needs a count"
        extra = {r: n for r, n in (("P", a.extra_p), ("T", a.extra_t)) if n} if topup else None
        write_tp(a.out_dir, a.teacher_tp, a.out, extra, a.part or 1)
        return
    assert not (a.extra_p or a.extra_t), "--extra-p / --extra-t are for --botgen-tp"
    if a.teacher_round:
        assert not (a.teacher or a.botgen or a.dagger or a.distill or a.extra or a.part is not None or a.first_game is not None
                    or a.half), (
            "--teacher-round (a round t1 / t2, 300 games, or the t2 Zerg plan) takes only --round / --zerg / --memory-log / --checkpoint / --name / --out / --out-dir")
        rnd = a.round or "t1"
        assert rnd in TEACHER_ROUNDS, f"--teacher-round --round is one of {', '.join(TEACHER_ROUNDS)}"
        assert a.zerg >= 0, "--zerg is a game count"
        assert not a.zerg or rnd in TEACHER_ROUND_Z_ROUNDS, f"--zerg goes with --round {' / '.join(TEACHER_ROUND_Z_ROUNDS)}"
        assert a.checkpoint, "--teacher-round needs --checkpoint (044-R8: the model of the round)"
        write_teacher_round(a.out_dir, a.checkpoint, a.out, a.name, rnd, a.zerg)
        print(f"follow-up: {teacher_round_follow_up(rnd, zerg=bool(a.zerg))}")
        return
    assert not a.zerg, "--zerg needs --teacher-round --round t2"
    assert a.round is None or a.round.isdigit(), "--dagger / --distill --round is a round number"
    rnd = 1 if a.round is None else int(a.round)
    assert not (a.dagger and a.distill), "--dagger and --distill are separate plans"
    assert a.dagger or a.distill or not a.half, "--half needs --dagger or --distill"
    assert a.dagger or a.distill or not a.memory_log, "--memory-log needs --dagger or --distill"
    assert a.distill or not (a.extra or a.part is not None), "--extra / --part need --distill"
    assert a.extra or a.part is None, "--part needs --extra"
    assert not a.extra or a.part is not None, "--extra needs --part K (1 for the round's first top-up, 2 for the next, ...)"
    if a.distill:
        assert a.checkpoint, "--distill needs --checkpoint (044-R1: d1 = the 041-R29 (4) selection S, d2 = the d1 model)"
        write_distill(a.out_dir, rnd, a.checkpoint, a.out, a.half, a.name, a.extra, a.part or 1)
        return
    if a.dagger:
        write_dagger(a.out_dir, rnd, a.checkpoint or DAGGER_CHECKPOINT, a.out, a.half, a.memory_log)
        return
    if a.first_game is not None:
        assert a.botgen and a.name, "--first-game needs --botgen and --name"
        write_botgen_extra(a.out_dir, a.first_game, a.name)
        return
    if a.botgen:
        write_botgen(a.out_dir)
        return
    name = a.name or ("teacher-check" if a.teacher else "rush5-check")
    out = a.out or (TEACHER_OUT if a.teacher else OUT)
    out.write_bytes(render(None if a.teacher else (a.checkpoint or CHECKPOINT), name).encode("utf-8"))
    print(f"{out}: {len(OPPONENTS) * len(MAPS)} matchups x {GAMES} games")
