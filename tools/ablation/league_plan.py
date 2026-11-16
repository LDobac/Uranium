import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "config" / "plans" / "arch-abl-pt-league.toml"
MODELS = [
    "core-abl-gtrxl-pt",
    "core-abl-lstm-pt",
    "arch-abl-terrain-pt",
    "arch-abl-ctx-pt",
    "arch-abl-pool-pt",
    "arch-abl-outer-pt",
    "arch-abl-all-pt",
]
REUSED = {("core-abl-gtrxl-pt", "core-abl-lstm-pt")}
MAPS = ("fighting_spirit", "circuit_breaker")
GAMES = 25
SEED_BASE = 35000


def pairs():
    return [p for p in itertools.combinations(MODELS, 2) if p not in REUSED]


def player(run: str) -> str:
    return f'{{ kind = "model", checkpoint = "checkpoints/{run}/ckpt-10000", sample = "stochastic" }}'


def render() -> str:
    lines = [
        'name = "arch-abl-pt-league"',
        f"seed_base = {SEED_BASE}",
        "save_replay = true",
        "wall_timeout_s = 7200",
        "zero_last_failed = true",
    ]
    for a, b in pairs():
        for m in MAPS:
            for p0, p1 in ((a, b), (b, a)):
                lines += ["[[matchup]]", f'map = "{m}"', f"games = {GAMES}", f"p0 = {player(p0)}", f"p1 = {player(p1)}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUT.write_bytes(render().replace("\n", "\r\n").encode("utf-8"))
    print(f"{OUT.relative_to(ROOT)}: {len(pairs())} pairs, {len(pairs()) * 4 * GAMES} games")
