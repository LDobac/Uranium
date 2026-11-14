"""044-R1 (AMB-044-3 / AMB-044-4): the self-game distillation plan generator
`tools/eval/rush_check_plan.py --distill` (CPU, no game): composition, 10-minute cap, decision logs, no
Random, deterministic seeds disjoint from every tracked plan (050-R9 protocol templates included) and
between rounds, the d2 `--half` plan, and the runner's plan loader."""

import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

import pytest

from uranium import _core

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "config" / "plans"
sys.path.insert(0, str(ROOT / "tools" / "eval"))

import rush_check_plan as rcp  # noqa: E402

CKPT = "checkpoints/s-placeholder/ckpt-1"
MAPS = {"fighting_spirit", "circuit_breaker"}
# 044-R1: (p1 kind, bot name or T0 race build, race) -> games per map of a full round d1.
FULL = {
    ("t0", "standard", "protoss"): 120,
    ("t0", "standard", "terran"): 60,
    ("oss", "ualbertabot", "protoss"): 30,
    ("oss", "stardust", "protoss"): 30,
    ("oss", "ualbertabot", "terran"): 30,
    ("oss", "iron", "terran"): 30,
}


def _write(tmp_path: Path, rnd: int, half: bool) -> tuple[Path, dict]:
    out = tmp_path / f"d{rnd}{'h' if half else ''}.toml"
    rcp.write_distill(PLANS, rnd, CKPT, out, half)
    return out, tomllib.loads(out.read_text(encoding="utf-8"))


def _opponent(p1: dict) -> tuple[str, str, str]:
    return (p1["kind"], p1.get("build", "standard") if p1["kind"] == "t0" else p1["name"], p1["race"])


@pytest.mark.parametrize("half", [False, True])
def test_distill_plan_composition_044_r1(tmp_path, half):
    """044-R1: d1 600 games (T0 P 2 x 120, T0 T 2 x 60, UAlbertaBot P / Stardust / UAlbertaBot T / Iron 2 x 30;
    Protoss : Terran = 3 : 2), d2 `--half` 300 of the same composition; the model stochastic as players[0],
    10 min, replays and decision logs, races fixed, no E-Regression builds or McRave."""
    _, plan = _write(tmp_path, 1, half)
    div = 2 if half else 1
    assert plan["name"] == "distill-m2-d1"
    assert plan["max_frames"] == 14400 and plan["save_replay"] is True and plan["memory_log"] is True
    assert plan["zero_last_failed"] is False and "seed_base" not in plan and "same_seed" not in plan
    games = Counter()
    for m in plan["matchup"]:
        assert m["map"] in MAPS
        assert m["p0"] == {"kind": "model", "checkpoint": CKPT, "sample": "stochastic"}
        assert m.get("p1_random_ratio", 0) == 0 and m["p1"]["race"] in ("protoss", "terran")
        assert "build" not in m["p1"], "T0 standard builds only (no E-Regression scenario build)"
        games[(_opponent(m["p1"]), m["map"])] += m["games"]
    assert games == Counter({(o, mp): g // div for o, g in FULL.items() for mp in MAPS})
    total = sum(games.values())
    assert total == 600 // div == len(plan["seeds"])
    protoss = sum(g for ((_, _, race), _), g in games.items() if race == "protoss")
    assert protoss * 2 == (total - protoss) * 3, "Protoss : Terran = 3 : 2"
    # The runner's loader (020) reads it and finds the one model.
    assert _core.plan_models(str(tmp_path / ("d1h.toml" if half else "d1.toml"))) == [(CKPT, "stochastic")]


def test_distill_seeds_disjoint_044_r1(tmp_path):
    """044-R1: seeds are the game-number hash (deterministic), d2 half = n 601..900 (round stride 600), and
    none is a seed of any tracked plan (botgen, check, dagger, the 050-R9 protocol templates, in particular
    t0_tp's T0 standard Terran / Protoss 32 games) or of the other round."""
    _, d1 = _write(tmp_path, 1, False)
    _, d2 = _write(tmp_path, 2, True)
    s1, s2 = d1["seeds"], d2["seeds"]
    assert s1 == [rcp.splitmix64(80000 + n) >> 32 for n in range(1, 601)]
    assert s2 == [rcp.splitmix64(80000 + n) >> 32 for n in range(601, 901)]
    assert len(set(s1)) == 600 and len(set(s2)) == 300 and not set(s1) & set(s2)
    tracked = set()
    for p in PLANS.rglob("*.toml"):
        if not p.name.startswith("distill_m2_d"):  # a generated round plan is these seeds themselves
            tracked |= rcp.plan_seeds(p)
    t0_tp = rcp.plan_seeds(PLANS / "protocol" / "t0_tp.toml")
    assert len(t0_tp) == 64 and t0_tp <= tracked
    assert not tracked & (set(s1) | set(s2))
    # Deterministic: the same round renders the same text.
    assert rcp.render_distill(1, CKPT, "x", False) == rcp.render_distill(1, CKPT, "x", False)


def test_distill_rejects_taken_seed_044_r1(tmp_path):
    """044-R1: a plan anywhere under --out-dir (subfolders too, like config/plans/protocol) that uses a seed of
    the round stops the generator; the round's own output file is not counted against itself."""
    _, seeds = rcp.render_distill(1, CKPT, "x", False)
    sub = tmp_path / "protocol"
    sub.mkdir()
    (sub / "t0_x.toml").write_text(
        f'name = "x"\nseeds = [{seeds[17]}]\n[[matchup]]\nmap = "fighting_spirit"\ngames = 1\n'
        'p0 = { kind = "t0", race = "terran" }\np1 = { kind = "t0", race = "protoss" }\n', encoding="utf-8")
    with pytest.raises(AssertionError, match="another plan's"):
        rcp.write_distill(tmp_path, 1, CKPT, None)
    (sub / "t0_x.toml").unlink()
    out = rcp.write_distill(tmp_path, 1, CKPT, None)
    assert out == tmp_path / "distill_m2_d1.toml"
    first = out.read_bytes()
    assert rcp.write_distill(tmp_path, 1, CKPT, None).read_bytes() == first, "rewriting its own file is fine"


@pytest.mark.parametrize("extra", [20, 120, 1000])
def test_distill_topup_composition_044_r5(tmp_path, extra):
    """044-R5 ③ top-up (user decision 2026-11-14): `--extra N` writes N more games of the round with the same
    model and the exact 044-R1 composition (every block x N / 600), the same 10 min / replays / decision logs,
    plan name distill-m2-d<R>-x<K>."""
    out = tmp_path / "x.toml"
    rcp.write_distill(PLANS, 1, CKPT, out, extra=extra, part=2)
    plan = tomllib.loads(out.read_text(encoding="utf-8"))
    assert plan["name"] == "distill-m2-d1-x2"
    assert plan["max_frames"] == 14400 and plan["save_replay"] is True and plan["memory_log"] is True
    assert plan["zero_last_failed"] is False, "044-R1: the model sees its last failed command, as DAgger"
    games = Counter()
    for m in plan["matchup"]:
        assert m["p0"] == {"kind": "model", "checkpoint": CKPT, "sample": "stochastic"}
        assert m.get("p1_random_ratio", 0) == 0 and "build" not in m["p1"]
        games[(_opponent(m["p1"]), m["map"])] += m["games"]
    assert games == Counter({(o, mp): g * extra // 600 for o, g in FULL.items() for mp in MAPS})
    assert sum(games.values()) == extra == len(plan["seeds"])
    assert _core.plan_models(str(out)) == [(CKPT, "stochastic")]


def test_distill_topup_seeds_disjoint_044_r5(tmp_path):
    """044-R5 ③ top-up: the seeds are the game-number hash of their own block n = 100,000 + (R - 1) * 10,000 +
    (K - 1) * 1,000 + i, disjoint from every round block (d1 n 1..600, d2 601..900, up to n 6,000), from the
    other parts and rounds' top-ups and from every tracked plan; N not a multiple of 20 (the ratio) or above
    1,000, `--half` with `--extra` and a part below 1 are refused; a plan under --out-dir that takes one of its
    seeds stops the generator."""
    seeds = {}
    for rnd, part in ((1, 1), (1, 2), (2, 1)):
        out = tmp_path / f"x{rnd}{part}.toml"
        rcp.write_distill(PLANS, rnd, CKPT, out, extra=1000, part=part)
        s = tomllib.loads(out.read_text(encoding="utf-8"))["seeds"]
        first = 100000 + (rnd - 1) * 10000 + (part - 1) * 1000 + 1
        assert s == [rcp.splitmix64(80000 + n) >> 32 for n in range(first, first + 1000)]
        seeds[(rnd, part)] = set(s)
    assert len(set().union(*seeds.values())) == 3000, "parts and rounds disjoint"
    rounds = {rcp.splitmix64(80000 + n) >> 32 for n in range(1, 6001)}
    _, d1 = rcp.render_distill(1, CKPT, "x", False)
    _, d2 = rcp.render_distill(2, CKPT, "x", True)
    assert set(d1) | set(d2) <= rounds and not rounds & set().union(*seeds.values())
    tracked = set()
    for p in PLANS.rglob("*.toml"):
        if not p.name.startswith("distill_m2_d"):
            tracked |= rcp.plan_seeds(p)
    assert not tracked & set().union(*seeds.values())
    for bad in (dict(extra=30), dict(extra=1020), dict(extra=20, part=0)):
        with pytest.raises(AssertionError):
            rcp.render_distill(1, CKPT, "x", False, **bad)
    with pytest.raises(AssertionError, match="separate plans"):
        rcp.render_distill(1, CKPT, "x", True, extra=20)
    # The round plan and an earlier part under --out-dir are scanned; its own file is not counted.
    d = tmp_path / "plans"
    d.mkdir()
    rcp.write_distill(d, 1, CKPT, None)
    out = rcp.write_distill(d, 1, CKPT, None, extra=40)
    assert out == d / "distill_m2_d1_x1.toml" and rcp.write_distill(d, 1, CKPT, None, extra=40) == out
    _, x1 = rcp.render_distill(1, CKPT, "x", False, extra=40, part=1)
    (d / "clash.toml").write_text(
        f'name = "c"\nseeds = [{x1[5]}]\n[[matchup]]\nmap = "fighting_spirit"\ngames = 1\n'
        'p0 = { kind = "t0", race = "terran" }\np1 = { kind = "t0", race = "protoss" }\n', encoding="utf-8")
    with pytest.raises(AssertionError, match="another plan's"):
        rcp.write_distill(d, 1, CKPT, None, extra=40)


def test_distill_topup_cli_044_r5(tmp_path):
    """044-R5 ③ top-up: `--extra` / `--part` only with `--distill`, `--part` only with `--extra`."""
    tool = ROOT / "tools" / "eval" / "rush_check_plan.py"
    run = lambda *a: subprocess.run([sys.executable, str(tool), *a], capture_output=True, text=True, encoding="utf-8", errors="replace")  # noqa: E731
    r = run("--distill", "--round", "1", "--extra", "60", "--checkpoint", CKPT, "--out", str(tmp_path / "x.toml"))
    assert r.returncode == 0, r.stderr
    assert tomllib.loads((tmp_path / "x.toml").read_text(encoding="utf-8"))["name"] == "distill-m2-d1-x1"
    for args in (("--dagger", "--extra", "60"), ("--distill", "--part", "2", "--checkpoint", CKPT), ("--distill", "--extra", "50", "--checkpoint", CKPT)):
        r = run(*args, "--out", str(tmp_path / "bad.toml"))
        assert r.returncode != 0 and not (tmp_path / "bad.toml").exists(), args


def test_plan_seeds_runner_rule_020_r20(tmp_path):
    """020-R20 (plan.rs `seed_of`): the seed scan counts a plan without `seeds` / `seed_base` as the default
    seed_base 1,000 + n, and `same_seed` as seed_base alone."""
    body = '[[matchup]]\nmap = "fighting_spirit"\ngames = 3\np0 = { kind = "t0", race = "terran" }\np1 = { kind = "t0", race = "zerg" }\n'
    p = tmp_path / "p.toml"
    p.write_text('name = "x"\n' + body, encoding="utf-8")
    assert rcp.plan_seeds(p) == {1001, 1002, 1003}
    p.write_text('name = "x"\nseed_base = 10\nsame_seed = true\n' + body, encoding="utf-8")
    assert rcp.plan_seeds(p) == {10}
    p.write_text('name = "x"\nseed_base = 10\n' + body, encoding="utf-8")
    assert rcp.plan_seeds(p) == {11, 12, 13}


def test_distill_cli_needs_checkpoint_044_r1(tmp_path):
    """044-R1: the model is S (d1) or the d1 model (d2): `--distill` has no default checkpoint."""
    tool = ROOT / "tools" / "eval" / "rush_check_plan.py"
    r = subprocess.run([sys.executable, str(tool), "--distill", "--out", str(tmp_path / "x.toml")], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode != 0 and "--checkpoint" in r.stderr and not (tmp_path / "x.toml").exists()
    r = subprocess.run([sys.executable, str(tool), "--distill", "--round", "2", "--half", "--memory-log", "--checkpoint", CKPT,
                        "--out", str(tmp_path / "d2.toml")], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr
    assert tomllib.loads((tmp_path / "d2.toml").read_text(encoding="utf-8"))["name"] == "distill-m2-d2"
