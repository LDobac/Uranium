"""022-R11 / 022-R12 (AMB-022-1 / AMB-022-7): the 022 TvT / TvP teacher's plan generators
`tools/eval/rush_check_plan.py --teacher-tp` (check plan) and `--botgen-tp` (botgen plan) (CPU, no game):
composition, caps, no Random, the tracked plans are the generator's output, and the seeds are the game-number hash,
disjoint from each other, from every tracked plan (050-R9 protocol templates included), from the DAgger /
distillation round blocks not generated yet and from the teacher's development seeds 77,001..77,264."""

import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "config" / "plans"
sys.path.insert(0, str(ROOT / "tools" / "eval"))

import rush_check_plan as rcp  # noqa: E402

TEACHER = {"kind": "t0", "race": "terran", "build": "terran_tp"}
MAPS = {"fighting_spirit", "circuit_breaker"}
T0 = {("t0", "standard", "protoss"), ("t0", "standard", "terran")}
T1 = {("oss", "ualbertabot", "protoss"), ("oss", "stardust", "protoss"), ("oss", "ualbertabot", "terran"), ("oss", "iron", "terran")}
TRACKED = {True: PLANS / "teacher_tp_check.toml", False: PLANS / "teacher_botgen_tp.toml"}


def _opponent(p1: dict) -> tuple[str, str, str]:
    return (p1["kind"], p1.get("build", "standard") if p1["kind"] == "t0" else p1["name"], p1["race"])


def _load(p: Path) -> dict:
    return tomllib.loads(p.read_text(encoding="utf-8"))


def _text(p: Path) -> str:
    """A plan file's text with universal newlines (a Windows checkout of a tracked plan has CRLF)."""
    return p.read_text(encoding="utf-8")


@pytest.mark.parametrize("check", [True, False])
def test_tp_plan_composition_022_r11_r12(check):
    """022-R11: T0 standard Protoss / Terran 2 maps x 32 + T1 UAlbertaBot P, Stardust, UAlbertaBot T, Iron 2 maps x 10
    = 208 games, 60 min. 022-R12: the same opponents x 150 / x 25 = 800 games, 10 min. The teacher terran_tp is
    players[0], races fixed (no Random, 022-R2), T0 standard builds only, replays kept, no model."""
    plan = _load(TRACKED[check])
    t0, t1, total, frames = (32, 10, 208, 86400) if check else (150, 25, 800, 14400)
    assert plan["name"] == ("teacher-tp-check" if check else "teacher-botgen-tp")
    assert plan["max_frames"] == frames and plan["save_replay"] is True
    assert "seed_base" not in plan and "same_seed" not in plan
    games = Counter()
    for m in plan["matchup"]:
        assert m["map"] in MAPS and m["p0"] == TEACHER
        assert m.get("p1_random_ratio", 0) == 0 and m["p1"]["race"] in ("protoss", "terran")
        assert "build" not in m["p1"], "T0 standard builds only"
        games[(_opponent(m["p1"]), m["map"])] += m["games"]
    assert games == Counter({**{(o, mp): t0 for o in T0 for mp in MAPS}, **{(o, mp): t1 for o in T1 for mp in MAPS}})
    assert sum(games.values()) == total == len(plan["seeds"])
    # The tracked file is the generator's output byte for byte.
    text, seeds = rcp.render_tp(check, plan["name"])
    assert TRACKED[check].read_text(encoding="utf-8") == text and plan["seeds"] == seeds


def test_tp_seeds_disjoint_022_r11_r12():
    """022-R11 / R12: seeds are splitmix64(1,000,000 + n) >> 32 (check n = 1..208, botgen n = 1,001..1,800), the two
    plans share none, and none is a seed of another tracked plan (botgen, check, dagger, the 050-R9 protocol
    templates), of the DAgger / distillation rounds 1..10 and top-ups not generated yet, or a development seed."""
    check, botgen = (_load(TRACKED[c])["seeds"] for c in (True, False))
    assert check == [rcp.splitmix64(1_000_000 + n) >> 32 for n in range(1, 209)]
    assert botgen == [rcp.splitmix64(1_000_000 + n) >> 32 for n in range(1001, 1801)]
    assert len(set(check)) == 208 and len(set(botgen)) == 800 and not set(check) & set(botgen)
    tracked = set()
    for p in PLANS.rglob("*.toml"):
        if p not in TRACKED.values():
            tracked |= rcp.plan_seeds(p)
    t0_tp = rcp.plan_seeds(PLANS / "protocol" / "t0_tp.toml")
    assert len(t0_tp) == 64 and t0_tp <= tracked
    assert not tracked & (set(check) | set(botgen))
    reserved = rcp.reserved_seeds()
    assert set(range(77001, 77265)) <= reserved
    dev_tune = {rcp.splitmix64(1_000_000 + n) >> 32 for n in range(5001, 5133)}
    assert len(dev_tune) == 132 and dev_tune <= reserved, "022 tuning development seeds (2026-11-14)"
    assert not dev_tune & (set(check) | set(botgen) | tracked)
    assert {rcp.splitmix64(70000 + n) >> 32 for n in (1, 6000)} <= reserved, "DAgger rounds 1..10"
    assert {rcp.splitmix64(80000 + n) >> 32 for n in (1, 6000, 100_001, 200_000)} <= reserved, "distillation rounds and top-ups"
    assert not reserved & (set(check) | set(botgen))


def test_tp_generator_rejects_taken_seed_022_r11(tmp_path):
    """022-R11: a plan anywhere under --out-dir (subfolders too) that uses one of the plan's seeds stops the
    generator; its own output file is not counted against itself, and rewriting gives the same bytes."""
    _, seeds = rcp.render_tp(True, "x")
    sub = tmp_path / "protocol"
    sub.mkdir()
    (sub / "t0_x.toml").write_text(
        f'name = "x"\nseeds = [{seeds[40]}]\n[[matchup]]\nmap = "fighting_spirit"\ngames = 1\n'
        'p0 = { kind = "t0", race = "terran" }\np1 = { kind = "t0", race = "protoss" }\n', encoding="utf-8")
    with pytest.raises(AssertionError, match="another plan's"):
        rcp.write_tp(tmp_path, True)
    (sub / "t0_x.toml").unlink()
    out = rcp.write_tp(tmp_path, True)
    assert out == tmp_path / "teacher_tp_check.toml"
    assert _text(out) == _text(TRACKED[True]), "the tracked plan is the generator output (line endings aside: git autocrlf)"
    assert rcp.write_tp(tmp_path, True).read_bytes() == out.read_bytes()
    # The botgen plan beside it: disjoint, written.
    assert _text(rcp.write_tp(tmp_path, False)) == _text(TRACKED[False])


def test_tp_cli_takes_only_out_dir_022_r11(tmp_path):
    """--teacher-tp / --botgen-tp have a fixed composition: other plan options are refused."""
    script = ROOT / "tools" / "eval" / "rush_check_plan.py"
    run = lambda *a: subprocess.run([sys.executable, str(script), *a, "--out-dir", str(tmp_path)], capture_output=True, text=True)  # noqa: E731
    for bad in (["--teacher-tp", "--botgen-tp"], ["--teacher-tp", "--checkpoint", "c"], ["--botgen-tp", "--half"], ["--botgen-tp", "--name", "n"]):
        assert run(*bad).returncode != 0, bad
    ok = run("--botgen-tp")
    assert ok.returncode == 0, ok.stderr
    assert _text(tmp_path / "teacher_botgen_tp.toml") == _text(TRACKED[False])


@pytest.mark.parametrize("extra", [32, 224, rcp.TP_EXTRA_MAX])
def test_tp_botgen_topup_composition_044_r8(tmp_path, extra):
    """044-R8 / 022-R12 top-up (user decision 2026-11-14): `--botgen-tp --extra N --part K` writes N more 022 teacher
    botgen games in the exact 022-R12 ratio (every block x N / 800), the teacher players[0], 10 min, no Random, plan
    name teacher-botgen-tp-x<K>; half of them against each race (mix-fractions' extra_botgen_tp_games rule)."""
    out = rcp.write_tp(tmp_path, False, extra=extra, part=2)
    assert out == tmp_path / "teacher_botgen_tp_x2.toml"
    plan = _load(out)
    assert plan["name"] == "teacher-botgen-tp-x2" and plan["max_frames"] == 14400 and plan["save_replay"] is True
    games = Counter()
    for m in plan["matchup"]:
        assert m["map"] in MAPS and m["p0"] == TEACHER
        assert m.get("p1_random_ratio", 0) == 0 and m["p1"]["race"] in ("protoss", "terran") and "build" not in m["p1"]
        games[(_opponent(m["p1"]), m["map"])] += m["games"]
    assert games == Counter({**{(o, mp): 150 * extra // 800 for o in T0 for mp in MAPS}, **{(o, mp): 25 * extra // 800 for o in T1 for mp in MAPS}})
    assert sum(games.values()) == extra == len(plan["seeds"])
    protoss = sum(g for ((_, _, race), _), g in games.items() if race == "protoss")
    assert protoss * 2 == extra, "half of the games against each race"


def test_tp_botgen_topup_seeds_044_r8(tmp_path):
    """044-R8 top-up: seeds are the game-number hash of the part's own block n = 10,000 + (K - 1) * 1,000 + i of the
    022 key, disjoint between parts, from the check / botgen plans, every tracked plan and the reserved blocks
    (the teacher round t1 included); N not a multiple of 32 or above 992 and a part outside 1..10 are refused; a
    plan under --out-dir that takes one of its seeds stops the generator; a part is never rewritten with other
    seeds."""
    seeds = {}
    for part in range(1, rcp.TP_EXTRA_PARTS + 1):
        _, s = rcp.render_tp(False, "x", extra=rcp.TP_EXTRA_MAX, part=part)
        first = 10_000 + (part - 1) * 1_000 + 1
        assert s == [rcp.splitmix64(1_000_000 + n) >> 32 for n in range(first, first + rcp.TP_EXTRA_MAX)]
        seeds[part] = set(s)
        assert set(s) <= rcp.tp_extra_block(part)
        assert not rcp.reserved_seeds(skip_tp_extra_part=part) & rcp.tp_extra_block(part)
    allp = set().union(*seeds.values())
    assert len(allp) == rcp.TP_EXTRA_PARTS * rcp.TP_EXTRA_MAX, "parts disjoint"
    tracked = set()
    for p in PLANS.rglob("*.toml"):
        if not p.name.startswith("teacher_botgen_tp_x"):
            tracked |= rcp.plan_seeds(p)
    assert not tracked & set().union(*(rcp.tp_extra_block(k) for k in range(1, 11)))
    for bad in (dict(extra=30), dict(extra=1024), dict(extra=32, part=0), dict(extra=32, part=11)):
        with pytest.raises(AssertionError):
            rcp.render_tp(False, "x", **bad)
    with pytest.raises(AssertionError):
        rcp.render_tp(True, "x", extra=32)
    out = rcp.write_tp(tmp_path, False, extra=64, part=1)
    assert rcp.write_tp(tmp_path, False, extra=64, part=1) == out
    with pytest.raises(AssertionError, match="other seeds"):
        rcp.write_tp(tmp_path, False, extra=96, part=1)
    _, x2 = rcp.render_tp(False, "x", extra=64, part=2)
    (tmp_path / "clash.toml").write_text(
        f'name = "c"\nseeds = [{x2[5]}]\n[[matchup]]\nmap = "fighting_spirit"\ngames = 1\n'
        'p0 = { kind = "t0", race = "terran" }\np1 = { kind = "t0", race = "protoss" }\n', encoding="utf-8")
    with pytest.raises(AssertionError, match="another plan's"):
        rcp.write_tp(tmp_path, False, extra=64, part=2)


def test_tp_botgen_topup_cli_044_r8(tmp_path):
    """044-R8 top-up: `--extra` needs `--part` (and back), both only with --botgen-tp (or --distill)."""
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / "tools" / "eval" / "rush_check_plan.py"), *a, "--out-dir", str(tmp_path)], capture_output=True, text=True)  # noqa: E731
    ok = run("--botgen-tp", "--extra", "64", "--part", "3")
    assert ok.returncode == 0, ok.stderr
    assert _load(tmp_path / "teacher_botgen_tp_x3.toml")["name"] == "teacher-botgen-tp-x3"
    for bad in (["--botgen-tp", "--extra", "64"], ["--botgen-tp", "--part", "1"], ["--teacher-tp", "--extra", "64", "--part", "1"],
                ["--botgen-tp", "--extra", "50", "--part", "1"], ["--botgen-tp", "--extra", "64", "--part", "11"], ["--botgen-tp", "--round", "1"]):
        r = run(*bad)
        assert r.returncode != 0, bad
