import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

import pytest

from eval_helpers import ROOT, opponent

import rush_check_plan as rcp

PLANS = ROOT / "config" / "plans"

TEACHER = {"kind": "t0", "race": "terran", "build": "terran_tp"}
MAPS = {"fighting_spirit", "circuit_breaker"}
T0 = {("t0", "standard", "protoss"), ("t0", "standard", "terran")}
T1 = {("oss", "ualbertabot", "protoss"), ("oss", "stardust", "protoss"), ("oss", "ualbertabot", "terran"), ("oss", "iron", "terran")}
TRACKED = {True: PLANS / "teacher_tp_check.toml", False: PLANS / "teacher_botgen_tp.toml"}


def _load(p: Path) -> dict:
    return tomllib.loads(p.read_text(encoding="utf-8"))


def _text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


@pytest.mark.parametrize("check", [True, False])
def test_tp_plan_composition_022_r11_r12(check):
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
        games[(opponent(m["p1"]), m["map"])] += m["games"]
    assert games == Counter({**{(o, mp): t0 for o in T0 for mp in MAPS}, **{(o, mp): t1 for o in T1 for mp in MAPS}})
    assert sum(games.values()) == total == len(plan["seeds"])
    text, seeds = rcp.render_tp(check, plan["name"])
    assert TRACKED[check].read_text(encoding="utf-8") == text and plan["seeds"] == seeds


def test_tp_seeds_disjoint_022_r11_r12():
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
    assert _text(rcp.write_tp(tmp_path, False)) == _text(TRACKED[False])


def test_tp_cli_takes_only_out_dir_022_r11(tmp_path):
    script = ROOT / "tools" / "eval" / "rush_check_plan.py"
    run = lambda *a: subprocess.run([sys.executable, str(script), *a, "--out-dir", str(tmp_path)], capture_output=True, text=True)  # noqa: E731
    for bad in (["--teacher-tp", "--botgen-tp"], ["--teacher-tp", "--checkpoint", "c"], ["--botgen-tp", "--half"], ["--botgen-tp", "--name", "n"]):
        assert run(*bad).returncode != 0, bad
    ok = run("--botgen-tp")
    assert ok.returncode == 0, ok.stderr
    assert _text(tmp_path / "teacher_botgen_tp.toml") == _text(TRACKED[False])


@pytest.mark.parametrize("extra", [{"P": 16}, {"T": 224}, {"P": 400, "T": 96}, {"P": rcp.TP_EXTRA_MAX}])
def test_tp_botgen_topup_composition_044_r8(tmp_path, extra):
    out = rcp.write_tp(tmp_path, False, extra=extra, part=2)
    assert out == tmp_path / "teacher_botgen_tp_x2.toml"
    plan = _load(out)
    assert plan["name"] == "teacher-botgen-tp-x2" and plan["max_frames"] == 14400 and plan["save_replay"] is True
    games = Counter()
    for m in plan["matchup"]:
        assert m["map"] in MAPS and m["p0"] == TEACHER and m["games"] > 0
        assert m.get("p1_random_ratio", 0) == 0 and m["p1"]["race"] in ("protoss", "terran") and "build" not in m["p1"]
        games[(opponent(m["p1"]), m["map"])] += m["games"]
    want = Counter()
    for race, letter in (("protoss", "P"), ("terran", "T")):
        n = extra.get(letter, 0)
        for o in T0 | T1:
            if o[2] == race and n:
                want.update({(o, mp): (150 if o in T0 else 25) * n // 400 for mp in MAPS})
    assert games == want
    assert sum(games.values()) == sum(extra.values()) == len(plan["seeds"])
    for race, letter in (("protoss", "P"), ("terran", "T")):
        assert sum(g for ((_, _, r), _), g in games.items() if r == race) == extra.get(letter, 0), "only the short race(s)"


def test_tp_botgen_topup_seeds_044_r8(tmp_path):
    seeds = {}
    for part in range(1, rcp.TP_EXTRA_PARTS + 1):
        _, s = rcp.render_tp(False, "x", extra={"P": 496, "T": 496}, part=part)
        first = 10_000 + (part - 1) * 1_000 + 1
        assert s == [rcp.splitmix64(1_000_000 + n) >> 32 for n in range(first, first + 992)]
        seeds[part] = set(s)
        assert set(s) <= rcp.tp_extra_block(part)
        assert not rcp.reserved_seeds(skip_tp_extra_part=part) & rcp.tp_extra_block(part)
    allp = set().union(*seeds.values())
    assert len(allp) == rcp.TP_EXTRA_PARTS * 992, "parts disjoint"
    tracked = set()
    for p in PLANS.rglob("*.toml"):
        if not p.name.startswith("teacher_botgen_tp_x"):
            tracked |= rcp.plan_seeds(p)
    assert not tracked & set().union(*(rcp.tp_extra_block(k) for k in range(1, 11)))
    for bad in (dict(extra={"P": 30}), dict(extra={"P": 512, "T": 496}), dict(extra={"P": 0}), dict(extra={"Z": 16}),
                dict(extra={"P": 16}, part=0), dict(extra={"T": 16}, part=11)):
        with pytest.raises(AssertionError):
            rcp.render_tp(False, "x", **bad)
    with pytest.raises(AssertionError):
        rcp.render_tp(True, "x", extra={"P": 16})
    out = rcp.write_tp(tmp_path, False, extra={"P": 64}, part=1)
    assert rcp.write_tp(tmp_path, False, extra={"P": 64}, part=1) == out
    with pytest.raises(AssertionError, match="other seeds"):
        rcp.write_tp(tmp_path, False, extra={"P": 64, "T": 32}, part=1)
    for other in ({"T": 64}, {"P": 32, "T": 32}):
        with pytest.raises(AssertionError, match="other matchups"):
            rcp.write_tp(tmp_path, False, extra=other, part=1)
    assert all(m["p1"]["race"] == "protoss" for m in _load(out)["matchup"]), "the played plan is left as it was"
    _, x2 = rcp.render_tp(False, "x", extra={"T": 64}, part=2)
    (tmp_path / "clash.toml").write_text(
        f'name = "c"\nseeds = [{x2[5]}]\n[[matchup]]\nmap = "fighting_spirit"\ngames = 1\n'
        'p0 = { kind = "t0", race = "terran" }\np1 = { kind = "t0", race = "protoss" }\n', encoding="utf-8")
    with pytest.raises(AssertionError, match="another plan's"):
        rcp.write_tp(tmp_path, False, extra={"T": 64}, part=2)


def test_tp_botgen_topup_cli_044_r8(tmp_path):
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / "tools" / "eval" / "rush_check_plan.py"), *a, "--out-dir", str(tmp_path)], capture_output=True, text=True)  # noqa: E731
    ok = run("--botgen-tp", "--extra-t", "64", "--part", "3")
    assert ok.returncode == 0, ok.stderr
    plan = _load(tmp_path / "teacher_botgen_tp_x3.toml")
    assert plan["name"] == "teacher-botgen-tp-x3" and all(m["p1"]["race"] == "terran" for m in plan["matchup"])
    for bad in (["--botgen-tp", "--extra-p", "64"], ["--botgen-tp", "--part", "1"], ["--teacher-tp", "--extra-p", "64", "--part", "1"],
                ["--botgen-tp", "--extra-p", "50", "--part", "1"], ["--botgen-tp", "--extra-t", "64", "--part", "11"], ["--botgen-tp", "--round", "1"],
                ["--botgen-tp", "--extra", "64", "--part", "1"], ["--dagger", "--extra-p", "64"]):
        r = run(*bad)
        assert r.returncode != 0, bad
