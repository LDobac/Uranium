import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

import pytest

from eval_helpers import ROOT, opponent
from uranium import _core

from uranium.measure import rush_check_plan as rcp

PLANS = ROOT / "config" / "plans"
TOOL = ROOT / "tools" / "eval" / "rush_check_plan.py"

CKPT = "checkpoints/m2-distill-d1/ckpt-placeholder"
MAPS = {"fighting_spirit", "circuit_breaker"}
FULL = {
    ("t0", "standard", "protoss"): 120,
    ("t0", "standard", "terran"): 60,
    ("oss", "ualbertabot", "protoss"): 30,
    ("oss", "stardust", "protoss"): 30,
    ("oss", "ualbertabot", "terran"): 30,
    ("oss", "iron", "terran"): 30,
}


def _load(p: Path) -> dict:
    return tomllib.loads(p.read_text(encoding="utf-8"))


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True, encoding="utf-8", errors="replace")


def test_teacher_round_plan_composition_044_r8(tmp_path):
    out = tmp_path / "t.toml"
    rcp.write_teacher_round(PLANS, CKPT, out)
    plan = _load(out)
    assert plan["name"] == "dagger-m2-t1"
    assert plan["max_frames"] == 14400 and plan["save_replay"] is True and plan["memory_log"] is True
    assert plan["zero_last_failed"] is False and "seed_base" not in plan and "same_seed" not in plan
    games = Counter()
    for m in plan["matchup"]:
        assert m["map"] in MAPS
        assert m["p0"] == {"kind": "model", "checkpoint": CKPT, "sample": "stochastic"}
        assert m.get("p1_random_ratio", 0) == 0 and m["p1"]["race"] in ("protoss", "terran"), "no Random, no Zerg"
        assert "build" not in m["p1"], "T0 standard builds only"
        games[(opponent(m["p1"]), m["map"])] += m["games"]
    assert games == Counter({(o, mp): g // 2 for o, g in FULL.items() for mp in MAPS})
    assert sum(games.values()) == 300 == len(plan["seeds"])
    assert _core.plan_models(str(out)) == [(CKPT, "stochastic")]
    follow_up = rcp.TEACHER_ROUND_FOLLOW_UP
    assert "--relabel terran_tp --games" in follow_up
    assert "--until-frame 14400 --time-limit 14400" in follow_up and "11520" not in follow_up


def test_teacher_round_seeds_disjoint_044_r8():
    text, t1 = rcp.render_teacher_round(CKPT, "x")
    assert t1 == [rcp.splitmix64(3_000_000 + n) >> 32 for n in range(1, 301)]
    assert len(set(t1)) == 300
    block = {rcp.splitmix64(3_000_000 + n) >> 32 for n in range(1, 601)}
    assert len(block) == 600
    tracked = set()
    for p in PLANS.rglob("*.toml"):
        if not p.name.startswith("dagger_m2_t"):
            tracked |= rcp.plan_seeds(p)
    assert rcp.plan_seeds(PLANS / "protocol" / "t0_tp.toml") <= tracked
    assert not tracked & block
    others = rcp.reserved_seeds(skip_teacher_round=True)
    assert not others & block
    assert block <= rcp.reserved_seeds()
    assert set(range(77001, 77265)) <= others
    assert {rcp.splitmix64(70000 + n) >> 32 for n in (1, 6000)} <= others
    assert {rcp.splitmix64(80000 + n) >> 32 for n in (1, 6000, 100_001, 200_000)} <= others
    assert {rcp.splitmix64(1_000_000 + n) >> 32 for n in (10_001, 20_000)} <= others, "022 botgen top-up parts 1..10"
    assert (text, t1) == rcp.render_teacher_round(CKPT, "x")


def test_teacher_round_rejects_taken_seed_044_r8(tmp_path):
    _, seeds = rcp.render_teacher_round(CKPT, "x")
    sub = tmp_path / "protocol"
    sub.mkdir()
    (sub / "t0_x.toml").write_text(
        f'name = "x"\nseeds = [{seeds[17]}]\n[[matchup]]\nmap = "fighting_spirit"\ngames = 1\n'
        'p0 = { kind = "t0", race = "terran" }\np1 = { kind = "t0", race = "protoss" }\n', encoding="utf-8")
    with pytest.raises(AssertionError, match="another plan's"):
        rcp.write_teacher_round(tmp_path, CKPT)
    (sub / "t0_x.toml").unlink()
    out = rcp.write_teacher_round(tmp_path, CKPT)
    assert out == tmp_path / "dagger_m2_t1.toml"
    first = out.read_bytes()
    assert rcp.write_teacher_round(tmp_path, "checkpoints/other/ckpt-2").exists(), "same seeds, other model: fine"
    assert rcp.write_teacher_round(tmp_path, CKPT).read_bytes() == first
    (tmp_path / "dagger_m2_t1.toml").write_text(
        'name = "x"\nseeds = [1]\n[[matchup]]\nmap = "fighting_spirit"\ngames = 1\n'
        'p0 = { kind = "t0", race = "terran" }\np1 = { kind = "t0", race = "protoss" }\n', encoding="utf-8")
    with pytest.raises(AssertionError, match="other seeds"):
        rcp.write_teacher_round(tmp_path, CKPT)
    (tmp_path / "dagger_m2_t1.toml").unlink()
    rcp.write_distill(tmp_path, 1, CKPT, None)
    rcp.write_distill(tmp_path, 2, CKPT, None, half=True)
    rcp.write_tp(tmp_path, True)
    rcp.write_tp(tmp_path, False)
    rcp.write_tp(tmp_path, False, extra={"P": rcp.TP_EXTRA_MAX}, part=1)
    assert rcp.write_teacher_round(tmp_path, CKPT).read_bytes() == first


def test_teacher_round_cli_044_r8(tmp_path):
    r = _run("--teacher-round", "--memory-log", "--checkpoint", CKPT, "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    plan = _load(tmp_path / "dagger_m2_t1.toml")
    assert plan["name"] == "dagger-m2-t1" and len(plan["seeds"]) == 300
    bad = tmp_path / "bad.toml"
    for args in (("--teacher-round",), ("--teacher-round", "--round", "1", "--checkpoint", CKPT), ("--teacher-round", "--half", "--checkpoint", CKPT),
                 ("--teacher-round", "--distill", "--checkpoint", CKPT), ("--teacher-round", "--dagger", "--checkpoint", CKPT),
                 ("--teacher-round", "--extra", "32", "--part", "1", "--checkpoint", CKPT), ("--teacher-tp", "--teacher-round"),
                 ("--teacher-round", "--botgen", "--checkpoint", CKPT)):
        r = _run(*args, "--out", str(bad), "--out-dir", str(tmp_path))
        assert r.returncode != 0 and not bad.exists(), args
    r = _run("--distill", "--checkpoint", CKPT, "--out", str(tmp_path / "d1.toml"), "--out-dir", str(PLANS))
    assert r.returncode == 0, r.stderr
    assert _load(tmp_path / "d1.toml")["name"] == "distill-m2-d1"


GN = "checkpoints/m2-gn-t1e2/ckpt-2844"
Z_FULL = {
    ("t0", "zerg_rush", "zerg"): 40,
    ("oss", "ualbertabot12poollings", "zerg"): 40,
    ("oss", "ualbertabot9pool10lings", "zerg"): 40,
    ("oss", "ualbertabot9poolhydra", "zerg"): 40,
    ("oss", "ualbertabot2hatchhydra", "zerg"): 28,
    ("oss", "ualbertabot3hatchmuta", "zerg"): 28,
    ("oss", "steamhammer2hatchmuta", "zerg"): 28,
    ("t0", "zerg_muta", "zerg"): 28,
    ("t0", "standard", "zerg"): 28,
}


def _games(plan: dict, ckpt: str) -> Counter:
    games = Counter()
    for m in plan["matchup"]:
        assert m["map"] in MAPS
        assert m["p0"] == {"kind": "model", "checkpoint": ckpt, "sample": "stochastic"}
        assert m.get("p1_random_ratio", 0) == 0 and m["p1"]["race"] != "random", "no Random"
        games[(opponent(m["p1"]), m["map"])] += m["games"]
    return games


def _same_text(a: Path, b: Path) -> bool:
    return a.read_bytes().replace(b"\r\n", b"\n") == b.read_bytes().replace(b"\r\n", b"\n")


def test_teacher_round_t2_composition_044_r8(tmp_path):
    out = rcp.write_teacher_round(PLANS, GN, tmp_path / "t2.toml", rnd="t2")
    plan, t1 = _load(out), _load(PLANS / "dagger_m2_t1.toml")
    assert plan["name"] == "dagger-m2-t2"
    assert plan["max_frames"] == 14400 and plan["save_replay"] is True and plan["memory_log"] is True and plan["zero_last_failed"] is False
    assert [{k: m[k] for k in ("map", "games", "p1")} for m in plan["matchup"]] == [{k: m[k] for k in ("map", "games", "p1")} for m in t1["matchup"]]
    assert _games(plan, GN) == Counter({(o, mp): g // 2 for o, g in FULL.items() for mp in MAPS})
    assert all(m["p1"]["race"] in ("protoss", "terran") for m in plan["matchup"])
    assert plan["seeds"] == [rcp.splitmix64(3_000_000 + n) >> 32 for n in range(301, 601)]
    assert not set(plan["seeds"]) & set(t1["seeds"])
    assert _core.plan_models(str(out)) == [(GN, "stochastic")]
    assert _same_text(out, PLANS / "dagger_m2_t2.toml"), "the committed plan is the generator's"
    follow_up = rcp.teacher_round_follow_up("t2")
    assert "--relabel terran_tp --games <dir>/games.jsonl" in follow_up and "--until-frame 14400 --time-limit 14400" in follow_up
    assert follow_up.endswith("shards/dagger_m2/t2") and rcp.teacher_round_follow_up("t1") == rcp.TEACHER_ROUND_FOLLOW_UP


@pytest.mark.parametrize("n", [150, 600, 1350])
def test_teacher_round_t2_zerg_composition_044_r8(tmp_path, n):
    out = rcp.write_teacher_round(tmp_path, GN, rnd="t2", zerg=n)
    plan = _load(out)
    assert plan["name"] == "dagger-m2-t2-z"
    assert plan["max_frames"] == 14400 and plan["save_replay"] is True and plan["memory_log"] is True and plan["zero_last_failed"] is False
    assert all(m["p1"]["race"] == "zerg" for m in plan["matchup"])
    assert _games(plan, GN) == Counter({(o, mp): g * n // 600 for o, g in Z_FULL.items() for mp in MAPS})
    r1 = _load(PLANS / "dagger_m2_r1.toml")
    assert [(m["map"], m["p1"]) for m in plan["matchup"]] == [(m["map"], m["p1"]) for m in r1["matchup"]], "the DAgger round opponents in their order"
    assert plan["seeds"] == [rcp.splitmix64(5_000_000 + k) >> 32 for k in range(1, n + 1)] and len(set(plan["seeds"])) == n


def test_teacher_round_t2_zerg_seeds_044_r8(tmp_path):
    assert rcp.TEACHER_ROUND_Z_UNIT == 150
    committed = PLANS / "dagger_m2_t2_z.toml"
    n = len(_load(committed)["seeds"]) if committed.exists() else len(rcp.TEACHER_ROUND_Z_BLOCK)
    assert n % rcp.TEACHER_ROUND_Z_UNIT == 0
    out = rcp.write_teacher_round(PLANS, GN, tmp_path / "z.toml", rnd="t2", zerg=n)
    if committed.exists():
        assert _same_text(out, committed), "the committed plan is the generator's"
    block = rcp.teacher_round_z_block()
    assert block == {rcp.splitmix64(5_000_000 + k) >> 32 for k in range(1, 3001)} and len(block) == 3000
    tracked = set()
    for p in PLANS.rglob("*.toml"):
        if p.name != "dagger_m2_t2_z.toml":
            tracked |= rcp.plan_seeds(p)
    assert not tracked & block
    others = rcp.reserved_seeds(skip_teacher_round_z=True)
    assert not others & block and block <= rcp.reserved_seeds() and block <= rcp.reserved_seeds(skip_teacher_round=True)
    assert {rcp.splitmix64(3_000_000 + k) >> 32 for k in (1, 600)} <= others
    for bad in (0, 100, 160, 3150):
        with pytest.raises(AssertionError, match="--zerg must be a multiple of 150"):
            rcp.render_teacher_round_z(GN, "x", bad)
    with pytest.raises(AssertionError, match="--zerg goes with --round t2"):
        rcp.render_teacher_round_z(GN, "x", 150, "t1")
    own = tmp_path / "own"
    own.mkdir()
    rcp.write_teacher_round(own, GN, rnd="t2", zerg=300)
    with pytest.raises(AssertionError, match="other seeds"):
        rcp.write_teacher_round(own, GN, rnd="t2", zerg=600)
    assert rcp.write_teacher_round(own, "checkpoints/other/ckpt-2", rnd="t2", zerg=300).exists(), "same seeds, other model: fine"
    t1 = rcp.write_teacher_round(own, CKPT)
    t2 = rcp.write_teacher_round(own, GN, rnd="t2")
    assert not rcp.plan_seeds(t1) & rcp.plan_seeds(t2)


def test_teacher_round_t2_zerg_follow_up_022_r2():
    z = rcp.teacher_round_follow_up("t2", zerg=True)
    assert "--relabel terran_react" in z and "--games" not in z and "terran_tp" not in z
    assert "--memory-log <dir>/memory --until-frame 14400 --time-limit 14400" in z and z.endswith("shards/dagger_m2/t2z")


def test_teacher_round_t2_cli_044_r8(tmp_path):
    r = _run("--teacher-round", "--round", "t2", "--checkpoint", GN, "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert _load(tmp_path / "dagger_m2_t2.toml")["name"] == "dagger-m2-t2" and "dagger_m2/t2" in r.stdout
    r = _run("--teacher-round", "--round", "t2", "--zerg", "150", "--checkpoint", GN, "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert len(_load(tmp_path / "dagger_m2_t2_z.toml")["seeds"]) == 150 and "dagger_m2/t2z" in r.stdout
    bad = tmp_path / "bad.toml"
    for args in (("--teacher-round", "--round", "t3", "--checkpoint", GN), ("--teacher-round", "--zerg", "150", "--checkpoint", GN),
                 ("--teacher-round", "--round", "t2", "--zerg", "100", "--checkpoint", GN), ("--dagger", "--zerg", "150"),
                 ("--distill", "--round", "t2", "--checkpoint", GN), ("--teacher-round", "--round", "t2", "--zerg", "-150", "--checkpoint", GN)):
        r = _run(*args, "--out", str(bad), "--out-dir", str(tmp_path))
        assert r.returncode != 0 and not bad.exists(), args
