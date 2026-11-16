import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

import pytest

from uranium import _core

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "config" / "plans"
TOOL = ROOT / "tools" / "eval" / "rush_check_plan.py"
sys.path.insert(0, str(ROOT / "tools" / "eval"))

import rush_check_plan as rcp  # noqa: E402

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


def _opponent(p1: dict) -> tuple[str, str, str]:
    return (p1["kind"], p1.get("build", "standard") if p1["kind"] == "t0" else p1["name"], p1["race"])


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
        games[(_opponent(m["p1"]), m["map"])] += m["games"]
    assert games == Counter({(o, mp): g // 2 for o, g in FULL.items() for mp in MAPS})
    assert sum(games.values()) == 300 == len(plan["seeds"])
    assert _core.plan_models(str(out)) == [(CKPT, "stochastic")]
    guide = (ROOT / "docs" / "guide" / "commands.md").read_text(encoding="utf-8")
    assert "--relabel terran_tp --games" in guide
    section = guide[guide.index("### 교사 라운드 t1"):]
    section = section[:section.index("\n#", 1)]
    assert "--until-frame 14400 --time-limit 14400" in section and "11520" not in section


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
