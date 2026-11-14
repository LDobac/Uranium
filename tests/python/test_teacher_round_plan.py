"""044-R8 / 022-R12 (AMB-022-7): the teacher round self-game plan generator
`tools/eval/rush_check_plan.py --teacher-round` (CPU, no game): the round's TvT / TvP self games that the 022
teacher's shadow relabels into dagger_m2/t<R> — the 044-R1 composition (full and `--half`), the model required,
10-minute cap, decision logs, last failed command shown, no Random, seeds from their own block disjoint from every
tracked plan, from the reserved DAgger / distillation / other teacher round blocks and the 022 development seeds,
and the runner's plan loader."""

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
# 044-R1 (the composition of a round's self games): games per map of a full round.
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


@pytest.mark.parametrize("half", [False, True])
def test_teacher_round_plan_composition_044_r8(tmp_path, half):
    """044-R8: the round's self games are the 044-R1 composition (600, `--half` 300; Protoss : Terran = 3 : 2),
    the model stochastic as players[0], 10 min, replays and decision logs, zero_last_failed false, races fixed,
    T0 standard builds only; plan name dagger-m2-t<R>; the runner's loader finds the one model."""
    out = tmp_path / "t.toml"
    rcp.write_teacher_round(PLANS, 2, CKPT, out, half)
    plan = _load(out)
    div = 2 if half else 1
    assert plan["name"] == "dagger-m2-t2"
    assert plan["max_frames"] == 14400 and plan["save_replay"] is True and plan["memory_log"] is True
    assert plan["zero_last_failed"] is False and "seed_base" not in plan and "same_seed" not in plan
    games = Counter()
    for m in plan["matchup"]:
        assert m["map"] in MAPS
        assert m["p0"] == {"kind": "model", "checkpoint": CKPT, "sample": "stochastic"}
        assert m.get("p1_random_ratio", 0) == 0 and m["p1"]["race"] in ("protoss", "terran"), "no Random, no Zerg"
        assert "build" not in m["p1"], "T0 standard builds only"
        games[(_opponent(m["p1"]), m["map"])] += m["games"]
    assert games == Counter({(o, mp): g // div for o, g in FULL.items() for mp in MAPS})
    assert sum(games.values()) == 600 // div == len(plan["seeds"])
    # The same blocks as the round's distillation plan.
    _, d = rcp.render_distill(2, CKPT, "x", half)
    assert len(d) == len(plan["seeds"])
    assert _core.plan_models(str(out)) == [(CKPT, "stochastic")]
    assert "--relabel terran_tp --games" in out.read_text(encoding="utf-8")


def test_teacher_round_seeds_disjoint_044_r8(tmp_path):
    """044-R8: seeds are splitmix64(2,000,000 + n) >> 32 with round stride 600 (a half round the first half of its
    block), deterministic, disjoint between rounds, from every tracked plan (050-R9 protocol templates included),
    from the DAgger / distillation round blocks and top-ups, the other teacher round blocks and the 022
    development seeds; the 022 plans' reserved set covers the teacher round blocks too."""
    _, t1 = rcp.render_teacher_round(1, CKPT, "x", False)
    _, t2 = rcp.render_teacher_round(2, CKPT, "x", True)
    assert t1 == [rcp.splitmix64(2_000_000 + n) >> 32 for n in range(1, 601)]
    assert t2 == [rcp.splitmix64(2_000_000 + n) >> 32 for n in range(601, 901)]
    assert len(set(t1)) == 600 and len(set(t2)) == 300 and not set(t1) & set(t2)
    tracked = set()
    for p in PLANS.rglob("*.toml"):
        if not p.name.startswith("dagger_m2_t"):  # a generated teacher round plan is these seeds themselves
            tracked |= rcp.plan_seeds(p)
    assert rcp.plan_seeds(PLANS / "protocol" / "t0_tp.toml") <= tracked
    assert not tracked & (set(t1) | set(t2))
    for r, s in ((1, t1), (2, t2)):
        others = rcp.reserved_seeds(skip_teacher_round=r)
        assert not others & set(s)
        assert set(s) <= rcp.reserved_seeds(), "the full reserved set holds every teacher round block"
    others = rcp.reserved_seeds(skip_teacher_round=1)
    assert set(t2) <= others and set(range(77001, 77265)) <= others
    assert {rcp.splitmix64(70000 + n) >> 32 for n in (1, 6000)} <= others
    assert {rcp.splitmix64(80000 + n) >> 32 for n in (1, 6000, 100_001, 200_000)} <= others
    assert {rcp.splitmix64(2_000_000 + n) >> 32 for n in (6000,)} <= others and not {rcp.splitmix64(2_000_000 + 6001) >> 32} & rcp.reserved_seeds()
    assert rcp.render_teacher_round(1, CKPT, "x", False) == rcp.render_teacher_round(1, CKPT, "x", False)
    for bad in (0, 11):
        with pytest.raises(AssertionError, match="--round"):
            rcp.render_teacher_round(bad, CKPT, "x", False)


def test_teacher_round_rejects_taken_seed_044_r8(tmp_path):
    """044-R8: a plan anywhere under --out-dir (subfolders too) that uses a seed of the round stops the generator;
    its own output file is not counted against itself and is rewritten only with the same seeds."""
    _, seeds = rcp.render_teacher_round(1, CKPT, "x", False)
    sub = tmp_path / "protocol"
    sub.mkdir()
    (sub / "t0_x.toml").write_text(
        f'name = "x"\nseeds = [{seeds[17]}]\n[[matchup]]\nmap = "fighting_spirit"\ngames = 1\n'
        'p0 = { kind = "t0", race = "terran" }\np1 = { kind = "t0", race = "protoss" }\n', encoding="utf-8")
    with pytest.raises(AssertionError, match="another plan's"):
        rcp.write_teacher_round(tmp_path, 1, CKPT)
    (sub / "t0_x.toml").unlink()
    out = rcp.write_teacher_round(tmp_path, 1, CKPT)
    assert out == tmp_path / "dagger_m2_t1.toml"
    first = out.read_bytes()
    assert rcp.write_teacher_round(tmp_path, 1, CKPT).read_bytes() == first, "rewriting its own file is fine"
    with pytest.raises(AssertionError, match="other seeds"):
        rcp.write_teacher_round(tmp_path, 1, CKPT, half=True)
    # The round's distillation plan and the 022 plans beside it: disjoint.
    rcp.write_distill(tmp_path, 1, CKPT, None)
    rcp.write_tp(tmp_path, True)
    rcp.write_tp(tmp_path, False)
    assert rcp.write_teacher_round(tmp_path, 1, CKPT).read_bytes() == first


def test_teacher_round_cli_044_r8(tmp_path):
    """044-R8: `--teacher-round` needs `--checkpoint` (no default model) and takes no other plan kind, top-up or
    botgen option; `--memory-log` is accepted (always on)."""
    r = _run("--teacher-round", "--round", "2", "--half", "--memory-log", "--checkpoint", CKPT, "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    plan = _load(tmp_path / "dagger_m2_t2.toml")
    assert plan["name"] == "dagger-m2-t2" and len(plan["seeds"]) == 300
    bad = tmp_path / "bad.toml"
    for args in (("--teacher-round",), ("--teacher-round", "--distill", "--checkpoint", CKPT), ("--teacher-round", "--dagger", "--checkpoint", CKPT),
                 ("--teacher-round", "--extra", "20", "--part", "1", "--checkpoint", CKPT), ("--teacher-round", "--round", "11", "--checkpoint", CKPT),
                 ("--teacher-tp", "--teacher-round"), ("--teacher-round", "--botgen", "--checkpoint", CKPT)):
        r = _run(*args, "--out", str(bad), "--out-dir", str(tmp_path))
        assert r.returncode != 0 and not bad.exists(), args
