import subprocess
import sys
import tomllib
from collections import Counter
from fractions import Fraction

import pytest

from eval_helpers import ROOT
from uranium import _core
from uranium.measure import explore_plan as ep
from uranium.measure import rush_check_plan as rcp

PLANS = ROOT / "config" / "plans"
TOOL = ROOT / "tools" / "eval" / "explore_plan.py"
CKPT = "checkpoints/m2-gn-t1e2/ckpt-2844"
T1 = {"zerg": {"ualbertabot", "steamhammer", "mcrave", "zzzkbot"}, "protoss": {"ualbertabot", "steamhammer", "mcrave", "stardust"},
      "terran": {"ualbertabot", "steamhammer", "mcrave", "iron"}}


def _load(p) -> dict:
    return tomllib.loads(p.read_text(encoding="utf-8"))


def _kind(p1: dict) -> tuple[str, str, str]:
    if p1["kind"] == "t0":
        return ("t0", p1.get("build", "standard"), p1["race"])
    return ("t1", p1["name"], p1["race"])


def test_composition_054_r1(tmp_path):
    out = tmp_path / "e.toml"
    ep.write(PLANS, CKPT, "explore-pilot", 1000, 1, True, "shadow", out)
    plan = _load(out)
    assert (plan["name"], plan["save_replay"], plan["memory_log"], plan["act_log"], plan["max_frames"]) == ("explore-pilot", True, True, True, 14400)
    assert (plan["zero_last_failed"], plan["surrender"]) == (True, "shadow")
    assert plan["explore"] == {"points": 3, "window_s": [120, 540], "gap_s": 60, "k": 4, "epsilon": 0.5, "min_open": 2}, "020-R22 ⑤"
    games, maps = Counter(), Counter()
    for m in plan["matchup"]:
        assert m["p0"] == {"kind": "model", "checkpoint": CKPT, "sample": "stochastic"}
        assert m.get("p1_random_ratio", 0) == 0 and m["map"] in ("fighting_spirit", "circuit_breaker")
        games[_kind(m["p1"])] += m["games"]
        maps[m["map"]] += m["games"]
    assert sum(games.values()) == len(plan["seeds"]) == 1000
    race = Counter()
    for (kind, _, r), n in games.items():
        race[(r, kind)] += n
    assert race == Counter({("zerg", "t0"): 250, ("zerg", "t1"): 250, ("protoss", "t0"): 150, ("protoss", "t1"): 150, ("terran", "t0"): 100, ("terran", "t1"): 100})
    assert (games[("t0", "standard", "zerg")], games[("t0", "zerg_rush", "zerg")]) == (125, 125), "Zerg T0 standard and zerg_rush half and half"
    assert games[("t0", "standard", "protoss")] == 150 and games[("t0", "standard", "terran")] == 100, "021 standard builds"
    for r, bots in T1.items():
        got = {name: n for (kind, name, rr), n in games.items() if kind == "t1" and rr == r}
        assert set(got) == bots, r
        assert max(got.values()) - min(got.values()) <= 1, f"{r}: uniform over the 021 list {got}"
    assert abs(maps["fighting_spirit"] - maps["circuit_breaker"]) <= 1
    assert _core.plan_models(str(out)) == [(CKPT, "stochastic")], "the Rust plan loader takes it (020-R22 ⑤)"


def test_blocks_round_to_the_count_054_r1():
    assert ep.split(10, [1, 1, 1]) == [4, 3, 3] and ep.split(1000, [50, 30, 20]) == [500, 300, 200]
    for n in (20, 160, 999, 1000, 9000, 10_000):
        bs = ep.blocks(n)
        assert sum(b[-1] for b in bs) == n and bs == ep.blocks(n)
        ideal = {(r, k, s, m): n * w for r, k, s, m, w in ep.leaves()}
        race, kind, maps = Counter(), Counter(), Counter()
        for r, k, s, m, g in bs:
            assert abs(g - ideal[(r, k, s, m)]) < 2, (n, r, k, s, m, g)
            race[r] += g
            kind[(r, k)] += g
            maps[m] += g
        for r, w in ep.RACES:
            assert abs(race[r] - Fraction(n * w, 100)) < 1, (n, r)
            assert abs(kind[(r, "t0")] - kind[(r, "t1")]) <= 1, (n, r)
        assert abs(maps["fighting_spirit"] - maps["circuit_breaker"]) <= 1, (n, maps)
    exact = {(r, k, s, m): 160 * w for r, k, s, m, w in ep.leaves()}
    assert all(w.denominator == 1 for w in exact.values()), "160 games split exactly"
    assert {(r, k, s, m): g for r, k, s, m, g in ep.blocks(160)} == exact
    assert sum(w for *_, w in ep.leaves()) == Fraction(1)


def test_seeds_are_deterministic_and_disjoint_054_r1(tmp_path):
    a, s1 = ep.render(CKPT, "explore-pilot", 1000, 1, False, "shadow")
    b, s2 = ep.render(CKPT, "explore-pilot", 1000, 1, False, "shadow")
    assert a == b and s1 == s2, "same arguments, same plan"
    assert s1 == [rcp.splitmix64(7_000_000 + k) >> 32 for k in range(1, 1001)]
    _, more = ep.render(CKPT, "explore-more", 9000, 1001, False, "shadow")
    assert not set(s1) & set(more), "a later plan from --first 1001 takes new seeds"
    block = rcp.explore_block()
    assert len(block) == len(rcp.EXPLORE_BLOCK) and set(s1) | set(more) <= block
    others = rcp.taken_seeds(PLANS, ()) | rcp.reserved_seeds(skip_explore=True)
    assert not others & block, "the explore block is disjoint from every plan and reserved block"
    assert block <= rcp.reserved_seeds(), "other generators keep out of the explore block"
    assert not block <= rcp.reserved_seeds(skip_explore=True)
    with pytest.raises(AssertionError, match="seed block"):
        ep.render(CKPT, "x", 200, 10_000, False, "shadow")
    with pytest.raises(AssertionError, match="seed block"):
        ep.render(CKPT, "x", 20, 0, False, "shadow")
    out = tmp_path / "e.toml"
    ep.write(PLANS, CKPT, "explore-pilot", 20, 10_001, False, "on", out)
    ep.write(PLANS, CKPT, "explore-pilot", 20, 10_001, False, "on", out)
    with pytest.raises(AssertionError, match="refusing to overwrite"):
        ep.write(PLANS, CKPT, "explore-pilot", 20, 10_021, False, "on", out)


def test_tool_needs_the_f_settings_054_r1(tmp_path):
    out = tmp_path / "t.toml"
    r = subprocess.run([sys.executable, str(TOOL), "--checkpoint", CKPT, "--out", str(out), "--games", "20", "--first", "10001"], capture_output=True, text=True)
    assert r.returncode != 0 and "--zero-last-failed" in r.stderr
    r = subprocess.run([sys.executable, str(TOOL), "--checkpoint", CKPT, "--out", str(out), "--games", "20", "--first", "10001", "--zero-last-failed", "false",
                        "--surrender", "shadow", "--name", "explore-ac4"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    plan = _load(out)
    assert plan["name"] == "explore-ac4" and len(plan["seeds"]) == 20 and plan["zero_last_failed"] is False
