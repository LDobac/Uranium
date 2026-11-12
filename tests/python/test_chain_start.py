"""041-R30 ⑤ / 041-AC7 (AMB-041-19): scoring with the chain-start observation is an explicit option of bc-eval and
tools/eval/per_action.py (the config's [data] chain_start_sources is not followed); per_action reports both ways,
its existing fields being the current-observation score. CPU, synthetic shards (`chains`: botgen games whose
Gather_Unit steps are chained one frame after a Move_Pos)."""

import functools
import sys
from pathlib import Path

import pytest

from uranium import _core

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "eval"))

import per_action as pa  # noqa: E402

from test_aux import CPU  # noqa: E402


@pytest.fixture
def trained(tiny_config, tmp_path):
    from test_bc_eval_race import _tiny_make

    try:
        _core.write_test_shards(str(tmp_path / "probe"), "chains", 4)
    except ValueError:
        pytest.skip("uranium._core without the `chains` test shards (rebuild the extension)")
    chains, varied = tmp_path / "chains", tmp_path / "varied"
    _core.write_test_shards(str(chains), "chains", 4)
    _core.write_test_shards(str(varied), "varied", 4)
    out = tmp_path / "out"
    assert _core.run_bc(functools.partial(_tiny_make, None, None), str(tiny_config), out=str(out), shards=[str(chains)], steps=1, synthetic_maps=True) == "done"
    keyed = tmp_path / "bc_keyed.toml"
    keyed.write_text(tiny_config.read_text(encoding="utf-8").replace("[data]\n", '[data]\nchain_start_sources = ["botgen"]\n', 1), encoding="utf-8")
    assert "chain_start_sources" in keyed.read_text(encoding="utf-8")
    return tiny_config, keyed, chains, varied, out / "last"


def score(cfg, shards, ckpt, chain):
    return pa.score(str(ckpt), [], [str(shards)], "cpu", str(cfg), None, synthetic_maps=True, split="train", chain_start=chain)


def counts(res):
    return {r["action"]: r["n"] for r in res["per_action"]}


def test_per_action_both_ways_041_r30(trained):
    cfg, keyed, chains, varied, ckpt = trained
    cur = score(cfg, chains, ckpt, None)
    cs = score(cfg, chains, ckpt, ["botgen"])
    assert cs["steps"] == cur["steps"] and cs["games"] == cur["games"] and counts(cs) == counts(cur), "labels and step counts unchanged"
    p = lambda res, a: next(r["mean_p"] for r in res["per_action"] if r["action"] == a)  # noqa: E731
    assert p(cs, "Gather_Unit") != p(cur, "Gather_Unit"), "the chained steps see their chain start"
    # The config's key is not followed: the keyed config scores like the plain one without the option.
    assert score(keyed, chains, ckpt, None)["per_action"] == cur["per_action"]
    # Games of a source that is not listed (the `varied` games are `test`) read the same either way.
    assert score(cfg, varied, ckpt, ["botgen"])["per_action"] == score(cfg, varied, ckpt, None)["per_action"]


def test_bc_eval_option_041_r30(trained):
    from uranium.bc.eval import evaluate

    cfg, keyed, chains, _, ckpt = trained
    run = lambda c, **kw: evaluate([str(chains)], str(ckpt), "train", None, 1, str(c), CPU, "f32", synthetic_maps=True, **kw)  # noqa: E731
    cur = run(cfg)
    assert "chain_start_sources" not in cur
    pick = lambda r: (r["steps"], r["action_top1"], r["nll"], r["delay_mae_frames"])  # noqa: E731
    assert pick(run(keyed)) == pick(cur), "bc-eval does not follow the config's chain_start_sources"
    cs = run(cfg, chain_start_sources=["botgen"])
    assert cs["chain_start_sources"] == ["botgen"] and cs["steps"] == cur["steps"]
    assert cs["nll"] != cur["nll"]
    with pytest.raises(Exception, match="041-R30"):
        _core.EvalStream(str(cfg), [str(chains)], split="train", synthetic_maps=True, chain_start_sources=["stardata"])


def test_chain_start_option_text_041_r30():
    assert pa.chain_sources("botgen") == ["botgen"] and pa.chain_sources("botgen,dagger") == ["botgen", "dagger"]
    assert pa.chain_sources("none") == []
    for bad in ("stardata", "", "botgen,human"):
        with pytest.raises(SystemExit, match="041-R30"):
            pa.chain_sources(bad)
