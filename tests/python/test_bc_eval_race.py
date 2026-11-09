"""050-R9 `val500_race`: bc-eval's `by_enemy_race` and `protocol.py check-bc-eval`; bc-eval's aux
values for a model with the 032-R24 modules (041-R23 ⑤). CPU."""

import functools
import json
from pathlib import Path

import pytest

from uranium import _core, ckpt
from uranium.bc.trainer import BcTrainer

from test_aux import AUX, CPU, make_net


@pytest.fixture
def shards(tmp_path: Path) -> Path:
    d = tmp_path / "shards"
    _core.write_test_shards(str(d), "mixed", 2)
    return d


def _tiny_make(variant, aux, cfg_json, start, dir, entity_chunk):
    cfg = json.loads(cfg_json)
    if aux:
        cfg["aux"] = aux
    if start == "new":
        net = make_net(variant)
    else:
        net, _ = ckpt.load_model(Path(dir))
    t = BcTrainer(cfg, net, CPU, "f32", entity_chunk)
    if start == "resume":
        ckpt.load_optimizer(Path(dir), ckpt.OPTIM_MAIN, t.net, t.opt)
    return t


@pytest.mark.parametrize("variant", [None, AUX], ids=["plain", "aux"])
def test_bc_eval_by_enemy_race_050_r9(tiny_config, shards, tmp_path, variant):
    """`by_enemy_race` splits the scored steps by the enemy race of the view's matchup (the races
    sum to the whole), with top-1 and per-head NLL; a model with aux modules also reports the
    aux losses (041-R23 ⑤)."""
    from uranium.bc.eval import evaluate

    make = functools.partial(_tiny_make, variant, {"priv": 0.2, "jepa": 0.1} if variant else None)
    out = tmp_path / "out"
    assert _core.run_bc(make, str(tiny_config), out=str(out), shards=[str(shards)], steps=1, synthetic_maps=True) == "done"
    for split in ("train", "val", "test"):
        try:
            rep = evaluate([str(shards)], str(out / "last"), split, None, 1, str(tiny_config), CPU, "f32", synthetic_maps=True)
        except RuntimeError:
            continue
        if rep["steps"]:
            break
    per = rep["by_enemy_race"]
    assert {"terran", "protoss", "zerg"} <= set(per)
    assert sum(x["steps"] for x in per.values()) == rep["steps"] > 0
    top1 = sum(x["action_top1"] * x["steps"] for x in per.values() if x["steps"])
    assert top1 == pytest.approx(rep["action_top1"] * rep["steps"])
    nll = sum(x["nll"]["action"] * x["nll_steps"]["action"] for x in per.values())
    assert nll == pytest.approx(rep["nll"]["action"] * rep["nll_steps"]["action"])
    for x in per.values():
        assert x["action_top1"] is None if not x["steps"] else 0.0 <= x["action_top1"] <= 1.0
    if variant is None:
        assert "aux_priv" not in rep
    else:
        assert rep["aux_priv"] >= 0.0 and rep["aux_jepa"] >= 0.0 and rep["jepa_tgt_std"] >= 0.0 and rep["aux_jepa_slot3"] is None


def test_check_bc_eval_refuses_a_race_without_steps_050_r9(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "eval"))
    import protocol

    f = tmp_path / "v.json"
    per = {r: {"steps": 10, "action_top1": 0.7, "nll": {"action": 0.9}} for r in ("terran", "protoss", "zerg")}
    base = {"split": "val", "games": 500, "selection_seed": 1, "action_top1": 0.72}
    f.write_text(json.dumps({**base, "by_enemy_race": per}), encoding="utf-8")
    protocol.main(["check-bc-eval", str(f), "--metric", "val500_race"])
    per["zerg"] = {"steps": 0, "action_top1": None, "nll": {"action": 0.0}}
    f.write_text(json.dumps({**base, "by_enemy_race": per}), encoding="utf-8")
    with pytest.raises(SystemExit, match="by_enemy_race.zerg` has no scored step"):
        protocol.main(["check-bc-eval", str(f), "--metric", "val500_race"])
