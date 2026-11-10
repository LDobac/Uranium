"""050-R9 `val500_race`: bc-eval's `by_enemy_race` and `protocol.py check-bc-eval`; bc-eval's aux
values for a model with the 032-R24 modules (041-R23 ⑤). CPU."""

import functools
import json
from pathlib import Path

import pytest

from uranium import _core, ckpt
from uranium.bc.trainer import BcTrainer

from test_aux import AUX, CPU, make_net


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


def eval_all_splits(dirs: list, ckpt_dir: Path, config: Path) -> list:
    """bc-eval reports of every split with games (the synthetic games spread over the splits)."""
    from uranium.bc.eval import evaluate

    reps = []
    for split in ("train", "val", "test"):
        try:
            reps.append(evaluate([str(d) for d in dirs], str(ckpt_dir), split, None, 1, str(config), CPU, "f32", synthetic_maps=True))
        except RuntimeError:
            continue
    return reps


@pytest.mark.parametrize("variant", [None, AUX], ids=["plain", "aux"])
def test_bc_eval_by_enemy_race_050_r9(tiny_config, tmp_path, variant):
    """`by_enemy_race` keys each game by the race its enemy played (GameInfo): over every split,
    each race's steps equal the steps of a bc-eval over only that race's games, and the lobby
    Random game (matchup `TvR`, played Protoss) counts under protoss, never in an `r` / `random`
    bucket; top-1 and per-head NLL add up to the whole. A model with aux modules also reports the
    aux losses (041-R23 ⑤)."""
    shards = tmp_path / "races"
    _core.write_test_shards(str(shards), "races", 4)
    make = functools.partial(_tiny_make, variant, {"priv": 0.2, "jepa": 0.1} if variant else None)
    out = tmp_path / "out"
    assert _core.run_bc(make, str(tiny_config), out=str(out), shards=[str(shards)], steps=1, synthetic_maps=True) == "done"
    reps = eval_all_splits([shards], out / "last", tiny_config)
    assert reps
    per_race = {}
    for rep in reps:
        per = rep["by_enemy_race"]
        assert set(per) == {"terran", "protoss", "zerg"}, "no extra bucket (the lobby Random game is protoss)"
        assert sum(x["steps"] for x in per.values()) == rep["steps"]
        top1 = sum(x["action_top1"] * x["steps"] for x in per.values() if x["steps"])
        assert top1 == pytest.approx(rep["action_top1"] * rep["steps"])
        nll = sum(x["nll"]["action"] * x["nll_steps"]["action"] for x in per.values())
        assert nll == pytest.approx(rep["nll"]["action"] * rep["nll_steps"]["action"])
        for r, x in per.items():
            assert x["action_top1"] is None if not x["steps"] else 0.0 <= x["action_top1"] <= 1.0
            per_race[r] = per_race.get(r, 0) + x["steps"]
        if variant is None:
            assert "aux_priv" not in rep
        elif rep["steps"]:
            assert rep["aux_jepa"] >= 0.0 and rep["jepa_tgt_std"] >= 0.0 and rep["jepa_tgt_std_raw"] >= 0.0 and rep["aux_jepa_slot3"] is None
    for race in ("terran", "protoss", "zerg"):
        only = tmp_path / f"only_{race}"
        _core.write_test_shards(str(only), "races", 4, only_race=race)
        want = sum(r["steps"] for r in eval_all_splits([only], out / "last", tiny_config))
        assert want > 0 and per_race[race] == want, race
    assert per_race["protoss"] > 0 and sum(per_race.values()) == sum(r["steps"] for r in reps)


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
