"""041-R24 loss-weight selection run configs (P3, AMB-041-13a): config/bc_m2_lambda_*.toml."""

import json
from pathlib import Path

import pytest

from uranium import _core


def test_lambda_run_configs_041_r24():
    """The seven AMB-041-13a runs: config/bc.toml's S1 schedule and data seed 1, everything kept,
    one factor changed per run (λ, F1, aux, model seed); [model] keys known (032-AC13)."""
    import tomllib

    from uranium.bc.run import model_variant

    root = Path(__file__).resolve().parents[2] / "config"
    ref = json.loads(_core.bc_config(str(root / "bc.toml"), "s1")[0])
    runs = {}
    for name in ("base", "noaux", "priv2x", "jepa2x", "nof1", "nof1_noaux", "seed2"):
        p = root / f"bc_m2_lambda_{name}.toml"
        cfg = json.loads(_core.bc_config(str(p), "s1")[0])
        v = model_variant(str(p))
        runs[name] = (cfg.get("aux"), v.core_skip, v.aux_priv, v.aux_jepa, cfg["train"]["model_seed"])
        assert cfg["keep_checkpoints"] == 0 and cfg["monitor"] == {"spatial_every": 100}
        assert v.terrain and v.ctx_token and v.attn_pool and v.outer == 32 and v.bypass and v.spatial_preact
        same = {k: x for k, x in cfg.items() if k not in ("aux", "monitor", "train", "keep_checkpoints", "data")}
        assert same == {k: x for k, x in ref.items() if k not in ("keep_checkpoints", "data")}
        assert {k: x for k, x in cfg["data"].items() if k != "checkpoint_every"} == {k: x for k, x in ref["data"].items() if k != "checkpoint_every"}
        assert cfg["total_steps"] == 39096 and cfg["data"]["seed"] == 1 and cfg["lr"] == 3e-4 and cfg["warmup_steps"] == 2000
        assert tomllib.loads(p.read_text(encoding="utf-8")).get("aux") == cfg.get("aux")
    on = {"priv": pytest.approx(0.2), "jepa": pytest.approx(0.1)}
    assert runs["base"] == (on, True, True, [1, 8, 32], 1)
    assert runs["noaux"] == (None, True, False, [], 1)
    assert runs["priv2x"][0] == {"priv": pytest.approx(0.4), "jepa": pytest.approx(0.1)}
    assert runs["jepa2x"][0] == {"priv": pytest.approx(0.2), "jepa": pytest.approx(0.2)}
    assert runs["nof1"] == (on, False, True, [1, 8, 32], 1)
    assert runs["nof1_noaux"] == (None, False, False, [], 1)
    assert runs["seed2"] == (on, True, True, [1, 8, 32], 2)
