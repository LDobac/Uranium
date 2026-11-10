"""041-R24 loss-weight selection run configs (P3, AMB-041-13a): config/bc_m2_lambda_*.toml, and the
model 2 path configs with the confirmed λ (041-R29, AMB-041-13b): bc_m2.toml, bc_m2_e2.toml,
bc_m2_mix_e1.toml."""

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


# 032 §2.7 four (main BC model) + every 032 §2.8 switch with F1 (041-R29 ①).
_ALL_ON = dict(terrain=True, ctx_token=True, attn_pool=True, outer=32, bypass=True, spatial_preact=True, core_skip=True, aux_priv=True, aux_jepa=[1, 8, 32])
_STRUCT = ("terrain", "ctx_token", "attn_pool", "outer")


def _bc(path, stage):
    cfg, fp = _core.bc_config(str(path), stage)
    return json.loads(cfg), fp


def test_main_lineage_configs_041_r29():
    """041-R29 ①② / 041-R23 ⑥ (AMB-041-13b Resolved 2026-11-10): config/bc_m2.toml (m2-s1, m2-s2e2)
    and config/bc_m2_e2.toml (m2-s1e2) are the Checkpoint 1 lineage's config/bc.toml and
    config/bc_e2.toml plus [model] (§2.8 + F1), [aux] 0.2 / 0.1, [monitor] 100, [train] model_seed 1."""
    from uranium.bc.run import model_variant

    root = Path(__file__).resolve().parents[2] / "config"
    for new, old, stages in (("bc_m2.toml", "bc.toml", ("s1", "s2")), ("bc_m2_e2.toml", "bc_e2.toml", ("s1",))):
        assert vars(model_variant(str(root / new))) == _ALL_ON
        assert {k: vars(model_variant(str(root / old)))[k] for k in _STRUCT} == {k: _ALL_ON[k] for k in _STRUCT}
        for stage in stages:
            cfg, fp = _bc(root / new, stage)
            ref, ref_fp = _bc(root / old, stage)
            assert cfg["aux"] == {"priv": pytest.approx(0.2), "jepa": pytest.approx(0.1)}
            assert cfg["monitor"] == {"spatial_every": 100} and cfg["train"] == {"model_seed": 1}
            assert cfg["keep_checkpoints"] == 0, "041-R29 ⑨: every checkpoint kept"
            assert {k: x for k, x in cfg.items() if k not in ("aux", "monitor", "train", "keep_checkpoints")} == {k: x for k, x in ref.items() if k != "keep_checkpoints"}
            assert fp != ref_fp, "[aux] / [train] are in the B-3 fingerprint"
    # m2-s1 has the lambda base run's settings (041-R24 ①): only execution keys differ.
    assert _bc(root / "bc_m2.toml", "s1")[1] == _bc(root / "bc_m2_lambda_base.toml", "s1")[1]
    assert _bc(root / "bc_m2_e2.toml", "s1")[0]["data"]["seed"] == 2


def test_m2_mix_e1_aux_041_r29():
    """041-R29 ③: the teacher mix template carries the confirmed λ (AMB-041-13b) and the model 2 switches."""
    from uranium.bc.run import model_variant

    p = Path(__file__).resolve().parents[2] / "config" / "bc_m2_mix_e1.toml"
    cfg, _ = _bc(p, "s2")
    assert cfg["aux"] == {"priv": pytest.approx(0.2), "jepa": pytest.approx(0.1)} and cfg["monitor"] == {"spatial_every": 100}
    assert cfg["keep_checkpoints"] == 0, "041-R29 ⑨"
    assert vars(model_variant(str(p))) == _ALL_ON
