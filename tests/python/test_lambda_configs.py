"""041-R24 loss-weight selection run configs (P3, AMB-041-13a): config/bc_m2_lambda_*.toml, and the
model 2 path configs with the confirmed λ (041-R29, AMB-041-13b): bc_m2.toml, bc_m2_e2.toml,
bc_m2_mix_e1.toml ... e7, the DAgger round 1 bc_m2_dagger_r1.toml and its plan dagger_m2_r1.toml."""

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
    # TOML level: only the model 2 keys differ (a stray table or key would show up here).
    import tomllib

    def flat(t, pre=""):
        out = {}
        for k, x in t.items():
            out.update(flat(x, f"{pre}{k}.") if isinstance(x, dict) else {f"{pre}{k}": x})
        return out

    allowed = {"aux.priv", "aux.jepa", "monitor.spatial_every", "train.model_seed", "keep_checkpoints"} | {f"model.{k}" for k in ("bypass", "spatial_preact", "core_skip", "aux_priv", "aux_jepa")}
    for new, old in (("bc_m2.toml", "bc.toml"), ("bc_m2_e2.toml", "bc_e2.toml")):
        a, b = (flat(tomllib.loads((root / f).read_text(encoding="utf-8"))) for f in (new, old))
        assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} == allowed, new
    # m2-s1 has the lambda base run's settings (041-R24 ①): only execution keys differ.
    assert _bc(root / "bc_m2.toml", "s1")[1] == _bc(root / "bc_m2_lambda_base.toml", "s1")[1]
    assert _bc(root / "bc_m2_e2.toml", "s1")[0]["data"]["seed"] == 2


_M2_MIX = tuple(f"bc_m2_mix_e{e}.toml" for e in range(1, 8))


def test_m2_mix_e1_aux_041_r29():
    """041-R29 ③: the teacher mix chain (e1 ... e7) carries the confirmed λ (AMB-041-13b), the model 2
    switches and the 041-R30 chain-start observation for botgen."""
    from uranium.bc.run import model_variant

    root = Path(__file__).resolve().parents[2] / "config"
    for name in _M2_MIX:
        p = root / name
        cfg, _ = _bc(p, "s2")
        assert cfg["aux"] == {"priv": pytest.approx(0.2), "jepa": pytest.approx(0.1)} and cfg["monitor"] == {"spatial_every": 100}, name
        assert cfg["keep_checkpoints"] == 0, "041-R29 ⑨"
        assert vars(model_variant(str(p))) == _ALL_ON, name


def test_m2_mix_chain_041_r29():
    """041-R29 ③ / 041-R21 ⑥: config/bc_m2_mix_e2.toml ... e7.toml are bc_m2_mix_e1.toml except data.seed
    (epoch N: N + 2), the 041-R26 per-race human fractions and [steps] s2 / s2_warmup (recounted with
    `uranium-train mix-fractions` per seed, warmup 5% rounded half up, 041-R14); 041-R30
    chain_start_sources = ["botgen"] kept; the fingerprints differ (seed and subsample are in it)."""
    import tomllib

    root = Path(__file__).resolve().parents[2] / "config"

    def flat(t, pre=""):
        out = {}
        for k, x in t.items():
            out.update(flat(x, f"{pre}{k}.") if isinstance(x, dict) else {f"{pre}{k}": x})
        return out

    ref = flat(tomllib.loads((root / _M2_MIX[0]).read_text(encoding="utf-8")))
    allowed = {"data.seed", "data.subsample.fraction", "steps.s2", "steps.s2_warmup"} | {f"data.subsample.fraction_by_race.{r}" for r in "TPZ"}
    fps = set()
    for e, name in enumerate(_M2_MIX, start=1):
        t = flat(tomllib.loads((root / name).read_text(encoding="utf-8")))
        assert t.keys() == ref.keys(), name
        assert {k for k in t if t[k] != ref[k]} <= allowed, name
        assert t["data.seed"] == e + 2, name
        assert t["data.chain_start_sources"] == ["botgen"] and t["data.subsample.keep_sources"] == ["botgen"], name
        assert t["steps.s2_warmup"] == int(t["steps.s2"] * 0.05 + 0.5), name
        assert all(0 < t[f"data.subsample.fraction_by_race.{r}"] < 1 for r in "TPZ"), name
        cfg, fp = _bc(root / name, "s2")
        assert cfg["data"]["seed"] == e + 2 and cfg["total_steps"] == t["steps.s2"] and cfg["warmup_steps"] == t["steps.s2_warmup"], name
        fps.add(fp)
    assert len(fps) == len(_M2_MIX)


def _flat(t, pre=""):
    out = {}
    for k, x in t.items():
        out.update(_flat(x, f"{pre}{k}.") if isinstance(x, dict) else {f"{pre}{k}": x})
    return out


def _toml(path):
    import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8"))


_M2_TABLES = {"keep_checkpoints", "monitor.spatial_every", "aux.priv", "aux.jepa"} | {f"model.{k}" for k in ("bypass", "spatial_preact", "core_skip", "aux_priv", "aux_jepa")}
_RACE_FRACTIONS = {f"data.subsample.fraction_by_race.{r}" for r in "TPZ"}
_COUNTED = {"data.subsample.fraction", "steps.s2", "steps.s2_warmup"} | _RACE_FRACTIONS


def test_m2_dagger_r1_config_041_r29(tmp_path):
    """041-R29 ④ / 041-R22 ③ / 041-R26 / 041-R30 ⑥: config/bc_m2_dagger_r1.toml is the Checkpoint 1 path's
    bc_dagger_r1.toml (1 epoch, learning rate, data.seed 13, keep_sources botgen + dagger) with the model 2
    tables of bc_m2_mix_e7.toml ([model], [aux], [monitor], keep_checkpoints 0), chain_start_sources
    ["botgen"] (not dagger), and its own per-race human fractions and [steps]. Until the run driver's count
    fills them, the placeholder zero fraction_by_race makes the loader refuse the file (s2 = 0 would mean
    total_steps); filled, they are a valid race subsample and a 5% warmup (041-R14)."""
    from uranium.bc.run import model_variant

    root = Path(__file__).resolve().parents[2] / "config"
    p = root / "bc_m2_dagger_r1.toml"
    new, ck1, mix = (_flat(_toml(root / f)) for f in ("bc_m2_dagger_r1.toml", "bc_dagger_r1.toml", "bc_m2_mix_e7.toml"))
    # TOML level against Checkpoint 1's round 1: only the model 2 keys, 041-R30 and the counted values differ.
    diff = {k for k in new.keys() | ck1.keys() if new.get(k) != ck1.get(k)}
    assert _M2_TABLES | {"data.chain_start_sources"} | _RACE_FRACTIONS <= diff
    assert diff <= _M2_TABLES | {"data.chain_start_sources"} | _COUNTED, diff - _M2_TABLES - _COUNTED
    # ... and against the lineage's last mix epoch: same keys; the DAgger data seed and sources, counted values.
    assert new.keys() == mix.keys()
    assert {k for k in new if new[k] != mix[k]} <= {"data.seed", "data.subsample.keep_sources"} | _COUNTED
    assert new["data.seed"] == 13 and new["data.subsample.keep_sources"] == ["botgen", "dagger"]
    assert new["data.chain_start_sources"] == ["botgen"], "041-R30 ⑥: teacher botgen only, not the dagger relabels"
    assert vars(model_variant(str(p))) == _ALL_ON
    fr = [new[k] for k in sorted(_RACE_FRACTIONS)]
    filled = not all(f == 0 for f in fr)
    if not filled:
        # Placeholder: refused by every loader (the run driver counts on a copy without the table).
        assert (new["steps.s2"], new["steps.s2_warmup"]) == (0, 0)
        with pytest.raises(Exception, match="fraction_by_race"):
            _bc(p, "s2")
        text = p.read_text(encoding="utf-8")
        table = "[data.subsample.fraction_by_race]\nT = 0.0\nP = 0.0\nZ = 0.0\n"
        assert text.count(table) == 1
        p = tmp_path / "count.toml"
        p.write_text(text.replace(table, ""), encoding="utf-8")
    else:
        assert all(0 < f <= 1 for f in fr) and 0 < new["data.subsample.fraction"] <= 1
        assert new["steps.s2"] > 0 and new["steps.s2_warmup"] == int(new["steps.s2"] * 0.05 + 0.5)
        assert "PLACEHOLDER" not in p.read_text(encoding="utf-8")
    cfg, fp = _bc(p, "s2")
    ref, ref_fp = _bc(root / "bc_dagger_r1.toml", "s2")
    assert cfg["aux"] == {"priv": pytest.approx(0.2), "jepa": pytest.approx(0.1)} and cfg["monitor"] == {"spatial_every": 100}
    assert cfg["keep_checkpoints"] == 0 and "train" not in cfg, "041-R29 ⑨; model_seed unused with --init"
    assert cfg["data"]["chain_start_sources"] == ["botgen"] and cfg["data"]["subsample"]["keep_sources"] == ["botgen", "dagger"]
    skip = ("aux", "monitor", "keep_checkpoints", "data", "steps", "total_steps", "warmup_steps")
    assert {k: x for k, x in cfg.items() if k not in skip} == {k: x for k, x in ref.items() if k not in skip}
    dskip = ("chain_start_sources", "subsample")
    assert {k: x for k, x in cfg["data"].items() if k not in dskip} == {k: x for k, x in ref["data"].items() if k not in dskip}
    assert cfg["lr"] * cfg["s2_lr_scale"] == pytest.approx(3e-5) and cfg["min_lr"] * cfg["s2_lr_scale"] == pytest.approx(3e-7)
    assert fp != ref_fp
    if filled:
        assert (cfg["total_steps"], cfg["warmup_steps"]) == (new["steps.s2"], new["steps.s2_warmup"])


def test_m2_dagger_r1_plan_041_r28():
    """041-R29 ④ / 041-R22 ① / 041-R28 ①: config/plans/dagger_m2_r1.toml (rush_check_plan.py --dagger --round 1
    --memory-log --checkpoint checkpoints/m2-mix-e7/ckpt-1212) is the Checkpoint 1 path's round-1 plan (same 600
    seeds and matchups, stochastic, 10 min, replays) with the lineage-2 model and the decision logs."""
    root = Path(__file__).resolve().parents[2] / "config" / "plans"
    new, ck1 = _toml(root / "dagger_m2_r1.toml"), _toml(root / "dagger_r1.toml")
    assert new["memory_log"] is True and "memory_log" not in ck1 and new["save_replay"] is True
    assert {k: x for k, x in new.items() if k not in ("memory_log", "matchup")} == {k: x for k, x in ck1.items() if k != "matchup"}
    assert len(new["seeds"]) == 600 == sum(m["games"] for m in new["matchup"])
    for a, b in zip(new["matchup"], ck1["matchup"], strict=True):
        assert a["p0"] == {"kind": "model", "checkpoint": "checkpoints/m2-mix-e7/ckpt-1212", "sample": "stochastic"}
        assert {k: x for k, x in a.items() if k != "p0"} == {k: x for k, x in b.items() if k != "p0"}
