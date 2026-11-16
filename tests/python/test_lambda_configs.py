import json
from pathlib import Path

import pytest

import eval_helpers  # noqa: F401
from uranium import _core

import rush_check_plan as gen


def test_lambda_run_configs_041_r24():
    import tomllib

    from uranium.model.config import model_variant

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


_ALL_ON = dict(terrain=True, ctx_token=True, attn_pool=True, outer=32, bypass=True, spatial_preact=True, core_skip=True, aux_priv=True, aux_jepa=[1, 8, 32])
_STRUCT = ("terrain", "ctx_token", "attn_pool", "outer")


def _bc(path, stage):
    cfg, fp = _core.bc_config(str(path), stage)
    return json.loads(cfg), fp


def test_main_lineage_configs_041_r29():
    from uranium.model.config import model_variant

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
    assert _bc(root / "bc_m2.toml", "s1")[1] == _bc(root / "bc_m2_lambda_base.toml", "s1")[1]
    assert _bc(root / "bc_m2_e2.toml", "s1")[0]["data"]["seed"] == 2


_M2_MIX = tuple(f"bc_m2_mix_e{e}.toml" for e in range(1, 8))


def test_m2_mix_e1_aux_041_r29():
    from uranium.model.config import model_variant

    root = Path(__file__).resolve().parents[2] / "config"
    for name in _M2_MIX:
        p = root / name
        cfg, _ = _bc(p, "s2")
        assert cfg["aux"] == {"priv": pytest.approx(0.2), "jepa": pytest.approx(0.1)} and cfg["monitor"] == {"spatial_every": 100}, name
        assert cfg["keep_checkpoints"] == 0, "041-R29 ⑨"
        assert vars(model_variant(str(p))) == _ALL_ON, name


def test_m2_mix_chain_041_r29():
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


_M2_DAGGER = {
    1: (13, "bc_m2_mix_e7.toml", {"data.seed", "data.subsample.keep_sources"}, 1),
    2: (14, "bc_m2_dagger_r1.toml", {"data.seed"}, 2),
    3: (15, "bc_m2_dagger_r2.toml", {"data.seed"}, 2),
}


@pytest.mark.parametrize("rnd", sorted(_M2_DAGGER))
def test_m2_dagger_config_041_r29(tmp_path, rnd):
    from uranium.model.config import model_variant

    seed, before, may_differ, ck1_round = _M2_DAGGER[rnd]
    seed_apart = {"data.seed"} if ck1_round != rnd else set()
    root = Path(__file__).resolve().parents[2] / "config"
    p = root / f"bc_m2_dagger_r{rnd}.toml"
    new, ck1, prev = (_flat(_toml(root / f)) for f in (p.name, f"bc_dagger_r{ck1_round}.toml", before))
    diff = {k for k in new.keys() | ck1.keys() if new.get(k) != ck1.get(k)}
    assert _M2_TABLES | {"data.chain_start_sources"} | _RACE_FRACTIONS <= diff
    allowed = _M2_TABLES | {"data.chain_start_sources"} | _COUNTED | seed_apart
    assert diff <= allowed, diff - allowed
    assert new.keys() == prev.keys()
    assert {k for k in new if new[k] != prev[k]} <= may_differ | _COUNTED
    assert new["data.seed"] == seed and new["data.subsample.keep_sources"] == ["botgen", "dagger"]
    assert new["data.chain_start_sources"] == ["botgen"], "041-R30 ⑥: teacher botgen only, not the dagger relabels"
    assert vars(model_variant(str(p))) == _ALL_ON
    fr = [new[k] for k in sorted(_RACE_FRACTIONS)]
    filled = not all(f == 0 for f in fr)
    if not filled:
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
    ref, ref_fp = _bc(root / f"bc_dagger_r{ck1_round}.toml", "s2")
    assert cfg["aux"] == {"priv": pytest.approx(0.2), "jepa": pytest.approx(0.1)} and cfg["monitor"] == {"spatial_every": 100}
    assert cfg["keep_checkpoints"] == 0 and "train" not in cfg, "041-R29 ⑨; model_seed unused with --init"
    assert cfg["data"]["chain_start_sources"] == ["botgen"] and cfg["data"]["subsample"]["keep_sources"] == ["botgen", "dagger"]
    skip = ("aux", "monitor", "keep_checkpoints", "data", "steps", "total_steps", "warmup_steps")
    assert {k: x for k, x in cfg.items() if k not in skip} == {k: x for k, x in ref.items() if k not in skip}
    dskip = ("chain_start_sources", "subsample", *(("seed",) if seed_apart else ()))
    assert {k: x for k, x in cfg["data"].items() if k not in dskip} == {k: x for k, x in ref["data"].items() if k not in dskip}
    assert cfg["lr"] * cfg["s2_lr_scale"] == pytest.approx(3e-5) and cfg["min_lr"] * cfg["s2_lr_scale"] == pytest.approx(3e-7)
    assert fp != ref_fp
    if filled:
        assert (cfg["total_steps"], cfg["warmup_steps"]) == (new["steps.s2"], new["steps.s2_warmup"])


_M2_DAGGER_PLAN = {
    1: (600, "checkpoints/m2-mix-e7/ckpt-1212", 1),
    2: (300, "checkpoints/m2-dagger-r1/ckpt-1285", 2),
    3: (300, "checkpoints/m2-dagger-r2/ckpt-1408", 2),
}


@pytest.mark.parametrize("rnd", sorted(_M2_DAGGER_PLAN))
def test_m2_dagger_plan_041_r28(rnd):
    games, ckpt, ck1_round = _M2_DAGGER_PLAN[rnd]
    root = Path(__file__).resolve().parents[2] / "config" / "plans"
    new, ck1 = _toml(root / f"dagger_m2_r{rnd}.toml"), _toml(root / f"dagger_r{ck1_round}.toml")
    assert new["memory_log"] is True and "memory_log" not in ck1 and new["save_replay"] is True
    apart = ("memory_log", "matchup") + (("name", "seeds") if ck1_round != rnd else ())
    assert {k: x for k, x in new.items() if k not in apart} == {k: x for k, x in ck1.items() if k not in apart}
    if ck1_round != rnd:
        first = (rnd - 1) * gen.DAGGER_ROUND_STRIDE + 1
        assert new["name"] == f"dagger-r{rnd}"
        assert new["seeds"] == [gen.splitmix64(gen.DAGGER_SEED_KEY + k) >> 32 for k in range(first, first + games)]
    assert len(new["seeds"]) == games == sum(m["games"] for m in new["matchup"])
    for a, b in zip(new["matchup"], ck1["matchup"], strict=True):
        assert a["p0"] == {"kind": "model", "checkpoint": ckpt, "sample": "stochastic"}
        assert {k: x for k, x in a.items() if k != "p0"} == {k: x for k, x in b.items() if k != "p0"}


_SOURCE_FRACTIONS = {"data.subsample.fraction_by_source.distill"}


def test_m2_distill_d1_config_044_r5(tmp_path):
    from uranium.model.config import model_variant

    root = Path(__file__).resolve().parents[2] / "config"
    p = root / "bc_m2_distill_d1.toml"
    new, prev = (_flat(_toml(root / f)) for f in (p.name, "bc_m2_dagger_r2.toml"))
    assert new.keys() == prev.keys() | _SOURCE_FRACTIONS
    assert {k for k in prev if new[k] != prev[k]} <= {"data.seed"} | _COUNTED
    assert new["data.seed"] == 15 and new["data.subsample.keep_sources"] == ["botgen", "dagger"]
    assert new["data.chain_start_sources"] == ["botgen"], "041-R30 ⑥ / 044-R4: botgen only"
    assert vars(model_variant(str(p))) == _ALL_ON
    fr = [new[k] for k in sorted(_RACE_FRACTIONS | _SOURCE_FRACTIONS)]
    filled = not all(f == 0 for f in fr)
    text = p.read_text(encoding="utf-8")
    if not filled:
        assert (new["steps.s2"], new["steps.s2_warmup"]) == (0, 0)
        with pytest.raises(Exception, match="fraction_by_(race|source)"):
            _bc(p, "s2")
        tables = ("[data.subsample.fraction_by_race]\nT = 0.0\nP = 0.0\nZ = 0.0\n", "[data.subsample.fraction_by_source]\ndistill = 0.0\n")
        for t in tables:
            assert text.count(t) == 1
        p1 = tmp_path / "race_only.toml"
        p1.write_text(text.replace(tables[0], ""), encoding="utf-8")
        with pytest.raises(Exception, match="fraction_by_source"):
            _bc(p1, "s2")
        p = tmp_path / "count.toml"
        p.write_text(text.replace(tables[0], "").replace(tables[1], ""), encoding="utf-8")
    else:
        assert all(0 < f <= 1 for f in fr) and 0 < new["data.subsample.fraction"] <= 1
        assert new["steps.s2"] > 0 and new["steps.s2_warmup"] == int(new["steps.s2"] * 0.05 + 0.5)
        assert "PLACEHOLDER" not in text
    cfg, fp = _bc(p, "s2")
    ref, ref_fp = _bc(root / "bc_m2_dagger_r2.toml", "s2")
    assert cfg["data"]["seed"] == 15 and cfg["data"]["subsample"]["keep_sources"] == ["botgen", "dagger"]
    if filled:
        assert cfg["data"]["subsample"]["fraction_by_source"] == {"distill": new["data.subsample.fraction_by_source.distill"]}
        assert (cfg["total_steps"], cfg["warmup_steps"]) == (new["steps.s2"], new["steps.s2_warmup"])
    skip = ("data", "steps", "total_steps", "warmup_steps")
    assert {k: x for k, x in cfg.items() if k not in skip} == {k: x for k, x in ref.items() if k not in skip}
    dskip = ("seed", "subsample")
    assert {k: x for k, x in cfg["data"].items() if k not in dskip} == {k: x for k, x in ref["data"].items() if k not in dskip}
    assert cfg["lr"] * cfg["s2_lr_scale"] == pytest.approx(3e-5) and cfg["min_lr"] * cfg["s2_lr_scale"] == pytest.approx(3e-7)
    assert fp != ref_fp


def test_m2_teacher_t1_config_044_r8(tmp_path):
    from uranium.model.config import model_variant

    root = Path(__file__).resolve().parents[2] / "config"
    p = root / "bc_m2_teacher_t1.toml"
    new, prev = (_flat(_toml(root / f)) for f in (p.name, "bc_m2_dagger_r2.toml"))
    assert new.keys() == prev.keys(), "no distill share: no fraction_by_source"
    assert {k for k in prev if new[k] != prev[k]} <= {"data.seed"} | _COUNTED
    assert new["data.seed"] in (16, 17) and new["data.subsample.keep_sources"] == ["botgen", "dagger"]
    assert new["data.chain_start_sources"] == ["botgen"]
    assert vars(model_variant(str(p))) == _ALL_ON
    text = p.read_text(encoding="utf-8")
    fr = [new[k] for k in sorted(_RACE_FRACTIONS)]
    if all(f == 0 for f in fr):
        assert (new["steps.s2"], new["steps.s2_warmup"]) == (0, 0)
        with pytest.raises(Exception, match="fraction_by_race"):
            _bc(p, "s2")
        table = "[data.subsample.fraction_by_race]\nT = 0.0\nP = 0.0\nZ = 0.0\n"
        assert text.count(table) == 1
        p = tmp_path / "count.toml"
        p.write_text(text.replace(table, ""), encoding="utf-8")
    else:
        assert all(0 < f <= 1 for f in fr) and new["steps.s2"] > 0 and new["steps.s2_warmup"] == int(new["steps.s2"] * 0.05 + 0.5)
        assert "PLACEHOLDER" not in text
    cfg, fp = _bc(p, "s2")
    ref, ref_fp = _bc(root / "bc_m2_dagger_r2.toml", "s2")
    assert "fraction_by_source" not in cfg["data"]["subsample"]
    skip = ("data", "steps", "total_steps", "warmup_steps")
    assert {k: x for k, x in cfg.items() if k not in skip} == {k: x for k, x in ref.items() if k not in skip}
    dskip = ("seed", "subsample")
    assert {k: x for k, x in cfg["data"].items() if k not in dskip} == {k: x for k, x in ref["data"].items() if k not in dskip}
    assert cfg["lr"] * cfg["s2_lr_scale"] == pytest.approx(3e-5) and cfg["min_lr"] * cfg["s2_lr_scale"] == pytest.approx(3e-7)
    assert fp != ref_fp
