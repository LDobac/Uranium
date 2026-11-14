"""041-R24 loss-weight selection run configs (P3, AMB-041-13a): config/bc_m2_lambda_*.toml, and the
model 2 path configs with the confirmed λ (041-R29, AMB-041-13b): bc_m2.toml, bc_m2_e2.toml,
bc_m2_mix_e1.toml ... e7, the DAgger rounds bc_m2_dagger_r1.toml / r2 / r3 and their plans dagger_m2_r1.toml / r2 / r3."""

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


# Round -> (data.seed, the lineage's config before it, keys that may differ from that one besides the counted,
# the Checkpoint 1 path's round compared with: round 3 is lineage 2 only (user decision 2026-11-14), so it is held
# against Checkpoint 1's last round 2 with the data seed apart).
_M2_DAGGER = {
    1: (13, "bc_m2_mix_e7.toml", {"data.seed", "data.subsample.keep_sources"}, 1),
    2: (14, "bc_m2_dagger_r1.toml", {"data.seed"}, 2),
    3: (15, "bc_m2_dagger_r2.toml", {"data.seed"}, 2),
}


@pytest.mark.parametrize("rnd", sorted(_M2_DAGGER))
def test_m2_dagger_config_041_r29(tmp_path, rnd):
    """041-R29 ④ / 041-R22 ③ / 041-R26 / 041-R30 ⑥: config/bc_m2_dagger_r<R>.toml is the Checkpoint 1 path's
    bc_dagger_r<R>.toml (r3: bc_dagger_r2.toml; 1 epoch, learning rate, data.seed 13 / 14 / 15, keep_sources
    botgen + dagger) with the model 2 tables of the lineage's config before it (r1: bc_m2_mix_e7.toml, r2 / r3:
    the round before; [model],
    [aux], [monitor], keep_checkpoints 0), chain_start_sources ["botgen"] (not dagger), and its own per-race
    human fractions and [steps] (the same 0.5954 teacher + dagger target in every round). Until the run
    driver's count fills them, the placeholder zero fraction_by_race makes the loader refuse the file (s2 = 0
    would mean total_steps); filled, they are a valid race subsample and a 5% warmup (041-R14)."""
    from uranium.bc.run import model_variant

    seed, before, may_differ, ck1_round = _M2_DAGGER[rnd]
    seed_apart = {"data.seed"} if ck1_round != rnd else set()
    root = Path(__file__).resolve().parents[2] / "config"
    p = root / f"bc_m2_dagger_r{rnd}.toml"
    new, ck1, prev = (_flat(_toml(root / f)) for f in (p.name, f"bc_dagger_r{ck1_round}.toml", before))
    # TOML level against Checkpoint 1's same round: only the model 2 keys, 041-R30 and the counted values differ.
    diff = {k for k in new.keys() | ck1.keys() if new.get(k) != ck1.get(k)}
    assert _M2_TABLES | {"data.chain_start_sources"} | _RACE_FRACTIONS <= diff
    allowed = _M2_TABLES | {"data.chain_start_sources"} | _COUNTED | seed_apart
    assert diff <= allowed, diff - allowed
    # ... and against the lineage's config before it: same keys; the DAgger data seed (and sources), counted values.
    assert new.keys() == prev.keys()
    assert {k for k in new if new[k] != prev[k]} <= may_differ | _COUNTED
    assert new["data.seed"] == seed and new["data.subsample.keep_sources"] == ["botgen", "dagger"]
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


# Round -> (games, the model that plays: the lineage's model before the round, 041-R22 ③ / 041-R29 ④; the Checkpoint 1
# path's plan compared with: round 3 is lineage 2 only, held against Checkpoint 1's half plan of round 2).
_M2_DAGGER_PLAN = {
    1: (600, "checkpoints/m2-mix-e7/ckpt-1212", 1),
    2: (300, "checkpoints/m2-dagger-r1/ckpt-1285", 2),
    3: (300, "checkpoints/m2-dagger-r2/ckpt-1408", 2),
}


@pytest.mark.parametrize("rnd", sorted(_M2_DAGGER_PLAN))
def test_m2_dagger_plan_041_r28(rnd):
    """041-R29 ④ / 041-R22 ① / 041-R28 ①: config/plans/dagger_m2_r<R>.toml (rush_check_plan.py --dagger --round R
    [--half for rounds 2, 3] --memory-log --checkpoint <the lineage's model before the round>) is the Checkpoint 1
    path's same-round plan (same seeds and matchups: r1 600 games n = 1..600, r2 the half plan 300 games
    n = 601..900; stochastic, 10 min, replays) with the lineage-2 model and the decision logs. Round 3 (lineage 2
    only, user decision 2026-11-14) is the round-2 half plan's matchups with the round-3 seeds n = 1,201..1,500
    (splitmix64(70,000 + n) >> 32, the 600-wide round stride)."""
    import importlib.util

    games, ckpt, ck1_round = _M2_DAGGER_PLAN[rnd]
    root = Path(__file__).resolve().parents[2] / "config" / "plans"
    new, ck1 = _toml(root / f"dagger_m2_r{rnd}.toml"), _toml(root / f"dagger_r{ck1_round}.toml")
    assert new["memory_log"] is True and "memory_log" not in ck1 and new["save_replay"] is True
    apart = ("memory_log", "matchup") + (("name", "seeds") if ck1_round != rnd else ())
    assert {k: x for k, x in new.items() if k not in apart} == {k: x for k, x in ck1.items() if k not in apart}
    if ck1_round != rnd:
        spec = importlib.util.spec_from_file_location("rush_check_plan", root.parents[1] / "tools" / "eval" / "rush_check_plan.py")
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        first = (rnd - 1) * gen.DAGGER_ROUND_STRIDE + 1
        assert new["name"] == f"dagger-r{rnd}"
        assert new["seeds"] == [gen.splitmix64(gen.DAGGER_SEED_KEY + k) >> 32 for k in range(first, first + games)]
    assert len(new["seeds"]) == games == sum(m["games"] for m in new["matchup"])
    for a, b in zip(new["matchup"], ck1["matchup"], strict=True):
        assert a["p0"] == {"kind": "model", "checkpoint": ckpt, "sample": "stochastic"}
        assert {k: x for k, x in a.items() if k != "p0"} == {k: x for k, x in b.items() if k != "p0"}


_SOURCE_FRACTIONS = {"data.subsample.fraction_by_source.distill"}


def test_m2_distill_d1_config_044_r5(tmp_path):
    """044-R5 / 041-R31: config/bc_m2_distill_d1.toml is config/bc_m2_dagger_r2.toml (041-R22 ③ one epoch, learning
    rate 3e-5 / 3e-7, model 2 tables, keep_sources botgen + dagger, chain_start_sources ["botgen"]) with data.seed 15
    and the distill source's share ([data.subsample.fraction_by_source] distill, 044-R5 ③), its own per-race human
    fractions and [steps] (`mix-fractions --share teacher=0.45,distill=0.2`). Until the run driver's count fills
    them, the placeholder zero fraction_by_race / fraction_by_source make the loader refuse the file; filled, they
    are a valid subsample and a 5% warmup (041-R14)."""
    from uranium.bc.run import model_variant

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
        # Only the race table stripped: the zero distill share is still refused.
        p1 = tmp_path / "race_only.toml"
        p1.write_text(text.replace(tables[0], ""), encoding="utf-8")
        with pytest.raises(Exception, match="fraction_by_source"):
            _bc(p1, "s2")
        p = tmp_path / "count.toml"
        p.write_text(text.replace(tables[0], "").replace(tables[1], ""), encoding="utf-8")
    else:
        assert all(0 < f <= 1 for f in fr) and 0 < new["data.subsample.fraction"] <= 1
        assert new["steps.s2"] > 0 and new["steps.s2_warmup"] == int(new["steps.s2"] * 0.05 + 0.5)
        # S (the 041-R29 ④ selection) plays the d1 games, so it is named before the count.
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
