import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "config"
PLANS = ROOT / "plans"

NET = {"net.entity.d_model": 512, "net.entity.ffn": 2048, "net.entity.layers": 6, "net.entity.heads": 16,
       "net.core.hidden": 1024, "net.core.gtrxl_layers": 4, "net.scalar.hidden": 512, "net.heads.ar": 512,
       "net.value.hidden": 512, "net.spatial.patch_dim": 512}
CNN_ONLY = {"model.spatial_preact", "model.spatial_norm", "monitor.spatial_every"}
RACE = {f"data.subsample.fraction_by_race.{r}" for r in "TPZ"}
COUNTED = {"data.subsample.fraction", "steps.s2", "steps.s2_warmup"} | RACE | {"data.subsample.fraction_by_source.distill"}

FIXED = {
    "bc_m2b_s1": "bc_m2",
    "bc_m2b_s1e2": "bc_m2_e2",
    "bc_m2b_s2e2": "bc_m2",
    **{f"bc_m2b_mix_e{e}": f"bc_m2_mix_e{e}" for e in range(1, 8)},
}
RECOUNTED = {
    "bc_m2b_dagger_r1": ("bc_m2_dagger_r1", 13),
    "bc_m2b_dagger_r2": ("bc_m2_dagger_r2", 14),
    "bc_m2b_distill_d1": ("bc_m2_distill_d1", 15),
    "bc_m2b_teacher_t1": ("bc_m2_teacher_t1", 17),
    "bc_m2b_gn_t1": ("bc_m2_gn_t1", 17),
    "bc_m2b_teacher_t2": ("bc_m2_teacher_t2", 18),
}
PLAN_MAP = {
    "dagger_m2b_r1": ("dagger_m2_r1", "m2b-mix-e7", "bc_m2b_mix_e7"),
    "dagger_m2b_r2": ("dagger_m2_r2", "m2b-dagger-r1", "bc_m2b_dagger_r1"),
    "dagger_m2b_r3": ("dagger_m2_r3", "m2b-dagger-r2", "bc_m2b_dagger_r2"),
    "distill_m2b_d1": ("distill_m2_d1", "m2b-dagger-r2", "bc_m2b_dagger_r2"),
    "distill_m2b_d1_x1": ("distill_m2_d1_x1", "m2b-dagger-r2", "bc_m2b_dagger_r2"),
    "dagger_m2b_t1": ("dagger_m2_t1", "m2b-distill-d1", "bc_m2b_distill_d1"),
    "dagger_m2b_t2": ("dagger_m2_t2", "m2b-gn-t1e2", "bc_m2b_gn_t1"),
    "dagger_m2b_t2_z": ("dagger_m2_t2_z", "m2b-gn-t1e2", "bc_m2b_gn_t1"),
}


def _flat(t, pre=""):
    out = {}
    for k, x in t.items():
        out.update(_flat(x, f"{pre}{k}.") if isinstance(x, dict) else {f"{pre}{k}": x})
    return out


def _toml(path):
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _expected(orig):
    exp = {k: v for k, v in _flat(_toml(ROOT / f"{orig}.toml")).items() if k not in CNN_ONLY}
    return {**exp, "model.spatial_patch": 8, **NET}


def _present(names):
    return [n for n in names if (ROOT / f"{n}.toml").exists()]


def test_m2b_config_files_known_045_ac1():
    known = set(FIXED) | set(RECOUNTED)
    assert {p.stem for p in ROOT.glob("bc_m2b_*.toml")} <= known
    assert set(_present(FIXED)) == set(FIXED)
    assert {p.stem for p in PLANS.glob("*m2b*.toml")} <= set(PLAN_MAP)


@pytest.mark.parametrize("name", sorted(FIXED))
def test_m2b_fixed_config_diff_045_r1_ac1(name):
    assert _flat(_toml(ROOT / f"{name}.toml")) == _expected(FIXED[name])


@pytest.mark.parametrize("name", sorted(RECOUNTED))
def test_m2b_recounted_config_diff_045_r2_ac1(name):
    if not (ROOT / f"{name}.toml").exists():
        pytest.skip("written by the run driver after that stage's games (045-R2 3)")
    orig, seed = RECOUNTED[name]
    new, exp = _flat(_toml(ROOT / f"{name}.toml")), _expected(orig)
    assert new.keys() == exp.keys()
    assert {k: v for k, v in new.items() if k not in COUNTED} == {k: v for k, v in exp.items() if k not in COUNTED}
    assert new["data.seed"] == seed
    assert new["steps.s2"] > 0 and new["steps.s2_warmup"] == int(new["steps.s2"] * 0.05 + 0.5)
    assert all(0 < new[k] <= 1 for k in RACE | {"data.subsample.fraction"})


def test_m2b_gn_counts_are_t1s_045_r2():
    if not all((ROOT / f"{n}.toml").exists() for n in ("bc_m2b_gn_t1", "bc_m2b_teacher_t1")):
        pytest.skip("written after the m2b t1 count")
    gn, t1 = (_flat(_toml(ROOT / f"{n}.toml")) for n in ("bc_m2b_gn_t1", "bc_m2b_teacher_t1"))
    keys = COUNTED - {"data.subsample.fraction_by_source.distill"}
    assert {k: gn[k] for k in keys} == {k: t1[k] for k in keys}


@pytest.mark.parametrize("name", sorted(set(FIXED) | set(RECOUNTED)))
def test_m2b_model_and_net_045_r1(name):
    if not (ROOT / f"{name}.toml").exists():
        pytest.skip("not written yet")
    t = _toml(ROOT / f"{name}.toml")
    t2 = {k: v for k, v in _toml(ROOT / "bc_m2_teacher_t2.toml")["model"].items() if f"model.{k}" not in CNN_ONLY}
    assert t["model"] == {**t2, "spatial_patch": 8}
    assert _flat(t["net"], "net.") == NET
    assert "monitor" not in t


def test_m2b_fixed_steps_are_the_original_epochs_045_r2():
    steps = {n: _toml(ROOT / f"{n}.toml")["steps"] for n in FIXED}
    assert steps["bc_m2b_s1"]["s1"] == 39096 and steps["bc_m2b_s1e2"]["s1"] == 39094 and steps["bc_m2b_s2e2"]["s2"] == 11387
    assert sum(steps[f"bc_m2b_mix_e{e}"]["s2"] for e in range(1, 8)) == 8417
    seeds = [_toml(ROOT / f"bc_m2b_mix_e{e}.toml")["data"]["seed"] for e in range(1, 8)]
    assert seeds == list(range(3, 10))


@pytest.mark.parametrize("name", sorted(PLAN_MAP))
def test_m2b_plan_is_the_original_with_checkpoint_and_name_045_r3(name):
    p = PLANS / f"{name}.toml"
    if not p.exists():
        pytest.skip("written by the run driver after the stage that plays it")
    orig, run, cfg = PLAN_MAP[name]
    a, b = (x.read_text(encoding="utf-8").replace("\r\n", "\n") for x in (PLANS / f"{orig}.toml", p))
    ta, tb = tomllib.loads(a), tomllib.loads(b)
    assert ta["seeds"] == tb["seeds"]
    ck = set(re.findall(r'checkpoint = "(checkpoints/m2b-[^"]+)"', b))
    assert len(ck) == 1
    m = re.fullmatch(rf"checkpoints/{run}/ckpt-(\d+)", ck.pop())
    assert m
    if (ROOT / f"{cfg}.toml").exists():
        assert int(m.group(1)) == _toml(ROOT / f"{cfg}.toml")["steps"]["s2"]
    na = re.sub(r'checkpoint = "checkpoints/m2-[^"]+"', "CK", a)
    nb = re.sub(r'checkpoint = "checkpoints/m2b-[^"]+"', "CK", b)
    la, lb = na.split("\n"), nb.split("\n")
    assert len(la) == len(lb)
    diff = [(x, y) for x, y in zip(la, lb) if x != y]
    assert diff == [(f'name = "{ta["name"]}"', f'name = "{tb["name"]}"')]
    assert "m2b" in tb["name"]


COPIES = ("distill_m2_d1", "distill_m2_d1_x1", "dagger_m2_t1", "dagger_m2_t2", "dagger_m2_t2_z")


def _m2b_copy(src, dst, seeds=None):
    t = src.read_text(encoding="utf-8").replace("\r\n", "\n")
    t = re.sub(r'checkpoint = "checkpoints/m2-[^"]+"', 'checkpoint = "checkpoints/m2b-x/ckpt-1"', t)
    t = re.sub(r'^name = "([^"]+)"', r'name = "\1-m2b"', t, count=1, flags=re.M)
    if seeds is not None:
        t = re.sub(r"^seeds = \[.*?\]", "seeds = [" + ", ".join(map(str, seeds)) + "]", t, count=1, flags=re.M | re.S)
    dst.write_text(t, encoding="utf-8")


def _plans_with_copies(tmp_path):
    import shutil

    d = tmp_path / "plans"
    shutil.copytree(PLANS, d)
    for n in COPIES:
        _m2b_copy(d / f"{n}.toml", d / f"{n.replace('_m2_', '_m2b_', 1)}.toml")
    return d


def test_m2b_copy_skipped_when_original_regenerated_045_r3(tmp_path):
    from uranium.measure import rush_check_plan as rcp

    d = _plans_with_copies(tmp_path)
    for n in COPIES:
        orig, copy = d / f"{n}.toml", d / f"{n.replace('_m2_', '_m2b_', 1)}.toml"
        assert rcp.is_m2b_copy(copy) and not rcp.is_m2b_copy(orig)
        seeds = rcp.plan_seeds(orig)
        assert seeds == rcp.plan_seeds(copy)
        assert not rcp.taken_seeds(d, (orig,)) & seeds
        assert seeds <= rcp.taken_seeds(d, ())


def test_m2b_plan_with_other_seeds_counted_045_r3(tmp_path):
    from uranium.measure import rush_check_plan as rcp

    d = _plans_with_copies(tmp_path)
    copy = d / "distill_m2b_d1.toml"
    new = [7_700_000_001, 7_700_000_002]
    _m2b_copy(d / "distill_m2_d1.toml", copy, new)
    assert not rcp.is_m2b_copy(copy)
    assert set(new) <= rcp.taken_seeds(d, (d / "distill_m2_d1.toml",))


def test_m2b_plan_without_original_counted_045_r3(tmp_path):
    from uranium.measure import rush_check_plan as rcp

    d = _plans_with_copies(tmp_path)
    lone = d / "distill_m2b_d1_x9.toml"
    new = [7_700_000_003, 7_700_000_004]
    _m2b_copy(d / "distill_m2_d1.toml", lone, new)
    assert not rcp.is_m2b_copy(lone)
    assert set(new) <= rcp.taken_seeds(d, ())


def test_m2b_copies_do_not_block_regeneration_045_r3(tmp_path):
    from uranium.measure import rush_check_plan as rcp

    d = _plans_with_copies(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    for name, plan in (("d1", rcp.write_distill(d, 1, "checkpoints/x/ckpt-1", out / "d1.toml")),
                       ("x1", rcp.write_distill(d, 1, "checkpoints/x/ckpt-1", out / "x1.toml", extra=500, part=1)),
                       ("t1", rcp.write_teacher_round(d, "checkpoints/x/ckpt-1", out / "t1.toml", rnd="t1")),
                       ("t2", rcp.write_teacher_round(d, "checkpoints/x/ckpt-1", out / "t2.toml", rnd="t2")),
                       ("t2z", rcp.write_teacher_round(d, "checkpoints/x/ckpt-1", out / "t2z.toml", rnd="t2", zerg=600))):
        orig = {"d1": "distill_m2_d1", "x1": "distill_m2_d1_x1", "t1": "dagger_m2_t1", "t2": "dagger_m2_t2", "t2z": "dagger_m2_t2_z"}[name]
        assert rcp.plan_seeds(plan) == rcp.plan_seeds(d / f"{orig}.toml"), name
