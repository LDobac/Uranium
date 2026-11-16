import json
import os
from pathlib import Path

import numpy as np
import pytest

from uranium import _core
from uranium.bc.feed_digest import GROUP_FIELDS, group_digest, groups

from testkit import run


def test_group_fields_match_rust():
    assert _core.group_fields() == GROUP_FIELDS


def test_start_slots_generator_020_r21():
    headers = [(43001, 2, 3), (43002, 1, 3), (43003, 1, 3), (43004, 0, 3), (43005, 1, 3), (43006, 1, 0), (43007, 0, 1), (43008, 1, 0)]
    for seed, a, b in headers:
        assert _core.start_slots(seed) == (a, b, a == min(i for i in range(4) if i != b))
    assert _core.start_slots(43006) == (1, 0, True)
    assert any(_core.start_slots(s, 1) != _core.start_slots(s) for s in range(100))
    seeds = _core.rush_check_seeds()
    assert seeds == _core.rush_check_seeds(4, 45001) and len(seeds) == 128 and seeds[0] == 45001
    for k in range(4):
        assert sum(_core.start_slots(s)[2] for s in seeds[32 * k:32 * (k + 1)]) == 16
    import tomllib

    plan = tomllib.loads((Path(__file__).resolve().parents[2] / "config" / "plans" / "rush5_check.toml").read_text(encoding="utf-8"))
    assert plan["seeds"] == seeds


def test_feeder_groups_have_the_030_layout(tiny_config, tmp_path):
    shards = tmp_path / "shards"
    _core.write_test_shards(str(shards), "mixed")
    f = _core.Feeder(str(tiny_config), [str(shards)], prefetch=0, synthetic_maps=True)
    b = f.next()
    assert b is not None and b.lanes == 2
    g = b.group(0, 2, 4, 64, 12)
    r, n = 2 * 4, g["n"]
    assert n % 64 == 0
    assert g["entity_cat"].shape == (r, n, 4) and g["entity_cat"].dtype == np.int32
    assert g["entity_cont"].dtype == np.float32 and g["entity_mask"].dtype == np.bool_
    assert g["spatial"].shape[0] == r and g["spatial"].shape[2:] == (128, 128)
    assert g["selected"].shape == (r, 12)
    assert g["keep"].shape == (2,) and not g["keep"].any(), "first batch: every lane starts a game"
    assert (g["selected"][:, -1] == n).all()


def test_epoch_walk_and_restore_061_r10(tiny_config, tmp_path):
    shards = tmp_path / "shards"
    _core.write_test_shards(str(shards), "varied")
    f = _core.Feeder(str(tiny_config), [str(shards)], prefetch=2, synthetic_maps=True)
    seq = []
    while (b := f.next()) is not None:
        seq.append((b.state_json(), [group_digest(g) for g in groups(b, 1, 4, 0, 12)]))
    table = json.loads(_core.epoch_table(str(tiny_config), [str(shards)]))
    assert len(seq) == table["s1"]["updates"]
    k = 3
    r = _core.Feeder(str(tiny_config), [str(shards)], prefetch=0, synthetic_maps=True, restore=seq[k][0])
    for want_state, want in seq[k + 1:]:
        b = r.next()
        assert b.state_json() == want_state
        got = [group_digest(g) for g in groups(b, 1, 4, 0, 12)]
        if want_state == seq[k + 1][0]:
            continue
        assert got == want


def test_lr_schedule_reference(tiny_config):
    lrs = [_core.lr_at(str(tiny_config), "s1", s) for s in range(5)]
    assert lrs[0] == pytest.approx(3e-3)
    assert all(lr == pytest.approx(3e-3) for lr in lrs)


@pytest.mark.parametrize("pack", [False, True])
def test_digest_matches_rust_feed_digest(tiny_config, tmp_path, train_bin, python_cmd, pack):
    shards = tmp_path / "shards"
    _core.write_test_shards(str(shards), "mixed")
    args = ["feed-digest", "--config", str(tiny_config), "--shards", str(shards), "--batches", "5", "--synthetic-maps"] + (["--pack"] if pack else [])
    rust = [json.loads(l) for l in run([str(train_bin)] + args).splitlines()]
    py = [json.loads(l) for l in run(python_cmd + args).splitlines()]
    assert len(rust) >= 3 and rust == py


@pytest.mark.data
def test_real_shards_match_rust(train_bin, python_cmd):
    dirs = [d for d in os.environ.get("URANIUM_TEST_SHARDS", "").split(os.pathsep) if d]
    if not dirs:
        pytest.skip("URANIUM_TEST_SHARDS not set")
    args = ["feed-digest", "--config", "config/bc.toml", "--batches", "2", "--group", "4"]
    for d in dirs:
        args += ["--shards", d]
    rust = [json.loads(l) for l in run([str(train_bin)] + args).splitlines()]
    py = [json.loads(l) for l in run(python_cmd + args).splitlines()]
    assert len(rust) == 2 and rust == py
