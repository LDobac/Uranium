"""`uranium._core` data side (041, AMB-000-4 step 1-2): the Python trainer reads exactly the
batches of the Rust Feeder."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from uranium import _core
from uranium.data import GROUP_FIELDS, group_digest, groups

from conftest import run


def test_group_fields_match_rust():
    assert _core.group_fields() == GROUP_FIELDS


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
    # END = padded n after the selection (032-R12).
    assert (g["selected"][:, -1] == n).all()


def test_epoch_walk_and_restore_061_r10(tiny_config, tmp_path):
    """The batch sequence is the Rust lane walk (041-R13 count) and a Feeder restored from a
    batch's state gives the batches that followed (061-R10)."""
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
        # Lanes restored mid-game start with a fresh core state (061-R10), so `keep` differs on
        # the first batch only; the data is the same.
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
    """Byte equality with the Rust Feeder through an independent hash (041-R18 packed or not)."""
    shards = tmp_path / "shards"
    _core.write_test_shards(str(shards), "mixed")
    args = ["feed-digest", "--config", str(tiny_config), "--shards", str(shards), "--batches", "5", "--synthetic-maps"] + (["--pack"] if pack else [])
    rust = [json.loads(l) for l in run([str(train_bin)] + args).splitlines()]
    py = [json.loads(l) for l in run(python_cmd + args).splitlines()]
    assert len(rust) >= 3 and rust == py


@pytest.mark.data
def test_real_shards_match_rust(train_bin, python_cmd):
    """Real BC shards and map precomputes (`URANIUM_TEST_SHARDS`, `URANIUM_MAPS_DIR`)."""
    dirs = [d for d in os.environ.get("URANIUM_TEST_SHARDS", "").split(os.pathsep) if d]
    if not dirs:
        pytest.skip("URANIUM_TEST_SHARDS not set")
    args = ["feed-digest", "--config", "config/bc.toml", "--batches", "2", "--group", "4"]
    for d in dirs:
        args += ["--shards", d]
    rust = [json.loads(l) for l in run([str(train_bin)] + args).splitlines()]
    py = [json.loads(l) for l in run(python_cmd + args).splitlines()]
    assert len(rust) == 2 and rust == py
