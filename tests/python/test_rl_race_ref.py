"""044-AC4 / 042-AC12 (042-R22 ①, 044-R9), CPU: race-aware KL reference. Without `[kl.reference]`
the update is as before (`kl_ref` empty); with every key on the `--bc` checkpoint the update is
bit for bit the one-reference update; lanes of different lobby races take the KL to their own
race's reference (Random / Unknown lobby = `random`); a reference with another model config is a
start error; keys naming the same checkpoint share one model."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from uranium import _core
from uranium.model import UraniumNet, UraniumNetConfig, action_columns, input_dims, to_labels, to_obs
from uranium.rl.learner import HEADS, REF_RACES, Learner, kl_per_head, lanes_by_ref

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "burn_parity"
CPU = torch.device("cpu")
WEIGHTS = [1.0, 0.3, 0.3, 0.05]
# Lanes of every lobby race; an Unknown lobby race (a lobby Random as BWAPI reports it) is `random`.
RACES = ["zerg", "terran", "protoss", "random", "zerg", "unknown"]
STAT_KEYS = ("loss", "policy", "value", "kl", "win_open", "kl_heads", "entropy_heads", "rho_mean", "adv_mean", "grad_norm", "adv_rms", "adv_rms_channels", "value_loss_channels", "kl_p99", "kl_max")


def tiny_net(seed=5) -> UraniumNet:
    cfg = UraniumNetConfig.from_dict(json.loads((FIX / "tiny-lstm" / "meta.json").read_text())["config"])
    cfg.input = input_dims()
    torch.manual_seed(seed)
    return UraniumNet(cfg, action_columns())


def rl_cfg(lanes: int) -> dict:
    cfg = json.loads(_core.rl_config(str(ROOT / "config" / "rl.toml")))
    cfg.update(lanes=lanes, seq_len=4, micro_lanes=1)
    cfg["optimizer"]["lr"] = 1e-3
    return cfg


class OnPolicy:
    """A segment batch whose behaviour log-probs are `net`'s own; `group` and `group_lanes` as
    `_core.SegmentBatch` (044-R9 lane groups by reference collate lane lists)."""

    def __init__(self, batch, net):
        self.batch, self.net = batch, net

    @property
    def lanes(self):
        return self.batch.lanes

    @property
    def trained(self):
        return self.batch.trained

    @property
    def versions(self):
        return self.batch.versions

    @property
    def ref_races(self):
        return self.batch.ref_races

    def behaviour(self, g, t):
        with torch.no_grad():
            out = self.net.forward_seq(to_obs(g, CPU), to_labels(g, CPU), torch.as_tensor(g["privileged"]).float(), self.net.core.init_state(g["b"], CPU), t)
        g["behaviour"] = sum(out["logp"][k] for k in HEADS).numpy().astype(np.float32)
        return g

    def group(self, start, end, t, max_select):
        return self.behaviour(self.batch.group(start, end, t, max_select), t)

    def group_lanes(self, lanes, t, max_select):
        return self.behaviour(self.batch.group_lanes(lanes, t, max_select), t)


def batch_of(races=RACES):
    return OnPolicy(_core.test_segments(len(races), 4, races=races), tiny_net(5))


def run(l: Learner, batch, updates: int = 2) -> tuple[list, dict]:
    stats = [l.update(batch, WEIGHTS) for _ in range(updates)]
    return stats, {k: v.detach().clone() for k, v in l.net.state_dict().items()}


def test_lobby_race_codes_and_lane_groups_044_r9():
    """The lane's `[kl.reference]` key index: zerg 0, terran 1, protoss 2, random 3 (a lobby
    Random given as Random or Unknown); `group` and `group_lanes` carry it; the lane groups by
    reference keep batch order within each reference."""
    b = _core.test_segments(len(RACES), 4, races=RACES)
    assert list(b.ref_races) == [0, 1, 2, 3, 0, 3]
    assert list(b.group(0, 6, 4, 12)["ref_race"]) == [0, 1, 2, 3, 0, 3]
    g = b.group_lanes([4, 0, 3], 4, 12)
    assert list(g["ref_race"]) == [0, 0, 3]
    whole = b.group(0, 6, 4, 12)
    for k in ("entity_cat", "action", "mask", "rewards", "terminal", "trained"):
        np.testing.assert_array_equal(b.group_lanes(list(range(6)), 4, 12)[k], whole[k])
    assert list(_core.test_segments(2, 4).ref_races) == [3, 3], "no lobby race given = unknown = random"
    with pytest.raises(ValueError):
        _core.test_segments(1, 4, races=["zergs"])
    with pytest.raises(ValueError):
        b.group_lanes([6], 4, 12)
    assert lanes_by_ref([0, 1, 2, 3, 0, 3], [1, 0, 0, 0], 6, 2) == [([1, 2], 0), ([3, 5], 0), ([0, 4], 1)]
    assert lanes_by_ref([0, 1, 2, 3, 0, 3], [1, 0, 0, 2], 5, 4) == [([1, 2], 0), ([0, 4], 1), ([3], 2)]


@pytest.mark.parametrize("micro_lanes", [1, 4])
def test_all_races_on_bc_is_bit_identical_044_ac4(micro_lanes):
    """044-AC4: every race on the `--bc` model (race_ref all 0) = the update without the table, bit
    for bit (statistics and weights over two updates); only `kl_ref` is added. Without the table
    `kl_ref` is empty (None)."""
    batch = batch_of()
    old, w_old = run(Learner(rl_cfg(6), tiny_net(5), tiny_net(5), CPU, "f32", "vtrace", micro_lanes=micro_lanes), batch)
    new, w_new = run(Learner(rl_cfg(6), tiny_net(5), tiny_net(5), CPU, "f32", "vtrace", micro_lanes=micro_lanes, race_ref=[0, 0, 0, 0]), batch)
    for a, b in zip(old, new):
        assert {k: a[k] for k in STAT_KEYS} == {k: b[k] for k in STAT_KEYS}
        assert a["kl_ref"] == [None] * 4
        assert all(x is not None for x in b["kl_ref"])
    assert all(torch.equal(w_old[k], w_new[k]) for k in w_old)
    # The table's statistics: per race the mean over its trained steps; their step-weighted mean is
    # the batch KL (every lane here is a queue lane taking the KL).
    trained = list(batch.trained)
    steps = [sum(x for r, x in zip(batch.ref_races, trained) if r == i) for i in range(4)]
    assert sum(k * s for k, s in zip(new[1]["kl_ref"], steps)) / sum(steps) == pytest.approx(new[1]["kl"], rel=1e-6)


def expected_kl_by_race(net, refs: list, race_ref: list, batch) -> list:
    """Per lobby race the mean per-step KL(π ‖ π_ref(race)) (used heads summed) of the trained
    steps, from one forward of every lane (the update's first KL, before its step)."""
    g = batch.group(0, batch.lanes, 4, 12)
    obs, labels, priv = to_obs(g, CPU), to_labels(g, CPU), torch.as_tensor(g["privileged"]).float()
    with torch.no_grad():
        out = net.forward_seq(obs, labels, priv, net.init_state(g["b"], CPU), 4, phased=True)
        per_ref = []
        for r in refs:
            o = r.forward_seq(obs, labels, priv, r.init_state(g["b"], CPU), 4, phased=True)
            kl = kl_per_head(out["dists"], o["dists"])
            per_ref.append((sum(kl[h] for h in HEADS) * torch.as_tensor(g["mask"])).view(g["b"], 4).sum(1))
    sums, steps = [0.0] * 4, [0.0] * 4
    for lane, race in enumerate(g["ref_race"]):
        sums[race] += float(per_ref[race_ref[race]][lane])
        steps[race] += float(g["trained"][lane])
    return [s / c if c else None for s, c in zip(sums, steps)]


@pytest.mark.parametrize("micro_lanes", [1, 2, 0])
def test_lanes_take_their_race_reference_044_ac4(micro_lanes):
    """044-AC4: lanes of different lobby races each take the KL to their race's reference: zerg →
    S, random (Random and Unknown lobby) → R, terran / protoss → --bc. `kl_ref` per race matches the
    KL against that race's model; the batch KL is their step-weighted mean."""
    net, bc, s, r = tiny_net(5), tiny_net(7), tiny_net(9), tiny_net(11)
    race_ref = [1, 0, 0, 2]
    batch = batch_of()
    want = expected_kl_by_race(net, [bc, s, r], race_ref, batch)
    l = Learner(rl_cfg(6), copy.deepcopy(net), copy.deepcopy(bc), CPU, "f32", "vtrace", micro_lanes=micro_lanes, refs=[copy.deepcopy(s), copy.deepcopy(r)], race_ref=race_ref)
    st = l.update(batch, WEIGHTS)
    assert st["kl_ref"] == pytest.approx(want, rel=1e-4)
    trained = list(batch.trained)
    steps = [sum(x for q, x in zip(batch.ref_races, trained) if q == i) for i in range(4)]
    assert st["kl"] == pytest.approx(sum(k * c for k, c in zip(want, steps)) / sum(steps), rel=1e-4)
    # Not the KL to --bc alone: the zerg and random lanes use other models.
    only_bc = expected_kl_by_race(net, [bc], [0, 0, 0, 0], batch)
    assert only_bc[0] != pytest.approx(want[0], rel=1e-3) and only_bc[3] != pytest.approx(want[3], rel=1e-3)
    assert only_bc[1] == pytest.approx(want[1], rel=1e-4) and only_bc[2] == pytest.approx(want[2], rel=1e-4)
    # A race without lanes in the batch: no value.
    st = Learner(rl_cfg(2), copy.deepcopy(net), copy.deepcopy(bc), CPU, "f32", "vtrace", micro_lanes=micro_lanes, refs=[copy.deepcopy(s)], race_ref=[1, 0, 0, 0]).update(batch_of(["zerg", "terran"]), WEIGHTS)
    assert st["kl_ref"][2] is None and st["kl_ref"][3] is None and st["kl_ref"][0] is not None


def test_lane_groups_by_reference_train_like_one_group_044_r9():
    """044-R9: a second reference with the --bc weights (another model object, so the lanes are
    grouped by reference) gives the same update as one reference within float rounding: the
    regrouped lanes collate and train as the ranges do."""
    batch = batch_of()
    one, w_one = run(Learner(rl_cfg(6), tiny_net(5), tiny_net(7), CPU, "f32", "vtrace", micro_lanes=2), batch)
    two, w_two = run(Learner(rl_cfg(6), tiny_net(5), tiny_net(7), CPU, "f32", "vtrace", micro_lanes=2, refs=[tiny_net(7)], race_ref=[1, 0, 1, 0]), batch)
    for a, b in zip(one, two):
        for k in ("loss", "policy", "value", "kl", "grad_norm"):
            assert b[k] == pytest.approx(a[k], rel=1e-5, abs=1e-7), k
    assert all(torch.allclose(w_one[k], w_two[k], rtol=1e-5, atol=1e-7) for k in w_one)


def test_reset_to_bc_and_bad_race_ref_044_r9():
    """The exploiter reset target stays --bc (042-R22 ①); a race_ref outside the references is an
    error; the reference learner (one KL reference) refuses a table."""
    from uranium.rl.reference import ReferenceLearner

    bc, s = tiny_net(7), tiny_net(9)
    l = Learner(rl_cfg(2), tiny_net(5), bc, CPU, "f32", "vtrace", refs=[s], race_ref=[1, 0, 0, 0])
    l.reset_to_bc()
    assert all(torch.equal(a, b) for a, b in zip(l.net.state_dict().values(), bc.state_dict().values()))
    with pytest.raises(ValueError, match="race_ref"):
        Learner(rl_cfg(2), tiny_net(5), tiny_net(7), CPU, "f32", "vtrace", refs=[s], race_ref=[2, 0, 0, 0])
    with pytest.raises(ValueError, match="race_ref"):
        Learner(rl_cfg(2), tiny_net(5), tiny_net(7), CPU, "f32", "vtrace", race_ref=[0, 0, 0])
    with pytest.raises(NotImplementedError, match="044-R9"):
        ReferenceLearner(rl_cfg(2), tiny_net(5), tiny_net(7), CPU, "f32", "vtrace", race_ref=[0, 0, 0, 0]).update(batch_of(["zerg", "terran"]), WEIGHTS)
    assert ReferenceLearner(rl_cfg(2), tiny_net(5), tiny_net(7), CPU, "f32", "vtrace").update(batch_of(["zerg", "terran"]), WEIGHTS)["kl_ref"] == [None] * 4


# --- `python -m uranium rl` wiring: [kl.reference] → models (044-R9) -------------------------


def save_ckpt(d: Path, net: UraniumNet) -> Path:
    from uranium import ckpt

    d.mkdir(parents=True, exist_ok=True)
    ckpt.save_model(d, net)
    ckpt.save_meta(d, net.cfg, {}, "f32")
    return d


def refs_of(table: dict, bc_dir: Path):
    from uranium import ckpt
    from uranium.rl.run import _no_dropout, kl_references

    bc, _ = ckpt.load_model(bc_dir, CPU, adjust=_no_dropout)
    cfg = rl_cfg(2)
    cfg["kl"]["reference"] = table
    models, race_ref, record = kl_references(cfg, str(bc_dir), bc, CPU)
    assert (record is None) == (race_ref is None)
    return models, race_ref


def test_reference_table_dedupes_and_resolves_044_r9(tmp_path, monkeypatch):
    """No table: no other model, no race map. Keys naming --bc (also a copy of its folder, and a
    path relative to URANIUM_SHARED_DIR) use --bc itself: no model loaded, race_ref all 0, so the
    update is the one-reference update (044-AC4). Keys naming the same other checkpoint share one
    model; races without a key use --bc."""
    shared = tmp_path / "shared"
    bc = save_ckpt(shared / "checkpoints" / "b" / "ckpt-1", tiny_net(5))
    copy_dir = shared / "checkpoints" / "b-copy" / "ckpt-1"
    shutil.copytree(bc, copy_dir)
    s = save_ckpt(shared / "checkpoints" / "s" / "ckpt-2", tiny_net(9))
    monkeypatch.setenv("URANIUM_SHARED_DIR", str(shared))
    assert refs_of({}, bc) == ([], None)
    models, race_ref = refs_of({"zerg": str(bc), "terran": "checkpoints/b-copy/ckpt-1", "protoss": "checkpoints/b/ckpt-1", "random": str(copy_dir)}, bc)
    assert (models, race_ref) == ([], [0, 0, 0, 0])
    models, race_ref = refs_of({"zerg": "checkpoints/s/ckpt-2", "random": str(s)}, bc)
    assert len(models) == 1 and race_ref == [1, 0, 0, 1]
    assert models[0].cfg.entity.dropout == 0.0, "dropout 0 as --bc in RL"
    assert all(torch.equal(a, b) for a, b in zip(models[0].state_dict().values(), tiny_net(9).state_dict().values()))
    monkeypatch.delenv("URANIUM_SHARED_DIR")
    with pytest.raises(ValueError, match="URANIUM_SHARED_DIR"):
        refs_of({"zerg": "checkpoints/s/ckpt-2"}, bc)
    with pytest.raises(FileNotFoundError):
        refs_of({"zerg": str(tmp_path / "missing")}, bc)


def test_reference_with_another_model_is_a_start_error_044_ac4(tmp_path):
    """044-AC4: a reference whose model config differs from --bc's fails at start (042 출발 정책)."""
    bc = save_ckpt(tmp_path / "bc", tiny_net(5))
    cfg = tiny_net().cfg
    cfg.core.kind = "Gtrxl" if cfg.core.kind != "Gtrxl" else "Lstm"
    torch.manual_seed(3)
    other = save_ckpt(tmp_path / "other", UraniumNet(cfg, action_columns()))
    with pytest.raises(ValueError, match=r"\[kl.reference\] zerg = .*model config differs from the --bc model \(core\.kind"):
        refs_of({"zerg": str(other)}, bc)


def test_learners_share_the_references_044_r9(monkeypatch, tmp_path):
    """`Learners` (the three agents of `python -m uranium rl`) get the race map and the same
    reference model objects (loaded once); the reset target stays each learner's --bc copy."""
    from uranium.rl import run

    bc = save_ckpt(tmp_path / "bc", tiny_net(5))
    s = save_ckpt(tmp_path / "s", tiny_net(9))
    monkeypatch.setattr(run, "_n_max", lambda: 0)  # no server warm-up (tiny net, CPU)
    cfg = rl_cfg(2)
    cfg["kl"]["reference"] = {"zerg": str(s)}
    ls = run.Learners(cfg, str(bc), None, None, CPU, "f32", "vtrace", tmp_path / "logs", 4, 5)
    for sv in ls.servers:
        sv.stop.set()
        sv.join(timeout=10)
        assert sv.error is None
    assert [l.race_ref for l in ls.learners] == [[1, 0, 0, 0]] * 3
    assert len({id(l.refs[1]) for l in ls.learners}) == 1, "one model for the three learners"
    assert all(l.refs[0] is l.bc for l in ls.learners)
    st = ls.learners[0].update(batch_of(["zerg", "terran"]), WEIGHTS)
    assert st["kl_ref"][0] is not None and st["kl_ref"][1] is not None
    # Every checkpoint records the race → checkpoint map with the weights' sha256 (044-R9).
    from uranium import ckpt
    from uranium.rl.run import KL_REFERENCE_KEY

    ls.save(0, str(tmp_path / "ck" / "main"), json.dumps({"update": 1, "agent": "main"}))
    rec = ckpt.read_meta(tmp_path / "ck" / "main")["train"][KL_REFERENCE_KEY]
    assert rec["zerg"] == {"path": str(s), "abs_path": str(s.resolve()), "sha256": ckpt.policy_version(s)}
    assert all(rec[r] == {"path": "--bc", "abs_path": str(bc.resolve()), "sha256": ckpt.policy_version(bc)} for r in ("terran", "protoss", "random"))


def test_rl_config_reference_json_044_r9():
    """The config JSON the learners read: no `reference` key without the table (as before)."""
    cfg = json.loads(_core.rl_config(str(ROOT / "config" / "rl.toml")))
    assert "reference" not in cfg["kl"]
    assert REF_RACES == ("zerg", "terran", "protoss", "random")


def test_bench_rl_with_two_references_060_ac7():
    """060-AC7 ② (044-R9): `bench rl --references 2` runs the race-aware reference path with a
    second resident reference model (a copy of the checkpoint, so the update equals the
    one-reference update within float rounding) and records the count."""
    from uranium.bench.rl import SyntheticSegments, compare, race_refs, run_learner

    assert race_refs(1) == (0, None)
    assert race_refs(2) == (1, [1, 0, 0, 0]) and race_refs(3) == (2, [1, 2, 0, 0]) and race_refs(4) == (3, [1, 2, 3, 0])
    net = tiny_net(5)
    batch = SyntheticSegments(net, 8, 4, (3, 70), templates=2, seed=3)
    assert list(batch.group(0, 8, 4, 12)["ref_race"]) == [0, 1, 2, 3, 0, 1, 2, 3] == batch.ref_races
    cfg = rl_cfg(8)
    one = run_learner("new", 2, net, cfg, batch, CPU, "f32", "vtrace", 1, False, log=lambda m: None)
    two = run_learner("new", 2, net, cfg, batch, CPU, "f32", "vtrace", 1, False, log=lambda m: None, references=2)
    assert (one["references"], two["references"]) == (1, 2)
    c = compare(one, two)
    assert c["updates"] == 2 and c["finite"] and c["loss"]["max_rel"] < 1e-4 and c["kl"]["max_abs"] < 1e-6, c
