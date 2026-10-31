"""041 BC trainer in PyTorch (AMB-000-4) on synthetic shards, CPU: micro-lane groups sum to the
full-batch gradient (041-R3), checkpoints round-trip (061-AC5), a resumed run continues bit for
bit (061-R10, AC1 on CPU), and `_core.run_bc` runs, checkpoints and resumes a local run."""

import functools
import json
from pathlib import Path

import pytest
import torch

from uranium import _core, ckpt
from uranium.bc.trainer import BcTrainer
from uranium.model import UraniumNet, UraniumNetConfig, action_columns, input_dims

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "burn_parity"
CPU = torch.device("cpu")


def tiny_net_config(kind: str = "Lstm", win_open: bool = False) -> UraniumNetConfig:
    """The tiny parity net with this build's input dims (the collate's real shapes)."""
    cfg = UraniumNetConfig.from_dict(json.loads((FIX / "tiny-lstm" / "meta.json").read_text())["config"])
    cfg.input = input_dims()
    cfg.core.kind = kind
    cfg.value.win_open = win_open
    return cfg


def tiny_net(kind="Lstm", seed=7, win_open=False) -> UraniumNet:
    torch.manual_seed(seed)
    return UraniumNet(tiny_net_config(kind, win_open), action_columns())


def bc_cfg(config: Path, **over) -> dict:
    cfg = json.loads(_core.bc_config(str(config), "s1")[0])
    cfg.update(over)
    return cfg


def feeder(config: Path, shards: Path, restore: str | None = None):
    return _core.Feeder(str(config), [str(shards)], prefetch=0, synthetic_maps=True, check_maps=False, restore=restore)


@pytest.fixture
def shards(tmp_path: Path) -> Path:
    d = tmp_path / "shards"
    _core.write_test_shards(str(d), "mixed", 2)
    return d


def params_of(net) -> dict:
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def test_micro_lanes_match_full_batch_041_r3(tiny_config, shards):
    """Groups of one lane and one group of all lanes give the same update (float rounding)."""
    b = feeder(tiny_config, shards).next()
    after = {}
    for m in (1, 0):
        t = BcTrainer(bc_cfg(tiny_config, micro_lanes=m), tiny_net(), CPU, "f32")
        t.train_batch(b, 3e-3, 0)
        after[m] = params_of(t.net)
    for k in after[0]:
        assert torch.allclose(after[1][k], after[0][k], atol=1e-6, rtol=1e-5), k


@pytest.mark.parametrize("kind", ["Lstm", "Gtrxl"])
def test_resume_continues_bit_identically_061_r10(tiny_config, shards, tmp_path, kind):
    cfg = bc_cfg(tiny_config)
    lr = lambda s: _core.lr_at(str(tiny_config), "s1", s)  # noqa: E731
    a = BcTrainer(cfg, tiny_net(kind), CPU, "f32")
    fa = feeder(tiny_config, shards)
    for s in range(3):
        b = fa.next()
        a.train_batch(b, lr(s), s)
    state = b.state_json()
    a.save(str(tmp_path / "ck"), json.dumps({"step": 3}))
    a.states = []  # the resume point truncates the core state (lanes restart, 061-R10)
    kept = [json.loads(a.train_batch(fa.next(), lr(s), s))["total"] for s in (3, 4)]

    r = BcTrainer.make(json.dumps(cfg), "resume", str(tmp_path / "ck"), 0, kind.lower(), CPU, "f32")
    fb = feeder(tiny_config, shards, restore=state)
    got = [json.loads(r.train_batch(fb.next(), lr(s), s))["total"] for s in (3, 4)]
    assert got == kept, "resumed updates differ from the run that kept going"
    pa, pr = params_of(a.net), params_of(r.net)
    assert all(torch.equal(pa[k], pr[k]) for k in pa)

    # Control: the weights alone (fresh optimizer) do not reproduce the run.
    c = BcTrainer.make(json.dumps(cfg), "init", str(tmp_path / "ck"), 0, kind.lower(), CPU, "f32")
    fc = feeder(tiny_config, shards, restore=state)
    assert [json.loads(c.train_batch(fc.next(), lr(s), s))["total"] for s in (3, 4)] != kept


def test_checkpoint_round_trip_061_ac5(tiny_config, shards, tmp_path):
    cfg = bc_cfg(tiny_config)
    cfg["value"]["win_open"] = 0.3  # 041-R19: the head trains, so it has optimizer state too
    t = BcTrainer(cfg, tiny_net("Gtrxl", win_open=True), CPU, "f32")
    t.train_batch(feeder(tiny_config, shards).next(), 3e-3, 0)
    t.save(str(tmp_path / "a"), json.dumps({"step": 1}))
    t.save(str(tmp_path / "b"), json.dumps({"step": 1}))
    # 061-R10: same weights -> same bytes, so the policy version (020-R5) is stable.
    assert (tmp_path / "a" / ckpt.WEIGHTS_FILE).read_bytes() == (tmp_path / "b" / ckpt.WEIGHTS_FILE).read_bytes()
    assert ckpt.policy_version(tmp_path / "a") == ckpt.policy_version(tmp_path / "b")
    meta = ckpt.read_meta(tmp_path / "a")
    assert meta["framework"] == "pytorch" and meta["precision"] == "f32" and meta["config"]["value"]["win_open"]

    r = BcTrainer.make(json.dumps(t.cfg), "resume", str(tmp_path / "a"), 0, "gtrxl", CPU, "f32")
    pa, pr = params_of(t.net), params_of(r.net)
    assert pa.keys() == pr.keys() and all(torch.equal(pa[k], pr[k]) for k in pa)
    names = dict(t.net.named_parameters())
    rnames = dict(r.net.named_parameters())
    assert len(t.opt.state) == len(r.opt.state) > 0
    for k, p in names.items():
        if p not in t.opt.state:  # no gradient reached it (e.g. unused embedding rows' table)
            assert rnames[k] not in r.opt.state, k
            continue
        sa, sr = t.opt.state[p], r.opt.state[rnames[k]]
        assert torch.equal(sa["exp_avg"], sr["exp_avg"]) and torch.equal(sa["exp_avg_sq"], sr["exp_avg_sq"]), k
        assert float(sa["step"]) == float(sr["step"]) == 1.0


def test_burn_checkpoint_is_refused_061_r10(tmp_path):
    (tmp_path / "model.bpk").write_bytes(b"burn")
    with pytest.raises(ValueError, match="Burn checkpoint"):
        ckpt.read_meta(tmp_path)


def test_feature_schema_6_checkpoint_loads_030_r5(tmp_path):
    """030-R5 / 032-R6 (2026-10-31): feature schema 7 changed masks only, so a schema-6 checkpoint
    (the BC runs) loads as it is; any other schema difference is refused. A new save writes 7."""
    fv, av = ckpt.schema_versions()
    assert (fv, ckpt.feature_load_compat()) == (7, [6])
    net = tiny_net()
    ckpt.save_model(tmp_path, net)
    ckpt.save_meta(tmp_path, net.cfg, {}, "f32")
    meta_path = tmp_path / ckpt.META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["feature_version"] == fv
    for v, ok in [(fv, True), (6, True), (5, False), (fv + 1, False)]:
        meta_path.write_text(json.dumps({**meta, "feature_version": v}), encoding="utf-8")
        if ok:
            loaded, m = ckpt.load_model(tmp_path)
            assert m["feature_version"] == v
            pa, pl = params_of(net), params_of(loaded)
            assert pa.keys() == pl.keys() and all(torch.equal(pa[k], pl[k]) for k in pa)
        else:
            with pytest.raises(ValueError, match="feature schema"):
                ckpt.read_meta(tmp_path)


def _tiny_make(kind, cfg_json, start, dir, entity_chunk):
    cfg = json.loads(cfg_json)
    if start == "new":
        return BcTrainer(cfg, tiny_net(kind), CPU, "f32", entity_chunk)
    return BcTrainer.make(cfg_json, start, str(dir), entity_chunk, kind.lower(), CPU, "f32")


@pytest.mark.parametrize("chunk", ["0", "5"])
def test_run_bc_local_checkpoints_and_resumes(tiny_config, shards, tmp_path, monkeypatch, chunk):
    """The Rust run loop drives the Python trainer: `last/` with every checkpoint file, then a
    resume continues from its step and data position."""
    monkeypatch.setenv("URANIUM_BC_ENTITY_CHUNK", chunk)
    monkeypatch.setenv("URANIUM_BC_PREFETCH", "1")
    make = functools.partial(_tiny_make, "Lstm")
    out = tmp_path / "out"
    end = _core.run_bc(make, str(tiny_config), out=str(out), shards=[str(shards)], steps=2, synthetic_maps=True)
    assert end == "done"
    last = out / "last"
    for f in ("model.safetensors", "optim-main.safetensors", "meta.json", "train_state.json"):
        assert (last / f).is_file(), f
    assert (last / "config").is_dir()
    st = json.loads((last / "train_state.json").read_text())
    assert st["kind"] == "bc" and st["step"] == 2 and "data" in st["extra"]

    out2 = tmp_path / "out2"
    end = _core.run_bc(make, str(tiny_config), out=str(out2), shards=[str(shards)], steps=4, resume=str(last), synthetic_maps=True)
    assert end == "done"
    st = json.loads((out2 / "last" / "train_state.json").read_text())
    assert st["step"] == 4
    assert st["extra"]["games"] >= json.loads((last / "train_state.json").read_text())["extra"]["games"]


def test_bc_eval_scores_every_game_once_041(tiny_config, shards, tmp_path):
    """bc-eval over the three splits reads each synthetic game once and fills the 041 report."""
    from uranium.bc.eval import evaluate

    make = functools.partial(_tiny_make, "Gtrxl")
    out = tmp_path / "out"
    assert _core.run_bc(make, str(tiny_config), out=str(out), shards=[str(shards)], steps=1, synthetic_maps=True) == "done"
    total, steps = 0, 0
    for split in ("train", "val", "test"):
        try:
            rep = evaluate([str(shards)], str(out / "last"), split, None, 1, str(tiny_config), CPU, "f32", synthetic_maps=True)
        except RuntimeError as e:
            assert "no game" in str(e).lower() or "empty" in str(e).lower(), e
            continue
        total += rep["games"]
        steps += rep["steps"]
        assert 0.0 <= rep["action_top1"] <= rep["action_top5"] <= 1.0
        assert set(rep["nll"]) == {"action", "delay", "queued", "selected", "target_unit", "target_loc"}
        assert rep["win_open_mse"] is None
    assert total == 5 and steps > 0


def test_bc_eval_out_writes_the_report_032_r20(tiny_config, shards, tmp_path, capsys):
    """032-R20 ③ val scoring: `bc-eval --out` writes the printed JSON report to a file too."""
    import argparse

    from uranium.bc import eval as bc_eval

    make = functools.partial(_tiny_make, "Gtrxl")
    out = tmp_path / "out"
    assert _core.run_bc(make, str(tiny_config), out=str(out), shards=[str(shards)], steps=1, synthetic_maps=True) == "done"
    report = tmp_path / "val" / "r.json"
    a = argparse.Namespace(dataset=[], shards=[str(shards)], checkpoint=str(out / "last"), split="train", backend="cpu", core=None, games=None, seed=1, config=str(tiny_config), maps_dir=None, synthetic_maps=True, out=str(report))
    assert bc_eval.main(a) == 0
    printed = json.loads(capsys.readouterr().out)
    assert json.loads(report.read_text(encoding="utf-8")) == printed
    assert printed["games"] > 0


def test_run_bc_refuses_another_run_ident(tiny_config, shards, tmp_path):
    """B-3: a resume with another config is refused before training."""
    make = functools.partial(_tiny_make, "Lstm")
    out = tmp_path / "out"
    assert _core.run_bc(make, str(tiny_config), out=str(out), shards=[str(shards)], steps=1, synthetic_maps=True) == "done"
    other = tmp_path / "other.toml"
    other.write_text(tiny_config.read_text().replace("lr = 3e-3", "lr = 1e-3"), encoding="utf-8")
    with pytest.raises(RuntimeError):
        _core.run_bc(make, str(other), out=str(tmp_path / "o2"), shards=[str(shards)], steps=2, resume=str(out / "last"), synthetic_maps=True)


def test_entity_recompute_keeps_the_update_041_r17(tiny_config, shards):
    """041-R17: the entity encoder in recomputed row chunks gives the same update (dropout 0 in
    the tiny net, so the replayed draws do not matter)."""
    b = feeder(tiny_config, shards).next()
    after = {}
    for chunk in (0, 3):
        t = BcTrainer(bc_cfg(tiny_config), tiny_net(), CPU, "f32", entity_chunk=chunk)
        t.train_batch(b, 3e-3, 0)
        after[chunk] = params_of(t.net)
    for k in after[0]:
        assert torch.allclose(after[3][k], after[0][k], atol=1e-6, rtol=1e-5), k
