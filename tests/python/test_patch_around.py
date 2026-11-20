import argparse
import copy
import json
from pathlib import Path

import pytest
import torch

from uranium import ckpt
from uranium.bench.cases import Ctx, run_suites
from uranium.bench.cli import rl_net, suite_net
from uranium.bench.realbc import run_group
from uranium.bench.rl import SyntheticSegments, run_learner
from uranium.model import UraniumNet, to_obs
from uranium.model.config import VariantConfig, model_net, new_net_config
from uranium.rl.learner import Learner

import testkit
from testkit import CPU, HEADS, KS, PATCH, PATCH_NET_TOML, WEIGHTS, patch_checkpoint, patch_net, rl_cfg

CNN_ONLY = {"URANIUM_CHANNELS_LAST": "on", "URANIUM_STEM_PAD": "on", "URANIUM_LOC_OUT": "mulsum", "URANIUM_CONV_WGRAD": "bmm"}
AUX_JEPA = VariantConfig(aux_jepa=KS)


def patch_config(tiny_config: Path) -> Path:
    p = tiny_config.with_name("bc_tiny_patch.toml")
    p.write_text(tiny_config.read_text(encoding="utf-8") + PATCH_NET_TOML, encoding="utf-8")
    return p


def numel(net) -> int:
    return sum(p.numel() for p in net.parameters())


def want_net(config: Path, core: str = "lstm") -> UraniumNet:
    variant, sizes = model_net(str(config))
    return UraniumNet(new_net_config(core, False, variant, sizes))


def test_bench_bc_builds_the_config_net_032_r28(tiny_config, shards, monkeypatch):
    monkeypatch.setenv("URANIUM_BC_PREFETCH", "0")
    cfg = patch_config(tiny_config)
    a = argparse.Namespace(config=str(cfg), stage="s1", core="lstm", dataset=[], shards=[str(shards)], synthetic_maps=True, seed=3, updates=1, ops=0, trace=None)
    r = run_group(a, 1, CPU, "f32", log=lambda m: None)
    want = want_net(cfg)
    assert want.patch == PATCH and r["params"] == numel(want)
    assert len(r["updates"]) == 2 and all(u["loss"]["total"] == u["loss"]["total"] for u in r["updates"])
    a.config = str(tiny_config)
    assert run_group(a, 1, CPU, "f32", log=lambda m: None)["params"] == numel(UraniumNet(new_net_config("lstm", False)))


def test_bc_profile_builds_the_config_net_032_r28(tiny_config, shards, monkeypatch, capsys):
    from uranium.bc import profile

    monkeypatch.setenv("URANIUM_BC_PREFETCH", "0")
    cfg = patch_config(tiny_config)
    built = []

    class Recorded(UraniumNet):
        def __init__(self, c, *rest):
            super().__init__(c, *rest)
            built.append(self)

    monkeypatch.setattr("uranium.model.UraniumNet", Recorded)
    a = argparse.Namespace(config=str(cfg), core="lstm", dataset=[], shards=[str(shards)], synthetic_maps=True, batches=1, backend="cpu", ops=0, trace=None)
    assert profile.main(a) == 0
    assert len(built) == 1 and built[0].patch == PATCH and numel(built[0]) == numel(want_net(cfg))
    assert json.loads(capsys.readouterr().out)["updates"]


def test_bench_suite_model_cases_follow_config_and_checkpoint_032_r28(tiny_config, tmp_path):
    cfg = patch_config(tiny_config)
    base = suite_net(argparse.Namespace(config=str(cfg), checkpoint=None))
    assert base.patch() == PATCH and base.entity.d_model == 32 and base.spatial.patch_dim == 16
    ck = patch_checkpoint(tmp_path / "ck")
    from_ck = suite_net(argparse.Namespace(config=None, checkpoint=str(ck)))
    assert from_ck.to_dict() == ckpt.read_meta(ck)["config"]
    assert suite_net(argparse.Namespace(config=None, checkpoint=None)) is None
    for served in (False, True):
        rows = run_suites(["model"], Ctx(CPU, "f32", False, 0.05, net=base, served=served), "net_act_lstm", log=lambda m: None)
        assert [r.status for r in rows] == ["ok"], [(r.case, r.note) for r in rows]
        assert f"{numel(want_net(cfg))} params" in rows[0].note and ("inference server" in rows[0].note) == served
    rows = run_suites(["model"], Ctx(CPU, "f32", False, 0.05, net=from_ck), "net_train_lstm", log=lambda m: None)
    assert [r.status for r in rows] == ["ok"], [(r.case, r.note) for r in rows]


def test_bench_rl_model_config_runs_the_patch_learner_032_r28_r29(tiny_config):
    cfg = patch_config(tiny_config)
    a = argparse.Namespace(checkpoint=None, model_config=str(cfg), core="lstm", model_seed=0)
    net = rl_net(a, CPU)
    assert net.patch == PATCH and net.cfg.entity.dropout == 0.0 and numel(net) == numel(want_net(cfg))
    batch = SyntheticSegments(net, 2, 4, (3, 40), templates=2, seed=3)
    run = run_learner("new", 1, net, rl_cfg(lanes=2), batch, CPU, "f32", "vtrace", 1, False, log=lambda m: None)
    assert all(u["stats"]["loss"] == u["stats"]["loss"] for u in run["updates"])


def test_rl_learner_jepa_moments_on_the_patch_path_032_r29():
    net = patch_net(5, AUX_JEPA)
    assert net.cfg.jepa_width() == net.cfg.entity.d_model
    batch = SyntheticSegments(net, 2, 4, (3, 40), templates=2, seed=3)
    cfg = rl_cfg(lanes=2)
    cfg["aux"] = {"jepa": 0.1}
    run = run_learner("new", 1, net, cfg, batch, CPU, "f32", "vtrace", 1, False, log=lambda m: None)
    for u in run["updates"]:
        assert u["stats"]["aux_jepa"] == u["stats"]["aux_jepa"]


def test_rl_learner_refuses_monitor_on_the_patch_path_032_r25():
    net = patch_net()
    with pytest.raises(ValueError, match="spatial_patch"):
        Learner(rl_cfg(monitor={"spatial_every": 10}), net, copy.deepcopy(net), CPU, "f32", "vtrace")


def test_rl_config_with_net_is_refused_032_r28(tmp_path):
    from uranium.__main__ import main
    from uranium.rl.run import refuse_net

    src = (testkit.ROOT / "config" / "rl.toml").read_text(encoding="utf-8")
    ok, bad = tmp_path / "rl.toml", tmp_path / "rl_net.toml"
    ok.write_text(src, encoding="utf-8")
    bad.write_text(src + "\n[net.entity]\nd_model = 512\n", encoding="utf-8")
    refuse_net(str(ok))
    with pytest.raises(SystemExit, match="032-R28"):
        refuse_net(str(bad))
    with pytest.raises(SystemExit, match="032-R28"):
        main(["rl", "--run", "x", "--bc", str(tmp_path / "none"), "--config", str(bad), "--backend", "cpu", "--logs", str(tmp_path / "logs")])
    with pytest.raises(SystemExit, match="032-R28"):
        main(["bench", "rl", "--checkpoint", str(tmp_path / "none"), "--config", str(bad), "--backend", "cpu"])


def forward_and_grads(group) -> tuple:
    net = patch_net(5, AUX_JEPA)
    obs, labels = to_obs(group, "cpu"), net.labels(group, "cpu")
    out = net.forward_seq(obs, labels, torch.as_tensor(group["privileged"]).float(), net.init_state(group["b"], "cpu"), 4, aux=True)
    loss = sum(out["logp"][k].sum() for k in HEADS) + sum(g.square().sum() for g in out["aux_jepa"].values())
    loss.backward()
    act = net.eval().act(obs, net.init_state(obs["entity_mask"].shape[0], "cpu"), 1.0, torch.Generator().manual_seed(3))
    return torch.stack([out["logp"][k] for k in HEADS]).detach(), {k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None}, act["log_prob"]


@pytest.fixture
def group(tiny_config, tmp_path):
    from uranium import _core

    _core.write_test_shards(str(tmp_path / "s"), "mixed", 2)
    cfg = json.loads(_core.bc_config(str(tiny_config), "s1")[0])
    b = _core.Feeder(str(tiny_config), [str(tmp_path / "s")], prefetch=0, synthetic_maps=True, check_maps=False).next()
    return b.group(0, cfg["lanes"], cfg["seq_len"], cfg["pad_entities"], 12)


def test_cnn_only_switches_leave_the_patch_path_alone_032_r29(group, monkeypatch):
    for k in CNN_ONLY:
        monkeypatch.delenv(k, raising=False)
    off = forward_and_grads(group)
    for k, v in CNN_ONLY.items():
        monkeypatch.setenv(k, v)
    on = forward_and_grads(group)
    assert torch.equal(on[0], off[0]) and torch.equal(on[2], off[2])
    assert on[1].keys() == off[1].keys() and all(torch.equal(on[1][k], g) for k, g in off[1].items())


def test_rl_channels_last_leaves_the_patch_learner_alone_032_r29(monkeypatch):
    from uranium.device import RL_CHANNELS_LAST

    def update():
        net = patch_net(5)
        l = Learner(rl_cfg(lanes=2), net, copy.deepcopy(net), CPU, "f32", "vtrace")
        batch = SyntheticSegments(patch_net(6), 2, 4, (3, 40), templates=2, seed=3)
        stats = l.update(batch, WEIGHTS)
        return stats, {k: v.detach().clone() for k, v in l.net.state_dict().items()}

    for k in (RL_CHANNELS_LAST, *CNN_ONLY):
        monkeypatch.delenv(k, raising=False)
    off = update()
    for k, v in ({RL_CHANNELS_LAST: "on"} | CNN_ONLY).items():
        monkeypatch.setenv(k, v)
    on = update()
    assert {k: v for k, v in on[0].items() if k != "phases"} == {k: v for k, v in off[0].items() if k != "phases"}
    assert all(torch.equal(on[1][k], v) for k, v in off[1].items())
