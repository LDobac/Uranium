"""AMB-000-4 step 2 optimisations keep the model's results (032-R19), CPU: the location head row
list from the host labels (exact / bucketed) gives the same log-probs and gradients as the device
`nonzero` path, and the 1x1 output written as multiply-and-sum equals the conv."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from uranium import _core
from uranium.model import UraniumNet, UraniumNetConfig, action_columns, input_dims, to_obs
from uranium.model.heads import TARGET_LOC, location_rows

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "burn_parity"
HEADS = ("action", "delay", "queued", "selected", "target_unit", "target_loc")


def tiny_net(seed: int = 7) -> UraniumNet:
    cfg = UraniumNetConfig.from_dict(json.loads((FIX / "tiny-gtrxl" / "meta.json").read_text())["config"])
    cfg.input = input_dims()
    torch.manual_seed(seed)
    return UraniumNet(cfg, action_columns())


def test_location_rows_buckets():
    tk = np.array([0, 2, 1, 2])
    actions = np.array([1, 0, 3, 3, 2, 1, 0, 0, 0, 0])  # rows 0, 2, 3, 5 use a location
    rows, k = location_rows(actions, tk, "exact")
    assert k == 4 and rows.tolist() == [0, 2, 3, 5]
    rows, k = location_rows(actions, tk, "mult8")
    assert k == 4 and rows.tolist() == [0, 2, 3, 5, 0, 0, 0, 0]
    rows, k = location_rows(actions[:6], tk, "mult8")  # the bucket covers the batch: every row
    assert k == 6 and rows.tolist() == list(range(6))
    rows, k = location_rows(np.zeros(5, dtype=np.int64), tk, "pow2")  # no used row: every row
    assert k == 5 and rows.tolist() == list(range(5))
    rows, k = location_rows(np.array([1, 1, 1, 0, 0, 0, 0, 0]), tk, "pow2")
    assert k == 3 and len(rows) == 4
    rows, k = location_rows(np.array([1, 0, 0, 0, 0, 0, 0, 0]), tk, "pow2")  # never one row
    assert k == 1 and rows.tolist() == [0, 0]


def forward(net, hg, mode, monkeypatch):
    monkeypatch.setenv("URANIUM_LOC_ROWS", mode)
    net.zero_grad(set_to_none=True)
    obs, labels = to_obs(hg, "cpu"), net.labels(hg, "cpu")
    out = net.forward_seq(obs, labels, torch.as_tensor(hg["privileged"]).float(), net.init_state(hg["b"], "cpu"), hg["t"] if "t" in hg else 4)
    loss = sum(out["logp"][k].sum() for k in HEADS)
    loss.backward()
    return out["logp"]["target_loc"].detach(), {k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None}


@pytest.fixture
def group(tiny_config, tmp_path):
    _core.write_test_shards(str(tmp_path / "s"), "mixed", 2)
    cfg = json.loads(_core.bc_config(str(tiny_config), "s1")[0])
    b = _core.Feeder(str(tiny_config), [str(tmp_path / "s")], prefetch=0, synthetic_maps=True, check_maps=False).next()
    return b.group(0, cfg["lanes"], cfg["seq_len"], cfg["pad_entities"], 12)


@pytest.mark.parametrize("mode", ["exact", "mult8", "pow2"])
def test_host_location_rows_keep_the_result(group, mode, monkeypatch):
    net = tiny_net()
    # Make some rows use a location so the sub-batch path runs.
    tk = np.asarray(net._cols["target_kind"])
    loc_actions = np.flatnonzero(tk == TARGET_LOC)
    hg = dict(group)
    act = np.array(hg["action"]).copy()
    act[::3] = loc_actions[0]
    hg["action"] = act
    ref_lp, ref_g = forward(net, hg, "off", monkeypatch)
    lp, g = forward(net, hg, mode, monkeypatch)
    assert torch.allclose(lp, ref_lp, atol=1e-5, rtol=1e-5)
    for k in ref_g:
        assert torch.allclose(g[k], ref_g[k], atol=1e-6, rtol=1e-4), k


def test_location_out_mulsum_equals_conv(group, monkeypatch):
    outs, grads = {}, {}
    for mode in ("conv", "mulsum"):
        monkeypatch.setenv("URANIUM_LOC_OUT", mode)
        net = tiny_net()
        h = net.heads.location
        assert h.out_mode == mode
        _, _, sp, _ = net.encode(to_obs(group, "cpu"))
        ar = torch.randn(sp["s8"].shape[0], net.cfg.heads.ar, generator=torch.Generator().manual_seed(1))
        y = h(ar, sp)
        y.square().sum().backward()
        outs[mode] = y.detach()
        grads[mode] = {k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None}
    assert torch.allclose(outs["mulsum"], outs["conv"], atol=1e-5, rtol=1e-5)
    assert grads["mulsum"].keys() == grads["conv"].keys()
    for k, g in grads["conv"].items():
        assert torch.allclose(grads["mulsum"][k], g, atol=1e-5, rtol=1e-4), k


def test_bias_gather_modes_equal_embedding(group, monkeypatch):
    """URANIUM_BIAS_GRAD: index_add and the transposed-table gather give the embedding path's
    bias and table gradient (032-R13)."""
    grads = {}
    obs = to_obs(group, "cpu")
    for mode in ("embedding", "index_add", "table_t"):
        monkeypatch.setenv("URANIUM_BIAS_GRAD", mode)
        net = tiny_net()
        e, pool = net.entity(obs)
        (e.square().sum() + pool.sum()).backward()
        grads[mode] = {k: p.grad.clone() for k, p in net.entity.named_parameters() if p.grad is not None}
    assert grads["index_add"]["bias_table"].abs().sum() > 0 and grads["table_t"]["bias_table"].abs().sum() > 0
    for mode in ("index_add", "table_t"):
        for k, g in grads["embedding"].items():
            assert torch.allclose(grads[mode][k], g, atol=1e-6, rtol=1e-4), (mode, k)


def test_apply_malloc_threshold(monkeypatch):
    """URANIUM_MALLOC_MMAP_THRESHOLD: set through mallopt on Linux glibc, ignored elsewhere."""
    import sys

    from uranium import env

    monkeypatch.delenv("URANIUM_MALLOC_MMAP_THRESHOLD", raising=False)
    assert env.apply_malloc() is None
    monkeypatch.setenv("URANIUM_MALLOC_MMAP_THRESHOLD", "1048576")
    assert env.apply_malloc() == (1048576 if sys.platform.startswith("linux") else None)


def test_apply_malloc_arenas(monkeypatch):
    """URANIUM_MALLOC_ARENA_MAX: set through mallopt on Linux glibc, ignored elsewhere."""
    import sys

    from uranium import env

    monkeypatch.delenv("URANIUM_MALLOC_ARENA_MAX", raising=False)
    assert env.apply_malloc_arenas() is None
    monkeypatch.setenv("URANIUM_MALLOC_ARENA_MAX", "2")
    assert env.apply_malloc_arenas() == (2 if sys.platform.startswith("linux") else None)


def test_conv_wgrad_bmm_equals_vendor(group, monkeypatch):
    """URANIUM_CONV_WGRAD=bmm: the spatial encoder's gradients equal the library conv's."""
    obs = to_obs(group, "cpu")
    grads = {}
    for mode in ("vendor", "bmm"):
        monkeypatch.setenv("URANIUM_CONV_WGRAD", mode)
        net = tiny_net()
        e = torch.randn(obs["entity_cat"].shape[0], obs["entity_cat"].shape[1], net.cfg.entity.d_model, generator=torch.Generator().manual_seed(2), requires_grad=True)
        sp = net.spatial(obs, e)
        (sp["pool"].square().sum() + sum(s.sum() for s in sp["skips"])).backward()
        grads[mode] = {k: p.grad.clone() for k, p in net.spatial.named_parameters() if p.grad is not None} | {"e": e.grad.clone()}
    assert grads["bmm"].keys() == grads["vendor"].keys()
    for k, g in grads["vendor"].items():
        assert torch.allclose(grads["bmm"][k], g, atol=1e-5, rtol=1e-4), k


def test_splitk_mm_equals_mm():
    from uranium.model.layers import splitk_mm

    g = torch.Generator().manual_seed(2)
    for k in (5, 64, 100, 1037):
        a, b = torch.randn(k, 7, generator=g), torch.randn(k, 9, generator=g)
        for chunk in (4, 16, 512):
            assert torch.allclose(splitk_mm(a, b, chunk), a.t() @ b, atol=1e-4, rtol=1e-5), (k, chunk)


def test_linear_wgrad_splitk_equals_vendor(group, monkeypatch):
    """URANIUM_LINEAR_WGRAD=splitk: every Linear weight gradient of a full update equals the
    library path (pieces small enough that the split runs on the test group)."""
    import uranium.model.layers as layers

    monkeypatch.setattr(layers, "SPLITK_CHUNK", 8)
    res = {}
    for mode in ("vendor", "splitk"):
        monkeypatch.setenv("URANIUM_LINEAR_WGRAD", mode)
        net = tiny_net()
        obs, labels = to_obs(group, "cpu"), net.labels(group, "cpu")
        out = net.forward_seq(obs, labels, torch.as_tensor(group["privileged"]).float(), net.init_state(group["b"], "cpu"), 4)
        sum(out["logp"][k].sum() for k in HEADS).backward()
        res[mode] = {k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None}
    assert res["splitk"].keys() == res["vendor"].keys()
    for k, g in res["vendor"].items():
        assert torch.allclose(res["splitk"][k], g, atol=1e-5, rtol=1e-4), k


def test_switches_fall_back_on_other_layer_shapes(group, monkeypatch):
    """A layer changed away from the shape an optimisation is written for takes the library
    path: a 5x5 / dilated / grouped / stride-3 conv under URANIUM_CONV_WGRAD=bmm, a 3x3 location
    output conv under URANIUM_LOC_OUT=mulsum."""
    import torch.nn as nn

    from uranium.model.layers import conv_wgrad

    monkeypatch.setenv("URANIUM_CONV_WGRAD", "bmm")
    g = torch.Generator().manual_seed(4)
    for conv in (nn.Conv2d(4, 6, 5, padding=2), nn.Conv2d(4, 6, 3, padding=2, dilation=2), nn.Conv2d(4, 6, 3, padding=1, groups=2), nn.Conv2d(4, 6, 3, stride=3, padding=1)):
        x = torch.randn(2, 4, 9, 9, generator=g, requires_grad=True)
        conv_wgrad(conv, x).square().sum().backward()
        gw = conv.weight.grad.clone()
        conv.weight.grad = None
        conv(x).square().sum().backward()
        assert torch.allclose(gw, conv.weight.grad, atol=1e-5, rtol=1e-4), conv
    outs = {}
    for mode in ("conv", "mulsum"):
        monkeypatch.setenv("URANIUM_LOC_OUT", mode)
        net = tiny_net()
        torch.manual_seed(9)
        net.heads.location.out = nn.Conv2d(32, 1, 3, padding=1)
        _, _, sp, _ = net.encode(to_obs(group, "cpu"))
        ar = torch.randn(sp["s8"].shape[0], net.cfg.heads.ar, generator=torch.Generator().manual_seed(1))
        outs[mode] = net.heads.location(ar, sp).detach()
    assert torch.equal(outs["mulsum"], outs["conv"])


def test_channels_last_keeps_the_result_032_r19(group, monkeypatch):
    """URANIUM_CHANNELS_LAST=on: the spatial encoder and location head on channels-last
    activations give the NCHW path's log-probs and gradients."""
    res = {}
    for mode in ("off", "on"):
        monkeypatch.setenv("URANIUM_CHANNELS_LAST", mode)
        net = tiny_net()
        obs, labels = to_obs(group, "cpu"), net.labels(group, "cpu")
        out = net.forward_seq(obs, labels, torch.as_tensor(group["privileged"]).float(), net.init_state(group["b"], "cpu"), 4)
        sum(out["logp"][k].sum() for k in HEADS).backward()
        res[mode] = (torch.stack([out["logp"][k] for k in HEADS]).detach(), {k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None})
    assert torch.allclose(res["on"][0], res["off"][0], atol=1e-5, rtol=1e-5)
    assert res["on"][1].keys() == res["off"][1].keys()
    for k, g in res["off"][1].items():
        assert torch.allclose(res["on"][1][k], g, atol=1e-5, rtol=1e-4), k


def test_spatial_bf16_is_bit_identical_under_autocast_032_r19(group, monkeypatch):
    """URANIUM_SPATIAL_BF16=on rounds the spatial planes to bf16 on the host; under bf16 autocast
    their consumers round them the same way, so log-probs and gradients are bit-identical (CPU
    autocast stands in for the CUDA host). With f32 compute the switch does nothing."""
    from uranium.device import autocast
    from uranium.model.net import stage

    monkeypatch.setenv("URANIUM_SPATIAL_BF16", "on")
    staged = stage(group, "cpu", "bf16")
    assert staged["spatial"].dtype == torch.bfloat16 and stage(group, "cpu", "f32") is group
    res = []
    for hg in (group, staged):
        net = tiny_net()
        obs, labels = to_obs(hg, "cpu"), net.labels(hg, "cpu")
        with autocast(torch.device("cpu"), "bf16"):
            out = net.forward_seq(obs, labels, torch.as_tensor(group["privileged"]).float(), net.init_state(group["b"], "cpu"), 4)
        sum(out["logp"][k].float().sum() for k in HEADS).backward()
        res.append((torch.stack([out["logp"][k].float() for k in HEADS]).detach(), {k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None}))
    assert torch.equal(res[0][0], res[1][0])
    assert res[0][1].keys() == res[1][1].keys()
    for k, g in res[0][1].items():
        assert torch.equal(res[1][1][k], g), k


def test_stem_pad_keeps_the_result_032_r19(group, monkeypatch):
    """URANIUM_STEM_PAD=on: the stem conv over zero-padded input channels (and zero weight
    columns) gives the plain conv's log-probs and gradients."""
    res = {}
    for mode in ("off", "on"):
        monkeypatch.setenv("URANIUM_STEM_PAD", mode)
        net = tiny_net()
        obs, labels = to_obs(group, "cpu"), net.labels(group, "cpu")
        out = net.forward_seq(obs, labels, torch.as_tensor(group["privileged"]).float(), net.init_state(group["b"], "cpu"), 4)
        sum(out["logp"][k].sum() for k in HEADS).backward()
        res[mode] = (torch.stack([out["logp"][k] for k in HEADS]).detach(), {k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None})
    assert torch.allclose(res["on"][0], res["off"][0], atol=1e-5, rtol=1e-5)
    assert res["on"][1].keys() == res["off"][1].keys()
    for k, g in res["off"][1].items():
        assert torch.allclose(res["on"][1][k], g, atol=1e-5, rtol=1e-4), k
