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
    assert location_rows(actions[:6], tk, "mult8") is None  # the bucket covers the batch
    assert location_rows(np.zeros(5, dtype=np.int64), tk, "pow2") is None  # no used row
    rows, k = location_rows(np.array([1, 1, 1, 0, 0, 0, 0, 0]), tk, "pow2")
    assert k == 3 and len(rows) == 4


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
