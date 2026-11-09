"""Dead-channel statistics (`uranium.model.monitor`, reports/jepa_survey.md 10.4) on the tiny
Burn-parity batch, CPU: every ReLU-bearing point is found (spatial stages incl. s8, entity FFN,
head ResMlp, core FFN) in the default model and with the 032 §2.7 variants, the counts match the
activations computed directly, the row mask drops rows, a dead block reads as dead, the forward
is unchanged, and the results stay tensors until the caller reads them."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from uranium.model import UraniumNet, UraniumNetConfig, action_columns, to_labels, to_obs
from uranium.model.config import VariantConfig
from uranium.model.monitor import ROWS_KEY, ReluMonitor, channel_stats, relu_stats, stats_to_host

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "burn_parity"


def tiny(name="tiny-gtrxl", variant=None):
    meta = json.loads((FIX / name / "meta.json").read_text())
    inp = load_file(FIX / name / "inputs.safetensors")
    cfg = UraniumNetConfig.from_dict(meta["config"])
    if variant is not None:
        cfg.variant = variant
    torch.manual_seed(3)
    net = UraniumNet(cfg, action_columns()).eval()
    return net, meta["b"], meta["t"], to_obs(inp, "cpu"), to_labels(inp, "cpu"), inp["privileged"]


def spatial_points(net):
    pts = {"spatial.stem", "spatial.s8"}
    for i in range(len(net.spatial.downs)):
        pts |= {f"spatial.down{i}.conv", f"spatial.down{i}.inner", f"spatial.down{i}.res"}
    for j in range(len(net.spatial.bottleneck)):
        pts |= {f"spatial.bottleneck{j}", f"spatial.bottleneck{j}.inner"}
    return pts


@pytest.mark.parametrize("name,variant", [("tiny-gtrxl", None), ("tiny-lstm", None), ("tiny-gtrxl", VariantConfig(terrain=True, ctx_token=True, attn_pool=True, outer=2))])
def test_every_relu_point_is_counted(name, variant):
    net, b, t, obs, labels, priv = tiny(name, variant)
    st = relu_stats(net, obs, labels, priv, net.init_state(b), t)
    want = spatial_points(net) | {"entity.L0.ffn", "heads.type_in", "heads.type_res0.inner", "heads.type_res0.out", "heads.type_res1.inner", "heads.type_res1.out"}
    if net.cfg.core.kind == "Gtrxl":
        want |= {f"core.L{l}.{p}" for l in range(2) for p in ("attn", "ffn", "ffn_hidden")}
    assert set(st) == want
    for k, v in st.items():
        assert isinstance(v["pos_frac"], torch.Tensor) and isinstance(v["never_active"], torch.Tensor), k  # no host read inside
        assert 0.0 <= float(v["pos_frac"]) <= 1.0 and 0 <= int(v["never_active"]) <= v["channels"], k
    # the s8 channels are the last stage's
    assert st["spatial.s8"]["channels"] == net.cfg.spatial.stages[-1]


def test_counts_match_the_activations():
    net, b, t, obs, labels, priv = tiny()
    st = relu_stats(net, obs, labels, priv, net.init_state(b), t)
    with torch.no_grad():
        _, e, sp, sc = net.encode(obs)
    s8 = sp["s8"]
    assert float(st["spatial.s8"]["pos_frac"]) == pytest.approx(float((s8 > 0).float().mean()))
    assert int(st["spatial.s8"]["never_active"]) == int((s8.amax((0, 2, 3)) <= 0).sum())
    skip2 = sp["skips"][2]
    assert float(st["spatial.down2.res"]["pos_frac"]) == pytest.approx(float((skip2 > 0).float().mean()))
    # entity FFN: real tokens only
    m = obs["entity_mask"]
    assert int(st["entity.L0.ffn"]["elements"]) == int(m.sum()) * net.cfg.entity.ffn


def test_row_mask_drops_rows():
    net, b, t, obs, labels, priv = tiny()
    rows = torch.zeros(b * t, dtype=torch.bool)
    rows[: t] = True  # the first sequence only
    st = relu_stats(net, obs, labels, priv, net.init_state(b), t, rows=rows)
    with torch.no_grad():
        _, _, sp, _ = net.encode(obs)
    want = channel_stats(sp["s8"][:t], channel_dim=1)
    assert float(st["spatial.s8"]["pos_frac"]) == pytest.approx(float(want["pos_frac"]))
    assert int(st["spatial.s8"]["elements"]) == int(want["elements"])
    assert int(st["heads.type_in"]["elements"]) == t * net.cfg.heads.ar
    assert int(st["core.L0.ffn"]["elements"]) == t * net.cfg.core.hidden


def test_dead_block_reads_dead():
    """10.4: a bottleneck whose second conv only pushes below zero gives s8 = 0 everywhere."""
    net, b, t, obs, labels, priv = tiny()
    last = net.spatial.bottleneck[-1]
    with torch.no_grad():
        last.c1.weight.zero_()
        last.c1.bias.fill_(1.0)
        last.c2.weight.fill_(-100.0)
        last.c2.bias.fill_(-100.0)
    st = stats_to_host(relu_stats(net, obs, labels, priv, net.init_state(b), t))
    assert st["spatial.s8"]["pos_frac"] == 0.0 and st["spatial.s8"]["never_active"] == st["spatial.s8"]["channels"]
    assert st["spatial.bottleneck0"]["pos_frac"] > 0.0


def test_monitor_leaves_the_forward_and_mode_unchanged():
    net, b, t, obs, labels, priv = tiny()
    net.train()
    torch.manual_seed(0)
    with torch.no_grad():
        ref = net.forward_seq(obs, labels, priv, net.init_state(b), t)
    relu_stats(net, obs, labels, priv, net.init_state(b), t)
    assert net.training
    o = dict(obs)
    o[ROWS_KEY] = torch.ones(b * t, dtype=torch.bool)
    torch.manual_seed(0)
    with torch.no_grad(), ReluMonitor(net) as mon:
        got = net.forward_seq(o, labels, priv, net.init_state(b), t)
    assert mon.handles == [] and mon.stats()
    for k in ref["logp"]:
        assert torch.equal(ref["logp"][k], got["logp"][k]), k


def test_row_chunks_give_the_same_counts(monkeypatch):
    """The counts and the recomputed points (stem, ResBlock inner) are taken in row chunks to
    bound scratch memory; tiny chunks give the same numbers."""
    from uranium.model import monitor

    net, b, t, obs, labels, priv = tiny()
    rows = torch.tensor([True, False, True, True] * (b * t // 4))
    whole = stats_to_host(relu_stats(net, obs, labels, priv, net.init_state(b), t, rows=rows))
    monkeypatch.setattr(monitor, "CHUNK_ELEMENTS", 7)
    chunked = stats_to_host(relu_stats(net, obs, labels, priv, net.init_state(b), t, rows=rows))
    assert chunked.keys() == whole.keys()
    for k in whole:
        assert chunked[k]["elements"] == whole[k]["elements"] and chunked[k]["never_active"] == whole[k]["never_active"], k
        assert chunked[k]["pos_frac"] == pytest.approx(whole[k]["pos_frac"], abs=1e-6), k


def test_channel_stats_of_given_activations():
    x = torch.tensor([[[1.0, -1.0, 0.0], [2.0, -3.0, 0.0]], [[0.5, -1.0, 0.0], [-1.0, -2.0, 0.0]]])  # [r=2, n=2, C=3]
    st = channel_stats(x)
    assert float(st["pos_frac"]) == pytest.approx(3 / 12) and int(st["never_active"]) == 2 and st["channels"] == 3
    mask = torch.tensor([[True, False], [True, False]])
    st = channel_stats(x, mask=mask)
    assert float(st["pos_frac"]) == pytest.approx(2 / 6) and int(st["elements"]) == 6
    st = channel_stats(x, rows=torch.tensor([False, True]))
    assert float(st["pos_frac"]) == pytest.approx(1 / 6) and int(st["never_active"]) == 2
