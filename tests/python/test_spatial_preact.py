"""032-R22 `spatial_preact` (032 §2.8, 032-AC13): with the switch on, the spatial encoder's 6
ResBlocks (4 stages, 2 bottleneck) compute `x + c2(relu(c1(relu(x))))` with no final ReLU, in
both ResBlock code paths (the bmm weight-gradient path for <= 64 channels and the library path);
the monitor's inner point (032-R25 ①) follows; no parameter is added and the shared initial
weights are the default model's; step = sequence (032-R5, outer segment boundary included);
masks, entity recompute and the checkpoint round trip hold; off (the default) is the model as it
was (Burn golden tests and AC11 stay as they are). CPU, tiny Burn-parity batch."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from uranium import ckpt
from uranium.model import UraniumNetConfig
from uranium.model.config import VariantConfig
from uranium.model.encoders import SpatialEncoder
from uranium.model.layers import RES_BMM_MAX_CH, ResBlock, conv_layout
from uranium.model.monitor import channel_stats, relu_stats

from test_variants import HEADS, VARIANTS, net_of, rows_at, tiny

PREACT = VariantConfig(spatial_preact=True)
PREACT_ALL = replace(VARIANTS["all"], spatial_preact=True)
CONFIGS = {"preact": PREACT, "preact_all": PREACT_ALL}


def blocks(net):
    sp = net.spatial
    return [d.res for d in sp.downs] + list(sp.bottleneck)


def formula(blk: ResBlock, x, preact: bool):
    """The 032 §2.2 / 032-R22 definitions written with the library conv."""
    c1 = lambda v: F.conv2d(v, blk.c1.weight, blk.c1.bias, 1, 1)  # noqa: E731
    c2 = lambda v: F.conv2d(v, blk.c2.weight, blk.c2.bias, 1, 1)  # noqa: E731
    if preact:
        return x + c2(F.relu(c1(F.relu(x))))
    return F.relu(x + c2(F.relu(c1(x))))


# Channels 8 take the `conv_wgrad` branch (<= RES_BMM_MAX_CH), 72 the library branch; under
# `URANIUM_CONV_WGRAD=bmm` the first one's weight gradient comes from the batched GEMMs.
@pytest.mark.parametrize("wgrad", ["vendor", "bmm"])
@pytest.mark.parametrize("ch", [8, RES_BMM_MAX_CH + 8])
@pytest.mark.parametrize("preact", [False, True])
def test_resblock_matches_definition_032_r22(monkeypatch, preact, ch, wgrad):
    monkeypatch.setenv("URANIUM_CONV_WGRAD", wgrad)
    torch.manual_seed(1)
    blk = ResBlock(ch, preact)
    x = torch.randn(3, ch, 6, 6, requires_grad=True)
    y = blk(x)
    with torch.no_grad():
        want = formula(blk, x, preact)
    assert torch.equal(y, want), "same convs, same order: bit-identical forward"
    if preact:
        assert (y < 0).any(), "no final ReLU"
        assert torch.equal(blk.inner_act(x), F.relu(blk.c1(F.relu(x))))
    else:
        assert (y >= 0).all()
        assert torch.equal(blk.inner_act(x), F.relu(blk.c1(x)))
    g = torch.randn_like(y)
    got = torch.autograd.grad(y, [x, *blk.parameters()], g)
    x2 = x.detach().clone().requires_grad_(True)
    want_g = torch.autograd.grad(formula(blk, x2, preact), [x2, *blk.parameters()], g)
    for a, b in zip(got, want_g):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_preact_adds_no_parameters_032_r22():
    """Same parameter names, shapes and initial values as the switch-off model, the global RNG
    left where it was (032-AC13 ①), and every one of the 6 ResBlocks switched."""
    cfg, *_ = tiny()
    on = net_of(replace(cfg, variant=PREACT), 9)
    after_on = torch.rand(3)
    off = net_of(cfg, 9)
    after_off = torch.rand(3)
    sa, sb = on.state_dict(), off.state_dict()
    assert sa.keys() == sb.keys()
    for k, v in sb.items():
        assert torch.equal(sa[k], v), k
    assert torch.equal(after_on, after_off)
    assert len(blocks(on)) == 6 and all(b.preact for b in blocks(on))
    assert not any(b.preact for b in blocks(off))


@pytest.mark.parametrize("name", CONFIGS)
def test_preact_changes_the_spatial_outputs(name):
    """The switch reaches the encoder: S8 / skips take negative values and the outputs differ."""
    cfg, b, t, obs, labels, _ = tiny(CONFIGS[name])
    off_cfg = replace(cfg, variant=replace(cfg.variant, spatial_preact=False))
    on, off = net_of(cfg), net_of(off_cfg)
    on.load_state_dict(off.state_dict())
    seen = []
    hook = on.spatial.register_forward_hook(lambda m, i, o: seen.append(o))
    with torch.no_grad():
        x = on.forward_seq(obs, labels, None, on.init_state(b), t)["value"]
    hook.remove()
    with torch.no_grad():
        y = off.forward_seq(obs, labels, None, off.init_state(b), t)["value"]
    assert seen and all((sp["s8"] < 0).any() and any((s < 0).any() for s in sp["skips"]) for sp in seen)
    assert not torch.equal(x, y)


def test_spatial_encoder_matches_definition_032_r22():
    """The whole encoder against the 032 §2.2 stages written out, with the preact blocks."""
    cfg, b, t, obs, _, _ = tiny(PREACT)
    net = net_of(cfg)
    sp_ = net.spatial
    with torch.no_grad():
        _, e, sp, _, _ = net.encode(obs)
        spatial = obs["spatial"]
        x = F.relu(sp_.stem(conv_layout(torch.cat([spatial, sp_.scatter_mean(obs, e, spatial.shape[2], spatial.shape[3])], 1))))
        outs = []
        for d in sp_.downs:
            x = formula(d.res, F.relu(d.conv(x)), True)
            outs.append(x)
        for r in sp_.bottleneck:
            x = formula(r, x, True)
    assert torch.allclose(sp["s8"], x, atol=1e-6)
    for got, want in zip(sp["skips"], outs[:3]):
        assert torch.allclose(got, want, atol=1e-6)
    assert torch.allclose(sp["pool"], sp_.pool(x.flatten(1)), atol=1e-6)


def test_monitor_inner_point_under_preact_032_r25():
    """`.inner` = relu(c1(relu(x))) of each block's input, `.res` / bottleneck / s8 = the block
    outputs (negative values allowed), against a direct computation."""
    cfg, b, t, obs, labels, priv = tiny(PREACT)
    net = net_of(cfg)
    st = relu_stats(net, obs, labels, priv, net.init_state(b), t)
    sp_ = net.spatial
    with torch.no_grad():
        _, e, sp, _, _ = net.encode(obs)
        spatial = obs["spatial"]
        x = F.relu(sp_.stem(conv_layout(torch.cat([spatial, sp_.scatter_mean(obs, e, spatial.shape[2], spatial.shape[3])], 1))))
        acts = {}
        for i, d in enumerate(sp_.downs):
            y = F.relu(d.conv(x))
            acts[f"spatial.down{i}.conv"] = y
            acts[f"spatial.down{i}.inner"] = F.relu(F.conv2d(F.relu(y), d.res.c1.weight, d.res.c1.bias, 1, 1))
            x = formula(d.res, y, True)
            acts[f"spatial.down{i}.res"] = x
        for j, r in enumerate(sp_.bottleneck):
            acts[f"spatial.bottleneck{j}.inner"] = F.relu(F.conv2d(F.relu(x), r.c1.weight, r.c1.bias, 1, 1))
            x = formula(r, x, True)
            acts[f"spatial.bottleneck{j}"] = x
        acts["spatial.s8"] = x
    assert (acts["spatial.s8"] < 0).any()
    # The bottleneck inputs are block outputs with negative values: the pre-activation ReLU matters.
    assert not torch.equal(acts["spatial.bottleneck1.inner"], F.relu(F.conv2d(acts["spatial.bottleneck0"], sp_.bottleneck[1].c1.weight, sp_.bottleneck[1].c1.bias, 1, 1)))
    for name, act in acts.items():
        want = channel_stats(act, channel_dim=1)
        assert float(st[name]["pos_frac"]) == pytest.approx(float(want["pos_frac"]), abs=1e-6), name
        assert int(st[name]["never_active"]) == int(want["never_active"]), name


@pytest.mark.parametrize("name", CONFIGS)
def test_step_matches_sequence_032_ac13(name):
    """One decision at a time = the training pass (032-R5), across the outer segment boundary
    for the all-on combination, and the phased RL pass gives the same."""
    from uranium.agent.infer import leaves

    cfg, b, t, obs, labels, _ = tiny(CONFIGS[name])
    net = net_of(cfg)
    with torch.no_grad():
        seq = net.forward_seq(obs, labels, None, net.init_state(b), t)
        s = net.init_state(b)
        values = []
        for k in range(t):
            out = net.act(rows_at(obs, b, t, k), s, 0.0)
            s = out["state"]
            values.append(out["value"])
        if net.outer is not None:
            ph = net.forward_seq(obs, labels, None, net.init_state(b), t, phased=True)
            assert torch.allclose(ph["value"], seq["value"], atol=1e-5)
            for k in HEADS:
                assert torch.allclose(ph["logp"][k], seq["logp"][k], atol=1e-4), k
    assert torch.allclose(seq["value"].view(b, t, -1), torch.stack(values, 1), atol=1e-5)
    for x, y in zip(leaves(seq["state"]), leaves(s)):
        assert x.dtype == y.dtype and torch.allclose(x.float(), y.float(), atol=1e-5)


@pytest.mark.parametrize("name", CONFIGS)
def test_masks_and_entity_recompute_032_ac13(name):
    """Sampled actions stay inside the type mask (032-AC4); the entity-chunked recompute gives
    the same gradients (041-R17)."""
    cfg, b, t, obs, labels, priv = tiny(CONFIGS[name])
    net = net_of(cfg)
    g = torch.Generator().manual_seed(5)
    o = rows_at(obs, b, t, 0)
    s = net.init_state(b)
    with torch.no_grad():
        for _ in range(10):
            out = net.act(o, s, 1.0, g)
            assert o["type_mask"][torch.arange(b), out["action"]["action"]].all()
            s = out["state"]
    grads = []
    for chunk in (0, 3):
        n = net_of(cfg).train()
        out = n.forward_seq(obs, labels, priv, n.init_state(b), t, chunk)
        (sum(out["logp"][k].sum() for k in HEADS) + out["value"].sum()).backward()
        grads.append({k: p.grad.clone() for k, p in n.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    assert any(k.startswith("spatial.bottleneck") for k in grads[0])
    for k, gr in grads[0].items():
        assert torch.allclose(gr, grads[1][k], atol=1e-5, rtol=1e-4), k


@pytest.mark.parametrize("name", CONFIGS)
def test_checkpoint_round_trip_032_ac13(name, tmp_path):
    cfg, b, t, obs, labels, _ = tiny(CONFIGS[name])
    net = net_of(cfg)
    ckpt.save_model(tmp_path, net)
    ckpt.save_meta(tmp_path, net.cfg, {}, "f32")
    back, meta = ckpt.load_model(tmp_path)
    assert back.cfg.variant == cfg.variant and meta["config"]["variant"]["spatial_preact"] is True
    assert all(blk.preact for blk in blocks(back))
    with torch.no_grad():
        x = net.eval().forward_seq(obs, labels, None, net.init_state(b), t)["value"]
        y = back.eval().forward_seq(obs, labels, None, back.init_state(b), t)["value"]
    assert torch.equal(x, y)


def test_default_is_off_032_ac13():
    """The default config, an old config dict (no `spatial_preact`) and a bare encoder / block
    are the post-activation model as before."""
    assert VariantConfig().spatial_preact is False and ResBlock(4).preact is False
    cfg, *_ = tiny()
    assert cfg.variant.spatial_preact is False
    net = net_of(cfg)
    assert not any(blk.preact for blk in blocks(net))
    assert UraniumNetConfig.from_dict(UraniumNetConfig().to_dict()).variant.spatial_preact is False
    enc = SpatialEncoder(cfg.input, cfg.spatial, cfg.entity.d_model)
    assert not any(blk.preact for blk in [d.res for d in enc.downs] + list(enc.bottleneck))


@pytest.mark.gpu
def test_preact_bmm_matches_vendor_on_device(monkeypatch):
    """The bmm weight-gradient path and the vendor path agree on the device (f32; TF32 off so
    both paths sum the same f32 products)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA / ROCm device")
    torch.manual_seed(2)
    blk = ResBlock(RES_BMM_MAX_CH, True).cuda()
    x0 = torch.randn(16, RES_BMM_MAX_CH, 32, 32, device="cuda")
    g = torch.randn_like(x0)
    res = {}
    tf32 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for mode in ("vendor", "bmm"):
            monkeypatch.setenv("URANIUM_CONV_WGRAD", mode)
            x = x0.clone().requires_grad_(True)
            y = blk(x)
            res[mode] = (y.detach(), *torch.autograd.grad(y, [x, *blk.parameters()], g))
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32
    for a, b in zip(res["vendor"], res["bmm"]):
        assert (a - b).abs().max() <= 1e-5 * a.abs().max()
