import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from uranium import ckpt
from uranium.bc.trainer import BcTrainer
from uranium.model import UraniumNetConfig
from uranium.model.config import VariantConfig, model_variant
from uranium.model.layers import GROUP_NORM_EPS, RES_BMM_MAX_CH, ResBlock, conv_layout, group_norm
from uranium.model.monitor import channel_stats, relu_stats

from testkit import CPU, HEADS, VARIANTS, bc_cfg, feeder, make_net, net_of, params_of, rows_at, tiny

ROOT = Path(__file__).resolve().parents[2]
PER = 4
PREACT = VariantConfig(spatial_preact=True)
NORM = VariantConfig(spatial_preact=True, spatial_norm=PER)
NORM_ALL = replace(VARIANTS["all"], spatial_preact=True, spatial_norm=PER)
CONFIGS = {"norm": NORM, "norm_all": NORM_ALL}
PREFIX = ckpt.SPATIAL_NORM_PREFIX


def shake(norms, seed: int = 11):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in norms.modules():
            if isinstance(m, nn.GroupNorm):
                m.weight.copy_(1.0 + 0.5 * torch.randn(m.weight.shape, generator=g))
                m.bias.copy_(0.5 * torch.randn(m.bias.shape, generator=g))
    return norms


def norm_net(cfg, seed: int = 3):
    net = net_of(cfg, seed)
    shake(net.variant.spatial_norm)
    return net


def gn(m: nn.GroupNorm, x):
    b, c = x.shape[:2]
    y = x.reshape(b, m.num_groups, -1)
    y = (y - y.mean(2, keepdim=True)) / (y.var(2, unbiased=False, keepdim=True) + GROUP_NORM_EPS).sqrt()
    return y.reshape(x.shape) * m.weight.view(1, c, 1, 1) + m.bias.view(1, c, 1, 1)


def conv(m: nn.Conv2d, x):
    return F.conv2d(x, m.weight, m.bias, m.stride, 1)


def block(blk: ResBlock, x, n1, n2):
    h = x if n1 is None else F.relu(gn(n1, x))
    return x + conv(blk.c2, F.relu(gn(n2, conv(blk.c1, h))))


def spatial_acts(net, obs, film=None):
    sp_, n = net.spatial, net.variant.spatial_norm
    _, e, sp, _, _ = net.encode(obs)
    if film is not None:
        sp = sp_(obs, e, film)
    spatial = obs["spatial"]
    x = F.relu(gn(n.stem, conv(sp_.stem, conv_layout(torch.cat([spatial, sp_.scatter_mean(obs, e, spatial.shape[2], spatial.shape[3])], 1)))))
    acts = {"spatial.stem": x}
    if film is not None:
        x = x * (1.0 + film[0][:, :, None, None]) + film[1][:, :, None, None]
    for i, d in enumerate(sp_.downs):
        y = F.relu(gn(n.down[i], conv(d.conv, x)))
        acts[f"spatial.down{i}.conv"] = y
        acts[f"spatial.down{i}.inner"] = F.relu(gn(n.down_res[i], conv(d.res.c1, y)))
        x = block(d.res, y, None, n.down_res[i])
        acts[f"spatial.down{i}.res"] = x
    for j, r in enumerate(sp_.bottleneck):
        acts[f"spatial.bottleneck{j}.inner"] = F.relu(gn(n.bottleneck_mid[j], conv(r.c1, F.relu(gn(n.bottleneck_in[j], x)))))
        x = block(r, x, n.bottleneck_in[j], n.bottleneck_mid[j])
        acts[f"spatial.bottleneck{j}"] = x
    return sp, acts


def test_default_is_off_032_ac16():
    v = VariantConfig()
    assert v.spatial_norm == 0 and not v.any() and ResBlock(4).norms is None
    assert "variant" not in UraniumNetConfig().to_dict()
    d = UraniumNetConfig(variant=PREACT).to_dict()
    assert "spatial_norm" not in d["variant"]
    assert UraniumNetConfig.from_dict(d).variant == PREACT
    cfg, *_ = tiny(PREACT)
    net = net_of(cfg)
    assert net.variant.spatial_norm is None and net.spatial.norms is None
    assert not any(k.startswith(PREFIX) for k in net.state_dict())
    assert all(m.norms is None for m in [*net.spatial.downs, *(x.res for x in net.spatial.downs), *net.spatial.bottleneck])


def test_model_table_032_r26():
    assert VariantConfig.from_table({"spatial_preact": True, "spatial_norm": 16}) == VariantConfig(spatial_preact=True, spatial_norm=16)
    assert VariantConfig.from_table({"spatial_norm": 0}) == VariantConfig()
    for t in ({"spatial_norm": 16}, {"spatial_preact": True, "spatial_norm": -1}, {"spatial_preact": True, "spatial_norm": True}, {"spatial_preact": True, "spatial_norm": 16.0}):
        with pytest.raises(ValueError):
            VariantConfig.from_table(t)
    on = UraniumNetConfig(variant=NORM)
    assert on.to_dict()["variant"]["spatial_norm"] == PER
    assert UraniumNetConfig.from_dict(json.loads(json.dumps(on.to_dict()))).variant == NORM
    cfg, *_ = tiny(replace(NORM, spatial_norm=3))
    with pytest.raises(ValueError, match="032-R26"):
        net_of(cfg)
    cfg.variant = VariantConfig(spatial_norm=PER)
    with pytest.raises(ValueError, match="spatial_preact"):
        net_of(cfg)


@pytest.mark.parametrize("name", CONFIGS)
def test_norm_adds_only_groupnorm_032_ac16(name):
    cfg, *_ = tiny(CONFIGS[name])
    on = net_of(cfg, 9)
    after_on = torch.rand(3)
    off = net_of(replace(cfg, variant=replace(cfg.variant, spatial_norm=0)), 9)
    after_off = torch.rand(3)
    sa, sb = on.state_dict(), off.state_dict()
    added = [k for k in sa if k not in sb]
    assert list(sa)[:len(sb)] == list(sb), "the GroupNorm tensors come after every existing tensor"
    assert [k for k, _ in on.named_parameters()][:len(list(off.named_parameters()))] == [k for k, _ in off.named_parameters()]
    assert len(added) == 26 and all(k.startswith(PREFIX) for k in added)
    for k, v in sb.items():
        assert torch.equal(sa[k], v), k
    assert torch.equal(after_on, after_off), "no global RNG"
    for k in added:
        assert torch.equal(sa[k], torch.ones_like(sa[k]) if k.endswith("weight") else torch.zeros_like(sa[k])), k
    n, c = on.variant.spatial_norm, cfg.spatial
    mods = [n.stem, *n.down, *n.down_res, *n.bottleneck_in, *n.bottleneck_mid]
    chans = [c.stem_channels, *c.stages, *c.stages, *[c.stages[-1]] * 4]
    assert len(mods) == 13
    for m, ch in zip(mods, chans, strict=True):
        assert (m.num_groups, m.num_channels, m.eps, m.affine) == (ch // PER, ch, 1e-5, True)
    assert all(d.res.norms[0] is None and d.res.nonneg_input for d in on.spatial.downs)
    twin = copy.deepcopy(on)
    assert twin.spatial.norms[0] is twin.variant.spatial_norm.stem and twin.spatial.norms[0] is not n.stem
    assert twin.spatial.bottleneck[1].norms == (twin.variant.spatial_norm.bottleneck_in[1], twin.variant.spatial_norm.bottleneck_mid[1])


@pytest.mark.parametrize("wgrad", ["vendor", "bmm"])
@pytest.mark.parametrize("ch", [8, RES_BMM_MAX_CH + 8])
@pytest.mark.parametrize("nonneg", [False, True])
def test_resblock_matches_definition_032_ac16(monkeypatch, nonneg, ch, wgrad):
    monkeypatch.setenv("URANIUM_CONV_WGRAD", wgrad)
    torch.manual_seed(1)
    blk = ResBlock(ch, True, nonneg_input=nonneg)
    norms = shake(nn.ModuleList([group_norm(ch, PER), group_norm(ch, PER)]))
    blk.norms = (None if nonneg else norms[0], norms[1])
    x0 = torch.randn(3, ch, 6, 6)
    x0 = F.relu(x0) if nonneg else x0
    params = [*blk.parameters(), *(p for m in blk.norms if m is not None for p in m.parameters())]
    x = x0.clone().requires_grad_(True)
    y = blk(x)
    x2 = x0.clone().requires_grad_(True)
    want = block(blk, x2, None if nonneg else norms[0], norms[1])
    assert torch.allclose(y, want, atol=1e-5, rtol=1e-5)
    assert (y < 0).any()
    with torch.no_grad():
        h = x0 if nonneg else F.relu(gn(norms[0], x0))
        assert torch.allclose(blk.inner_act(x0), F.relu(gn(norms[1], conv(blk.c1, h))), atol=1e-5, rtol=1e-5)
    g = torch.randn_like(y)
    for a, b in zip(torch.autograd.grad(y, [x, *params], g), torch.autograd.grad(want, [x2, *params], g), strict=True):
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)


def test_group_norm_is_per_sample_032_r26():
    m = shake(nn.ModuleList([group_norm(8, PER)]))[0]
    x = torch.randn(5, 8, 6, 6, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        full = m(x)
        assert torch.allclose(full, gn(m, x), atol=1e-5)
        for i in range(5):
            assert torch.allclose(m(x[i:i + 1]), full[i:i + 1], atol=1e-6)
    with pytest.raises(ValueError, match="032-R26"):
        group_norm(8, 3)


@pytest.mark.parametrize("wgrad", ["vendor", "bmm"])
@pytest.mark.parametrize("with_film", [False, True])
def test_spatial_encoder_matches_definition_032_ac16(monkeypatch, with_film, wgrad):
    monkeypatch.setenv("URANIUM_CONV_WGRAD", wgrad)
    cfg, b, t, obs, _, _ = tiny(NORM)
    net = norm_net(cfg)
    g = torch.Generator().manual_seed(6)
    film = tuple(torch.randn(b * t, cfg.spatial.stem_channels, generator=g) for _ in range(2)) if with_film else None
    params = [p for p in [*net.spatial.parameters(), *net.variant.spatial_norm.parameters()] if p is not net.spatial.pool.weight and p is not net.spatial.pool.bias]
    sp, acts = spatial_acts(net, obs, film)
    x = acts["spatial.bottleneck1"]
    if with_film:
        pre = acts["spatial.stem"]
        swapped = F.relu(pre * (1.0 + film[0][:, :, None, None]) + film[1][:, :, None, None])
        assert not torch.equal(swapped, pre * (1.0 + film[0][:, :, None, None]) + film[1][:, :, None, None]), "the FiLM output goes negative, so its place matters"
    assert torch.allclose(sp["s8"], x, atol=1e-5)
    for got, i in zip(sp["skips"], range(3), strict=True):
        assert torch.allclose(got, acts[f"spatial.down{i}.res"], atol=1e-5)
    assert torch.allclose(sp["pool"], net.spatial.pool(x.flatten(1)), atol=1e-5)
    g = torch.randn(sp["s8"].shape, generator=torch.Generator().manual_seed(4))
    for a, w, p in zip(torch.autograd.grad(sp["s8"], params, g), torch.autograd.grad(x, params, g), params, strict=True):
        assert torch.allclose(a, w, atol=1e-4, rtol=1e-3), tuple(p.shape)


def test_stem_pad_and_channels_last_032_ac16():
    cfg, b, t, obs, _, _ = tiny(NORM)
    net = norm_net(cfg)
    with torch.no_grad():
        want = net.encode(obs)[2]
        got = net.conv_layout(True, True).encode(obs)[2]
    assert torch.allclose(got["s8"], want["s8"], atol=1e-5) and torch.allclose(got["pool"], want["pool"], atol=1e-5)


def test_monitor_points_under_norm_032_r25():
    cfg, b, t, obs, labels, priv = tiny(NORM)
    net = norm_net(cfg)
    st = relu_stats(net, obs, labels, priv, net.init_state(b), t)
    with torch.no_grad():
        _, acts = spatial_acts(net, obs)
    acts["spatial.s8"] = acts["spatial.bottleneck1"]
    with torch.no_grad():
        plain = F.relu(conv(net.spatial.downs[0].conv, acts["spatial.stem"]))
    assert not torch.equal(plain > 0, acts["spatial.down0.conv"] > 0)
    for name, act in acts.items():
        want = channel_stats(act, channel_dim=1)
        assert float(st[name]["pos_frac"]) == pytest.approx(float(want["pos_frac"]), abs=1e-6), name
        assert int(st[name]["never_active"]) == int(want["never_active"]), name


@pytest.mark.parametrize("name", CONFIGS)
def test_step_matches_sequence_032_ac16(name):
    from uranium.model.state import leaves

    cfg, b, t, obs, labels, _ = tiny(CONFIGS[name])
    net = norm_net(cfg)
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


def test_checkpoint_round_trip_032_ac16(tmp_path):
    cfg, b, t, obs, labels, _ = tiny(NORM_ALL)
    net = norm_net(cfg)
    ckpt.save_model(tmp_path, net)
    ckpt.save_meta(tmp_path, net.cfg, {}, "f32")
    back, meta = ckpt.load_model(tmp_path)
    assert back.cfg.variant == cfg.variant and meta["config"]["variant"]["spatial_norm"] == PER
    with torch.no_grad():
        x = net.forward_seq(obs, labels, None, net.init_state(b), t)["value"]
        y = back.eval().forward_seq(obs, labels, None, back.init_state(b), t)["value"]
    assert torch.equal(x, y)


def test_weights_only_graft_032_ac16(tmp_path):
    cfg, *_ = tiny(PREACT)
    off = net_of(cfg)
    ckpt.save_model(tmp_path, off)
    ckpt.save_meta(tmp_path, off.cfg, {}, "f32")
    net, _ = ckpt.load_model(tmp_path, graft=NORM)
    assert net.cfg.variant == NORM
    sa, sb = net.state_dict(), off.state_dict()
    assert {k for k in sa if k not in sb} == {k for k in sa if k.startswith(PREFIX)} and len(sa) == len(sb) + 26
    for k, v in sb.items():
        assert torch.equal(sa[k], v), k
    for k in sa.keys() - sb.keys():
        assert torch.equal(sa[k], torch.ones_like(sa[k]) if k.endswith("weight") else torch.zeros_like(sa[k])), k
    same, _ = ckpt.load_model(tmp_path, graft=PREACT)
    assert same.cfg.variant == PREACT and same.state_dict().keys() == sb.keys()
    other, _ = ckpt.load_model(tmp_path, graft=replace(NORM, core_skip=True))
    assert other.cfg.variant == PREACT, "any other [model] difference is not grafted"
    on = tmp_path / "on"
    on.mkdir()
    ckpt.save_model(on, net)
    ckpt.save_meta(on, net.cfg, {}, "f32")
    assert ckpt.load_model(on, graft=PREACT)[0].cfg.variant == NORM


def test_bc_init_grafts_and_resume_refuses_041_r4(tiny_config, shards, tmp_path, capsys):
    cfg = bc_cfg(tiny_config)
    a = BcTrainer(cfg, make_net(PREACT), CPU, "f32")
    a.train_batch(feeder(tiny_config, shards).next(), 3e-3, 0)
    start = tmp_path / "start"
    a.save(str(start), json.dumps({"step": 1}))
    cfg_json = json.dumps(cfg)
    with pytest.raises(ValueError, match=r"\[model\]"):
        BcTrainer.make(cfg_json, "resume", str(start), 0, "lstm", CPU, "f32", NORM)
    with pytest.raises(ValueError, match=r"\[model\]"):
        BcTrainer.make(cfg_json, "init", str(start), 0, "lstm", CPU, "f32", replace(NORM, core_skip=True))
    assert "spatial_norm" not in capsys.readouterr().out, "no graft message before a refusal"
    t = BcTrainer.make(cfg_json, "init", str(start), 0, "lstm", CPU, "f32", NORM)
    assert "spatial_norm = 4 added" in capsys.readouterr().out
    assert t.net.cfg.variant == NORM
    pa, pt = params_of(a.net), params_of(t.net)
    assert all(torch.equal(pa[k], pt[k]) for k in pa) and len(pt) == len(pa) + 26
    out = json.loads(t.train_batch(feeder(tiny_config, shards).next(), 3e-3, 0))
    assert out["total"] is not None
    moved = params_of(t.net)
    assert any(not torch.equal(moved[k], pt[k]) for k in moved if k.startswith(PREFIX)), "GroupNorm scale and bias are trained"
    run = tmp_path / "run"
    t.save(str(run), json.dumps({"step": 1}))
    assert json.loads((run / "meta.json").read_text(encoding="utf-8"))["config"]["variant"]["spatial_norm"] == PER
    r = BcTrainer.make(cfg_json, "resume", str(run), 0, "lstm", CPU, "f32", NORM)
    pr = params_of(r.net)
    assert all(torch.equal(moved[k], pr[k]) for k in moved)
    with pytest.raises(ValueError, match=r"\[model\]"):
        BcTrainer.make(cfg_json, "init", str(run), 0, "lstm", CPU, "f32", PREACT)


def test_gn_t1_config_032_r26():
    import tomllib

    def flat(t, pre=""):
        out = {}
        for k, x in t.items():
            out.update(flat(x, f"{pre}{k}.") if isinstance(x, dict) else {f"{pre}{k}": x})
        return out

    root = ROOT / "config"
    new, old = (flat(tomllib.loads((root / f).read_text(encoding="utf-8"))) for f in ("bc_m2_gn_t1.toml", "bc_m2_teacher_t1.toml"))
    assert {k for k in new.keys() | old.keys() if new.get(k) != old.get(k)} == {"model.spatial_norm", "steps.s2", "steps.s2_warmup"}
    assert (new["model.spatial_norm"], new["steps.s2"], new["steps.s2_warmup"]) == (16, 3000, 150)
    v = model_variant(str(root / "bc_m2_gn_t1.toml"))
    assert v.grafts_onto(model_variant(str(root / "bc_m2_teacher_t1.toml")))
    c = UraniumNetConfig().spatial
    assert not any(ch % v.spatial_norm for ch in (c.stem_channels, *c.stages))
