import json
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from uranium import ckpt
from uranium.model.config import model_variant
from uranium.bc.trainer import BcTrainer
from uranium.model import UraniumNet, UraniumNetConfig, action_columns, to_labels, to_obs
from uranium.model.config import VariantConfig

from testkit import HEADS, SEG, VARIANTS, net_of, perturb, rows_at, state_after, tiny, widen_scalar

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "burn_parity" / "tiny-gtrxl"
BYPASS_VARIANTS = {
    "bypass": VariantConfig(bypass=True),
    "bypass_all": VariantConfig(terrain=True, ctx_token=True, attn_pool=True, outer=SEG, bypass=True),
}
WITH_BYPASS = {**VARIANTS, **BYPASS_VARIANTS}


def test_config_table_032_ac11():
    assert not VariantConfig.from_table(None).any()
    assert VariantConfig.from_table({"outer": 32, "terrain": True}) == VariantConfig(terrain=True, outer=32)
    for bad in ({"terain": True}, {"terrain": 1}, {"outer": True}, {"outer": -1}, {"outer": 3.0}):
        with pytest.raises(ValueError):
            VariantConfig.from_table(bad)
    assert "variant" not in UraniumNetConfig().to_dict()
    on = UraniumNetConfig(variant=VariantConfig(attn_pool=True))
    assert UraniumNetConfig.from_dict(on.to_dict()).variant == on.variant
    assert UraniumNetConfig.from_dict(UraniumNetConfig().to_dict()).variant == VariantConfig()


def test_comparison_configs_032_r20():
    base = tomllib.loads((ROOT / "config" / "bc_core_ablation_pt.toml").read_text(encoding="utf-8"))
    want = {"terrain": VariantConfig(terrain=True), "ctx": VariantConfig(ctx_token=True), "pool": VariantConfig(attn_pool=True), "outer": VariantConfig(outer=32), "all": VariantConfig(True, True, True, 32)}
    for name, v in want.items():
        path = ROOT / "config" / f"bc_arch_abl_{name}_pt.toml"
        cfg = tomllib.loads(path.read_text(encoding="utf-8"))
        assert cfg.pop("model") is not None and cfg == base, name
        assert model_variant(str(path)) == v, name
    assert not model_variant(str(ROOT / "config" / "bc_core_ablation_pt.toml")).any()
    assert model_variant(str(ROOT / "config" / "bc.toml")) == want["all"]
    assert model_variant(str(ROOT / "config" / "bc_ft_empty.toml")) == want["all"]


@pytest.mark.parametrize("name", VARIANTS)
def test_shared_initial_weights_032_ac11(name):
    cfg, *_ = tiny()
    on = replace(cfg, variant=VARIANTS[name])
    a = net_of(on, 9)
    after_on = torch.rand(3)
    b = net_of(cfg, 9)
    after_off = torch.rand(3)
    sa, sb = a.state_dict(), b.state_dict()
    assert any(k.startswith("variant.") for k in sa) and not any(k.startswith("variant.") for k in sb)
    assert set(sb) <= set(sa)
    for k, v in sb.items():
        assert torch.equal(sa[k], v), k
    assert torch.equal(after_on, after_off), "the global RNG moved"


@pytest.mark.parametrize("name", VARIANTS)
def test_step_matches_sequence_032_ac11(name):
    cfg, b, t, obs, labels, _ = tiny(VARIANTS[name])
    net = net_of(cfg)
    with torch.no_grad():
        seq = net.forward_seq(obs, labels, None, net.init_state(b), t)
        s = net.init_state(b)
        values = []
        for k in range(t):
            out = net.act(rows_at(obs, b, t, k), s, 0.0)
            s = out["state"]
            values.append(out["value"])
    assert torch.allclose(seq["value"].view(b, t, -1), torch.stack(values, 1), atol=1e-5)
    from uranium.model.state import leaves

    for x, y in zip(leaves(seq["state"]), leaves(s)):
        assert x.dtype == y.dtype and torch.allclose(x.float(), y.float(), atol=1e-5)


@pytest.mark.parametrize("name", ["outer", "all"])
def test_outer_heads_inputs_match_per_decision(name):
    cfg, b, t, obs, labels, _ = tiny(VARIANTS[name])
    net = net_of(cfg)
    with torch.no_grad():
        seq = net.forward_seq(obs, labels, None, net.init_state(b), t)["logp"]
        cs, os_ = net.init_state(b)
        hs, es, sps, scs = [], [], [], []
        for k in range(t):
            core_in, e, sp, sc, _ = net.encode(rows_at(obs, b, t, k), h_o=os_[0])
            h, cs = net.core.step(core_in, cs)
            os_ = net.outer.step(h, os_)
            hs.append(h)
            es.append(e)
            sps.append(sp)
            scs.append(sc)
        rows = lambda xs: torch.stack(xs, 1).reshape(b * t, *xs[0].shape[1:])  # noqa: E731
        sp = {"s8": rows([x["s8"] for x in sps]), "skips": [rows([x["skips"][i] for x in sps]) for i in range(3)], "pool": rows([x["pool"] for x in sps])}
        sc = {k: rows([x[k] for x in scs]) for k in scs[0]}
        ref = net.heads(labels, 1.0, rows(hs), sc["c"], rows(es), sp, obs, net.table)["logp"]
    for k in HEADS:
        assert torch.allclose(seq[k], ref[k], atol=1e-4), k


@pytest.mark.parametrize("name", ["outer", "all"])
def test_outer_state_carries_across_chunks(name):
    cfg, b, t, obs, labels, _ = tiny(VARIANTS[name])
    net = net_of(cfg)
    half = t // 2
    cut = lambda d, lo: {k: x.view(b, t, *x.shape[1:])[:, lo:lo + half].reshape(b * half, *x.shape[1:]) for k, x in d.items()}  # noqa: E731
    with torch.no_grad():
        whole = net.forward_seq(obs, labels, None, net.init_state(b), t)["value"].view(b, t, -1)
        first = net.forward_seq(cut(obs, 0), cut(labels, 0), None, net.init_state(b), half)
        s = net.mask_state(net.detach_state(first["state"]), torch.ones(b, dtype=torch.bool))
        second = net.forward_seq(cut(obs, half), cut(labels, half), None, s, half)
    joined = torch.cat([first["value"].view(b, half, -1), second["value"].view(b, half, -1)], 1)
    assert torch.allclose(whole, joined, atol=1e-5)


def test_outer_state_updates_on_segments():
    cfg, b, t, obs, _, _ = tiny(VARIANTS["outer"])
    net = net_of(cfg)
    s = net.init_state(b)
    seen = []
    with torch.no_grad():
        for k in range(t):
            s = net.act(rows_at(obs, b, t, k), s, 0.0)["state"]
            seen.append(s[1][0].clone())
    assert torch.equal(s[1][3], torch.full((b,), float(t)))
    assert not seen[0].any() and seen[1].any()
    assert torch.equal(seen[1], seen[2]) and not torch.equal(seen[2], seen[3])
    kept = net.mask_state(s, torch.tensor([True, False]))
    assert kept[1][0][0].any() and not kept[1][0][1].any() and kept[1][3].tolist() == [float(t), 0.0]


@pytest.mark.parametrize("name", ["ctx", "outer", "all"])
def test_tokens_stay_out_of_entities(name):
    cfg, b, t, obs, _, _ = tiny(VARIANTS[name])
    net = net_of(cfg)
    n = obs["entity_mask"].shape[1]
    with torch.no_grad():
        _, e, _, _, _ = net.encode(rows_at(obs, b, t, 0), h_o=torch.zeros(b, 256) if net.outer is not None else None)
        assert e.shape[1] == n
        out = net.act(rows_at(obs, b, t, 0), net.init_state(b), 1.0, torch.Generator().manual_seed(1))
    assert (out["action"]["selected"] <= n).all() and (out["action"]["target_unit"] < n).all()


@pytest.mark.parametrize("name", ["ctx", "outer", "all", "bypass_all"])
def test_trimmed_entity_padding_keeps_the_update_032_r19(name):
    import numpy as np

    from uranium.model.net import ENTITY_KEYS, trim_entity_pad

    cfg, b, t, _, _, _ = tiny(WITH_BYPASS[name])
    cfg.entity.dropout = 0.0
    inp = {k: v.numpy() for k, v in load_file(FIX / "inputs.safetensors").items()}
    if cfg.variant.bypass:
        inp = widen_scalar(cfg, inp)
    k = cfg.variant.tokens()
    n = inp["entity_mask"].shape[1]
    padded = dict(inp)
    for key in ENTITY_KEYS:
        x = inp[key]
        padded[key] = np.concatenate([x, np.zeros((x.shape[0], k, *x.shape[2:]), x.dtype)], 1)
    padded["selected"] = np.where(inp["selected"] == n, n + k, inp["selected"])
    trimmed = trim_entity_pad(padded, k)
    assert trimmed["entity_mask"].shape[1] == n
    keep = dict(padded, entity_mask=padded["entity_mask"].copy())
    keep["entity_mask"][0, -1] = True
    assert trim_entity_pad(keep, k) is keep
    res = []
    for g in (padded, trimmed):
        net = perturb(net_of(cfg)).train()
        out = net.forward_seq(to_obs(g, "cpu"), to_labels(g, "cpu"), torch.as_tensor(g["privileged"]), net.init_state(b), t)
        loss = sum(out["logp"][h].sum() for h in HEADS) + out["value"].sum()
        loss.backward()
        res.append((torch.stack([out["logp"][h] for h in HEADS]).detach(), {p: q.grad.clone() for p, q in net.named_parameters() if q.grad is not None}))
    torch.testing.assert_close(res[0][0], res[1][0], atol=1e-5, rtol=1e-5)
    assert res[0][1].keys() == res[1][1].keys()
    for p, g in res[0][1].items():
        torch.testing.assert_close(g, res[1][1][p], atol=1e-5, rtol=1e-4, msg=p)


def test_attn_pool_empty_set():
    cfg, b, t, obs, _, _ = tiny(VARIANTS["pool"])
    net = net_of(cfg)
    e = torch.randn(b, obs["entity_mask"].shape[1], cfg.entity.d_model)
    own = obs["entity_mask"][:b]
    pool = net.variant.pool
    joined = {}
    handle = pool.out.register_forward_hook(lambda _m, args, _o: joined.update(x=args[0]))
    out = pool(e, own, torch.zeros_like(own))
    handle.remove()
    half = joined["x"].shape[1] // 2
    assert torch.isfinite(out).all()
    assert joined["x"][:, :half].abs().sum() > 0 and not joined["x"][:, half:].any()


@pytest.mark.parametrize("name", ["all"])
def test_masks_hold_032_ac4(name):
    cfg, b, t, obs, _, _ = tiny(VARIANTS[name])
    net = net_of(cfg)
    g = torch.Generator().manual_seed(5)
    o = rows_at(obs, b, t, 0)
    s = net.init_state(b)
    with torch.no_grad():
        for _ in range(20):
            out = net.act(o, s, 1.0, g)
            assert o["type_mask"][torch.arange(b), out["action"]["action"]].all()
            s = out["state"]


@pytest.mark.parametrize("name", WITH_BYPASS)
def test_privileged_only_reaches_value_032_r11(name):
    cfg, b, t, obs, labels, priv = tiny(WITH_BYPASS[name])
    net = perturb(net_of(cfg))
    priv = priv.clone().requires_grad_(True)
    out = net.forward_seq(obs, labels, priv, net.init_state(b), t)
    policy = sum(v.sum() for v in out["logp"].values())
    (g,) = torch.autograd.grad(policy, priv, allow_unused=True, retain_graph=True)
    assert g is None or not g.any()


@pytest.mark.parametrize("name", VARIANTS)
def test_entity_recompute_keeps_the_update_041_r17(name):
    cfg, b, t, obs, labels, priv = tiny(VARIANTS[name])
    grads = []
    for chunk in (0, 3):
        net = net_of(cfg).train()
        out = net.forward_seq(obs, labels, priv, net.init_state(b), t, chunk)
        loss = sum(out["logp"][k].sum() for k in HEADS) + out["value"].sum()
        loss.backward()
        grads.append({k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    assert any(k.startswith("variant.") for k in grads[0])
    for k, g in grads[0].items():
        assert torch.allclose(g, grads[1][k], atol=1e-5, rtol=1e-4), k


@pytest.mark.parametrize("name", VARIANTS)
def test_checkpoint_round_trip(name, tmp_path):
    cfg, b, t, obs, labels, _ = tiny(VARIANTS[name])
    net = net_of(cfg)
    ckpt.save_model(tmp_path, net)
    ckpt.save_meta(tmp_path, net.cfg, {}, "f32")
    back, meta = ckpt.load_model(tmp_path)
    assert back.cfg.variant == cfg.variant and meta["config"]["variant"]["outer"] == cfg.variant.outer
    with torch.no_grad():
        x = net.eval().forward_seq(obs, labels, None, net.init_state(b), t)["value"]
        y = back.eval().forward_seq(obs, labels, None, back.init_state(b), t)["value"]
    assert torch.equal(x, y)


def test_resume_with_another_variant_is_refused_032_ac11(tmp_path):
    cfg, *_ = tiny(VARIANTS["pool"])
    net = net_of(cfg)
    ckpt.save_model(tmp_path, net)
    ckpt.save_meta(tmp_path, net.cfg, {}, "f32")
    cfg_json = json.dumps({"lr": 1e-3, "weight_decay": 0.0, "seq_len": 4, "value": {}})
    with pytest.raises(ValueError, match=r"\[model\]"):
        BcTrainer.make(cfg_json, "resume", str(tmp_path), 0, "gtrxl", torch.device("cpu"), "f32", VARIANTS["terrain"])
    with pytest.raises(ValueError, match=r"\[model\]"):
        BcTrainer.make(cfg_json, "init", str(tmp_path), 0, "gtrxl", torch.device("cpu"), "f32", None)


def test_outer_segment_must_divide_seq_len():
    cfg_json = json.dumps({"lr": 1e-3, "weight_decay": 0.0, "seq_len": 64, "value": {}})
    with pytest.raises(ValueError, match="divide"):
        BcTrainer.make(cfg_json, "new", None, 0, "gtrxl", torch.device("cpu"), "f32", VariantConfig(outer=48))


def per_decision(net, obs, labels, s0, b: int, t: int):
    cs, os_ = s0
    hs, es, sps, scs = [], [], [], []
    for k in range(t):
        core_in, e, sp, sc, _ = net.encode(rows_at(obs, b, t, k), h_o=os_[0])
        h, cs = net.core.step(core_in, cs)
        os_ = net.outer.step(h, os_)
        hs.append(h)
        es.append(e)
        sps.append(sp)
        scs.append(sc)
    rows = lambda xs: torch.stack(xs, 1).reshape(b * t, *xs[0].shape[1:])  # noqa: E731
    sp = {"s8": rows([x["s8"] for x in sps]), "skips": [rows([x["skips"][i] for x in sps]) for i in range(3)], "pool": rows([x["pool"] for x in sps])}
    sc = {k: rows([x[k] for x in scs]) for k in scs[0]}
    logp = net.heads(labels, 1.0, rows(hs), sc["c"], rows(es), sp, obs, net.table)["logp"]
    return logp, net.value(rows(hs), sc["z"], sc["race"], None), (cs, os_)


@pytest.mark.parametrize("core", ["tiny-gtrxl", "tiny-lstm"])
@pytest.mark.parametrize("name", ["outer", "all"])
@pytest.mark.parametrize("counts", [(1, 2), (3, 0), (5, 1)])
def test_phased_matches_per_decision_032_ac11(name, counts, core):
    from uranium.model.state import leaves

    cfg, b, t, obs, labels, _ = tiny(VARIANTS[name], FIX.parent / core)
    net = net_of(cfg)
    s0 = state_after(net, obs, b, t, counts)
    seq = net.forward_seq(obs, labels, None, s0, t, phased=True)
    logp, value, state = per_decision(net, obs, labels, s0, b, t)
    for k in HEADS:
        assert torch.allclose(seq["logp"][k], logp[k], atol=1e-4), k
    assert torch.allclose(seq["value"], value, atol=1e-5)
    for x, y in zip(leaves(seq["state"]), leaves(state)):
        assert torch.allclose(x.float(), y.float(), atol=1e-5)
    params = [net.outer.weight_ih, next(net.core.parameters()), next(net.entity.parameters())]
    loss = lambda lp, v: sum(lp[k].sum() for k in HEADS) + v.sum()  # noqa: E731
    ga = torch.autograd.grad(loss(seq["logp"], seq["value"]), params)
    gb = torch.autograd.grad(loss(logp, value), params)
    for x, y in zip(ga, gb):
        assert torch.allclose(x, y, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("name", ["outer", "all"])
@pytest.mark.parametrize("counts", [(0, 0), (2, 4)])
def test_phased_zero_phase_matches_bc_path_032_ac11(name, counts):
    from uranium.model.state import leaves

    cfg, b, t, obs, labels, _ = tiny(VARIANTS[name])
    net = net_of(cfg)
    s0 = state_after(net, obs, b, t, counts)
    with torch.no_grad():
        bc = net.forward_seq(obs, labels, None, s0, t)
        rl = net.forward_seq(obs, labels, None, s0, t, phased=True)
    for k in HEADS:
        assert torch.allclose(bc["logp"][k], rl["logp"][k], atol=1e-5), k
    assert torch.allclose(bc["value"], rl["value"], atol=1e-6)
    for x, y in zip(leaves(bc["state"]), leaves(rl["state"])):
        assert torch.allclose(x.float(), y.float(), atol=1e-6)


@pytest.mark.gpu
def test_fused_entity_attention_with_tokens(monkeypatch):
    from testkit import kernel_or_skip, rel

    from uranium.model.config import new_net_config
    from uranium.bench.inputs import random_obs

    kernel_or_skip()
    cfg = new_net_config("gtrxl", True, VariantConfig(ctx_token=True, outer=32))
    cfg.entity.dropout = 0.0
    res = {}
    for mode in ("torch", "fused"):
        monkeypatch.setenv("URANIUM_ENTITY_ATTN", mode)
        torch.manual_seed(5)
        net = UraniumNet(cfg, action_columns()).cuda()
        obs = random_obs(4, 40, cfg.input, "cuda", torch.Generator().manual_seed(3), valid=33)
        h_o = torch.randn(4, 256, generator=torch.Generator().manual_seed(4)).cuda()
        _, e, _, _, _ = net.encode(obs, h_o=h_o)
        e.square().sum().backward()
        res[mode] = (e.detach(), {k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None and k.startswith(("entity", "variant"))})
    assert rel(res["fused"][0], res["torch"][0]) < 1e-4
    assert res["fused"][1].keys() == res["torch"][1].keys()
    assert {"variant.ctx_bias", "variant.outer.tok_bias"} <= res["torch"][1].keys()
    floor = 1e-3 * max(float(g.norm()) for g in res["torch"][1].values())
    for k, g in res["torch"][1].items():
        err = float((res["fused"][1][k] - g).norm()) / max(float(g.norm()), floor)
        assert err < 1e-3, (k, err)
    for k in ("variant.ctx_bias", "variant.outer.tok_bias"):
        g = res["torch"][1][k]
        err = float((res["fused"][1][k] - g).norm() / g.norm())
        assert err < 1e-2, (k, err)


@pytest.mark.gpu
@torch.no_grad()
def test_graphed_act_matches_eager_variant():
    from uranium.agent.infer import GraphedAct
    from uranium.model.state import leaves
    from uranium.model.config import new_net_config
    from uranium.bench.inputs import random_obs
    from uranium.device import autocast, resolve

    if not torch.cuda.is_available():
        pytest.skip("no CUDA / ROCm device")
    _, dev, prec = resolve(None)
    torch.manual_seed(0)
    net = UraniumNet(new_net_config("gtrxl", True, VariantConfig(True, True, True, 32)), action_columns()).to(dev).eval()
    graphed = GraphedAct(net.act, dev, prec)
    exact = torch.version.hip is None
    close = (lambda a, b: torch.equal(a, b)) if exact else (lambda a, b: torch.allclose(a.float(), b.float(), atol=1e-5, rtol=1e-5))
    obs = random_obs(3, 128, net.cfg.input, dev, torch.Generator().manual_seed(1), valid=119)
    pad = torch.tensor([0, 1, 2, 0], device=dev)
    obs4 = {k: v.index_select(0, pad) for k, v in obs.items()}
    s_eager, s_graph = net.init_state(4, dev), net.init_state(3, dev)
    for _ in range(40):
        with autocast(dev, prec):
            want = net.act(obs4, s_eager, 0.0)
        got = graphed(obs, s_graph, 0.0)
        for k in want["action"]:
            assert torch.equal(got["action"][k], want["action"][k][:3]), k
        assert close(got["value"], want["value"][:3])
        s_eager, s_graph = want["state"], got["state"]
    for x, y in zip(leaves(s_graph), leaves(s_eager)):
        assert close(x, y[:3])
