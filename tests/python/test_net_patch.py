import json
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from uranium import ckpt
from uranium.bc.trainer import BcTrainer
from uranium.model import UraniumNet, UraniumNetConfig, action_columns
from uranium.model.config import VariantConfig, model_net, net_sizes_from_table, net_sizes_of, new_net_config, size_mismatch
from uranium.model.encoders import PatchEncoder
from uranium.model.heads import PatchLocationHead
from uranium.model.monitor import ReluMonitor, relu_stats
from uranium.model.state import leaves

from testkit import CPU, HEADS, VARIANTS, bc_cfg, feeder, make_net, net_of, params_of, rows_at, state_after, tiny

P = 8
D_P = 24
PATCH = VariantConfig(spatial_patch=P)
PATCH_VARIANTS = {
    "patch": PATCH,
    "patch_outer": replace(VARIANTS["outer"], spatial_patch=P),
    "patch_all": replace(VARIANTS["all"], spatial_patch=P, bypass=True, core_skip=True),
}
B_NET = {
    "entity": {"d_model": 512, "ffn": 2048, "layers": 6, "heads": 16},
    "core": {"hidden": 1024, "gtrxl_layers": 4, "gtrxl_heads": 8, "gtrxl_memory": 64},
    "scalar": {"hidden": 512},
    "heads": {"ar": 512},
    "value": {"hidden": 512},
    "spatial": {"patch_dim": 512},
}
CNN_MODULES = ("spatial.stem", "spatial.downs", "spatial.bottleneck", "spatial.pool", "heads.location.ups")


def patch_tiny(variant=PATCH, d_p=D_P):
    cfg, b, t, obs, labels, priv = tiny(variant)
    cfg.spatial.patch_dim = d_p
    return cfg, b, t, obs, labels, priv


def test_net_absent_keeps_the_model_032_ac18():
    plain = new_net_config("gtrxl", True)
    assert new_net_config("gtrxl", True, None, {}) == plain == new_net_config("gtrxl", True, VariantConfig(), net_sizes_from_table(None))
    d = plain.to_dict()
    assert "patch_dim" not in d["spatial"] and "variant" not in d
    assert set(d["spatial"]) == {"scatter_channels", "stem_channels", "stages", "pool"}
    torch.manual_seed(3)
    a = UraniumNet(UraniumNetConfig(input=plain.input, core=replace(plain.core), value=replace(plain.value)), action_columns())
    torch.manual_seed(3)
    b = UraniumNet(new_net_config("gtrxl", True, None, {}), action_columns())
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_net_table_sets_sizes_032_r28():
    sizes = net_sizes_from_table({"entity": {"d_model": 64, "heads": 2, "dropout": 0.0}, "spatial": {"stages": [8, 8, 16, 16]}, "core": {"hidden": 96}})
    cfg = new_net_config("lstm", False, None, sizes)
    assert (cfg.entity.d_model, cfg.entity.heads, cfg.entity.dropout, cfg.spatial.stages, cfg.core.hidden) == (64, 2, 0.0, [8, 8, 16, 16], 96)
    assert cfg.entity.ffn == 1024 and cfg.core.kind == "Lstm"
    assert "patch_dim" not in cfg.to_dict()["spatial"]
    on = new_net_config("gtrxl", True, PATCH, net_sizes_from_table({"spatial": {"patch_dim": 64}}))
    assert on.spatial.patch_dim == 64 and on.to_dict()["spatial"]["patch_dim"] == 64 and on.to_dict()["variant"] == {**VariantConfig().to_table(), "spatial_patch": P}


@pytest.mark.parametrize("table", [
    {"entiy": {"d_model": 64}}, {"entity": {"d_modle": 64}}, {"entity": 64}, {"entity": {"d_model": 64.0}}, {"entity": {"d_model": True}},
    {"entity": {"d_model": 0}}, {"entity": {"d_model": -8}}, {"entity": {"dropout": 1}}, {"entity": {"dropout": 1.5}}, {"spatial": {"stages": [8, 8, 16]}},
    {"spatial": {"stages": (8, 8, 16, 16)}}, {"spatial": {"stages": [8, 8, 16, 0]}}, {"spatial": {"pool": "256"}}, {"input": {"map_size": 64}},
    {"core": {"kind": "Gtrxl"}}, {"value": {"win_open": True}}, {"heads": {"ar": [256]}},
])
def test_net_table_is_strict_032_ac18(table):
    with pytest.raises(ValueError, match=r"\[net\]"):
        net_sizes_from_table(table)


def test_model_net_reads_both_tables_032_r28_r29(tmp_path):
    p = tmp_path / "bc.toml"
    p.write_text("[model]\nspatial_patch = 8\nouter = 2\n[net.entity]\nd_model = 64\n[net.spatial]\npatch_dim = 32\n", encoding="utf-8")
    v, sizes = model_net(str(p))
    assert v == VariantConfig(outer=2, spatial_patch=8) and sizes == {"entity": {"d_model": 64}, "spatial": {"patch_dim": 32}}
    for text, match in (("[model]\nspatial_patch = 8\n[net.spatial]\npool = 128\n", "CNN-only"), ("[model]\nspatial_patch = 8\n[net.heads]\nloc_ar_channels = 8\n", "CNN-only"),
                        ("[net.spatial]\npatch_dim = 64\n", "without"), ("[net.spatail]\npool = 1\n", "unknown")):
        p.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            model_net(str(p))


@pytest.mark.parametrize("table", [{"spatial_patch": 8, "spatial_preact": True}, {"spatial_patch": 8, "spatial_preact": True, "spatial_norm": 16}, {"spatial_patch": -1}, {"spatial_patch": True}])
def test_patch_switch_conflicts_032_ac19(table):
    with pytest.raises(ValueError):
        VariantConfig.from_table(table)


def test_patch_must_divide_the_map_032_ac19():
    cfg, *_ = tiny(VariantConfig(spatial_patch=5))
    with pytest.raises(ValueError, match="divide"):
        net_of(cfg)
    with pytest.raises(ValueError, match="divide"):
        new_net_config("gtrxl", False, VariantConfig(spatial_patch=48))


def test_patch_model_has_no_cnn_032_ac19():
    cfg, *_ = patch_tiny(PATCH_VARIANTS["patch_all"])
    torch.manual_seed(3)
    net = UraniumNet(cfg, action_columns())
    names = [k for k, _ in net.named_parameters()]
    assert net.patch == P and isinstance(net.spatial, PatchEncoder) and isinstance(net.heads.location, PatchLocationHead)
    assert not any(k.startswith(CNN_MODULES) for k in names)
    assert {k for k in names if k.startswith("spatial.")} == {"spatial.scatter.weight", "spatial.scatter.bias", "spatial.proj.weight", "spatial.proj.bias", "spatial.pos"}
    assert net.core.input.in_features == cfg.entity.d_model + cfg.scalar.hidden
    assert net.variant.core_skip.in_features == net.variant.outer.core_add.out_features == cfg.entity.d_model + cfg.scalar.hidden
    assert net.variant.outer.film.out_features == 2 * D_P
    torch.manual_seed(3)
    again = UraniumNet(cfg, action_columns())
    assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), again.state_dict().values()))
    fixed = ("spatial.proj", "spatial.pos", "heads.location", "variant.outer.film")
    torch.manual_seed(99)
    other = UraniumNet(cfg, action_columns())
    sa, so = net.state_dict(), other.state_dict()
    for k in sa:
        if k.startswith(fixed):
            assert torch.equal(sa[k], so[k]), k
    k_proj = 1.0 / (net.spatial.proj.in_features ** 0.5)
    assert net.spatial.proj.weight.abs().max() <= k_proj and 0.01 < net.spatial.pos.std() < 0.03


def test_new_modules_leave_the_global_rng_032_r29():
    cfg, *_ = patch_tiny()
    torch.manual_seed(4)
    PatchEncoder(cfg.input, cfg.spatial, cfg.entity.d_model, P)
    PatchLocationHead(cfg.input, cfg.heads, cfg.spatial, P)
    a = torch.rand(4)
    torch.manual_seed(4)
    torch.nn.Linear(cfg.entity.d_model, cfg.spatial.scatter_channels)
    assert torch.equal(a, torch.rand(4)), "only the scatter Linear uses the global RNG"


def test_patch_projection_and_location_match_definition_032_ac19():
    cfg, b, t, obs, labels, _ = patch_tiny()
    net = net_of(cfg)
    o = rows_at(obs, b, t, 1)
    with torch.no_grad():
        e, _ = net.entity(o)
        g = torch.randn(b, D_P, generator=torch.Generator().manual_seed(2))
        beta = torch.randn(b, D_P, generator=torch.Generator().manual_seed(3))
        got = net.spatial(o, e, (g, beta))["patch"]
        side = cfg.input.map_size
        n = side // P
        sc = torch.zeros(b, cfg.spatial.scatter_channels, side, side)
        v = net.spatial.scatter(e) * o["entity_mask"][..., None].float()
        cnt = torch.zeros(b, side, side)
        for i in range(b):
            for j in range(e.shape[1]):
                cell = int(o["entity_cell"][i, j])
                sc[i, :, cell // side, cell % side] += v[i, j]
                cnt[i, cell // side, cell % side] += float(o["entity_mask"][i, j])
        x = torch.cat([o["spatial"], sc / cnt.clamp_min(1.0)[:, None]], 1)
        want = torch.zeros(b, n * n, D_P)
        for gy in range(n):
            for gx in range(n):
                block = x[:, :, gy * P:(gy + 1) * P, gx * P:(gx + 1) * P].reshape(b, -1)
                f = net.spatial.proj(block) + net.spatial.pos[gy * n + gx]
                want[:, gy * n + gx] = f * (1 + g) + beta
        assert torch.allclose(got, want, atol=1e-5)
        ar = torch.randn(b, cfg.heads.ar, generator=torch.Generator().manual_seed(4))
        loc = net.heads.location(ar, {"patch": got})
        head = net.heads.location
        for gy in range(n):
            for gx in range(n):
                y = head.out(F.relu(got[:, gy * n + gx] + head.ar(ar)))
                for r in range(P):
                    for c in range(P):
                        cell = (gy * P + r) * side + gx * P + c
                        assert torch.allclose(loc[:, cell], y[:, r * P + c], atol=1e-6)


@pytest.mark.parametrize("name", PATCH_VARIANTS)
def test_patch_step_matches_sequence_032_ac19(name):
    cfg, b, t, obs, labels, _ = patch_tiny(PATCH_VARIANTS[name])
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
    for x, y in zip(leaves(seq["state"]), leaves(s)):
        assert x.dtype == y.dtype and torch.allclose(x.float(), y.float(), atol=1e-5)


def per_decision(net, obs, labels, s0, b: int, t: int):
    cs, os_ = s0
    hs, es, sps, scs, ins = [], [], [], [], []
    for k in range(t):
        core_in, e, sp, sc, _ = net.encode(rows_at(obs, b, t, k), h_o=os_[0])
        h, cs = net.core.step(core_in, cs)
        os_ = net.outer.step(h, os_)
        hs.append(net._skip(h, core_in))
        es.append(e)
        sps.append(sp)
        scs.append(sc)
    rows = lambda xs: torch.stack(xs, 1).reshape(b * t, *xs[0].shape[1:])  # noqa: E731
    sp = {k: rows([x[k] for x in sps]) for k in sps[0]}
    sc = {k: rows([x[k] for x in scs]) for k in scs[0]}
    h, e = rows(hs), rows(es)
    logp = net.heads(labels, 1.0, h, sc["c"], e, sp, obs, net.table, u=net._bypass(h, sc, e, obs))["logp"]
    return logp, net.value(h, sc["z"], sc["race"], None), (cs, os_)


@pytest.mark.parametrize("name", ["patch_outer", "patch_all"])
@pytest.mark.parametrize("counts", [(0, 0), (1, 2), (5, 1)])
def test_patch_phased_matches_per_decision_032_ac19(name, counts):
    cfg, b, t, obs, labels, _ = patch_tiny(PATCH_VARIANTS[name])
    net = net_of(cfg)
    s0 = state_after(net, obs, b, t, counts)
    seq = net.forward_seq(obs, labels, None, s0, t, phased=True)
    logp, value, state = per_decision(net, obs, labels, s0, b, t)
    for k in HEADS:
        assert torch.allclose(seq["logp"][k], logp[k], atol=1e-4), k
    assert torch.allclose(seq["value"], value, atol=1e-5)
    for x, y in zip(leaves(seq["state"]), leaves(state)):
        assert torch.allclose(x.float(), y.float(), atol=1e-5)
    params = [net.spatial.proj.weight, net.heads.location.out.weight, net.variant.outer.film.weight]
    loss = lambda lp, v: sum(lp[k].sum() for k in HEADS) + v.sum()  # noqa: E731
    ga = torch.autograd.grad(loss(seq["logp"], seq["value"]), params)
    gb = torch.autograd.grad(loss(logp, value), params)
    for x, y in zip(ga, gb):
        assert torch.allclose(x, y, atol=1e-4, rtol=1e-4)
        assert x.abs().sum() > 0
    if counts == (0, 0):
        with torch.no_grad():
            bc = net.forward_seq(obs, labels, None, s0, t)
        for k in HEADS:
            assert torch.allclose(bc["logp"][k], seq["logp"][k].detach(), atol=1e-5), k


@pytest.mark.parametrize("name", ["patch", "patch_outer"])
def test_patch_jepa_target_is_e_pool_032_ac19(name):
    v = replace(PATCH_VARIANTS[name], aux_priv=True, aux_jepa=[1, 2])
    cfg, b, t, obs, labels, priv = patch_tiny(v)
    net = net_of(cfg).train()
    assert all(m.l2.out_features == cfg.entity.d_model for m in net.variant.aux_jepa.values())
    out = net.forward_seq(obs, labels, priv, net.init_state(b), t, aux=True)
    y = out["jepa_target"]
    assert y.shape == (b * t, cfg.entity.d_model) and not y.requires_grad
    with torch.no_grad():
        s = net.init_state(b)
        ys = []
        for k in range(t):
            o = rows_at(obs, b, t, k)
            _, _, _, _, e_pool = net.encode(o, h_o=None if net.outer is None else s[1][0])
            ys.append(e_pool)
            s = net.act(o, s, 0.0)["state"]
    assert torch.allclose(y, torch.stack(ys, 1).reshape(b * t, -1), atol=1e-5)


@pytest.mark.parametrize("name", PATCH_VARIANTS)
def test_patch_masks_privileged_and_recompute_032_ac19(name):
    cfg, b, t, obs, labels, priv = patch_tiny(PATCH_VARIANTS[name])
    net = net_of(cfg)
    g = torch.Generator().manual_seed(5)
    o = rows_at(obs, b, t, 0)
    s = net.init_state(b)
    with torch.no_grad():
        for _ in range(10):
            out = net.act(o, s, 1.0, g)
            assert o["type_mask"][torch.arange(b), out["action"]["action"]].all()
            s = out["state"]
    pr = priv.clone().requires_grad_(True)
    out = net.forward_seq(obs, labels, pr, net.init_state(b), t)
    (gp,) = torch.autograd.grad(sum(v.sum() for v in out["logp"].values()), pr, allow_unused=True)
    assert gp is None or not gp.any()
    grads = []
    for chunk in (0, 3):
        n = net_of(cfg).train()
        out = n.forward_seq(obs, labels, priv, n.init_state(b), t, chunk)
        (sum(out["logp"][k].sum() for k in HEADS) + out["value"].sum()).backward()
        grads.append({k: p.grad.clone() for k, p in n.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys() and "spatial.proj.weight" in grads[0]
    for k, x in grads[0].items():
        assert torch.allclose(x, grads[1][k], atol=1e-5, rtol=1e-4), k


@pytest.mark.parametrize("name", PATCH_VARIANTS)
def test_patch_checkpoint_round_trip_032_ac19(name, tmp_path):
    cfg, b, t, obs, labels, _ = patch_tiny(PATCH_VARIANTS[name])
    net = net_of(cfg)
    ckpt.save_model(tmp_path, net)
    ckpt.save_meta(tmp_path, net.cfg, {}, "f32")
    meta = json.loads((tmp_path / ckpt.META_FILE).read_text(encoding="utf-8"))
    assert meta["config"]["variant"]["spatial_patch"] == P and meta["config"]["spatial"]["patch_dim"] == D_P
    back, _ = ckpt.load_model(tmp_path)
    assert back.cfg == net.cfg and back.patch == P
    with torch.no_grad():
        x = net.eval().forward_seq(obs, labels, None, net.init_state(b), t)
        y = back.eval().forward_seq(obs, labels, None, back.init_state(b), t)
    assert torch.equal(x["value"], y["value"]) and torch.equal(x["dists"]["target_loc"], y["dists"]["target_loc"])


def test_patch_off_meta_has_no_patch_keys_032_ac19(tmp_path):
    cfg, *_ = tiny(VARIANTS["all"])
    net = net_of(cfg)
    ckpt.save_meta(tmp_path, net.cfg, {}, "f32")
    meta = json.loads((tmp_path / ckpt.META_FILE).read_text(encoding="utf-8"))["config"]
    assert "patch_dim" not in meta["spatial"] and "spatial_patch" not in meta["variant"]


def test_patch_monitor_032_r25_r29():
    cfg, b, t, obs, labels, _ = patch_tiny()
    net = net_of(cfg)
    with pytest.raises(ValueError, match="032-R25"):
        ReluMonitor(net, points="spatial")
    st = relu_stats(net, obs, labels, None, net.init_state(b), t, points="all")
    assert st and not any(k.startswith("spatial") for k in st)
    assert net.conv_layout(True, True) is net


def test_bc_refuses_size_mismatch_032_ac18(tiny_config, shards, tmp_path):
    cfg = bc_cfg(tiny_config)
    a = BcTrainer(cfg, make_net(VariantConfig()), CPU, "f32")
    a.train_batch(feeder(tiny_config, shards).next(), 3e-3, 0)
    a.save(str(tmp_path / "ck"), json.dumps({"step": 1}))
    sizes = net_sizes_of(a.net.cfg)
    cfg_json = json.dumps(cfg)
    for start in ("resume", "init"):
        t = BcTrainer.make(cfg_json, start, str(tmp_path / "ck"), 0, "lstm", CPU, "f32", None, sizes)
        assert size_mismatch(t.net.cfg, a.net.cfg) == []
        for bad in (None, {**sizes, "core": {**sizes["core"], "hidden": 48}}, {**sizes, "spatial": {**sizes["spatial"], "stages": [8, 8, 16, 32]}}):
            with pytest.raises(ValueError, match="032-R28"):
                BcTrainer.make(cfg_json, start, str(tmp_path / "ck"), 0, "lstm", CPU, "f32", None, bad)
        with pytest.raises(ValueError, match=r"\[model\]"):
            BcTrainer.make(cfg_json, start, str(tmp_path / "ck"), 0, "lstm", CPU, "f32", PATCH, net_sizes_of(replace(a.net.cfg, variant=PATCH)))
    drop = {**sizes, "entity": {**sizes["entity"], "dropout": 0.5}}
    with pytest.raises(ValueError, match="entity.dropout"):
        BcTrainer.make(cfg_json, "resume", str(tmp_path / "ck"), 0, "lstm", CPU, "f32", None, drop)


def test_bc_new_run_uses_net_sizes_032_ac18(tiny_config, shards, tmp_path):
    cfg = bc_cfg(tiny_config) | {"train": {"model_seed": 1}}
    cfg_json = json.dumps(cfg)
    sizes = net_sizes_from_table({"entity": {"d_model": 48, "heads": 4, "ffn": 96, "layers": 1}, "core": {"hidden": 40, "gtrxl_heads": 4, "gtrxl_memory": 8},
                                  "scalar": {"hidden": 40}, "heads": {"ar": 40}, "value": {"hidden": 40}, "spatial": {"patch_dim": 32}})
    v = replace(VARIANTS["outer"], outer=cfg["seq_len"] // 2, spatial_patch=16)
    t = BcTrainer.make(cfg_json, "new", None, 0, "gtrxl", CPU, "f32", v, sizes)
    c = t.net.cfg
    assert (c.entity.d_model, c.core.hidden, c.scalar.hidden, c.heads.ar, c.value.hidden, c.spatial.patch_dim, c.patch()) == (48, 40, 40, 40, 40, 32, 16)
    assert t.net.core.input.in_features == 48 + 40
    out = json.loads(t.train_batch(feeder(tiny_config, shards).next(), 3e-3, 0))
    assert out["total"] is not None
    t.save(str(tmp_path / "run"), json.dumps({"step": 1}))
    r = BcTrainer.make(cfg_json, "resume", str(tmp_path / "run"), 0, "gtrxl", CPU, "f32", v, sizes)
    pa, pr = params_of(t.net), params_of(r.net)
    assert pa.keys() == pr.keys() and all(torch.equal(pa[k], pr[k]) for k in pa)
    with pytest.raises(ValueError, match="032-R28"):
        BcTrainer.make(cfg_json, "resume", str(tmp_path / "run"), 0, "gtrxl", CPU, "f32", v, None)
    with pytest.raises(ValueError, match=r"\[model\]"):
        BcTrainer.make(cfg_json, "init", str(tmp_path / "run"), 0, "gtrxl", CPU, "f32", replace(v, spatial_patch=8), sizes)
    with pytest.raises(ValueError, match="spatial_every"):
        BcTrainer.make(json.dumps(cfg | {"monitor": {"spatial_every": 10}}), "new", None, 0, "gtrxl", CPU, "f32", v, sizes)
    aux = BcTrainer.make(json.dumps(cfg | {"aux": {"priv": 0.2, "jepa": 0.1}}), "new", None, 0, "gtrxl", CPU, "f32", replace(v, aux_priv=True, aux_jepa=[1, 2]), sizes)
    out = json.loads(aux.train_batch(feeder(tiny_config, shards).next(), 3e-3, 0))
    assert out["aux_jepa"] is not None and out["jepa_tgt_std"] is not None


def test_b_configuration_size_045_r1():
    v = VariantConfig(terrain=True, ctx_token=True, attn_pool=True, outer=32, bypass=True, core_skip=True, aux_priv=True, aux_jepa=[1, 8, 32], spatial_patch=8)
    cfg = new_net_config("gtrxl", True, v, net_sizes_from_table(B_NET))
    with torch.device("meta"):
        net = UraniumNet(cfg, action_columns())
    n = sum(p.numel() for p in net.parameters())
    assert 135_000_000 < n < 142_000_000
    assert net.spatial.proj.in_features == 58 * 64 and net.spatial.pos.shape == (256, 512) and net.heads.location.out.out_features == 64


@pytest.mark.gpu
@torch.no_grad()
def test_graphed_act_matches_eager_patch_032_ac19():
    from uranium.agent.infer import GraphedAct
    from uranium.bench.inputs import random_obs
    from uranium.device import autocast, resolve

    if not torch.cuda.is_available():
        pytest.skip("no CUDA / ROCm device")
    _, dev, prec = resolve(None)
    torch.manual_seed(0)
    net = UraniumNet(new_net_config("gtrxl", True, replace(VariantConfig(True, True, True, 32), spatial_patch=8)), action_columns()).to(dev).eval()
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
