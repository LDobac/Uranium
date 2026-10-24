"""032 §2.7 / 032-AC11: the architecture-comparison variants (`[model]` switches). Every switch and
all of them together: shared initial weights equal to the default model, step = sequence (outer
segment boundaries included), masks, no privileged path, entity recompute, checkpoint round trip,
config handling. Inputs are the tiny Burn-parity batch (B 2, T 4); the outer segment is 2 there."""

import json
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from uranium import ckpt
from uranium.bc.run import model_variant
from uranium.bc.trainer import BcTrainer
from uranium.model import UraniumNet, UraniumNetConfig, action_columns, to_labels, to_obs
from uranium.model.config import VariantConfig

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "burn_parity" / "tiny-gtrxl"
HEADS = ("action", "delay", "queued", "selected", "target_unit", "target_loc")
SEG = 2
VARIANTS = {
    "terrain": VariantConfig(terrain=True),
    "ctx": VariantConfig(ctx_token=True),
    "pool": VariantConfig(attn_pool=True),
    "outer": VariantConfig(outer=SEG),
    "all": VariantConfig(terrain=True, ctx_token=True, attn_pool=True, outer=SEG),
}


def tiny(variant: VariantConfig | None = None):
    meta = json.loads((FIX / "meta.json").read_text())
    inp = load_file(FIX / "inputs.safetensors")
    cfg = UraniumNetConfig.from_dict(meta["config"])
    if variant is not None:
        cfg.variant = variant
    return cfg, meta["b"], meta["t"], to_obs(inp, "cpu"), to_labels(inp, "cpu"), inp["privileged"]


def net_of(cfg, seed=3):
    torch.manual_seed(seed)
    return UraniumNet(cfg, action_columns()).eval()


def rows_at(obs: dict, b: int, t: int, k: int) -> dict:
    """Step k of every sequence (rows are b-major)."""
    return {key: x.view(b, t, *x.shape[1:])[:, k] for key, x in obs.items()}


def test_config_table_032_ac11():
    assert not VariantConfig.from_table(None).any()
    assert VariantConfig.from_table({"outer": 32, "terrain": True}) == VariantConfig(terrain=True, outer=32)
    for bad in ({"terain": True}, {"terrain": 1}, {"outer": True}, {"outer": -1}, {"outer": 3.0}):
        with pytest.raises(ValueError):
            VariantConfig.from_table(bad)
    # The default model's config keeps its old form (no "variant" in the checkpoint meta).
    assert "variant" not in UraniumNetConfig().to_dict()
    on = UraniumNetConfig(variant=VariantConfig(attn_pool=True))
    assert UraniumNetConfig.from_dict(on.to_dict()).variant == on.variant
    assert UraniumNetConfig.from_dict(UraniumNetConfig().to_dict()).variant == VariantConfig()


def test_comparison_configs_032_r20():
    """config/bc_arch_abl_<v>_pt.toml = bc_core_ablation_pt.toml + [model] (032-R20 ②)."""
    base = tomllib.loads((ROOT / "config" / "bc_core_ablation_pt.toml").read_text(encoding="utf-8"))
    want = {"terrain": VariantConfig(terrain=True), "ctx": VariantConfig(ctx_token=True), "pool": VariantConfig(attn_pool=True), "outer": VariantConfig(outer=32), "all": VariantConfig(True, True, True, 32)}
    for name, v in want.items():
        path = ROOT / "config" / f"bc_arch_abl_{name}_pt.toml"
        cfg = tomllib.loads(path.read_text(encoding="utf-8"))
        assert cfg.pop("model") is not None and cfg == base, name
        assert model_variant(str(path)) == v, name
    assert not model_variant(str(ROOT / "config" / "bc_core_ablation_pt.toml")).any()


@pytest.mark.parametrize("name", VARIANTS)
def test_shared_initial_weights_032_ac11(name):
    """A variant adds its own modules after the default ones and leaves the global RNG where it
    was, so every shared parameter starts as in the default model (032 §2.7)."""
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
    """`act` one decision at a time gives the training pass's values and final state (032-R5),
    across the outer segment boundary."""
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
    from uranium.agent.infer import leaves

    for x, y in zip(leaves(seq["state"]), leaves(s)):
        assert x.dtype == y.dtype and torch.allclose(x.float(), y.float(), atol=1e-5)


def test_outer_state_updates_on_segments():
    """h_o changes only after every SEG decisions and the counter counts decisions."""
    cfg, b, t, obs, _, _ = tiny(VARIANTS["outer"])
    net = net_of(cfg)
    s = net.init_state(b)
    seen = []
    with torch.no_grad():
        for k in range(t):
            s = net.act(rows_at(obs, b, t, k), s, 0.0)["state"]
            seen.append(s[1][0].clone())
    assert torch.equal(s[1][3], torch.full((b,), float(t)))
    assert not seen[0].any() and seen[1].any()  # updated at decision 2
    assert torch.equal(seen[1], seen[2]) and not torch.equal(seen[2], seen[3])
    kept = net.mask_state(s, torch.tensor([True, False]))
    assert kept[1][0][0].any() and not kept[1][0][1].any() and kept[1][3].tolist() == [float(t), 0.0]


@pytest.mark.parametrize("name", ["ctx", "outer", "all"])
def test_tokens_stay_out_of_entities(name):
    """The extra tokens are dropped from E: same entity count, and the heads' pointers never pick
    them (selection and targets index entities only)."""
    cfg, b, t, obs, _, _ = tiny(VARIANTS[name])
    net = net_of(cfg)
    n = obs["entity_mask"].shape[1]
    with torch.no_grad():
        _, e, _, _ = net.encode(rows_at(obs, b, t, 0), h_o=torch.zeros(b, 256) if net.outer is not None else None)
        assert e.shape[1] == n
        out = net.act(rows_at(obs, b, t, 0), net.init_state(b), 1.0, torch.Generator().manual_seed(1))
    assert (out["action"]["selected"] <= n).all() and (out["action"]["target_unit"] < n).all()


def test_attn_pool_empty_set():
    """No enemy seen: the enemy queries give zeros, the pool stays finite."""
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


@pytest.mark.parametrize("name", VARIANTS)
def test_privileged_only_reaches_value_032_r11(name):
    cfg, b, t, obs, labels, priv = tiny(VARIANTS[name])
    net = net_of(cfg)
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


def test_rl_refuses_outer():
    from uranium.rl.learner import Learner

    cfg, *_ = tiny(VARIANTS["outer"])
    net = net_of(cfg)
    with pytest.raises(ValueError, match="outer"):
        Learner({}, net, net, torch.device("cpu"), "f32", "vtrace")


# --- GPU (pytest -m gpu): the fused attention kernel with the extra tokens' bias buckets and the
# inference graph of a variant model (032-AC11, run on the CUDA host before the league games).


@pytest.mark.gpu
def test_fused_entity_attention_with_tokens(monkeypatch):
    from test_kernels import kernel_or_skip, rel

    from uranium.bc.trainer import new_net_config
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
        _, e, _, _ = net.encode(obs, h_o=h_o)
        e.square().sum().backward()
        res[mode] = (e.detach(), net.variant.ctx_bias.grad.clone(), net.variant.outer.tok_bias.grad.clone())
    for a, b in zip(res["fused"], res["torch"]):
        assert rel(a, b) < 1e-3


@pytest.mark.gpu
@torch.no_grad()
def test_graphed_act_matches_eager_variant():
    from uranium.agent.infer import GraphedAct, leaves
    from uranium.bc.trainer import new_net_config
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
    s_eager = s_graph = net.init_state(3, dev)
    for _ in range(40):  # past an outer update (every 32 decisions)
        with autocast(dev, prec):
            want = net.act(obs, s_eager, 0.0)
        got = graphed(obs, s_graph, 0.0)
        for k in want["action"]:
            assert torch.equal(got["action"][k], want["action"][k]), k
        assert close(got["value"], want["value"])
        s_eager, s_graph = want["state"], got["state"]
    for x, y in zip(leaves(s_graph), leaves(s_eager)):
        assert close(x, y)
