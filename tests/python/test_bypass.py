"""032 §2.8 `bypass` (032-R21, AC13): the ActionType head's cross-attention over the entity tokens
with the raw scalars `s_raw`. The s_raw columns against the feature schema; the zero-initialised
output (the model starts as the one without it, also a §2.8-only model against the default one);
gradients; the empty entity set; step = sequence (outer segments, RL phased); entity recompute;
checkpoint round trip; and on a GPU the inference graph. The tiny Burn-parity batch has a 10-wide
scalar, so these tests widen it to the 030 §4 layout (`tiny_b`)."""

import json
import re
from pathlib import Path

import pytest
import torch

from uranium import ckpt
from uranium.model import UraniumNetConfig
from uranium.model.config import VariantConfig, input_dims
from uranium.model.variants import BYPASS_DIM, S_RAW, SCALAR_BLOCKS, SCALAR_COLUMNS, SEED_BYPASS, scalar_layout, s_raw_slices

from test_variants import HEADS, SEG, net_of, rows_at, tiny

ROOT = Path(__file__).resolve().parents[2]
EXTRACT = ROOT / "crates" / "uranium-env" / "src" / "features" / "extract.rs"
SCHEMA = ROOT / "crates" / "uranium-env" / "src" / "features" / "schema.rs"

BYPASS = VariantConfig(bypass=True)
# §2.7 variants the bypass is combined with (032-AC13: each switch and with the §2.7 four on).
WITH = {
    "only": VariantConfig(),
    "pool": VariantConfig(attn_pool=True),
    "outer": VariantConfig(outer=SEG),
    "all": VariantConfig(terrain=True, ctx_token=True, attn_pool=True, outer=SEG),
}
SEEN = 6  # enemy max-seen columns of the widened tiny scalar


def on(v: VariantConfig) -> VariantConfig:
    return VariantConfig(**{**v.__dict__, "bypass": True})


def tiny_b(variant: VariantConfig | None = None):
    """The tiny batch with its scalar widened to the 030 §4 layout (SEEN enemy columns): the
    fixture's 10 columns first, then fixed pseudo-random values in [0, 1)."""
    cfg, b, t, obs, labels, priv = tiny(variant)
    fixed = sum(w for _, w in SCALAR_BLOCKS) + len(SCALAR_COLUMNS) + 2 * cfg.input.z_units
    width = fixed + SEEN + cfg.input.actions + cfg.heads.delay_buckets + 1
    s = obs["scalar"]
    extra = torch.rand(s.shape[0], width - s.shape[1], generator=torch.Generator().manual_seed(21))
    obs = dict(obs, scalar=torch.cat([s, extra.to(s.dtype)], 1))
    cfg.input.scalar = width
    return cfg, b, t, obs, labels, priv


def perturb(net, scale: float = 0.05):
    """Non-zero output projection (as after training), so the bypass changes the outputs."""
    g = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for p in net.variant.bypass.out.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale)
    return net


# --- s_raw columns (032-R21 ④, AMB-032-12; supply total / max left out, main interim decision).


def test_scalar_layout_matches_the_schema_032_r21():
    """The layout's widths sum to the Rust schema's scalar width, its enemy max-seen block is the
    one the 013 type tables give (diag tables: one column per observed type) and the last-action
    block is A + delay buckets + 1."""
    from uranium import _core
    from uranium.diag import tables as T

    d = json.loads(_core.feature_dims())
    cfg = UraniumNetConfig(input=input_dims())
    assert cfg.heads.delay_buckets == len(d["delay_buckets"])
    lay = scalar_layout(cfg)
    names = list(lay)
    assert names[:len(SCALAR_COLUMNS)] == list(SCALAR_COLUMNS)
    assert [lay[n].start for n in SCALAR_COLUMNS] == list(range(len(SCALAR_COLUMNS)))
    assert list(lay.values())[-1].stop == d["scalar"]
    assert all(a.stop == b.start for a, b in zip(list(lay.values()), list(lay.values())[1:])), "contiguous"
    assert lay["enemy_seen"] == T.sc_enemy_cols()
    assert lay["enemy_seen"].stop - lay["enemy_seen"].start == len(T.observed())
    assert lay["own"].start == T.sc_own_cols(T.terran()[0])[0] and lay["own"].stop - lay["own"].start == 2 * d["z_units"]
    assert lay["last_action"].stop - lay["last_action"].start == d["actions"] + len(d["delay_buckets"]) + 1
    # s_raw: frame, log frame, minerals, gas, supply used (columns 0..5), then the enemy block.
    assert S_RAW == ("frame", "log_frame", "minerals", "gas", "supply_used", "enemy_seen")
    assert s_raw_slices(cfg) == [slice(0, 5), T.sc_enemy_cols()]


def test_scalar_layout_matches_the_extractor_032_r21():
    """The names follow the extractor's push order (`fn scalar`, extract.rs) and the block widths
    the schema's `FeatureDims::scalar` formula."""
    src = EXTRACT.read_text(encoding="utf-8")
    body = src[src.index("fn scalar(w: &World)"):]
    body = body[:re.search(r"\n\}\r?\n", body).start()]
    pushes = re.findall(r"v\.push\((.*)\);", body)[:len(SCALAR_COLUMNS)]
    want = {"frame": "86400", "log_frame": "/ 12.0", "minerals": "me.minerals", "gas": "me.gas", "supply_used": "me.supply_used", "supply_total": "me.supply_total", "supply_max": ".min(400.0)"}
    for name, expr in zip(SCALAR_COLUMNS, pushes):
        assert want[name] in expr, (name, expr)
    loops = [int(n) for n in re.findall(r"for i in 0\.\.(\d+)", body)]
    assert loops == [w for _, w in SCALAR_BLOCKS]
    order = [body.index(k) for k in ("for i in 0..63", "&d.terran_types", "&d.observed_types", "w.mem.last_action")]
    assert order == sorted(order), "own counts, then enemy max-seen, then the previous action"
    schema = SCHEMA.read_text(encoding="utf-8")
    assert "2 + 5 + 63 + 47 + 47 + 63 + 2 * self.terran_types.len() + self.observed_types.len() + self.actions + DELAY_BUCKETS.len() + 1" in schema


def test_scalar_layout_refuses_a_narrow_scalar():
    cfg, *_ = tiny(BYPASS)
    with pytest.raises(ValueError, match="030"):
        scalar_layout(cfg)
    with pytest.raises(ValueError, match="030"):
        net_of(cfg)


def test_bypass_module_shapes_and_seeds_032_r21():
    """d_b 256, 8 x 32; q [h, c, s_raw] -> 256; u [attention output, s_raw] -> ar, zero; seeds from
    SEED_BYPASS (+0 query, +1 K, +2 V) whatever the global RNG."""
    cfg, *_ = tiny_b(BYPASS)
    a, b = net_of(cfg, seed=3), net_of(cfg, seed=9)
    m = a.variant.bypass
    n_s = 5 + SEEN
    assert m.query.weight.shape == (BYPASS_DIM, cfg.core.hidden + cfg.scalar.hidden + n_s)
    assert m.k.weight.shape == m.v.weight.shape == (BYPASS_DIM, cfg.entity.d_model)
    assert m.out.weight.shape == (cfg.heads.ar, BYPASS_DIM + n_s)
    assert m.heads == 8 and BYPASS_DIM // m.heads == 32
    assert not m.out.weight.any() and not m.out.bias.any()
    for k, v in a.variant.bypass.state_dict().items():
        assert torch.equal(v, b.variant.bypass.state_dict()[k]), k
    g = torch.Generator().manual_seed(SEED_BYPASS + 1)
    k = 1.0 / cfg.entity.d_model ** 0.5
    assert torch.equal(m.k.weight, torch.empty(BYPASS_DIM, cfg.entity.d_model).uniform_(-k, k, generator=g))


# --- Zero start (032-R21 ②, 032-AC13 ① ②).


@pytest.mark.parametrize("name", WITH)
def test_bypass_starts_as_the_model_without_it_032_ac13(name):
    """Turned on, the model's shared weights and every output (BC sequence, RL phased, act) are
    those of the model without it: for `only` that is the default model (no variant module at
    all), the §2.8-only check the U3 review asked for."""
    cfg, b, t, obs, labels, priv = tiny_b(WITH[name])
    ref = net_of(cfg).eval()
    cfg_on, *_ = tiny_b(on(WITH[name]))
    net = net_of(cfg_on).eval()
    if name == "only":
        assert ref.variant is None
    sd, sd_ref = net.state_dict(), ref.state_dict()
    assert set(sd) - set(sd_ref) == {k for k in sd if k.startswith("variant.bypass.")}
    for k, v in sd_ref.items():
        assert torch.equal(v, sd[k]), k
    with torch.no_grad():
        for kw in ({}, {"phased": True}) if ref.outer is not None else ({},):
            x = ref.forward_seq(obs, labels, priv, ref.init_state(b), t, **kw)
            y = net.forward_seq(obs, labels, priv, net.init_state(b), t, **kw)
            for k in HEADS:
                assert torch.equal(x["logp"][k], y["logp"][k]), (kw, k)
                assert torch.equal(x["dists"][k], y["dists"][k]), (kw, k)
            assert torch.equal(x["value"], y["value"])
        s_a, s_b = ref.init_state(b), net.init_state(b)
        for k in range(t):
            o = rows_at(obs, b, t, k)
            p, q = ref.act(o, s_a, 0.0), net.act(o, s_b, 0.0)
            for h in HEADS:
                assert torch.equal(p["action"][h], q["action"][h]), (k, h)
            assert torch.equal(p["log_prob"], q["log_prob"]) and torch.equal(p["value"], q["value"])
            s_a, s_b = p["state"], q["state"]


# --- Gradients, empty set.


def test_gradient_reaches_the_bypass_032_r21():
    """At the start only the zero output projection gets a gradient (u's slope in the attention
    is its weight, 0); once it is not zero, the query, K and V get one too."""
    cfg, b, t, obs, labels, priv = tiny_b(BYPASS)

    def grads(net):
        net.train().zero_grad()
        out = net.forward_seq(obs, labels, priv, net.init_state(b), t)
        sum(out["logp"][k].sum() for k in HEADS).backward()
        return {k: p.grad for k, p in net.variant.bypass.named_parameters()}

    g = grads(net_of(cfg))
    assert g["out.weight"].abs().sum() > 0 and g["out.bias"].abs().sum() > 0
    for k in ("query.weight", "query.bias", "k.weight", "v.weight", "v.bias"):
        assert g[k] is not None and not g[k].any(), k
    g = grads(perturb(net_of(cfg)))
    for k, v in g.items():
        assert v is not None and v.abs().sum() > 0 and torch.isfinite(v).all(), k


def test_empty_entity_set_gives_zero_attention_032_r21():
    """A row with no real entity: attention output exactly 0, u finite (= the projection of
    [0, s_raw]), gradients finite; the other rows are unchanged by it."""
    cfg, *_ = tiny_b(BYPASS)
    m = perturb(net_of(cfg)).variant.bypass.train()
    g = torch.Generator().manual_seed(5)
    b, n = 3, 7
    h = torch.randn(b, cfg.core.hidden, generator=g, requires_grad=True)
    c = torch.randn(b, cfg.scalar.hidden, generator=g)
    e = (torch.randn(b, n, cfg.entity.d_model, generator=g) * 50).requires_grad_()
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[0] = False
    mask[2, 3:] = False
    obs = {"scalar": torch.rand(b, cfg.input.scalar, generator=g), "entity_mask": mask}
    s = m.s_raw(obs["scalar"])
    att = m.attend(m.query(torch.cat([h, c, s], 1)), e, mask)
    assert torch.equal(att[0], torch.zeros(BYPASS_DIM)) and att[1:].abs().sum() > 0
    u = m(h, c, e, obs)
    assert torch.isfinite(u).all()
    assert torch.allclose(u[0], m.out(torch.cat([torch.zeros(BYPASS_DIM), s[0]])), atol=1e-6)
    u.square().sum().backward()
    assert torch.isfinite(h.grad).all() and torch.isfinite(e.grad).all()
    assert not e.grad[0].any(), "no gradient into the masked tokens"
    # Padding rows of row 2 do not change its output.
    e2 = e.detach().clone()
    e2[2, 3:] = 1e3
    assert torch.allclose(m(h, c, e2, obs), u, atol=1e-5)


# --- step = sequence (032 §2.8, 032-AC13 ③), entity recompute, checkpoint.


@pytest.mark.parametrize("name", WITH)
def test_bypass_step_matches_sequence_032_ac13(name):
    """With a trained-like (non-zero) bypass: u of each decision in `act` is the u of that row in
    the training pass (and in the RL phased pass), the ActionType distribution agrees and the
    state is the same."""
    from uranium.agent.infer import leaves

    cfg, b, t, obs, labels, _ = tiny_b(on(WITH[name]))
    net = perturb(net_of(cfg))
    seen = []
    net.variant.bypass.register_forward_hook(lambda _m, _i, out: seen.append(out.detach().clone()))
    with torch.no_grad():
        seq = net.forward_seq(obs, labels, None, net.init_state(b), t)
        u_seq = seen.pop()
        if net.outer is not None:
            ph = net.forward_seq(obs, labels, None, net.init_state(b), t, phased=True)
            assert torch.allclose(seen.pop(), u_seq, atol=1e-5)
            assert torch.allclose(ph["dists"]["action"], seq["dists"]["action"], atol=1e-4)
        assert u_seq.abs().sum() > 0
        s = net.init_state(b)
        lp = seq["dists"]["action"].view(b, t, -1)
        for k in range(t):
            out = net.act(rows_at(obs, b, t, k), s, 0.0)
            assert torch.allclose(seen.pop(), u_seq.view(b, t, -1)[:, k], atol=1e-5), k
            want = lp[:, k].argmax(1)
            assert torch.equal(out["action"]["action"], want), k
            assert torch.allclose(out["head_log_prob"][:, 0], lp[:, k].gather(1, want[:, None]).squeeze(1), atol=1e-4), k
            s = out["state"]
    for x, y in zip(leaves(seq["state"]), leaves(s)):
        assert torch.allclose(x.float(), y.float(), atol=1e-5)


@pytest.mark.parametrize("name", ["only", "all"])
def test_entity_recompute_keeps_the_bypass_update_041_r17(name):
    cfg, b, t, obs, labels, priv = tiny_b(on(WITH[name]))
    grads = []
    for chunk in (0, 3):
        net = perturb(net_of(cfg)).train()
        out = net.forward_seq(obs, labels, priv, net.init_state(b), t, chunk)
        (sum(out["logp"][k].sum() for k in HEADS) + out["value"].sum()).backward()
        grads.append({k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys() and any(k.startswith("variant.bypass.") for k in grads[0])
    for k, g in grads[0].items():
        assert torch.allclose(g, grads[1][k], atol=1e-5, rtol=1e-4), k


@pytest.mark.parametrize("name", ["only", "all"])
def test_bypass_checkpoint_round_trip_032_ac13(name, tmp_path):
    cfg, b, t, obs, labels, _ = tiny_b(on(WITH[name]))
    net = perturb(net_of(cfg))
    ckpt.save_model(tmp_path, net)
    ckpt.save_meta(tmp_path, net.cfg, {}, "f32")
    back, meta = ckpt.load_model(tmp_path)
    assert back.cfg.variant == cfg.variant and meta["config"]["variant"]["bypass"] is True
    assert meta["config"]["input"]["scalar"] == cfg.input.scalar
    for k, v in net.state_dict().items():
        assert torch.equal(v, back.state_dict()[k]), k
    with torch.no_grad():
        x = net.eval().forward_seq(obs, labels, None, net.init_state(b), t)
        y = back.eval().forward_seq(obs, labels, None, back.init_state(b), t)
    for k in HEADS:
        assert torch.equal(x["logp"][k], y["logp"][k]), k


# --- GPU (pytest -m gpu): the inference graph with the bypass (032-AC13: act = eager, no recompile).


@pytest.mark.gpu
@torch.no_grad()
def test_graphed_act_matches_eager_bypass():
    from uranium.agent.infer import GraphedAct
    from uranium.bc.trainer import new_net_config
    from uranium.bench.inputs import random_obs
    from uranium.device import autocast, resolve

    if not torch.cuda.is_available():
        pytest.skip("no CUDA / ROCm device")
    _, dev, prec = resolve(None)
    torch.manual_seed(0)
    net = perturb(net_of_full(new_net_config("gtrxl", True, VariantConfig(True, True, True, 32, bypass=True)))).to(dev).eval()
    graphed = GraphedAct(net.act, dev, prec)
    exact = torch.version.hip is None
    close = (lambda a, b: torch.equal(a, b)) if exact else (lambda a, b: torch.allclose(a.float(), b.float(), atol=1e-5, rtol=1e-5))
    obs = random_obs(4, 128, net.cfg.input, dev, torch.Generator().manual_seed(1), valid=119)
    obs["entity_mask"][1] = False  # an empty entity set inside the captured graph
    s_eager, s_graph = net.init_state(4, dev), net.init_state(4, dev)
    for _ in range(3):
        with autocast(dev, prec):
            want = net.act(obs, s_eager, 0.0)
        got = graphed(obs, s_graph, 0.0)
        assert len(graphed.graphs) == 1
        for k in want["action"]:
            assert torch.equal(got["action"][k], want["action"][k]), k
        assert close(got["log_prob"], want["log_prob"]) and torch.isfinite(got["log_prob"]).all()
        s_eager, s_graph = want["state"], got["state"]


def net_of_full(cfg):
    from uranium.model import UraniumNet, action_columns

    return UraniumNet(cfg, action_columns())
