"""032-R23 F1 `core_skip` (032 §2.8, 032-AC13): h' = h + Linear(core input), the Linear zero at
init. Heads, value and win_open read h'; the core's recurrent state and the outer LSTM's h_sum keep
the pre-skip h. Checked on the GTrXL and the LSTM core, alone and with the four §2.7 switches on
(the tiny Burn-parity batches, B 2, T 4, outer segment 2)."""

import copy
from dataclasses import replace

import pytest
import torch

from uranium import ckpt
from uranium.agent.infer import leaves
from uranium.model import UraniumNet, action_columns
from uranium.model.config import VariantConfig
from uranium.model.variants import core_skip_linear

from test_variants import FIX, HEADS, VARIANTS, net_of, rows_at, state_after, tiny

CORES = ["tiny-gtrxl", "tiny-lstm"]
# Each case: (the model without the skip, the same model with it).
OFF = {"skip": VariantConfig(), "skip_all": VARIANTS["all"]}
ON = {"skip": VariantConfig(core_skip=True), "skip_all": replace(VARIANTS["all"], core_skip=True)}
SKIP_KEYS = {"variant.core_skip.weight", "variant.core_skip.bias"}


def setup(core: str, arch: str):
    cfg, b, t, obs, labels, priv = tiny(ON[arch], FIX.parent / core)
    return cfg, b, t, obs, labels, priv


def trained_skip(net, seed: int = 11):
    """The skip as after some training: non-zero weight and bias (a fixed generator)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.variant.core_skip.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return net


def same(a, b, eq=torch.equal) -> bool:
    """Equal tensors, or both None (a model without the win_open head, 032-R18)."""
    return a is None and b is None if a is None or b is None else eq(a, b)


def paths(net):
    """The training passes of this model: BC, and the RL phased pass when it has `outer`."""
    return ({}, {"phased": True}) if net.outer is not None else ({},)


def test_core_skip_linear_is_zero_032_r23():
    with torch.random.fork_rng(devices=[]):  # as inside `Variants`
        lin = core_skip_linear(7, 5)
    assert lin.weight.shape == (5, 7) and lin.bias.shape == (5,)
    assert not lin.weight.any() and not lin.bias.any()


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("arch", ON)
def test_init_equals_skip_off_032_ac13(core, arch):
    """032-AC13 ①②: the shared modules start from the weights of the model without the skip (the
    global RNG not moved) and, the skip being zero, every output and state is that model's, on the
    BC, RL phased and step paths."""
    cfg_on, b, t, obs, labels, _ = setup(core, arch)
    cfg_off = replace(cfg_on, variant=OFF[arch])
    on = net_of(cfg_on, 9)
    after_on = torch.rand(3)
    off = net_of(cfg_off, 9)
    after_off = torch.rand(3)
    assert torch.equal(after_on, after_off), "the global RNG moved"
    sa, sb = on.state_dict(), off.state_dict()
    assert set(sa) - set(sb) == SKIP_KEYS and set(sb) <= set(sa)
    for k, v in sb.items():
        assert torch.equal(sa[k], v), k
    assert not sa["variant.core_skip.weight"].any() and not sa["variant.core_skip.bias"].any()
    with torch.no_grad():
        for kw in paths(on):
            x = off.forward_seq(obs, labels, None, off.init_state(b), t, **kw)
            y = on.forward_seq(obs, labels, None, on.init_state(b), t, **kw)
            for k in HEADS:
                assert torch.equal(x["logp"][k], y["logp"][k]), (kw, k)
            assert torch.equal(x["value"], y["value"]) and same(x["win_open"], y["win_open"]), kw
            for p, q in zip(leaves(x["state"]), leaves(y["state"])):
                assert torch.equal(p, q), kw
        sx, sy = off.init_state(b), on.init_state(b)
        for k in range(t):
            o = rows_at(obs, b, t, k)
            x, y = off.act(o, sx, 0.0), on.act(o, sy, 0.0)
            for h in x["action"]:
                assert torch.equal(x["action"][h], y["action"][h]), (k, h)
            for key in ("log_prob", "value", "win_open"):
                assert same(x[key], y[key]), (k, key)
            sx, sy = x["state"], y["state"]
    for p, q in zip(leaves(sx), leaves(sy)):
        assert torch.equal(p, q)


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("arch", ON)
def test_gradients_reach_the_skip_032_r23(core, arch):
    """The zero skip gets a gradient (weight and bias) from the policy and value losses, and adds
    nothing to the other parameters' gradients (it is zero, so no gradient flows back through
    it): the first update is the one of the model without it, plus the skip's own."""
    cfg_on, b, t, obs, labels, priv = setup(core, arch)
    cfg_off = replace(cfg_on, variant=OFF[arch])
    for kw in paths(net_of(cfg_on)):
        grads = []
        for cfg in (cfg_off, cfg_on):
            net = net_of(cfg).train()
            torch.manual_seed(1)
            out = net.forward_seq(obs, labels, priv, net.init_state(b), t, **kw)
            loss = sum(out["logp"][k].sum() for k in HEADS) + out["value"].sum()
            loss.backward()
            grads.append({k: p.grad for k, p in net.named_parameters() if p.grad is not None})
        off, on = grads
        assert set(on) - set(off) == SKIP_KEYS, kw
        for k in SKIP_KEYS:
            assert torch.isfinite(on[k]).all() and on[k].abs().sum() > 0, (kw, k)
        for k, g in off.items():
            torch.testing.assert_close(on[k], g, atol=1e-6, rtol=1e-6, msg=lambda m, k=k: f"{kw} {k}: {m}")


def per_decision(net, obs, labels, s0, b: int, t: int):
    """Head log-probs, values and final state from one decision at a time, the skip written out:
    the core and the outer LSTM step on the pre-skip h, the heads and value read h + skip."""
    s = s0
    hs, es, sps, scs = [], [], [], []
    for k in range(t):
        o = rows_at(obs, b, t, k)
        if net.outer is None:
            core_in, e, sp, sc, _ = net.encode(o)
            h, s = net.core.step(core_in, s)
        else:
            cs, os_ = s
            core_in, e, sp, sc, _ = net.encode(o, h_o=os_[0])
            h, cs = net.core.step(core_in, cs)
            s = (cs, net.outer.step(h, os_))
        hs.append(h + net.variant.core_skip(core_in))
        es.append(e)
        sps.append(sp)
        scs.append(sc)
    rows = lambda xs: torch.stack(xs, 1).reshape(b * t, *xs[0].shape[1:])  # noqa: E731
    sp = {"s8": rows([x["s8"] for x in sps]), "skips": [rows([x["skips"][i] for x in sps]) for i in range(3)], "pool": rows([x["pool"] for x in sps])}
    sc = {k: rows([x[k] for x in scs]) for k in scs[0]}
    h = rows(hs)
    logp = net.heads(labels, 1.0, h, sc["c"], rows(es), sp, obs, net.table)["logp"]
    return logp, net.value(h, sc["z"], sc["race"], None), net.value.win_open(h, sc["z"], sc["race"]), s


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("arch", ON)
def test_step_matches_sequence_032_ac13(core, arch):
    """032-AC13 ③ with a trained (non-zero) skip: `act` one decision at a time, the BC pass and
    the RL phased pass from different decisions inside an outer segment give the per-decision
    log-probs, values, final state and gradients (032-R5)."""
    cfg, b, t, obs, labels, _ = setup(core, arch)
    net = trained_skip(net_of(cfg))
    starts = [None] + ([(1, 2), (3, 0)] if net.outer is not None else [])
    for counts in starts:
        s0 = net.init_state(b) if counts is None else state_after(net, obs, b, t, counts)
        seq = net.forward_seq(obs, labels, None, s0, t, phased=counts is not None)
        logp, value, win_open, state = per_decision(net, obs, labels, s0, b, t)
        for k in HEADS:
            assert torch.allclose(seq["logp"][k], logp[k], atol=1e-4), (counts, k)
        assert torch.allclose(seq["value"], value, atol=1e-5), counts
        assert same(seq["win_open"], win_open, lambda a, b: torch.allclose(a, b, atol=1e-5)), counts
        for x, y in zip(leaves(seq["state"]), leaves(state)):
            assert torch.allclose(x.float(), y.float(), atol=1e-5), counts
        params = [net.variant.core_skip.weight, net.variant.core_skip.bias, next(net.core.parameters()), next(net.entity.parameters())]
        loss = lambda lp, v: sum(lp[k].sum() for k in HEADS) + v.sum()  # noqa: E731
        ga = torch.autograd.grad(loss(seq["logp"], seq["value"]), params)
        gb = torch.autograd.grad(loss(logp, value), params)
        for x, y in zip(ga, gb):
            assert torch.allclose(x, y, atol=1e-4, rtol=1e-4), counts
    # `act` (the inference path) gives the BC pass's values and final state.
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
        assert torch.allclose(x.float(), y.float(), atol=1e-5)


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("arch", ON)
def test_states_take_the_pre_skip_h_032_r23(core, arch):
    """The trained skip changes the value but no state: the core state (LSTM h, c / GTrXL memory)
    and the outer LSTM state equal those of the same model with the skip zeroed, on every path."""
    cfg, b, t, obs, labels, _ = setup(core, arch)
    zero = net_of(cfg)
    net = trained_skip(copy.deepcopy(zero))
    with torch.no_grad():
        for kw in paths(net):
            x = zero.forward_seq(obs, labels, None, zero.init_state(b), t, **kw)
            y = net.forward_seq(obs, labels, None, net.init_state(b), t, **kw)
            assert not torch.equal(x["value"], y["value"]), ("the skip reaches the value", kw)
            for p, q in zip(leaves(x["state"]), leaves(y["state"])):
                assert torch.equal(p, q), kw
        sx, sy = zero.init_state(b), net.init_state(b)
        for k in range(t):
            o = rows_at(obs, b, t, k)
            sx, sy = zero.act(o, sx, 0.0)["state"], net.act(o, sy, 0.0)["state"]
            for p, q in zip(leaves(sx), leaves(sy)):
                assert torch.equal(p, q), k


@pytest.mark.parametrize("core", CORES)
def test_outer_h_sum_is_the_pre_skip_h_032_r23(core):
    """After the first decision (the outer segment is 2, so no update yet) the outer h_sum is the
    core output h itself, not h' = h + skip."""
    cfg, b, t, obs, _, _ = setup(core, "skip_all")
    net = trained_skip(net_of(cfg))
    o = rows_at(obs, b, t, 0)
    with torch.no_grad():
        cs, os_ = net.init_state(b)
        core_in, *_ = net.encode(o, h_o=os_[0])
        h, _ = net.core.step(core_in, cs)
        skip = net.variant.core_skip(core_in)
        h_sum = net.act(o, net.init_state(b), 0.0)["state"][1][2]
    assert skip.abs().sum() > 0
    assert torch.equal(h_sum, h.float())


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("arch", ON)
def test_entity_recompute_keeps_the_update_041_r17(core, arch):
    """032-AC13 ④ / 041-R17 with a trained skip: the entity encoder recomputed per 3-row chunk
    gives the gradients of one pass, the skip's and the entity encoder's included."""
    cfg, b, t, obs, labels, priv = setup(core, arch)
    grads = []
    for chunk in (0, 3):
        net = trained_skip(net_of(cfg)).train()
        out = net.forward_seq(obs, labels, priv, net.init_state(b), t, chunk)
        loss = sum(out["logp"][k].sum() for k in HEADS) + out["value"].sum()
        loss.backward()
        grads.append({k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    assert SKIP_KEYS <= grads[0].keys() and any(k.startswith("entity.") for k in grads[0])
    for k, g in grads[0].items():
        assert torch.allclose(g, grads[1][k], atol=1e-5, rtol=1e-4), k


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("arch", ON)
def test_masks_hold_032_ac4(core, arch):
    """032-AC4 with a trained skip: sampled action types stay inside the type mask."""
    cfg, b, t, obs, _, _ = setup(core, arch)
    net = trained_skip(net_of(cfg))
    g = torch.Generator().manual_seed(5)
    o = rows_at(obs, b, t, 0)
    s = net.init_state(b)
    with torch.no_grad():
        for _ in range(20):
            out = net.act(o, s, 1.0, g)
            assert o["type_mask"][torch.arange(b), out["action"]["action"]].all()
            s = out["state"]


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("arch", ON)
def test_checkpoint_round_trip_032_ac13(core, arch, tmp_path):
    """032-AC13 ④: the trained skip and the switch survive save and load."""
    cfg, b, t, obs, labels, _ = setup(core, arch)
    net = trained_skip(net_of(cfg))
    ckpt.save_model(tmp_path, net)
    ckpt.save_meta(tmp_path, net.cfg, {}, "f32")
    back, meta = ckpt.load_model(tmp_path)
    assert back.cfg.variant == cfg.variant and meta["config"]["variant"]["core_skip"] is True
    assert SKIP_KEYS <= set(back.state_dict())
    assert torch.equal(back.variant.core_skip.weight, net.variant.core_skip.weight)
    with torch.no_grad():
        x = net.eval().forward_seq(obs, labels, None, net.init_state(b), t)
        y = back.eval().forward_seq(obs, labels, None, back.init_state(b), t)
    assert torch.equal(x["value"], y["value"])
    for k in HEADS:
        assert torch.equal(x["logp"][k], y["logp"][k]), k


# --- GPU (pytest -m gpu): the inference graph of a model with the skip (032-AC13, act = eager).


@pytest.mark.gpu
@torch.no_grad()
def test_graphed_act_matches_eager_core_skip_032_ac13():
    from uranium.agent.infer import GraphedAct
    from uranium.bc.trainer import new_net_config
    from uranium.bench.inputs import random_obs
    from uranium.device import autocast, resolve

    if not torch.cuda.is_available():
        pytest.skip("no CUDA / ROCm device")
    _, dev, prec = resolve(None)
    torch.manual_seed(0)
    v = VariantConfig(terrain=True, ctx_token=True, attn_pool=True, outer=32, core_skip=True)
    net = trained_skip(UraniumNet(new_net_config("gtrxl", True, v), action_columns())).to(dev).eval()
    graphed = GraphedAct(net.act, dev, prec)
    exact = torch.version.hip is None
    close = (lambda a, b: torch.equal(a, b)) if exact else (lambda a, b: torch.allclose(a.float(), b.float(), atol=1e-5, rtol=1e-5))
    # As test_graphed_act_matches_eager_variant: batch 3, padded to 4 rows by GraphedAct; eager
    # runs the same 4 rows.
    obs = random_obs(3, 128, net.cfg.input, dev, torch.Generator().manual_seed(1), valid=119)
    pad = torch.tensor([0, 1, 2, 0], device=dev)
    obs4 = {k: x.index_select(0, pad) for k, x in obs.items()}
    s_eager, s_graph = net.init_state(4, dev), net.init_state(3, dev)
    for _ in range(40):  # past an outer update (every 32 decisions)
        with autocast(dev, prec):
            want = net.act(obs4, s_eager, 0.0)
        got = graphed(obs, s_graph, 0.0)
        for k in want["action"]:
            assert torch.equal(got["action"][k], want["action"][k][:3]), k
        assert close(got["value"], want["value"][:3])
        s_eager, s_graph = want["state"], got["state"]
    for x, y in zip(leaves(s_graph), leaves(s_eager)):
        assert close(x, y[:3])
