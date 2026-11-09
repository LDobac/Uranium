"""032-R25 dead-channel statistics (`uranium.model.monitor`; 041-R25 / 042-R21 record, 032-AC15) on
the tiny Burn-parity batch, CPU: every point is found (spatial stages incl. s8 and the `const`
counts; with `points="all"` also entity FFN, head ResMlp, core FFN) in the default model and with
the 032 §2.7 variants; the counts, the stem and ResBlock inner points and `const` match direct
computations; the row mask drops rows; the phased RL pass is counted once; a dead block reads as
dead and constant; `packed` follows the 041-R25 columns; the forward and the mode are unchanged;
and BC training with the statistics on and off is bit-identical (032-AC15)."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from uranium.model import UraniumNet, UraniumNetConfig, action_columns, to_labels, to_obs
from uranium.model.config import VariantConfig
from uranium.model.layers import conv_layout
from uranium.model.monitor import (CONST_POINTS, ROWS_KEY, SPEC_COLUMNS, SPEC_LAYERS, ReluMonitor, channel_stats, const_stats, packed, relu_stats,
                                   stats_to_host)

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "burn_parity"
OUTER = VariantConfig(terrain=True, ctx_token=True, attn_pool=True, outer=2)


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
    pts = {"spatial.stem", "spatial.s8", *CONST_POINTS}
    for i in range(len(net.spatial.downs)):
        pts |= {f"spatial.down{i}.conv", f"spatial.down{i}.inner", f"spatial.down{i}.res"}
    for j in range(len(net.spatial.bottleneck)):
        pts |= {f"spatial.bottleneck{j}", f"spatial.bottleneck{j}.inner"}
    return pts


@pytest.mark.parametrize("name,variant", [("tiny-gtrxl", None), ("tiny-lstm", None), ("tiny-gtrxl", OUTER)])
def test_every_point_is_counted_032_r25(name, variant):
    net, b, t, obs, labels, priv = tiny(name, variant)
    sp = relu_stats(net, obs, labels, priv, net.init_state(b), t)
    assert set(sp) == spatial_points(net)  # the trainers' default: spatial only
    st = relu_stats(net, obs, labels, priv, net.init_state(b), t, points="all")
    want = spatial_points(net) | {"entity.L0.ffn", "heads.type_in", "heads.type_res0.inner", "heads.type_res0.out", "heads.type_res1.inner", "heads.type_res1.out"}
    if net.cfg.core.kind == "Gtrxl":
        want |= {f"core.L{l}.{p}" for l in range(2) for p in ("attn", "ffn", "ffn_hidden")}
    assert set(st) == want
    for k, v in st.items():
        if k in CONST_POINTS:
            assert isinstance(v["const"], torch.Tensor) and 0 <= int(v["const"]) <= v["channels"], k
            continue
        assert isinstance(v["pos_frac"], torch.Tensor) and isinstance(v["never_active"], torch.Tensor), k  # no host read inside
        assert 0.0 <= float(v["pos_frac"]) <= 1.0 and 0 <= int(v["never_active"]) <= v["channels"], k
    assert st["spatial.s8"]["channels"] == net.cfg.spatial.stages[-1]
    assert st["spatial.pool.const"]["channels"] == net.cfg.spatial.pool and st["spatial.s8.const"]["dtype"] == "float32"


def test_points_match_direct_computation_032_r25():
    """pos / dead of s8, a skip, the stem (before FiLM) and every ResBlock's inner ReLU, and
    const of S8 / s_pool, against the activations computed here."""
    net, b, t, obs, labels, priv = tiny()
    st = relu_stats(net, obs, labels, priv, net.init_state(b), t, points="all")
    sp_ = net.spatial
    with torch.no_grad():
        _, e, sp, _, _ = net.encode(obs)
        spatial = obs["spatial"]
        x = F.relu(sp_.stem(conv_layout(torch.cat([spatial, sp_.scatter_mean(obs, e, spatial.shape[2], spatial.shape[3])], 1))))
        stem = x
        inner = {}
        for i, d in enumerate(sp_.downs):
            y = F.relu(d.conv(x))
            inner[f"spatial.down{i}.inner"] = F.relu(d.res.c1(y))
            x = d.res(y)
        for j, r in enumerate(sp_.bottleneck):
            inner[f"spatial.bottleneck{j}.inner"] = F.relu(r.c1(x))
            x = r(x)
    for name, act in [("spatial.s8", sp["s8"]), ("spatial.down2.res", sp["skips"][2]), ("spatial.stem", stem), *inner.items()]:
        want = channel_stats(act, channel_dim=1)
        assert float(st[name]["pos_frac"]) == pytest.approx(float(want["pos_frac"])), name
        assert int(st[name]["never_active"]) == int(want["never_active"]), name
    for name, act in (("spatial.s8.const", sp["s8"]), ("spatial.pool.const", sp["pool"])):
        std = act.double().transpose(0, 1).reshape(act.shape[1], -1).std(1, unbiased=False)
        assert torch.allclose(st[name]["std"], std, atol=1e-9), name
        assert int(st[name]["const"]) == int((std < 1e-4).sum())
    m = obs["entity_mask"]
    assert int(st["entity.L0.ffn"]["elements"]) == int(m.sum()) * net.cfg.entity.ffn


def test_const_counts_flat_channels():
    x = torch.randn(6, 3, 4, 4)
    x[:, 1] = 0.25  # a constant channel
    x[:, 2] = 0.25
    x[0, 2, 0, 0] = 0.26  # nearly: std about 6e-4, above 1e-4
    st = const_stats(x)
    assert int(st["const"]) == 1 and st["channels"] == 3
    assert int(const_stats(x, threshold=1e-2)["const"]) == 2
    # rows: only rows 1.. kept, channel 2 is constant there
    assert int(const_stats(x, rows=torch.tensor([False, True, True, True, True, True]))["const"]) == 2
    p = torch.randn(5, 4)
    p[:, 0] = 3.0
    assert int(const_stats(p, channel_dim=1)["const"]) == 1


def test_row_mask_drops_rows():
    net, b, t, obs, labels, priv = tiny()
    rows = torch.zeros(b * t, dtype=torch.bool)
    rows[: t] = True  # the first sequence only
    st = relu_stats(net, obs, labels, priv, net.init_state(b), t, rows=rows, points="all")
    with torch.no_grad():
        _, _, sp, _, _ = net.encode(obs)
    want = channel_stats(sp["s8"][:t], channel_dim=1)
    assert float(st["spatial.s8"]["pos_frac"]) == pytest.approx(float(want["pos_frac"]))
    assert int(st["spatial.s8"]["elements"]) == int(want["elements"])
    assert torch.allclose(st["spatial.pool.const"]["std"], sp["pool"][:t].double().std(0, unbiased=False), atol=1e-9)
    assert int(st["heads.type_in"]["elements"]) == t * net.cfg.heads.ar
    assert int(st["core.L0.ffn"]["elements"]) == t * net.cfg.core.hidden


def test_phased_pass_counts_each_row_once():
    """The RL learner's phased outer pass (032 §2.7): rows start inside an outer segment, the
    encoders run per phase on a subset of rows and the core once per phase over every row. The
    spatial counts equal the same rows' activations; the core points are skipped, not doubled,
    and the row mask works (no crash)."""
    net, b, t, obs, labels, priv = tiny(variant=OUTER)
    s0 = net.init_state(b)
    s0 = (s0[0], (*s0[1][:3], torch.tensor([1.0, 0.0])))  # row 0 starts one decision into a segment
    rows = torch.ones(b * t, dtype=torch.bool)
    rows[t - 1] = False
    for r in (None, rows):
        with ReluMonitor(net, points="all") as mon:
            relu_stats(net, obs, labels, priv, s0, t, rows=r, phased=True, monitor=mon)
            st = mon.stats()
        assert mon.core_skipped > 0 and not any(k.startswith("core.") for k in st)
        with torch.no_grad():
            _, _, sp, _, _, _ = net._outer_seq_phased(obs, s0, t, 0)
        want = channel_stats(sp["s8"], rows=r, channel_dim=1)
        assert int(st["spatial.s8"]["elements"]) == int(want["elements"])
        assert float(st["spatial.s8"]["pos_frac"]) == pytest.approx(float(want["pos_frac"]))
        std = const_stats(sp["pool"], rows=r)["std"]
        assert torch.allclose(st["spatial.pool.const"]["std"], std, atol=1e-9)
        assert int(st["heads.type_in"]["elements"]) == (b * t if r is None else int(r.sum())) * net.cfg.heads.ar


def test_dead_block_reads_dead_and_constant():
    """10.4: a bottleneck whose second conv only pushes below zero gives s8 = 0 everywhere: every
    channel dead and constant; s_pool is then constant too."""
    net, b, t, obs, labels, priv = tiny()
    last = net.spatial.bottleneck[-1]
    with torch.no_grad():
        last.c1.weight.zero_()
        last.c1.bias.fill_(1.0)
        last.c2.weight.fill_(-100.0)
        last.c2.bias.fill_(-100.0)
    st = stats_to_host(relu_stats(net, obs, labels, priv, net.init_state(b), t))
    assert st["spatial.s8"]["pos_frac"] == 0.0 and st["spatial.s8"]["never_active"] == st["spatial.s8"]["channels"]
    assert st["spatial.s8.const"]["const"] == st["spatial.s8.const"]["channels"]
    assert st["spatial.pool.const"]["const"] == st["spatial.pool.const"]["channels"]
    assert st["spatial.bottleneck0"]["pos_frac"] > 0.0


def test_packed_follows_the_042_r21_columns():
    """042-R21 ③: every sp_pos in 041-R25 layer order, then every sp_dead in the same order,
    then sp_const_s8, sp_const_pool (grouped, not interleaved)."""
    net, b, t, obs, labels, priv = tiny()
    st = relu_stats(net, obs, labels, priv, net.init_state(b), t)
    v = packed(st)
    assert v.dtype == torch.float64 and v.shape == (len(SPEC_COLUMNS),) == (2 * len(SPEC_LAYERS) + 2,)
    names = [n for n, _ in SPEC_LAYERS]
    assert names == ["stem", "c0", "c1", "c2", "c3", "r0", "r1", "r2", "r3", "r4", "r5", "d0", "d1", "d2", "d3", "b0", "b1"]
    assert list(SPEC_COLUMNS) == [f"sp_pos_{n}" for n in names] + [f"sp_dead_{n}" for n in names] + ["sp_const_s8", "sp_const_pool"]
    pts = [p for _, p in SPEC_LAYERS]
    assert v[:17].tolist() == pytest.approx([float(st[p]["pos_frac"]) for p in pts])
    assert v[17:34].tolist() == [int(st[p]["never_active"]) for p in pts]
    col = dict(zip(SPEC_COLUMNS, v.tolist()))
    assert col["sp_pos_b1"] == pytest.approx(float(st["spatial.bottleneck1"]["pos_frac"]))
    assert col["sp_dead_c2"] == int(st["spatial.down2.conv"]["never_active"])
    assert col["sp_pos_r4"] == pytest.approx(float(st["spatial.bottleneck0.inner"]["pos_frac"]))
    assert col["sp_const_s8"] == int(st["spatial.s8.const"]["const"]) and col["sp_const_pool"] == int(st["spatial.pool.const"]["const"])


def test_missing_points_refuse():
    """No pass with the monitor (or a compiled wrapper that does not call the hooks): no silent
    empty statistics."""
    net, *_ = tiny()
    with ReluMonitor(net) as mon:
        pass
    with pytest.raises(RuntimeError, match="no values"):
        mon.stats()
    with pytest.raises(ValueError):
        ReluMonitor(net, points="core")


def test_monitor_leaves_the_forward_and_mode_unchanged():
    net, b, t, obs, labels, priv = tiny()
    net.train()
    torch.manual_seed(0)
    with torch.no_grad():
        ref = net.forward_seq(obs, labels, priv, net.init_state(b), t)
    relu_stats(net, obs, labels, priv, net.init_state(b), t, points="all")
    assert net.training
    o = dict(obs)
    o[ROWS_KEY] = torch.ones(b * t, dtype=torch.bool)
    torch.manual_seed(0)
    with torch.no_grad(), ReluMonitor(net, points="all") as mon:
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
    whole = stats_to_host(relu_stats(net, obs, labels, priv, net.init_state(b), t, rows=rows, points="all"))
    monkeypatch.setattr(monitor, "CHUNK_ELEMENTS", 7)
    chunked = stats_to_host(relu_stats(net, obs, labels, priv, net.init_state(b), t, rows=rows, points="all"))
    assert chunked.keys() == whole.keys()
    for k in whole:
        if "const" in whole[k]:
            assert chunked[k]["const"] == whole[k]["const"] and chunked[k]["std_min"] == pytest.approx(whole[k]["std_min"], abs=1e-9), k
            continue
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


@pytest.mark.parametrize("kind", ["Gtrxl", "Lstm"])
def test_bc_training_with_and_without_statistics_is_bit_identical_032_ac15(tiny_config, tmp_path, kind):
    """032-AC15 / 041-R25 ④: the trainer contract (a separate no_grad eval pass per lane group
    with the pre-update s0, state dropped, no RNG) leaves BC training bit-identical on CPU."""

    from uranium import _core
    from uranium.bc.trainer import BcTrainer
    from uranium.model import input_dims

    shards = tmp_path / "shards"
    _core.write_test_shards(str(shards), "mixed", 2)
    cfg_net = UraniumNetConfig.from_dict(json.loads((FIX / "tiny-lstm" / "meta.json").read_text())["config"])
    cfg_net.input = input_dims()
    cfg_net.core.kind = kind
    cfg_net.entity.dropout = 0.1  # dropout on: the statistics pass must not use the training RNG
    cfg = json.loads(_core.bc_config(str(tiny_config), "s1")[0])
    seen = []

    class Monitored(BcTrainer):
        def group_loss(self, hg, group, valid, value_valid, step, *rest):
            net, dev, t = self.net, self.device, self.cfg["seq_len"]
            s = self.states[group]
            keep = torch.as_tensor(hg["keep"]).bool()
            s0 = net.init_state(hg["b"], dev) if s is None else net.mask_state(s, keep)
            st = relu_stats(net, to_obs(hg, dev), net.labels(hg, dev), torch.as_tensor(hg["privileged"]).float(), s0, t, rows=torch.as_tensor(hg["weights"]) > 0)
            seen.append(packed(st))
            return super().group_loss(hg, group, valid, value_valid, step, *rest)

    params = {}
    for cls in (BcTrainer, Monitored):
        torch.manual_seed(7)
        trainer = cls(cfg, UraniumNet(cfg_net, action_columns()), torch.device("cpu"), "f32")
        feeder = _core.Feeder(str(tiny_config), [str(shards)], prefetch=0, synthetic_maps=True, check_maps=False)
        losses = [json.loads(trainer.train_batch(feeder.next(), 3e-3, s))["total"] for s in range(3)]
        params[cls.__name__] = ({k: v.detach().clone() for k, v in trainer.net.state_dict().items()}, losses)
    (pa, la), (pb, lb) = params["BcTrainer"], params["Monitored"]
    assert la == lb and all(torch.equal(pa[k], pb[k]) for k in pa)
    assert seen and all(v.shape == (len(SPEC_COLUMNS),) for v in seen)
