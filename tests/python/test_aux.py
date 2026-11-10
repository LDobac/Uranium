"""032-R24 aux modules, the 041-R23 / 042-R21 aux losses, the 041-R25 / 042-R21 ③ dead-channel
monitor in the trainers (032-AC14, AC15, 041-AC6), the bc-eval per-race report (050-R9
`val500_race`) and `[train] model_seed` (041-R24 ⑦). CPU."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from uranium import _core, ckpt
from uranium.bc.trainer import AUX_PARTS, BcTrainer, bc_aux_counts
from uranium.model import UraniumNet, UraniumNetConfig, action_columns, input_dims, to_labels, to_obs
from uranium.model.auxiliary import AUX_JEPA_HIDDEN, AUX_PRIV_HIDDEN, jepa_mean, jepa_moments, jepa_std, jepa_sums, priv_sum
from uranium.model.config import VariantConfig
from uranium.model.monitor import ALARM_FROM, SPEC_COLUMNS, spatial_alarm
from uranium.model.variants import SEED_AUX_PRIV, seed_aux_jepa
from uranium.rl.learner import HEADS, Learner, rl_aux_counts, state0_of
from uranium.rl.reference import ReferenceLearner

from test_rl_opt import WEIGHTS, Segments, rl_cfg

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "burn_parity"
CPU = torch.device("cpu")
KS = [1, 2, 8]  # 8 >= the tiny configs' T = 4: no pair (slot 3 empty)
AUX = VariantConfig(aux_priv=True, aux_jepa=KS)
ARCH = dict(terrain=True, ctx_token=True, attn_pool=True, outer=2)


def net_cfg(variant: VariantConfig | None = None, kind: str = "Lstm", dropout: float | None = None) -> UraniumNetConfig:
    cfg = UraniumNetConfig.from_dict(json.loads((FIX / "tiny-lstm" / "meta.json").read_text())["config"])
    cfg.input = input_dims()
    cfg.core.kind = kind
    if dropout is not None:
        cfg.entity.dropout = dropout
    if variant is not None:
        cfg.variant = variant
    return cfg


def make_net(variant: VariantConfig | None = None, seed: int = 7, **kw) -> UraniumNet:
    torch.manual_seed(seed)
    return UraniumNet(net_cfg(variant, **kw), action_columns())


def params_of(net) -> dict:
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def bc_cfg(config: Path, **over) -> dict:
    cfg = json.loads(_core.bc_config(str(config), "s1")[0])
    cfg.update(over)
    return cfg


@pytest.fixture
def shards(tmp_path: Path) -> Path:
    d = tmp_path / "shards"
    _core.write_test_shards(str(d), "mixed", 2)
    return d


def feeder(config: Path, shards: Path):
    return _core.Feeder(str(config), [str(shards)], prefetch=0, synthetic_maps=True, check_maps=False)


# --- modules (032-R24, AMB-032-13) ---------------------------------------------------------------


def test_aux_modules_shapes_and_fixed_seeds_032_r24():
    """AuxPriv MLP(h -> 256 -> P), one MLP(h -> 512 -> d_model + pool) per k; built last from fixed
    seeds: the shared modules keep the default model's initial weights, the aux weights do not
    depend on the global RNG, and g_k does not depend on the other k."""
    cfg = net_cfg(AUX)
    net, ref = make_net(AUX), make_net()
    v = net.variant
    assert v.aux_priv.l1.weight.shape == (AUX_PRIV_HIDDEN, cfg.core.hidden) and v.aux_priv.l2.weight.shape == (cfg.input.privileged, AUX_PRIV_HIDDEN)
    d_y = cfg.entity.d_model + cfg.spatial.pool
    assert list(v.aux_jepa) == [str(k) for k in KS]
    for g in v.aux_jepa.values():
        assert g.l1.weight.shape == (AUX_JEPA_HIDDEN, cfg.core.hidden) and g.l2.weight.shape == (d_y, AUX_JEPA_HIDDEN)
    shared = ref.state_dict()
    for k, x in net.state_dict().items():
        if k.startswith("variant.aux_"):
            continue
        assert torch.equal(x, shared[k]), k
    other = make_net(AUX, seed=99)
    for k, x in net.state_dict().items():
        if k.startswith("variant.aux_"):
            assert torch.equal(x, other.state_dict()[k]), k
    alone = make_net(VariantConfig(aux_jepa=[8]), seed=5).variant.aux_jepa["8"]
    assert all(torch.equal(a, b) for a, b in zip(alone.parameters(), v.aux_jepa["8"].parameters()))
    # Own generators: the first layer of AuxPriv is the seed's U(-k, k) draw.
    g = torch.Generator().manual_seed(SEED_AUX_PRIV)
    k = 1.0 / math.sqrt(cfg.core.hidden)
    assert torch.equal(v.aux_priv.l1.weight, torch.empty(AUX_PRIV_HIDDEN, cfg.core.hidden).uniform_(-k, k, generator=g))
    assert seed_aux_jepa(1) != seed_aux_jepa(8)


def test_aux_checkpoint_round_trip_032_ac13(tmp_path):
    net = make_net(AUX)
    ckpt.save_model(tmp_path, net)
    ckpt.save_meta(tmp_path, net.cfg, {}, "f32", {})
    back, _ = ckpt.load_model(tmp_path)
    assert back.cfg.variant == AUX
    a, b = net.state_dict(), back.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# --- loss functions (041-R23 ②, 042-R21 ①) -----------------------------------------------------


def test_priv_sum_masks_rows_041_r23():
    g = torch.Generator().manual_seed(1)
    pred, tgt = torch.randn(5, 7, generator=g), torch.randn(5, 7, generator=g)
    rows = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
    want = sum(((pred[i] - tgt[i]) ** 2).mean() for i in (0, 2))
    assert torch.allclose(priv_sum(pred, tgt, rows), want)


def test_jepa_pairs_stay_inside_a_row_and_the_masks_032_ac14():
    """Pairs (t, t+k) only inside one row (a lane's chunk / segment), source and target masks
    apart, a k >= T has none; the target is LayerNorm'd without parameters; 1 - cos."""
    b, t, d = 3, 5, 6
    g = torch.Generator().manual_seed(2)
    y = torch.randn(b * t, d, generator=g)
    preds = {k: torch.randn(b * t, d, generator=g) for k in (1, 3, 5)}
    src = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    tgt = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    sums = jepa_sums(preds, y, src, tgt, [1, 3, 5])
    yn = F.layer_norm(y, (d,))
    for i, k in enumerate((1, 3, 5)):
        want, n = 0.0, 0
        for r in range(b):
            for s in range(t):
                if s + k < t and src[r, s] and tgt[r, s + k]:
                    want += 1.0 - float(F.cosine_similarity(preds[k][r * t + s], yn[r * t + s + k], dim=0))
                    n += 1
        assert math.isclose(float(sums[i]), want, rel_tol=1e-5, abs_tol=1e-6), k
    assert float(sums[2]) == 0.0, "k = T: no pair"
    total, per = jepa_mean(sums, [9, 2, 0])
    assert torch.allclose(total, (sums[0] / 9 + sums[1] / 2) / 2), "a k without a pair is left out of the mean"
    assert float(per[2]) == 0.0
    m = jepa_moments(y, tgt.reshape(-1).float())
    sel = yn[tgt.reshape(-1)].double()
    assert torch.allclose(jepa_std(m[:2], int(tgt.sum())), sel.std(0, unbiased=False).mean())
    raw = y[tgt.reshape(-1)].double()
    assert torch.allclose(jepa_std(m[2:], int(tgt.sum())), raw.std(0, unbiased=False).mean())


# --- BC trainer (041-R23) ----------------------------------------------------------------------


def manual_bc_aux(net, batch, t: int, ks: list) -> dict:
    """The 041-R23 aux values of one batch from their definition, over one group of every lane."""
    hg = batch.group(0, batch.lanes, t, 0, net.cfg.heads.max_select)
    obs, labels = to_obs(hg, CPU), net.labels(hg, CPU)
    priv = torch.as_tensor(hg["privileged"]).float()
    with torch.no_grad():
        out = net.eval().forward_seq(obs, labels, priv, net.init_state(hg["b"]), t, aux=True)
    real, pp = torch.as_tensor(hg["real"]), torch.as_tensor(hg["priv_present"])
    rows = real & pp
    res = {"aux_priv": float(((out["aux_priv"] - priv) ** 2).mean(1)[rows].mean()) if rows.any() else None}
    yn = F.layer_norm(out["jepa_target"], (out["jepa_target"].shape[1],))
    per = []
    for k in ks:
        terms = [1.0 - float(F.cosine_similarity(out["aux_jepa"][k][r * t + s], yn[r * t + s + k], dim=0))
                 for r in range(hg["b"]) for s in range(t) if s + k < t and real[r * t + s] and real[r * t + s + k]]
        per.append(sum(terms) / len(terms) if terms else None)
    live = [x for x in per if x is not None]
    res["aux_jepa"] = sum(live) / len(live)
    res.update({f"aux_jepa_slot{i + 1}": x for i, x in enumerate(per)})
    res["jepa_tgt_std"] = float(yn[real].std(0, unbiased=False).mean())
    res["jepa_tgt_std_raw"] = float(out["jepa_target"][real].std(0, unbiased=False).mean())
    return res


def test_bc_aux_losses_match_the_definition_041_r23(tiny_config, shards):
    """Lane groups of one lane give the batch's aux means (batch-wide divisors), the log has them
    before λ, and the total adds λ_priv L_priv + λ_jepa L_jepa."""
    b = feeder(tiny_config, shards).next()
    t = bc_cfg(tiny_config)["seq_len"]
    net = make_net(AUX)
    want = manual_bc_aux(copy.deepcopy(net), b, t, KS)
    on = json.loads(BcTrainer(bc_cfg(tiny_config, aux={"priv": 0.2, "jepa": 0.1}), net, CPU, "f32").train_batch(b, 3e-3, 0))
    for k, x in want.items():
        if x is None:
            assert on[k] is None, k
        else:
            assert on[k] == pytest.approx(x, rel=1e-5, abs=1e-6), k
    assert on["aux_jepa_slot3"] is None, "k = 8 >= T: no pair"
    zero = json.loads(BcTrainer(bc_cfg(tiny_config, aux={"priv": 0.0, "jepa": 0.0}), make_net(AUX), CPU, "f32").train_batch(b, 3e-3, 0))
    assert zero["aux_priv"] == on["aux_priv"] and zero["aux_jepa"] == on["aux_jepa"]
    assert on["total"] == pytest.approx(zero["total"] + 0.2 * on["aux_priv"] + 0.1 * on["aux_jepa"], rel=1e-5)
    for k in HEADS:
        assert on[k] == zero[k], k


def test_bc_aux_counts_match_the_group_masks_041_r23(tiny_config, shards):
    """The host divisors equal the collated masks: real & priv_present rows, and pairs of real
    steps inside a lane's chunk (padding out, one game per chunk)."""
    f = feeder(tiny_config, shards)
    t = 4
    for _ in range(4):
        b = f.next()
        if b is None:
            break
        hg = b.group(0, b.lanes, t, 0, 12)
        real, pp = hg["real"].reshape(-1, t), hg["priv_present"].reshape(-1, t)
        c = bc_aux_counts(b, t, KS)
        assert c["real"] == real.sum() and c["priv"] == (real & pp).sum()
        assert c["pairs"] == [int((real[:, :t - k] & real[:, k:]).sum()) if k < t else 0 for k in KS]


def test_bc_aux_loss_masks_padding_and_missing_privileged_041_ac6():
    """`BcTrainer.aux_loss`: padding rows and steps without a privileged summary are out of L_priv;
    a real step with a void label (weight 0) is in; JEPA pairs need both steps real."""
    net = make_net(AUX)
    tr = BcTrainer({"lr": 1e-3, "weight_decay": 0.0, "seq_len": 3, "lanes": 2, "micro_lanes": 0, "value": {}, "aux": {"priv": 1.0, "jepa": 1.0}}, net, CPU, "f32")
    b, t, p, d = 2, 3, net.cfg.input.privileged, net.cfg.entity.d_model + net.cfg.spatial.pool
    g = torch.Generator().manual_seed(4)
    out = {"aux_priv": torch.randn(b * t, p, generator=g), "aux_jepa": {k: torch.randn(b * t, d, generator=g) for k in KS}, "jepa_target": torch.randn(b * t, d, generator=g)}
    priv = torch.randn(b * t, p, generator=g)
    hg = {"real": np.array([1, 1, 1, 1, 0, 0], bool), "priv_present": np.array([0, 1, 1, 1, 1, 1], bool), "weights": np.array([1, 0, 1, 1, 0, 0], np.float32)}
    aux = {"real": 4, "priv": 3, "pairs": [3, 1, 0]}
    moments = torch.zeros(4, d, dtype=torch.float64)
    loss, parts = tr.aux_loss(out, hg, priv, b, t, aux, moments, torch.zeros(()))
    rows = [1, 2, 3]  # real & priv_present; row 1 has a void label and still counts
    want_priv = sum(((out["aux_priv"][i] - priv[i]) ** 2).mean() for i in rows) / 3
    assert torch.allclose(parts[0], want_priv)
    yn = F.layer_norm(out["jepa_target"], (d,))
    pairs1 = [(0, 1), (1, 2)]  # lane 1: rows 3 (real) -> 4 (padding) is out
    j1 = sum(1 - F.cosine_similarity(out["aux_jepa"][1][a], yn[c], dim=0) for a, c in pairs1) / 3
    j2 = (1 - F.cosine_similarity(out["aux_jepa"][2][0], yn[2], dim=0)) / 1
    assert torch.allclose(parts[2], j1) and torch.allclose(parts[3], j2) and float(parts[4]) == 0.0
    assert torch.allclose(parts[1], (j1 + j2) / 2)
    assert torch.allclose(loss, parts[0] + parts[1])
    assert len(parts) == len(AUX_PARTS)


def test_bc_aux_weight_zero_equals_no_module_032_ac14(tiny_config, shards):
    """λ = 0 with the modules: the shared parameters and the loss parts are those of the model
    without them, bit for bit (the aux parameters get no gradient and do not move); without [aux]
    and modules the log has no aux key (the earlier update)."""
    f = feeder(tiny_config, shards)
    batches = [f.next() for _ in range(3)]
    plain = BcTrainer(bc_cfg(tiny_config), make_net(), CPU, "f32")
    zero = BcTrainer(bc_cfg(tiny_config, aux={"priv": 0.0, "jepa": 0.0}), make_net(AUX), CPU, "f32")
    aux0 = {k: v.clone() for k, v in params_of(zero.net).items() if k.startswith("variant.aux_")}
    for s, b in enumerate(batches):
        a, z = json.loads(plain.train_batch(b, 3e-3, s)), json.loads(zero.train_batch(b, 3e-3, s))
        assert not {"aux_priv", "aux_jepa", "jepa_tgt_std", *SPEC_COLUMNS} & a.keys()
        assert {k: z[k] for k in a} == a
        assert z["aux_priv"] is not None and z["aux_jepa"] is not None
    pa, pz = params_of(plain.net), params_of(zero.net)
    assert all(torch.equal(pa[k], pz[k]) for k in pa)
    assert all(torch.equal(aux0[k], pz[k]) for k in aux0)


def test_bc_non_finite_aux_part_skips_the_update_041_r3(tiny_config, shards):
    """041-R3: a non-finite aux loss part skips the optimizer step like any loss part."""
    b = feeder(tiny_config, shards).next()
    tr = BcTrainer(bc_cfg(tiny_config, aux={"priv": 0.2, "jepa": 0.1}), make_net(AUX), CPU, "f32")
    orig = tr.aux_loss

    def poisoned(*a):
        loss, parts = orig(*a)
        return loss, [parts[0] * float("nan"), *parts[1:]]

    tr.aux_loss = poisoned
    before = params_of(tr.net)
    out = json.loads(tr.train_batch(b, 3e-3, 0))
    assert out["skipped"] is True and out["aux_priv"] is None
    after = params_of(tr.net)
    assert all(torch.equal(before[k], after[k]) for k in before)
    # λ_priv = 0: the part is only logged (null), the update is not skipped.
    tr0 = BcTrainer(bc_cfg(tiny_config, aux={"priv": 0.0, "jepa": 0.1}), make_net(AUX), CPU, "f32")
    orig0 = tr0.aux_loss
    tr0.aux_loss = lambda *a: (lambda r: (r[0], [r[1][0] * float("nan"), *r[1][1:]]))(orig0(*a))
    out = json.loads(tr0.train_batch(b, 3e-3, 0))
    assert "skipped" not in out and out["aux_priv"] is None and out["aux_jepa"] is not None


def test_aux_log_nulls_without_rows_041_r23(tiny_config):
    """A batch without a privileged step (or a k without a pair) logs null, not 0 (BC and RL)."""
    from uranium.rl.learner import aux_stats

    tr = BcTrainer(bc_cfg(tiny_config, aux={"priv": 0.2, "jepa": 0.1}), make_net(AUX), CPU, "f32")
    log = tr.aux_log([0.0, 0.5, 0.5, 0.0, 0.0], {"priv": 0, "pairs": [4, 0, 0]}, (1.0, 2.0))
    assert log["aux_priv"] is None and log["aux_jepa"] == 0.5 and log["aux_jepa_slot2"] is None
    assert (log["jepa_tgt_std"], log["jepa_tgt_std_raw"]) == (1.0, 2.0)
    st = aux_stats({"priv": 0.2, "jepa": 0.1}, KS, [0.0, 0.5, 0.5, 0.0, 0.0], {"priv": 0, "pairs": [4, 0, 0]}, (1.0, 2.0))
    assert st["aux_priv"] is None and st["aux_jepa"] == 0.5 and st["aux_jepa_slots"] == [0.5, None, None] and st["jepa_tgt_std_raw"] == 2.0


# --- monitor in the BC trainer (041-R25, 032-AC15) ----------------------------------------------


@pytest.mark.parametrize("variant", [None, AUX], ids=["plain", "aux"])
def test_bc_monitor_on_off_is_bit_identical_032_ac15(tiny_config, shards, variant):
    """`[monitor] spatial_every` on (every update, every 2nd) vs off: the same parameters and losses
    bit for bit (dropout on), the statistics logged on the due updates only, in SPEC_COLUMNS."""
    f = feeder(tiny_config, shards)
    batches = [f.next() for _ in range(3)]
    aux = {"aux": {"priv": 0.2, "jepa": 0.1}} if variant else {}
    runs = {}
    for every in (0, 1, 2):
        mon = {"monitor": {"spatial_every": every}} if every else {}
        tr = BcTrainer(bc_cfg(tiny_config, **aux, **mon), make_net(variant, dropout=0.1), CPU, "f32")
        logs = [json.loads(tr.train_batch(b, 3e-3, s)) for s, b in enumerate(batches)]
        runs[every] = (params_of(tr.net), logs)
    p0, l0 = runs[0]
    for every in (1, 2):
        p, logs = runs[every]
        assert all(torch.equal(p0[k], p[k]) for k in p0)
        for s, (a, b) in enumerate(zip(l0, logs)):
            due = (s + 1) % every == 0
            assert {k: b[k] for k in a} == a
            assert (set(SPEC_COLUMNS) <= b.keys()) == due and not (set(SPEC_COLUMNS) & a.keys())
            if due:
                assert 0.0 <= b["sp_pos_stem"] <= 1.0 and b["sp_dead_r5"] >= 0.0 and b["sp_const_pool"] >= 0.0


def test_bc_monitor_alarm_line_041_r25(tiny_config, shards, capsys):
    """The alarm is a log warning line from update 600 on; training goes on."""
    b = feeder(tiny_config, shards).next()
    tr = BcTrainer(bc_cfg(tiny_config, monitor={"spatial_every": 1}), make_net(), CPU, "f32")
    rec = {c: 0.0 for c in SPEC_COLUMNS} | {"sp_dead_c3": 16.0}
    channels = {f"sp_dead_{n}": 16 for n in ("stem", "c0", "c1", "c2", "c3", "r0", "r1", "r2", "r3", "r4", "r5")} | {"sp_const_s8": 16, "sp_const_pool": 32}
    tr.monitor_log(ALARM_FROM - 1, list(rec.values()), channels)
    assert "WARNING" not in capsys.readouterr().err
    assert "spatial_alarm" not in tr.monitor_log(ALARM_FROM - 1, list(rec.values()), channels)
    logged = tr.monitor_log(ALARM_FROM, list(rec.values()), channels)
    assert logged["spatial_alarm"] == ["sp_dead_c3 = 16 of 16"]
    assert "spatial encoder collapse alarm (041-R25): sp_dead_c3 = 16 of 16" in capsys.readouterr().err
    json.loads(tr.train_batch(b, 3e-3, ALARM_FROM - 1))  # an update at 600 runs (alarm or not)


def test_spatial_alarm_thresholds_041_r25():
    ch = {f"sp_dead_{n}": 8 for n in ("stem", "c0", "c1", "c2", "c3", "r0", "r1", "r2", "r3", "r4", "r5")} | {"sp_const_s8": 16, "sp_const_pool": 32}
    quiet = {c: 0.0 for c in SPEC_COLUMNS}
    assert spatial_alarm(10_000, quiet, ch) == []
    assert spatial_alarm(ALARM_FROM, quiet | {"sp_dead_r2": 4.0}, ch) == ["sp_dead_r2 = 4 of 8"]
    assert spatial_alarm(ALARM_FROM, quiet | {"sp_dead_r2": 3.0}, ch) == []
    assert spatial_alarm(ALARM_FROM - 1, quiet | {"sp_dead_r2": 8.0}, ch) == []
    assert spatial_alarm(ALARM_FROM, quiet | {"sp_const_s8": 8.0, "sp_const_pool": 16.0}, ch) == ["sp_const_s8 = 8 of 16", "sp_const_pool = 16 of 32"]
    assert spatial_alarm(ALARM_FROM, quiet | {"sp_dead_d3": 8.0, "sp_dead_b1": 8.0}, ch) == [], "block outputs are not watched"


# --- RL learner (042-R21) -----------------------------------------------------------------------


def rl_net(variant: VariantConfig | None, outer: bool = False, seed: int = 5, dropout: float | None = None) -> UraniumNet:
    v = variant if variant is not None else VariantConfig()
    if outer:
        v = VariantConfig(**{**v.__dict__, **ARCH})
    torch.manual_seed(seed)
    return UraniumNet(net_cfg(v if v.any() else None, dropout=dropout), action_columns())


def rl_run(variant, cfg_over: dict, outer: bool = False, updates: int = 2, lanes: int = 4, group: int = 1, sil_returns: tuple = (), priv_from: int = 1, net=None, dropout=None):
    cfg = rl_cfg(lanes=lanes)
    cfg.update(cfg_over)
    net = net if net is not None else rl_net(variant, outer, dropout=dropout)
    l = Learner(cfg, net, copy.deepcopy(net), CPU, "f32", "vtrace", micro_lanes=group)
    batch = Segments(lanes, 4, rl_net(None, outer, seed=6), 2 if outer else 0, sil_returns=sil_returns, priv_from=priv_from)
    stats = [l.update(batch, WEIGHTS) for _ in range(updates)]
    return stats, params_of(l.net), batch


def manual_rl_aux(net, batch, t: int, ks: list) -> dict:
    """042-R21 ① from its definition over one group of every lane: L_priv over trained steps with
    a summary, JEPA source trained / target real, queue lanes only."""
    g = batch.group(0, batch.lanes, t, net.cfg.heads.max_select)
    obs, labels = to_obs(g, CPU), to_labels(g, CPU)
    priv = torch.as_tensor(g["privileged"]).float()
    with torch.no_grad():
        out = net.eval().forward_seq(obs, labels, priv, state0_of(net, g, CPU), t, phased=True, aux=True)
    b = g["b"]
    queue = torch.as_tensor(~np.isfinite(g["sil_return"])).repeat_interleave(t)
    trained = torch.as_tensor(g["mask"] > 0) & queue
    real = torch.as_tensor(g["real"])
    rows = trained & torch.as_tensor(g["priv_present"])
    res = {"aux_priv": float(((out["aux_priv"] - priv) ** 2).mean(1)[rows].mean())}
    yn = F.layer_norm(out["jepa_target"], (out["jepa_target"].shape[1],))
    per = []
    for k in ks:
        terms = [1.0 - float(F.cosine_similarity(out["aux_jepa"][k][r * t + s], yn[r * t + s + k], dim=0))
                 for r in range(b) for s in range(t) if s + k < t and trained[r * t + s] and real[r * t + s + k]]
        per.append(sum(terms) / len(terms) if terms else None)
    live = [x for x in per if x is not None]
    res["aux_jepa"] = sum(live) / len(live)
    res["aux_jepa_slots"] = per
    res["jepa_tgt_std"] = float(yn[real & queue].std(0, unbiased=False).mean())
    res["jepa_tgt_std_raw"] = float(out["jepa_target"][real & queue].std(0, unbiased=False).mean())
    return res


@pytest.mark.parametrize("sil", [None, "replay", "sil"])
def test_rl_aux_losses_match_the_definition_042_r21(sil):
    """Lane groups of one lane give the batch's aux means: L_priv over trained steps (bootstrap row
    out) with a summary, JEPA source trained / target real (bootstrap in), 042-R16 buffer lanes
    out (as ordinary lanes in "replay", their own groups in "sil")."""
    over = {"aux": {"priv": 0.2, "jepa": 0.1}}
    sil_returns = ()
    if sil:
        over["sil"] = {"enabled": True, "mode": sil}
        sil_returns = (None, None, None, 1.0)
    net = rl_net(AUX)
    batch = Segments(4, 4, rl_net(None, seed=6), 0, sil_returns=sil_returns, priv_from=1)
    want = manual_rl_aux(copy.deepcopy(net), batch, 4, KS)
    cfg = rl_cfg(lanes=4)
    cfg.update(over)
    s = Learner(cfg, net, copy.deepcopy(net), CPU, "f32", "vtrace", micro_lanes=1).update(batch, WEIGHTS)
    assert s["aux_priv"] == pytest.approx(want["aux_priv"], rel=1e-5)
    assert s["aux_jepa"] == pytest.approx(want["aux_jepa"], rel=1e-5)
    assert s["jepa_tgt_std"] == pytest.approx(want["jepa_tgt_std"], rel=1e-5)
    assert s["jepa_tgt_std_raw"] == pytest.approx(want["jepa_tgt_std_raw"], rel=1e-5)
    for x, y in zip(s["aux_jepa_slots"], want["aux_jepa_slots"]):
        assert (x is None and y is None) or x == pytest.approx(y, rel=1e-5)
    assert s["aux_jepa_slots"][2] is None and s["spatial"] is None


def test_rl_aux_counts_match_the_group_masks_042_r21():
    """Batch-wide divisors: trained steps with a summary, pairs (trained t, real t+k) of a segment,
    real steps; queue lanes only (the buffer lane last)."""
    t = 4
    batch = _core.test_segments(4, t, sil_returns=[None, None, None, 1.0], priv_from=1)
    g = batch.group(0, 4, t, 12)
    mask, real, pp = g["mask"].reshape(4, t)[:3] > 0, g["real"].reshape(4, t)[:3], g["priv_present"].reshape(4, t)[:3]
    c = rl_aux_counts(batch, t, 3, KS)
    assert c["priv"] == (mask & pp).sum() and c["real"] == real.sum()
    assert c["pairs"] == [int((mask[:, :t - k] & real[:, k:]).sum()) if k < t else 0 for k in KS]
    # The non-terminal lane's bootstrap step: a JEPA target, never a source or a priv row.
    assert not mask[1, -1] and real[1, -1]


def test_rl_aux_loss_masks_fill_bootstrap_and_buffer_rows_042_r21():
    """`Learner.aux_loss` on crafted rows: fill rows (real false), the bootstrap row (mask 0, real),
    a step without a summary and a buffer lane."""
    net = rl_net(AUX)
    cfg = rl_cfg(lanes=3)
    cfg["aux"] = {"priv": 1.0, "jepa": 1.0}
    l = Learner(cfg, net, copy.deepcopy(net), CPU, "f32", "vtrace", micro_lanes=0)
    b, t, p, d = 3, 3, net.cfg.input.privileged, net.cfg.entity.d_model + net.cfg.spatial.pool
    gen = torch.Generator().manual_seed(8)
    out = {"aux_priv": torch.randn(b * t, p, generator=gen), "aux_jepa": {k: torch.randn(b * t, d, generator=gen) for k in KS}, "jepa_target": torch.randn(b * t, d, generator=gen)}
    priv = torch.randn(b * t, p, generator=gen)
    # lane 0: terminal, 2 steps + 1 fill; lane 1: 2 trained + bootstrap; lane 2: buffer lane
    mask = torch.tensor([1, 1, 0, 1, 1, 0, 1, 1, 1], dtype=torch.float32)
    real = torch.tensor([1, 1, 0, 1, 1, 1, 1, 1, 1], dtype=torch.bool)
    g = {"b": b, "priv_present": np.array([0, 1, 1, 1, 1, 1, 1, 1, 1], bool)}
    lane_rows = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.bool)
    aux = {"priv": 3, "real": 5, "pairs": [3, 1, 0]}
    moments = torch.zeros(4, d, dtype=torch.float64)
    loss, parts = l.aux_loss(out, g, priv, mask, real, lane_rows, t, aux, moments, torch.zeros(()))
    want_priv = sum(((out["aux_priv"][i] - priv[i]) ** 2).mean() for i in (1, 3, 4)) / 3
    assert torch.allclose(parts[0], want_priv)
    yn = F.layer_norm(out["jepa_target"], (d,))
    cos = lambda k, a, c: 1 - F.cosine_similarity(out["aux_jepa"][k][a], yn[c], dim=0)  # noqa: E731
    j1 = (cos(1, 0, 1) + cos(1, 3, 4) + cos(1, 4, 5)) / 3  # 1 -> 2 is a fill row; 4 -> 5 the bootstrap
    j2 = cos(2, 3, 5) / 1  # 0 -> 2 is a fill row
    assert torch.allclose(parts[2], j1) and torch.allclose(parts[3], j2) and float(parts[4]) == 0.0
    assert torch.allclose(parts[1], (j1 + j2) / 2) and torch.allclose(loss, parts[0] + parts[1])
    sel = yn[torch.tensor([0, 1, 3, 4, 5])].double()
    assert torch.allclose(jepa_std(moments[:2], 5), sel.std(0, unbiased=False).mean())


@pytest.mark.parametrize("outer", [False, True])
def test_rl_aux_weight_zero_equals_no_module_042_r21(outer):
    """λ = 0 with the modules: the learner's update and statistics are those of the model without
    them (shared parameters bit for bit); the BC reference forward runs no aux module."""
    plain, p_plain, _ = rl_run(None, {}, outer)
    zero, p_zero, _ = rl_run(AUX, {"aux": {"priv": 0.0, "jepa": 0.0}}, outer)
    for a, z in zip(plain, zero):
        shared = [k for k in a if not k.startswith(("aux_", "jepa_"))]
        assert {k: z[k] for k in shared} == {k: a[k] for k in shared}
        assert a["aux_priv"] is None and a["aux_jepa_slots"] == [None] * 3 and a["spatial"] is None
        assert z["aux_priv"] is not None and z["aux_jepa"] is not None
    assert all(torch.equal(p_plain[k], p_zero[k]) for k in p_plain)


@pytest.mark.parametrize("outer", [False, True])
def test_rl_monitor_on_off_is_bit_identical_032_ac15(outer):
    """[monitor] spatial_every = 1 vs off: the same update bit for bit (dropout on in the trained
    net), `spatial` filled (in SPEC_COLUMNS order, phased RL pass), with the aux losses on too."""
    over = {"aux": {"priv": 0.2, "jepa": 0.1}}
    off, p_off, _ = rl_run(AUX, over, outer, dropout=0.1)
    on, p_on, _ = rl_run(AUX, over | {"monitor": {"spatial_every": 1}}, outer, dropout=0.1)
    for a, b in zip(off, on):
        assert {k: v for k, v in a.items() if k != "spatial"} == {k: v for k, v in b.items() if k != "spatial"}
        assert a["spatial"] is None and len(b["spatial"]) == len(SPEC_COLUMNS)
        assert 0.0 <= b["spatial"][0] <= 1.0
    assert all(torch.equal(p_off[k], p_on[k]) for k in p_off)


def test_rl_aux_off_matches_the_reference_042_r21():
    """Default settings: the learner's statistics (aux and spatial keys `None`) equal the reference
    learner's bit for bit; the reference refuses aux and [monitor]."""
    cfg = rl_cfg(lanes=4)
    net = rl_net(None)
    batch = Segments(4, 4, rl_net(None, seed=6), 0)
    a = Learner(cfg, copy.deepcopy(net), copy.deepcopy(net), CPU, "f32", "vtrace", micro_lanes=1)
    a.trim_pad = False
    r = ReferenceLearner(cfg, copy.deepcopy(net), copy.deepcopy(net), CPU, "f32", "vtrace", micro_lanes=1)
    sa, sr = a.update(batch, WEIGHTS), r.update(batch, WEIGHTS)
    assert sa == sr
    assert sa["aux_priv"] is None and sa["spatial"] is None and sa["aux_jepa_slots"] == [None] * 3
    cfg2 = dict(cfg, monitor={"spatial_every": 1})
    with pytest.raises(NotImplementedError):
        ReferenceLearner(cfg2, copy.deepcopy(net), copy.deepcopy(net), CPU, "f32", "vtrace", micro_lanes=1).update(batch, WEIGHTS)


# --- every 032 §2.8 switch with the §2.7 ones (032-AC13 ①) -------------------------------------

ALL_ON = VariantConfig(**ARCH, bypass=True, spatial_preact=True, core_skip=True, aux_priv=True, aux_jepa=KS)


def test_all_switches_train_bc_and_rl_032_ac13(tiny_config, shards):
    """Every §2.8 switch (bypass, spatial_preact, core_skip, aux_priv, aux_jepa) and the four §2.7
    ones: the shared modules start from the all-off model's weights (032-AC13 ①), a BC update with
    aux and the monitor and an RL update run, with finite losses and the logs filled."""
    on, off = make_net(ALL_ON), make_net()
    base = off.state_dict()
    for k, x in on.state_dict().items():
        if k in base:
            assert torch.equal(x, base[k]), k
    assert any(k.startswith("variant.aux_") for k in on.state_dict()) and set(base) <= set(on.state_dict())
    cfg = bc_cfg(tiny_config, aux={"priv": 0.2, "jepa": 0.1}, monitor={"spatial_every": 1}, seq_len=4)
    tr = BcTrainer(cfg, on, CPU, "f32")
    log = json.loads(tr.train_batch(feeder(tiny_config, shards).next(), 3e-3, 0))
    assert math.isfinite(log["total"]) and log["aux_priv"] is not None and log["aux_jepa"] is not None
    assert set(SPEC_COLUMNS) <= log.keys()
    stats, _, _ = rl_run(ALL_ON, {"aux": {"priv": 0.2, "jepa": 0.1}, "monitor": {"spatial_every": 1}}, updates=1)
    s = stats[0]
    assert math.isfinite(s["loss"]) and s["aux_priv"] is not None and s["aux_jepa"] is not None and len(s["spatial"]) == len(SPEC_COLUMNS)


# --- [train] model_seed (041-R24 ⑦) ------------------------------------------------------------


def test_model_seed_sets_a_new_nets_weights_041_r24(tiny_config, tmp_path):
    """`[train] model_seed` seeds a new net (same seed = same weights whatever the global RNG did
    before, another seed = other weights); absent keeps the global RNG's draw; resume ignores it.
    The Rust config passes it and the fingerprint has it only when present."""
    js, fp = _core.bc_config(str(tiny_config), "s1")
    assert "train" not in json.loads(js)
    p = tiny_config.with_name("bc_seed.toml")
    p.write_text(tiny_config.read_text(encoding="utf-8") + "\n[train]\nmodel_seed = 2\n", encoding="utf-8")
    js2, fp2 = _core.bc_config(str(p), "s1")
    assert json.loads(js2)["train"] == {"model_seed": 2} and fp2 != fp
    bad = tiny_config.with_name("bc_bad_seed.toml")
    bad.write_text(tiny_config.read_text(encoding="utf-8") + "\n[train]\nseed = 2\n", encoding="utf-8")
    with pytest.raises(Exception):
        _core.bc_config(str(bad), "s1")

    def new(seed):
        cfg = json.loads(js)
        if seed is not None:
            cfg["train"] = {"model_seed": seed}
        torch.randn(17)  # move the global RNG: a seeded build does not see it
        return BcTrainer.make(json.dumps(cfg), "new", None, 0, "lstm", CPU, "f32").net

    a, b, c = new(1), new(1), new(2)
    first = lambda n: next(n.parameters())  # noqa: E731
    assert torch.equal(first(a), first(b)) and not torch.equal(first(a), first(c))
    # Absent: no reseed, the global RNG's draw as before the key.
    from uranium.bc.trainer import new_net_config

    torch.manual_seed(1)
    torch.randn(17)
    want = UraniumNet(new_net_config("lstm", False))
    torch.manual_seed(1)
    assert torch.equal(first(new(None)), first(want))
    # Resume: the checkpoint's weights, model_seed ignored.
    (tmp_path / "ck").mkdir()
    ckpt.save_model(tmp_path / "ck", c)
    ckpt.save_meta(tmp_path / "ck", c.cfg, {}, "f32", {})
    cfg = json.loads(js) | {"train": {"model_seed": 1}}
    tr = BcTrainer.make(json.dumps(cfg), "init", str(tmp_path / "ck"), 0, "lstm", CPU, "f32")
    assert torch.equal(first(tr.net), first(c))


# --- GPU: compile and CUDA graph replay with aux and the monitor (032-AC13, 032-R19) -------------


@pytest.mark.gpu
def test_bc_aux_compiled_graph_no_recompile_per_batch(tiny_config, shards, monkeypatch):
    """`URANIUM_COMPILE=reduce-overhead` (CUDA graph replay), bf16, aux modules and the monitor every
    2nd update: once the shapes settle a new batch compiles nothing (the eager monitor pass's
    hooks do not invalidate the compiled forward), and the losses are finite."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    import torch._dynamo.utils as du

    monkeypatch.setenv("URANIUM_COMPILE", "reduce-overhead")
    torch._dynamo.reset()
    dev = torch.device("cuda")
    tr = BcTrainer(bc_cfg(tiny_config, aux={"priv": 0.2, "jepa": 0.1}, monitor={"spatial_every": 2}), make_net(AUX), dev, "bf16")
    assert tr.graphs
    f = feeder(tiny_config, shards)
    frames = []
    for s in range(8):
        b = f.next()
        if b is None:  # the synthetic epoch is short: start it again (the same shapes come back)
            f = feeder(tiny_config, shards)
            b = f.next()
        out = json.loads(tr.train_batch(b, 3e-3, s))
        assert math.isfinite(out["total"]) and out["aux_priv"] is not None and out["aux_jepa"] is not None
        assert (set(SPEC_COLUMNS) <= out.keys()) == ((s + 1) % 2 == 0)
        frames.append(du.counters["frames"]["ok"])
    print("compiled frames after each batch", frames)
    assert frames[-1] > 0 and frames[-1] == frames[-4], f"compiled frames after each batch {frames}: a recompile after the shapes settled"
