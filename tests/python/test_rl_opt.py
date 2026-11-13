"""032-R19 ⑥ / 042 learner optimisation, CPU: the optimised learner gives the update of the learner
before it (`uranium.rl.reference`) bit for bit where the operations are the same (targets, phase
indices, core_state0, statistics, read-back order), within float rounding where the 032 §2.7
entity padding is cut (`trim_pad`, off by default); lane groups of 1, 2 and all lanes give the same update (float rounding);
`URANIUM_RL_MICRO_LANES` overrides the config's group size."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from uranium import _core
from uranium.agent.infer import leaves
from uranium.model import UraniumNet, UraniumNetConfig, action_columns, input_dims, to_labels, to_obs
from uranium.model.config import VariantConfig
from uranium.rl import reference, returns
from uranium.rl.learner import HEADS, MICRO_LANES_ENV, Learner, host_micro_lanes, state0_of
from uranium.rl.reference import ReferenceLearner

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "burn_parity"
CPU = torch.device("cpu")
WEIGHTS = [1.0, 0.3, 0.3, 0.05]


def lanes_of(seed: int, lengths: list[int], t: int):
    g = torch.Generator().manual_seed(seed)
    l = len(lengths)
    return torch.randn(l, t, generator=g), torch.randn(l, t + 1, generator=g), torch.randn(l, t, generator=g) * 0.5, torch.tensor(lengths)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_targets_are_bit_identical_to_the_reference_032_r19(seed):
    """The recursions with their step-independent terms computed up front give the per-step
    loop's values bit for bit (V-trace, UPGO, GAE; lengths 0..T, many lanes)."""
    t = 64
    lengths = [64, 63, 1, 0, 40, 64, 2, 63] * 4
    r, v, lr, length = lanes_of(seed, lengths, t)
    args = (0.99 if seed % 2 else 1.0, 0.8, 1.0, 1.0)
    for new, old in zip(returns.vtrace(r, v, lr, length, *args), reference.vtrace(r, v, lr, length, *args)):
        assert torch.equal(new, old)
    assert torch.equal(returns.upgo(r, v, lr, length, args[0], 1.0), reference.upgo(r, v, lr, length, args[0], 1.0))
    for new, old in zip(returns.gae(r, v, length, args[0], 0.95), reference.gae(r, v, length, args[0], 0.95)):
        assert torch.equal(new, old)


def tiny_net(outer: int, seed: int = 5) -> UraniumNet:
    """The tiny LSTM test model; `outer` > 0: with every 032 §2.7 variant, outer segments of that
    many decisions."""
    cfg = UraniumNetConfig.from_dict(json.loads((FIX / "tiny-lstm" / "meta.json").read_text())["config"])
    cfg.input = input_dims()
    if outer:
        cfg.variant = VariantConfig(terrain=True, ctx_token=True, attn_pool=True, outer=outer)
    torch.manual_seed(seed)
    return UraniumNet(cfg, action_columns())


class Segments:
    """`_core.test_segments` with behaviour log-probs from another network (off-policy) and a
    core_state0 on every lane but the first (a game's first segment starts from zero); for an
    outer model the outer counter puts lanes inside an outer segment at every phase and on its
    boundary (032 §2.7: 5, 4, 3, 7, ... decisions so far; segments of 2 and 3)."""

    def __init__(self, lanes: int, t: int, net, outer: int, r_value_off: tuple = (), pseudo_rewards: bool = False, pseudo_scale: tuple = (), sil_returns: tuple = (), priv_from: int = 0):
        self.batch, self.net = _core.test_segments(lanes, t, list(r_value_off), pseudo_rewards, list(pseudo_scale), list(sil_returns), priv_from), net
        g = torch.Generator().manual_seed(7)
        s = net.init_state(lanes, CPU)
        cs = [torch.randn(x.shape, generator=g) * 0.1 for x in leaves(s if not outer else s[0])]
        self.state = cs
        if outer:
            h_o, c_o, h_sum, _ = s[1]
            counter = torch.tensor([0.0] + [(5.0, 4.0, 3.0, 7.0)[(i - 1) % 4] for i in range(1, lanes)])
            self.state = (cs, (torch.randn(h_o.shape, generator=g), torch.randn(c_o.shape, generator=g), torch.randn(h_sum.shape, generator=g), counter))

    lanes = property(lambda self: self.batch.lanes)
    trained = property(lambda self: self.batch.trained)
    versions = property(lambda self: self.batch.versions)
    sil_returns = property(lambda self: self.batch.sil_returns)
    real_steps = property(lambda self: self.batch.real_steps)
    priv_trained = property(lambda self: self.batch.priv_trained)

    def group(self, start, end, t, max_select):
        g = self.batch.group(start, end, t, max_select)
        g["state0"] = [None if lane == 0 else [(tuple(x[lane:lane + 1].shape), x[lane:lane + 1].numpy().astype(np.float32).ravel()) for x in leaves(self.state)] for lane in range(start, end)]
        with torch.no_grad():
            out = self.net.forward_seq(to_obs(g, CPU), to_labels(g, CPU), torch.as_tensor(g["privileged"]).float(), state0_of(self.net, g, CPU), t, phased=True)
        g["behaviour"] = sum(out["logp"][k] for k in HEADS).numpy().astype(np.float32)
        return g


def rl_cfg(**over) -> dict:
    cfg = json.loads(_core.rl_config(str(ROOT / "config" / "rl.toml")))
    cfg.update(seq_len=4, micro_lanes=1)
    cfg["optimizer"]["lr"] = 1e-3
    cfg.update(over)
    return cfg


def run(cls, algo: str, outer: int, group: int, lanes: int = 4, updates: int = 2, trim: bool = True, r_value_off: tuple = (), weights=WEIGHTS, pseudo_rewards: bool = False, pseudo_scale: tuple = ()):
    l = cls(rl_cfg(lanes=lanes), tiny_net(outer), tiny_net(outer), CPU, "f32", algo, micro_lanes=group)
    l.trim_pad = trim
    batch = Segments(lanes, 4, tiny_net(outer, seed=6), outer, r_value_off, pseudo_rewards, pseudo_scale)
    stats = [l.update(batch, weights) for _ in range(updates)]
    grads = {k: p.grad.clone() for k, p in l.net.named_parameters() if p.grad is not None}
    return stats, {k: v.detach().clone() for k, v in l.net.state_dict().items()}, grads


@pytest.mark.parametrize("algo", ["vtrace", "ppo"])
@pytest.mark.parametrize("outer", [0, 2, 3])
def test_update_is_bit_identical_to_the_reference_032_r19(algo, outer):
    """Without the padding cut, the optimised update (host core_state0 and phase indices, staged
    groups, targets of the 4 components in one recursion, device-summed statistics, BC forward
    before the targets) equals the reference update bit for bit: statistics, gradients and
    weights after two updates (the second off-policy)."""
    new, ref = run(Learner, algo, outer, 1, trim=False), run(ReferenceLearner, algo, outer, 1)
    for s_new, s_ref in zip(new[0], ref[0]):
        assert s_new == s_ref
    for i in (1, 2):
        assert new[i].keys() == ref[i].keys()
        for k in new[i]:
            assert torch.equal(new[i][k], ref[i][k]), k


@pytest.mark.parametrize("algo", ["vtrace", "ppo"])
def test_r_value_lane_weight_matches_the_reference_amb_042_14(algo):
    """042 "r_value의 게임별 적용" (AMB-042-14): with r_value off in some lanes' games, the
    learner's update equals the reference's bit for bit; it differs from the all-on update, and
    all lanes off equals r_value weight 0 for the whole batch."""
    off = (1, 2)
    new, ref = run(Learner, algo, 0, 1, trim=False, r_value_off=off), run(ReferenceLearner, algo, 0, 1, r_value_off=off)
    assert new[0] == ref[0]
    for k in new[1]:
        assert torch.equal(new[1][k], ref[1][k]), k
    on = run(Learner, algo, 0, 1, trim=False)
    assert any(not torch.equal(on[1][k], new[1][k]) for k in on[1]), "r_value off changed the update"
    all_off = run(Learner, algo, 0, 1, trim=False, r_value_off=(0, 1, 2, 3))
    zero = run(Learner, algo, 0, 1, trim=False, weights=WEIGHTS[:3] + [0.0])
    assert all_off[0] == zero[0]
    for k in zero[1]:
        assert torch.equal(all_off[1][k], zero[1][k]), k


@pytest.mark.parametrize("algo", ["vtrace", "ppo"])
def test_r_value_lane_weight_per_lane_targets_amb_042_14(algo):
    """The targets of a lane group with r_value off in some lanes: those lanes' policy
    coefficients are the ones with r_value weight 0, the other lanes' the full ones; the value
    targets are the same in every case (the value head learns r_value in every game, 033-R3)."""
    t, lanes = 4, 4
    l = Learner(rl_cfg(lanes=lanes), tiny_net(0), tiny_net(0), CPU, "f32", algo, micro_lanes=0)
    g_on = Segments(lanes, t, tiny_net(0, seed=6), 0).group(0, lanes, t, 12)
    g_mix = Segments(lanes, t, tiny_net(0, seed=6), 0, r_value_off=(0, 3)).group(0, lanes, t, 12)
    assert g_on["r_value_policy"].tolist() == [True] * 4 and g_mix["r_value_policy"].tolist() == [False, True, True, False]
    gen = torch.Generator().manual_seed(3)
    logp, values = torch.randn(lanes * t, generator=gen) - 2.0, torch.randn(lanes * t, 4, generator=gen)
    full, mix, zero = l.targets(g_on, logp, values, WEIGHTS), l.targets(g_mix, logp, values, WEIGHTS), l.targets(g_on, logp, values, WEIGHTS[:3] + [0.0])
    rows = lambda x, ls: x.view(lanes, t)[ls]  # noqa: E731
    assert torch.equal(rows(mix["coef"], [0, 3]), rows(zero["coef"], [0, 3]))
    assert torch.equal(rows(mix["coef"], [1, 2]), rows(full["coef"], [1, 2]))
    assert not torch.equal(rows(full["coef"], [0, 3]), rows(zero["coef"], [0, 3]))
    assert torch.equal(mix["vs"], full["vs"]) and torch.equal(zero["vs"], full["vs"])
    # Without the per-lane array (a group built elsewhere): every lane on.
    del g_on["r_value_policy"]
    assert torch.equal(l.targets(g_on, logp, values, WEIGHTS)["coef"], full["coef"])


SCOUT_WEIGHTS = [1.0, 0.15, 0.1, 0.05]
# 033-R11 "scout" (AMB-042-16): lanes 1 and 2 are games against a model (r_scout x0.5, r_survive x0).
MIXED_SCALE = ((1.0, 1.0), (0.5, 0.0), (0.5, 0.0), (1.0, 1.0))


@pytest.mark.parametrize("algo", ["vtrace", "ppo"])
def test_pseudo_lane_scale_matches_the_reference_033_r11(algo):
    """033-R11 / 042 (AMB-042-16): with build / units multipliers per lane, the learner's update
    equals the reference's bit for bit; it differs from the unscaled update; every multiplier 1
    equals no multiplier array (the "z" mode and bot games), bit for bit."""
    kw = dict(trim=False, weights=SCOUT_WEIGHTS, pseudo_rewards=True)
    new, ref = run(Learner, algo, 0, 1, pseudo_scale=MIXED_SCALE, **kw), run(ReferenceLearner, algo, 0, 1, pseudo_scale=MIXED_SCALE, **kw)
    assert new[0] == ref[0]
    for k in new[1]:
        assert torch.equal(new[1][k], ref[1][k]), k
    plain = run(Learner, algo, 0, 1, **kw)
    assert any(not torch.equal(plain[1][k], new[1][k]) for k in plain[1]), "the multipliers changed the update"
    ones = run(Learner, algo, 0, 1, pseudo_scale=((1.0, 1.0),) * 4, **kw)
    assert ones[0] == plain[0]
    for i in (1, 2):
        for k in plain[i]:
            assert torch.equal(ones[i][k], plain[i][k]), k
    # Zero multipliers on every lane = channel weights 0 for the whole batch.
    zero = run(Learner, algo, 0, 1, pseudo_scale=((0.0, 0.0),) * 4, **kw)
    off = run(Learner, algo, 0, 1, trim=False, weights=[1.0, 0.0, 0.0, 0.05], pseudo_rewards=True)
    assert zero[0] == off[0]
    for k in off[1]:
        assert torch.equal(zero[1][k], off[1][k]), k


@pytest.mark.parametrize("algo", ["vtrace", "ppo"])
def test_pseudo_lane_scale_per_lane_targets_033_r11(algo):
    """The targets of a lane group with build / units multipliers: lanes at 0 have the policy
    coefficients of channel weights 0, lanes at 1 the full ones, a lane at 0.5 the half weight; the
    value targets are the same in every case (the value heads learn the raw values in every game)."""
    t, lanes = 4, 4
    l = Learner(rl_cfg(lanes=lanes), tiny_net(0), tiny_net(0), CPU, "f32", algo, micro_lanes=0)
    seg = lambda scale: Segments(lanes, t, tiny_net(0, seed=6), 0, pseudo_rewards=True, pseudo_scale=scale).group(0, lanes, t, 12)  # noqa: E731
    g_on, g_mix = seg(()), seg(((1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (0.5, 1.0)))
    assert g_on["pseudo_scale"].tolist() == [[1.0, 1.0]] * 4 and g_mix["pseudo_scale"].tolist() == [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.5, 1.0]]
    gen = torch.Generator().manual_seed(3)
    logp, values = torch.randn(lanes * t, generator=gen) - 2.0, torch.randn(lanes * t, 4, generator=gen)
    w = SCOUT_WEIGHTS
    full, mix, zero, half = (l.targets(g_on, logp, values, w), l.targets(g_mix, logp, values, w), l.targets(g_on, logp, values, [w[0], 0.0, 0.0, w[3]]), l.targets(g_on, logp, values, [w[0], w[1] * 0.5, w[2], w[3]]))
    rows = lambda x, ls: x.view(lanes, t)[ls]  # noqa: E731
    assert torch.equal(rows(mix["coef"], [1, 2]), rows(zero["coef"], [1, 2]))
    assert torch.equal(rows(mix["coef"], [0]), rows(full["coef"], [0]))
    torch.testing.assert_close(rows(mix["coef"], [3]), rows(half["coef"], [3]), rtol=1e-6, atol=1e-7)
    assert not torch.equal(rows(full["coef"], [1, 2]), rows(zero["coef"], [1, 2]))
    assert torch.equal(mix["vs"], full["vs"]) and torch.equal(zero["vs"], full["vs"])
    # Without the per-lane array (a group built elsewhere): every multiplier 1.
    del g_on["pseudo_scale"]
    assert torch.equal(l.targets(g_on, logp, values, w)["coef"], full["coef"])


def close_stats(a: dict, b: dict, rtol: float, atol: float) -> None:
    for k in ("loss", "policy", "value", "kl", "win_open", "rho_mean", "adv_mean", "grad_norm"):
        assert a[k] == pytest.approx(b[k], rel=rtol, abs=atol), k
    np.testing.assert_allclose(a["kl_heads"], b["kl_heads"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(a["entropy_heads"], b["entropy_heads"], rtol=rtol, atol=atol)


def test_entity_pad_cut_keeps_the_update_032_r19():
    """With the 032 §2.7 padding cut (outer model: 2 extra tokens, the synthetic games' last 2
    slots are padding), the update equals the reference within float rounding."""
    l = Learner(rl_cfg(lanes=4), tiny_net(2), tiny_net(2), CPU, "f32", "vtrace", micro_lanes=1)
    batch = Segments(4, 4, tiny_net(2, seed=6), 2)
    # Off by default (no measured gain, 032-R19).
    assert not l.trim_pad
    assert l.prepare(batch, (0, 1), 4)["entity_mask"].shape[1] == batch.group(0, 1, 4, 12)["entity_mask"].shape[1]
    l.trim_pad = True
    assert l.prepare(batch, (0, 1), 4)["entity_mask"].shape[1] == batch.group(0, 1, 4, 12)["entity_mask"].shape[1] - 2, "no padding cut"
    new, ref = run(Learner, "vtrace", 2, 1), run(ReferenceLearner, "vtrace", 2, 1)
    for s_new, s_ref in zip(new[0], ref[0]):
        close_stats(s_new, s_ref, 1e-5, 1e-6)
    for k in new[1]:
        assert torch.allclose(new[1][k], ref[1][k], atol=1e-6, rtol=1e-5), k


@pytest.mark.parametrize("outer", [0, 2])
def test_lane_groups_give_the_same_update_042(outer):
    """042 learner lane groups (032-R19 ②): groups of 1, 2 and all lanes give the same statistics,
    gradient norm, gradients and weights after one Adam step, up to the order of the sums."""
    res = {m: run(Learner, "vtrace", outer, m, lanes=4, updates=1) for m in (1, 2, 0)}
    for m in (2, 0):
        close_stats(res[m][0][0], res[1][0][0], 1e-5, 1e-6)
        for k in res[1][2]:
            assert torch.allclose(res[m][2][k], res[1][2][k], atol=1e-6, rtol=1e-4), k
        for k in res[1][1]:
            assert torch.allclose(res[m][1][k], res[1][1][k], atol=1e-6, rtol=1e-5), k


def test_micro_lanes_override_042(monkeypatch):
    """`URANIUM_RL_MICRO_LANES` (001-R9) over the config's `micro_lanes`; unset or empty = the
    config; 0 = all lanes; anything else than a lane count is an error."""
    cfg = rl_cfg(micro_lanes=1)
    monkeypatch.delenv(MICRO_LANES_ENV, raising=False)
    assert host_micro_lanes(cfg) == 1
    monkeypatch.setenv(MICRO_LANES_ENV, " ")
    assert host_micro_lanes(cfg) == 1
    monkeypatch.setenv(MICRO_LANES_ENV, "4")
    assert host_micro_lanes(cfg) == 4
    l = Learner(cfg, tiny_net(0), tiny_net(0), CPU, "f32", "vtrace")
    assert l.micro_lanes == 4 and l.group_size(32) == 4 and l.group_size(3) == 3
    monkeypatch.setenv(MICRO_LANES_ENV, "0")
    assert Learner(cfg, tiny_net(0), tiny_net(0), CPU, "f32", "vtrace").group_size(32) == 32
    for bad in ("-1", "two", "1.5"):
        monkeypatch.setenv(MICRO_LANES_ENV, bad)
        with pytest.raises(ValueError, match=MICRO_LANES_ENV):
            host_micro_lanes(cfg)


def test_spatial_bf16_staging_is_bit_identical_for_the_rl_model_032_r19():
    """The learner's bf16 spatial staging (bf16 autocast) on the 032 §2.7 model (terrain cells
    gathered from the planes, phased outer forward): log-probs, values and gradients equal those
    of f32 planes rounded by autocast (CPU autocast stands in for the CUDA host)."""
    from uranium.device import autocast
    from uranium.model.net import stage

    seg = Segments(2, 4, tiny_net(2, seed=6), 2)
    g = seg.group(0, 2, 4, 12)
    res = []
    for hg in (g, stage(g, CPU, "bf16", pinned=False, spatial_bf16=True)):
        net = tiny_net(2)
        with autocast(CPU, "bf16"):
            out = net.forward_seq(to_obs(hg, CPU), net.labels(hg, CPU), torch.as_tensor(g["privileged"]).float(), state0_of(net, g, CPU), 4, phased=True)
        (sum(out["logp"][k].float().sum() for k in HEADS) + out["value"].float().sum()).backward()
        res.append((torch.stack([out["logp"][k].float() for k in HEADS]).detach(), out["value"].float().detach(), {k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None}))
    assert res[1][0].dtype == torch.float32 and torch.equal(res[0][0], res[1][0]) and torch.equal(res[0][1], res[1][1])
    assert res[0][2].keys() == res[1][2].keys()
    for k, v in res[0][2].items():
        assert torch.equal(res[1][2][k], v), k


def test_phase_split_on_the_host_matches_the_device_split_032_r19():
    """`phase_split` gives the same row indices from a host counter as the phased forward computes
    from the state's counter (the RL learner passes the host ones)."""
    from uranium.model.net import phase_split

    counter = torch.tensor([0.0, 3.0, 31.0, 32.0, 45.0, 7.0])
    got = phase_split(counter, 64, 32)
    assert len(got) == 3 and torch.equal(torch.cat(got).sort().values, torch.arange(6 * 64))
    # Row r's first segment holds 32 - counter mod 32 steps (all 32 on a boundary), in step order.
    rows = lambda idx: {int(r): int(n) for r, n in zip(*torch.unique(torch.div(idx, 64, rounding_mode="floor"), return_counts=True))}  # noqa: E731
    assert rows(got[0]) == {0: 32, 1: 29, 2: 1, 3: 32, 4: 19, 5: 25}
    assert rows(got[1]) == {r: 32 for r in range(6)}
    assert rows(got[2]) == {1: 3, 2: 31, 4: 13, 5: 7}
    assert all(bool((torch.diff(x) > 0).all()) for x in got)


def test_relative_logs_become_absolute_042(tmp_path, monkeypatch):
    """042 `--logs` fix: a relative `--logs` is resolved against the working directory (the
    inference servers' model folders under it go to the Rust Broker); an absolute one is kept."""
    from uranium.rl.run import logs_dir

    monkeypatch.chdir(tmp_path)
    assert logs_dir("runs/rl") == tmp_path / "runs" / "rl" and logs_dir("runs/rl").is_absolute()
    assert logs_dir(str(tmp_path / "x")) == tmp_path / "x"


def test_bench_rl_runs_both_learners_032_r19():
    """`bench rl` pieces on CPU: the synthetic batch has the Rust group's keys and dtypes (entities
    padded to 64 per group, END labels at the group's count), and the new and reference learners
    run on it with phase times; their per-update statistics agree (`bench rl-parity`)."""
    from uranium.bench.rl import SyntheticSegments, compare, run_learner
    from uranium.model.net import ENTITY_KEYS

    net = tiny_net(2, seed=6)
    batch = SyntheticSegments(net, 4, 4, (3, 70), templates=2, seed=3)
    g, ref = batch.group(0, 4, 4, 12), _core.test_segments(1, 4).group(0, 1, 4, 12)
    assert set(ref) <= set(g)
    for k, v in ref.items():
        if isinstance(v, np.ndarray):
            assert g[k].dtype == v.dtype, k
            if k not in ENTITY_KEYS:
                assert g[k].shape[1:] == v.shape[1:], k
    assert g["entity_mask"].shape[1] % 64 == 0 and g["selected"].max() <= g["n"]
    assert g["state0"][0] is None and all(s is not None for s in g["state0"][1:])
    cfg = rl_cfg(lanes=4)
    runs = [run_learner(kind, m, net, cfg, batch, CPU, "f32", "vtrace", 1, True, log=lambda m: None, trim=trim) for kind, m, trim in (("reference", 1, False), ("new", 1, True), ("new", 2, False))]
    assert all(r["median_s"] is not None and "backward" in r["phase_median_s"] for r in runs)
    assert runs[1]["trim"] and not runs[2]["trim"]
    for r in runs[1:]:
        c = compare(runs[0], r)
        assert c["updates"] == 2 and c["finite"]
        assert c["loss"]["max_rel"] < 1e-4 and c["grad_norm"]["max_rel"] < 1e-4, c
    # Default (no padding cut): the new learner is the reference bit for bit (CPU).
    same = compare(runs[0], run_learner("new", 1, net, cfg, batch, CPU, "f32", "vtrace", 1, False, log=lambda m: None))
    assert all(v["max_abs"] == 0.0 for k, v in same.items() if isinstance(v, dict)), same


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["f32", "bf16"])
def test_update_on_the_device_matches_the_reference_032_r19(precision):
    """On the GPU (page-locked staging, read-back while the BC forward runs, bf16 spatial planes
    under bf16): the update equals the reference's up to GPU summation order."""
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    dev = torch.device("cuda")
    res = []
    for cls in (Learner, ReferenceLearner):
        l = cls(rl_cfg(lanes=4), tiny_net(2), tiny_net(2), dev, precision, "vtrace", micro_lanes=2)
        l.trim_pad = False
        batch = Segments(4, 4, tiny_net(2, seed=6), 2)
        stats = [l.update(batch, WEIGHTS) for _ in range(2)]
        res.append((stats, {k: v.detach().cpu() for k, v in l.net.state_dict().items()}))
    if precision == "bf16":
        assert Learner(rl_cfg(), tiny_net(2), tiny_net(2), dev, precision, "vtrace").spatial_bf16
    for a, b in zip(res[0][0], res[1][0]):
        close_stats(a, b, 1e-3, 1e-5)
    for k in res[0][1]:
        assert torch.allclose(res[0][1][k], res[1][1][k], atol=1e-5, rtol=1e-4), k


@pytest.mark.parametrize("algo", ["vtrace", "ppo"])
def test_update_on_sampled_actions_is_bit_identical_to_the_reference_032_r19(algo):
    """The same on `bench rl`'s synthetic batch, whose actions the network sampled (selection,
    target unit and location heads used, entity counts padded per group, outer counters at varied
    phases): statistics and weights after two updates equal the reference's bit for bit."""
    from uranium.bench.rl import SyntheticSegments

    batch = SyntheticSegments(tiny_net(3, seed=6), 6, 4, (3, 120), templates=3, seed=4)
    res = []
    for cls in (Learner, ReferenceLearner):
        l = cls(rl_cfg(lanes=6), tiny_net(3), tiny_net(3), CPU, "f32", algo, micro_lanes=2)
        l.trim_pad = False
        res.append(([l.update(batch, WEIGHTS) for _ in range(2)], {k: v.detach().clone() for k, v in l.net.state_dict().items()}))
    used = res[0][0][0]["entropy_heads"]
    assert used[3] > 0 and (used[4] > 0 or used[5] > 0), f"selection / target heads unused: {used}"
    assert res[0][0] == res[1][0]
    for k in res[0][1]:
        assert torch.equal(res[0][1][k], res[1][1][k]), k


# --- AMB-042-17: KL run settings and head multipliers (042-R9), 진단 로그 (042-R2), 자기 모방 (042-R16) ---


def run_cfg(cfg: dict, batch, cls=Learner, algo: str = "vtrace", group: int = 1, updates: int = 1, outer: int = 0):
    l = cls(cfg, tiny_net(outer), tiny_net(outer), CPU, "f32", algo, micro_lanes=group)
    stats = [l.update(batch, WEIGHTS) for _ in range(updates)]
    grads = {k: p.grad.clone() for k, p in l.net.named_parameters() if p.grad is not None}
    return stats, {k: v.detach().clone() for k, v in l.net.state_dict().items()}, grads


def with_kl(cfg: dict, **kl) -> dict:
    cfg = json.loads(json.dumps(cfg))
    cfg["kl"].update(kl)
    return cfg


DIAG_KEYS = ("adv_rms", "adv_rms_channels", "value_loss_channels", "kl_p99", "kl_max", "sil_loss", "sil_pos_frac")


@pytest.mark.parametrize("algo", ["vtrace", "ppo"])
@pytest.mark.parametrize("outer", [0, 2])
def test_diagnostics_match_the_reference_042_r2(algo, outer):
    """042-R2 진단 로그 / 042-AC7 (AMB-042-17): on a small CPU batch the advantage RMS (combined and
    per channel), the per-channel value loss and the per-step KL p99 / max equal the reference
    learner's (`reference.py`, trained steps picked out on the host) within float tolerance; they
    are consistent with the other statistics (value = weight x Σ channels, RMS >= |mean|,
    p99 <= max); the SIL statistics are empty with `[sil]` off."""
    cfg = rl_cfg(lanes=4)
    new = run_cfg(cfg, Segments(4, 4, tiny_net(outer, seed=6), outer), algo=algo, updates=2, outer=outer)
    ref = run_cfg(cfg, Segments(4, 4, tiny_net(outer, seed=6), outer), cls=ReferenceLearner, algo=algo, updates=2, outer=outer)
    for a, b in zip(new[0], ref[0]):
        for k in DIAG_KEYS:
            assert k in a and k in b, k
        assert a["adv_rms"] == pytest.approx(b["adv_rms"], rel=1e-6, abs=1e-9)
        np.testing.assert_allclose(a["adv_rms_channels"], b["adv_rms_channels"], rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(a["value_loss_channels"], b["value_loss_channels"], rtol=1e-6, atol=1e-9)
        assert a["kl_p99"] == pytest.approx(b["kl_p99"], rel=1e-6, abs=1e-9) and a["kl_max"] == pytest.approx(b["kl_max"], rel=1e-6, abs=1e-9)
        assert (a["sil_loss"], a["sil_pos_frac"]) == (None, None)
        assert a["value"] == pytest.approx(cfg["loss"]["value"] * sum(a["value_loss_channels"]), rel=1e-5)
        assert a["adv_rms"] >= abs(a["adv_mean"]) and min(a["adv_rms_channels"]) >= 0
        assert 0 <= a["kl_p99"] <= a["kl_max"]
        assert len(a["adv_rms_channels"]) == len(a["value_loss_channels"]) == 4
    # The rewards of the test segments: win and value channels move.
    assert new[0][0]["adv_rms_channels"][0] > 0 and new[0][0]["adv_rms_channels"][3] > 0


def test_diagnostics_by_hand_042_r2():
    """042-R2 진단 로그 on one lane group, by hand: each channel's squared PG advantage (from
    `targets`) equals the coefficient of that channel alone without UPGO; the KL p99 / max of a
    known per-step vector (NaN = untrained step)."""
    from uranium.rl.learner import kl_tail

    t, lanes = 4, 4
    l = Learner(rl_cfg(lanes=lanes), tiny_net(0), tiny_net(0), CPU, "f32", "vtrace", micro_lanes=0)
    g = Segments(lanes, t, tiny_net(0, seed=6), 0).group(0, lanes, t, 12)
    gen = torch.Generator().manual_seed(4)
    logp, values = torch.randn(lanes * t, generator=gen) - 2.0, torch.randn(lanes * t, 4, generator=gen)
    tg = l.targets(g, logp, values, WEIGHTS)
    valid = torch.as_tensor(g["mask"]).view(lanes, t) > 0
    cfg_pg = json.loads(json.dumps(l.cfg))
    cfg_pg["loss"]["upgo"] = 0.0
    lp = Learner(cfg_pg, tiny_net(0), tiny_net(0), CPU, "f32", "vtrace", micro_lanes=0)
    for c in range(4):
        w = [0.0] * 4
        w[c] = 1.0
        adv = lp.targets(g, logp, values, w)["coef"].view(lanes, t)
        assert tg["adv_sq_c"][c] == pytest.approx(float((adv[valid] ** 2).sum()), rel=1e-6, abs=1e-9)
    x = torch.tensor([0.0, 1.0, 2.0, 3.0, float("nan"), 100.0])
    p99, mx = kl_tail([x[:3], x[3:]], CPU).tolist()
    assert p99 == pytest.approx(float(torch.quantile(torch.tensor([0.0, 1.0, 2.0, 3.0, 100.0]), 0.99))) and mx == 100.0
    assert not any(np.isfinite(v) for v in kl_tail([torch.tensor([float("nan")])], CPU).tolist()), "no trained step: None in the stats"


def test_kl_schedule_is_a_run_setting_042_r9():
    """042-R9 (AMB-042-17): the coefficient from the run's `[kl]` (rush-s3 0.03 / 0.01 / 1,500),
    `init` at a new learner's update 0, halved every `half_life_updates`, then the floor; the
    learner reports it per update and a resumed learner goes on from its update count."""
    from uranium.rl.learner import kl_coef

    cfg = with_kl(rl_cfg(lanes=2), init=0.03, floor=0.01, half_life_updates=1500)
    assert kl_coef(cfg, 0) == pytest.approx(0.03)
    assert kl_coef(cfg, 1500) == pytest.approx(0.015)
    assert kl_coef(cfg, 2000) == pytest.approx(0.03 * 0.5 ** (2000 / 1500))
    assert kl_coef(cfg, 3000) == pytest.approx(0.01) and kl_coef(cfg, 10**7) == pytest.approx(0.01)
    l = Learner(cfg, tiny_net(0), tiny_net(0), CPU, "f32", "vtrace", micro_lanes=1)
    batch = Segments(2, 4, tiny_net(0, seed=6), 0)
    assert [l.update(batch, WEIGHTS)["kl_coef"] for _ in range(2)] == [pytest.approx(0.03), pytest.approx(0.03 * 0.5 ** (1 / 1500))]
    assert Learner(cfg, tiny_net(0), tiny_net(0), CPU, "f32", "vtrace", update=1500).update(batch, WEIGHTS)["kl_coef"] == pytest.approx(0.015)


@pytest.mark.parametrize("algo", ["vtrace", "ppo"])
def test_kl_head_scale_042_r9(algo):
    """042-R9 / 042-AC7 (AMB-042-17): multipliers all 1.0 (or absent) give the earlier update bit
    for bit; other multipliers multiply each head's KL term (the learner equals the reference
    with the same multipliers bit for bit, and differs from the unscaled update); every
    multiplier 0.5 is the coefficient halved (float rounding apart); `kl` / `kl_heads` stay
    unscaled, `loss` uses the scaled sum."""
    from uranium.rl.learner import kl_head_scale

    base = rl_cfg(lanes=4)
    seg = lambda: Segments(4, 4, tiny_net(0, seed=6), 0)  # noqa: E731
    assert kl_head_scale(base) is None
    absent = json.loads(json.dumps(base))
    del absent["kl"]["head_scale"]
    assert kl_head_scale(absent) is None
    plain, none = run_cfg(base, seg(), algo=algo, updates=2), run_cfg(absent, seg(), algo=algo, updates=2)
    assert plain[0] == none[0]
    for k in plain[1]:
        assert torch.equal(plain[1][k], none[1][k]), k
    scale = {"action": 2.0, "delay": 1.0, "queued": 0.0, "selected": 0.5, "target_unit": 1.0, "target_loc": 3.0}
    scaled = with_kl(base, head_scale=scale)
    assert kl_head_scale(scaled) == [2.0, 1.0, 0.0, 0.5, 1.0, 3.0]
    new, ref = run_cfg(scaled, seg(), algo=algo, updates=2), run_cfg(scaled, seg(), cls=ReferenceLearner, algo=algo, updates=2)
    assert new[0] == ref[0]
    for k in new[1]:
        assert torch.equal(new[1][k], ref[1][k]), k
    assert any(not torch.equal(new[2][k], plain[2][k]) for k in plain[2]), "the multipliers changed the gradients"
    s0 = new[0][0]
    assert s0["kl"] == plain[0][0]["kl"] and s0["kl_heads"] == plain[0][0]["kl_heads"], "first update: same weights, unscaled KL"
    bonus = sum(base["entropy"][h] * s0["entropy_heads"][i] for i, h in enumerate(HEADS))
    want = s0["policy"] + s0["value"] + s0["win_open"] + sum(scale[h] * x for h, x in zip(HEADS, s0["kl_heads"])) * s0["kl_coef"] - bonus
    assert s0["loss"] == pytest.approx(want, rel=1e-6)
    half = run_cfg(with_kl(base, head_scale={h: 0.5 for h in HEADS}), seg(), algo=algo)
    coef = run_cfg(with_kl(base, init=base["kl"]["init"] * 0.5, floor=base["kl"]["floor"] * 0.5), seg(), algo=algo)
    for k in half[2]:
        assert torch.allclose(half[2][k], coef[2][k], rtol=1e-5, atol=1e-7), k


def sil_cfg(mode: str = "sil", **over) -> dict:
    cfg = rl_cfg(lanes=4)
    cfg["optimizer"]["grad_clip"] = 1e9  # no clip: the gradients add up (042-R16 checks)
    cfg["sil"] = {**cfg["sil"], "enabled": True, "mode": mode, **over}
    return cfg


class FirstLanes:
    """The first `k` lanes of a `Segments` batch (same lanes, same core states), no buffer lanes."""

    def __init__(self, seg: Segments, k: int):
        self.seg, self.lanes = seg, k

    trained = property(lambda self: self.seg.trained[:self.lanes])
    versions = property(lambda self: self.seg.versions[:self.lanes])
    sil_returns = property(lambda self: [None] * self.lanes)

    def group(self, start, end, t, max_select):
        return self.seg.group(start, end, t, max_select)


def sil_by_hand(seg: Segments, start: int, end: int, rets: list, w_pg: float, w_v: float) -> tuple:
    """042-R16 "sil" loss of lanes `start..end` on a fresh copy of the learner's start weights:
    (loss, A+ > 0 share of the trained steps, gradients, V_win)."""
    t = 4
    net = tiny_net(0)
    g = seg.group(start, end, t, 12)
    out = net.forward_seq(to_obs(g, CPU), net.labels(g, CPU), torch.as_tensor(g["privileged"]).float(), state0_of(net, g, CPU), t, phased=True)
    mask = torch.as_tensor(g["mask"]).float()
    logp = sum(out["logp"][k].float() for k in HEADS)
    v = out["value"].float()[:, 0]
    r = torch.tensor(rets).repeat_interleave(t)
    a_plus = torch.clamp(r - v, min=0.0)
    n = mask.sum()
    loss = w_pg * (-(logp * a_plus.detach()) * mask).sum() / n + w_v * (0.5 * a_plus**2 * mask).sum() / n
    loss.backward()
    frac = float(((a_plus > 0).float() * mask).sum() / n)
    return float(loss.detach()), frac, {k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None}, v.detach()


@pytest.mark.parametrize("group", [1, 2])
def test_sil_lanes_042_r16(group):
    """042-R16 / 042-AC7 (AMB-042-17) "sil": buffer lanes (last) leave every other loss term as on
    the queue lanes alone (same statistics), and add exactly the self-imitation loss
    `w_pg · mean[−log π · A+] + w_v · mean[0.5 (A+)²]`, `A+ = max(R − V_win, 0)` (gradients =
    queue-only + the by-hand loss's; `sil_loss` and `sil_pos_frac` as by hand)."""
    cfg = sil_cfg(w_pg=0.05, w_v=0.01)
    rets = [1.0, 1.0]
    seg = Segments(4, 4, tiny_net(0, seed=6), 0, sil_returns=(None, None, *rets))
    assert list(seg.sil_returns) == [None, None, 1.0, 1.0]
    g = seg.group(0, 4, 4, 12)
    assert np.isnan(g["sil_return"][:2]).all() and g["sil_return"][2:].tolist() == rets
    full = run_cfg(cfg, seg, group=group)
    only = run_cfg(cfg, FirstLanes(seg, 2), group=group)
    a, b = full[0][0], only[0][0]
    for k in ("policy", "value", "kl", "win_open", "kl_heads", "entropy_heads", "rho_mean", "adv_mean", "steps", "adv_rms", "value_loss_channels", "kl_p99", "kl_max", "lag_mean"):
        assert a[k] == b[k], k
    loss, frac, grads, _ = sil_by_hand(seg, 2, 4, rets, 0.05, 0.01)
    assert a["sil_loss"] == pytest.approx(loss, rel=1e-5) and loss > 0
    assert a["sil_pos_frac"] == pytest.approx(frac) and b["sil_loss"] == 0.0 and b["sil_pos_frac"] is None
    assert a["loss"] == pytest.approx(b["loss"] + loss, rel=1e-5)
    for k in full[2]:
        torch.testing.assert_close(full[2][k], only[2][k] + grads.get(k, torch.zeros_like(full[2][k])), rtol=1e-4, atol=1e-7, msg=k)


def test_sil_loss_is_zero_without_positive_advantage_042_r16():
    """042-R16: with A+ = 0 everywhere (lost games, R = −1 below every V_win) the "sil" loss is 0
    and the update is the queue lanes' alone."""
    rets = [-1.0, -1.0]
    seg = Segments(4, 4, tiny_net(0, seed=6), 0, sil_returns=(None, None, *rets))
    _, frac, _, v = sil_by_hand(seg, 2, 4, rets, 0.05, 0.01)
    assert float(v.min()) > -1.0 and frac == 0.0, "precondition: V_win > -1"
    cfg = sil_cfg()
    full, only = run_cfg(cfg, seg), run_cfg(cfg, FirstLanes(seg, 2))
    assert full[0][0]["sil_loss"] == 0.0 and full[0][0]["sil_pos_frac"] == 0.0
    for k in full[2]:
        assert torch.equal(full[2][k], only[2][k]), k


@pytest.mark.parametrize("algo", ["vtrace", "ppo"])
def test_sil_replay_mode_042_r16(algo):
    """042-R16 "replay": buffer lanes are ordinary lanes (the update equals the same lanes without
    the buffer mark bit for bit, apart from the queue-only policy lag and `steps`, which counts the
    queue lanes only like the run loop's `trained_steps`); `sil_loss` is empty and `sil_pos_frac`
    is the share of the buffer lanes' steps with R − V_win > 0."""
    rets = [1.0, -1.0]
    seg = Segments(4, 4, tiny_net(0, seed=6), 0, sil_returns=(None, None, *rets))
    plain = Segments(4, 4, tiny_net(0, seed=6), 0)
    cfg = sil_cfg("replay")
    a, b = run_cfg(cfg, seg, algo=algo), run_cfg(cfg, plain, algo=algo)
    sa, sb = a[0][0], b[0][0]
    for k in ("loss", "policy", "value", "kl", "kl_heads", "entropy_heads", "rho_mean", "adv_mean", "adv_rms", "kl_p99"):
        assert sa[k] == sb[k], k
    assert sa["steps"] == sum(list(seg.trained)[:2]) and sb["steps"] == sum(list(plain.trained))
    for k in a[1]:
        assert torch.equal(a[1][k], b[1][k]), k
    assert sa["sil_loss"] is None and sb["sil_pos_frac"] is None
    _, _, _, v = sil_by_hand(seg, 2, 4, rets, 0.0, 0.0)
    mask = torch.as_tensor(seg.group(2, 4, 4, 12)["mask"]) > 0
    want = float(((torch.tensor(rets).repeat_interleave(4) - v) > 0)[mask].float().mean())
    assert sa["sil_pos_frac"] == pytest.approx(want)


def test_sil_off_keeps_the_update_042_r16():
    """042-R16: `[sil] enabled = false` (the default) — no buffer lanes, the update and statistics
    as before (the reference's bit for bit), SIL statistics empty; buffer lanes must come last."""
    from uranium.rl.learner import sil_lane_count

    cfg = rl_cfg(lanes=4)
    assert cfg["sil"]["enabled"] is False
    new, ref = run_cfg(cfg, Segments(4, 4, tiny_net(0, seed=6), 0), updates=2), run_cfg(cfg, Segments(4, 4, tiny_net(0, seed=6), 0), cls=ReferenceLearner, updates=2)
    assert new[0] == ref[0]
    assert all((s["sil_loss"], s["sil_pos_frac"]) == (None, None) for s in new[0])
    bad = Segments(4, 4, tiny_net(0, seed=6), 0, sil_returns=(None, 1.0, None, 1.0))
    with pytest.raises(ValueError, match="042-R16"):
        sil_lane_count(bad)
    assert sil_lane_count(Segments(3, 4, tiny_net(0, seed=6), 0, sil_returns=(None, 1.0, 1.0))) == 2
