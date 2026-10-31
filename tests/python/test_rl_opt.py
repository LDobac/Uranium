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

    def __init__(self, lanes: int, t: int, net, outer: int):
        self.batch, self.net = _core.test_segments(lanes, t), net
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


def run(cls, algo: str, outer: int, group: int, lanes: int = 4, updates: int = 2, trim: bool = True):
    l = cls(rl_cfg(lanes=lanes), tiny_net(outer), tiny_net(outer), CPU, "f32", algo, micro_lanes=group)
    l.trim_pad = trim
    batch = Segments(lanes, 4, tiny_net(outer, seed=6), outer)
    stats = [l.update(batch, WEIGHTS) for _ in range(updates)]
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
    assert l.prepare(batch, 0, 1, 4)["entity_mask"].shape[1] == batch.group(0, 1, 4, 12)["entity_mask"].shape[1]
    l.trim_pad = True
    assert l.prepare(batch, 0, 1, 4)["entity_mask"].shape[1] == batch.group(0, 1, 4, 12)["entity_mask"].shape[1] - 2, "no padding cut"
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
