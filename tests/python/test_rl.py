"""042 RL learner in PyTorch (AMB-000-4), CPU: the batched return targets equal the Rust
reference (`uranium_league::returns`), and the learner's update behaves like the Burn one it
replaces (trained steps, KL to itself 0, on-policy ρ 1, weights change, PPO, exploiter reset)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from uranium import _core
from uranium.model import UraniumNet, UraniumNetConfig, action_columns, input_dims, to_labels, to_obs
from uranium.rl.learner import HEADS, Learner, kl_coef
from uranium.rl.returns import gae, upgo, vtrace

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "burn_parity"
CPU = torch.device("cpu")


def lanes_of(seed: int, lengths: list[int], t: int):
    g = torch.Generator().manual_seed(seed)
    l = len(lengths)
    rewards = torch.randn(l, t, generator=g)
    values = torch.randn(l, t + 1, generator=g)
    log_rho = torch.randn(l, t, generator=g) * 0.5
    return rewards, values, log_rho, torch.tensor(lengths)


def close(a, b):
    np.testing.assert_allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_targets_match_rust_reference_042(seed):
    t, lengths = 6, [6, 3, 1, 5]
    r, v, lr, length = lanes_of(seed, lengths, t)
    gamma, lam, rho_bar, c_bar = 0.99, 0.8, 1.0, 1.0
    vs, pg, rho = vtrace(r, v, lr, length, gamma, lam, rho_bar, c_bar)
    up = upgo(r, v, lr, length, gamma, rho_bar)
    adv, ret = gae(r, v, length, gamma, 0.95)
    for i, n in enumerate(lengths):
        args = (r[i, :n].tolist(), v[i, : n + 1].tolist())
        rvs, rpg, rrho = _core.vtrace(*args, lr[i, :n].tolist(), gamma, lam, rho_bar, c_bar)
        close(vs[i, :n], rvs)
        close(pg[i, :n], rpg)
        close(rho[i, :n], rrho)
        close(up[i, :n], _core.upgo(*args, lr[i, :n].tolist(), gamma, rho_bar))
        radv, rret = _core.gae(*args, gamma, 0.95)
        close(adv[i, :n], radv)
        close(ret[i, :n], rret)
        assert (vs[i, n:] == 0).all() and (pg[i, n:] == 0).all() and (up[i, n:] == 0).all()


def test_kl_schedule_042():
    cfg = json.loads(_core.rl_config(str(ROOT / "config" / "rl.toml")))
    k = cfg["kl"]
    assert kl_coef(cfg, 0) == pytest.approx(k["init"])
    assert kl_coef(cfg, int(k["half_life_updates"])) == pytest.approx(max(k["floor"], k["init"] / 2))
    assert kl_coef(cfg, 10**9) == pytest.approx(k["floor"])


def tiny_net(seed=5) -> UraniumNet:
    cfg = UraniumNetConfig.from_dict(json.loads((FIX / "tiny-lstm" / "meta.json").read_text())["config"])
    cfg.input = input_dims()
    torch.manual_seed(seed)
    return UraniumNet(cfg, action_columns())


class OnPolicy:
    """A segment batch whose behaviour log-probs are the network's own (π = μ at the start)."""

    def __init__(self, batch, net):
        self.batch, self.net = batch, net

    @property
    def lanes(self):
        return self.batch.lanes

    @property
    def trained(self):
        return self.batch.trained

    @property
    def versions(self):
        return self.batch.versions

    def group(self, start, end, t, max_select):
        g = self.batch.group(start, end, t, max_select)
        with torch.no_grad():
            out = self.net.forward_seq(to_obs(g, CPU), to_labels(g, CPU), torch.as_tensor(g["privileged"]).float(), self.net.core.init_state(g["b"], CPU), t)
        g["behaviour"] = sum(out["logp"][k] for k in HEADS).numpy().astype(np.float32)
        return g


def learner(algo: str) -> tuple[Learner, OnPolicy]:
    cfg = json.loads(_core.rl_config(str(ROOT / "config" / "rl.toml")))
    cfg.update(lanes=2, seq_len=4, micro_lanes=1)
    cfg["optimizer"]["lr"] = 1e-3
    net = tiny_net()
    bc = tiny_net()
    l = Learner(cfg, net, bc, CPU, "f32", algo)
    return l, OnPolicy(_core.test_segments(2, 4), tiny_net())


def params(net) -> dict:
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def test_vtrace_update_042():
    """Trains the right steps (terminal lane: all 4, the other: all but the bootstrap step),
    starts with KL(π‖π_BC) = 0 and ρ = 1, changes the weights, and KL grows after."""
    l, batch = learner("vtrace")
    before = params(l.net)
    s = l.update(batch, [1.0, 0.3, 0.3, 0.05])
    assert s["steps"] == 4 + 3
    assert all(np.isfinite([s["loss"], s["policy"], s["value"]])), s
    assert abs(s["kl"]) < 1e-4, f"KL to itself {s['kl']}"
    assert abs(s["rho_mean"] - 1.0) < 1e-3, f"on-policy rho {s['rho_mean']}"
    assert l.update_count == 1 and s["update"] == 1
    after = params(l.net)
    assert any(not torch.equal(before[k], after[k]) for k in before), "weights changed"
    s2 = l.update(batch, [1.0, 0.3, 0.3, 0.05])
    assert s2["kl"] > 0.0, f"KL after a step {s2['kl']}"
    assert s2["entropy_heads"][0] > 0.0


def test_ppo_update_042():
    l, batch = learner("ppo")
    s = l.update(batch, [1.0, 0.0, 0.0, 0.0])
    assert np.isfinite(s["loss"]) and abs(s["kl"]) < 1e-4, s
    assert l.update_count == 1


def test_reset_to_bc_042():
    l, batch = learner("vtrace")
    bc = params(l.bc)
    l.update(batch, [1.0, 0.0, 0.0, 0.0])
    l.reset_to_bc()
    now = params(l.net)
    assert all(torch.equal(bc[k], now[k]) for k in bc)
    assert l.update_count == 0


def test_segment_group_layout_042():
    """Rows b-major, mask on trained steps, rewards on real steps, zero state0 as None."""
    g = _core.test_segments(2, 4).group(0, 2, 6, 12)
    assert g["b"] == 2 and g["t"] == 6
    assert g["mask"].tolist() == [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0]
    assert g["rewards"].shape == (12, 4) and g["rewards"][3, 0] == 1.0 and g["rewards"][4:6].sum() == 0
    assert g["terminal"].tolist() == [True, False] and g["trained"].tolist() == [4, 3]
    assert g["state0"] == [None, None]
    assert g["entity_mask"].shape[1] % 64 == 0


def test_segment_batch_totals_match_group():
    """The learner's step total and versions come from `SegmentBatch.trained` / `versions` before
    collating (the groups are collated one ahead, overlapped with training): same as `group`."""
    b = _core.test_segments(3, 4)
    g = b.group(0, b.lanes, 4, 12)
    assert list(b.trained) == [int(x) for x in g["trained"]]
    assert list(b.versions) == [int(x) for x in g["versions"]]
