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


class PhasedOnPolicy(OnPolicy):
    """Lanes whose core_state0 sits inside an outer segment and on its boundary (counters 3 and 0,
    h_sum 0 there as after an update, 032 §2.7)."""

    def __init__(self, batch, net):
        super().__init__(batch, net)
        g = torch.Generator().manual_seed(7)
        cs, (h_o, c_o, h_sum, _) = net.init_state(2, CPU)
        self.state = (cs, (torch.randn(h_o.shape, generator=g), torch.randn(c_o.shape, generator=g), torch.randn(h_sum.shape, generator=g) * torch.tensor([[1.0], [0.0]]), torch.tensor([3.0, 0.0])))

    def group(self, start, end, t, max_select):
        from uranium.agent.infer import leaves
        from uranium.rl.learner import state0_of

        g = self.batch.group(start, end, t, max_select)
        g["state0"] = [[(tuple(x[lane:lane + 1].shape), x[lane:lane + 1].numpy().ravel()) for x in leaves(self.state)] for lane in range(start, end)]
        with torch.no_grad():
            out = self.net.forward_seq(to_obs(g, CPU), to_labels(g, CPU), torch.as_tensor(g["privileged"]).float(), state0_of(self.net, g, CPU), t, phased=True)
        g["behaviour"] = sum(out["logp"][k] for k in HEADS).numpy().astype(np.float32)
        return g


@pytest.mark.parametrize("algo", ["vtrace", "ppo"])
def test_outer_update_from_inside_a_segment_042(algo):
    """032 §2.7 / 042: the learner trains an outer-LSTM model on segments that start inside an
    outer segment (on-policy: KL to itself 0, ρ 1) and changes the weights."""
    from uranium.model.config import VariantConfig

    cfg = json.loads(_core.rl_config(str(ROOT / "config" / "rl.toml")))
    cfg.update(lanes=2, seq_len=4, micro_lanes=1)
    cfg["optimizer"]["lr"] = 1e-3

    def outer_net():
        net = tiny_net()
        cfg_v = net.cfg
        cfg_v.variant = VariantConfig(terrain=True, ctx_token=True, attn_pool=True, outer=2)
        torch.manual_seed(5)
        return UraniumNet(cfg_v, action_columns())

    l = Learner(cfg, outer_net(), outer_net(), CPU, "f32", algo)
    batch = PhasedOnPolicy(_core.test_segments(2, 4), outer_net())
    before = params(l.net)
    s = l.update(batch, [1.0, 0.3, 0.3, 0.05])
    assert all(np.isfinite([s["loss"], s["policy"], s["value"]])), s
    assert abs(s["kl"]) < 1e-4 and abs(s["rho_mean"] - 1.0) < 1e-3, s
    after = params(l.net)
    assert any(not torch.equal(before[k], after[k]) for k in before), "weights changed"


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


# --- 042 출발 정책 `--init` (AMB-042-13) ---------------------------------------------------


def save_ckpt(d: Path, net: UraniumNet, train: dict) -> Path:
    from uranium import ckpt

    d.mkdir(parents=True, exist_ok=True)
    ckpt.save_model(d, net)
    ckpt.save_meta(d, net.cfg, train, "f32")
    return d


def rl_ckpt(root: Path, step: int, nets: dict, update: int) -> Path:
    """An RL checkpoint folder as the run loop writes it: `agents/<name>/` and `train_state.json`."""
    from uranium import ckpt
    from uranium.rl.run import STATE_FILE

    d = root / f"ckpt-{step}"
    for name, net in nets.items():
        save_ckpt(d / ckpt.AGENTS_DIR / name, net, {"update": update, "agent": name})
    (d / STATE_FILE).write_text(json.dumps({"kind": "rl", "step": step, "update": update}), encoding="utf-8")
    return d


def make_learners(monkeypatch, tmp_path: Path, bc: Path, init, resume):
    """`Learners` as `_make` builds it, on CPU; its inference servers are stopped right away."""
    from uranium.rl import run

    monkeypatch.setattr(run, "_n_max", lambda: 0)  # no server warm-up (tiny net, CPU)
    cfg = json.loads(_core.rl_config(str(ROOT / "config" / "rl.toml")))
    cfg.update(lanes=2, seq_len=4, micro_lanes=1)
    ls = run.Learners(cfg, str(bc), None if init is None else str(init), None if resume is None else str(resume), CPU, "f32", "vtrace", tmp_path / "logs", 4, 5)
    for s in ls.servers:
        s.stop.set()
        s.join(timeout=10)
        assert s.error is None
    return ls


def same(net, other) -> bool:
    a, b = net.state_dict(), other.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def saved_train(ls, i: int, d: Path) -> dict:
    """The train JSON `Learners.save` writes for agent `i` (the run loop's fields plus the start)."""
    from uranium import ckpt
    from uranium.rl.run import AGENTS

    ls.save(i, str(d), json.dumps({"update": ls.version(i), "agent": AGENTS[i]}))
    return ckpt.read_meta(d)["train"]


def rec(frm: str, path: Path, **kw) -> dict:
    """An expected start record: the path as given and resolved (042, AMB-042-13)."""
    return {"from": frm, "path": str(path), "abs_path": str(Path(path).resolve()), **kw}


@pytest.mark.parametrize("agent_folder", [False, True])
def test_init_main_from_rl_checkpoint_042(monkeypatch, tmp_path, agent_folder):
    """042 (AMB-042-13): `--init` gives Main its start weights only (update 0, empty optimizer);
    the KL reference and the exploiters stay the BC model; the start is in the train JSON. The
    checkpoint folder and its `agents/main` both work."""
    from uranium import ckpt

    bc_net, init_net = tiny_net(5), tiny_net(9)
    bc = save_ckpt(tmp_path / "bc" / "ckpt-11", bc_net, {"step": 11})
    root = rl_ckpt(tmp_path / "ac1", 3133569, {"main": init_net, "main-exploiter": tiny_net(13), "league-exploiter": tiny_net(13)}, update=1234)
    init = root / ckpt.AGENTS_DIR / "main" if agent_folder else root
    ls = make_learners(monkeypatch, tmp_path, bc, init, None)
    main, mx, lx = ls.learners
    assert same(main.net, init_net) and not same(main.net, bc_net)
    assert same(mx.net, bc_net) and same(lx.net, bc_net)
    assert all(same(l.bc, bc_net) for l in ls.learners), "KL reference is --bc"
    assert all(l.update_count == 0 for l in ls.learners) and not main.opt.state
    t = saved_train(ls, 0, tmp_path / "save" / "main")
    assert t["start"] == rec("init", init, update=1234, step=3133569)
    assert t["update"] == 0 and t["agent"] == "main"
    assert saved_train(ls, 1, tmp_path / "save" / "mx")["start"] == rec("bc", bc)
    # An exploiter reset goes back to BC, not to --init (042).
    lx.reset_to_bc()
    assert same(lx.net, bc_net)


def test_no_init_starts_all_from_bc_042(monkeypatch, tmp_path):
    bc_net = tiny_net(5)
    bc = save_ckpt(tmp_path / "bc", bc_net, {})
    ls = make_learners(monkeypatch, tmp_path, bc, None, None)
    assert all(same(l.net, bc_net) for l in ls.learners)
    assert saved_train(ls, 0, tmp_path / "save")["start"] == rec("bc", bc)


def test_resume_ignores_init_042(monkeypatch, tmp_path):
    """042-R1 / 061 over AMB-042-13: with a LATEST every agent continues its own folder (weights,
    update, optimizer) whatever `--init` says, and the start record written there is kept."""
    from uranium import ckpt
    from uranium.rl.run import AGENTS

    bc_net, init_net = tiny_net(5), tiny_net(9)
    bc = save_ckpt(tmp_path / "bc", bc_net, {})
    first = rl_ckpt(tmp_path / "ac1", 100, {"main": init_net}, update=7)
    ls = make_learners(monkeypatch, tmp_path / "a", bc, first, None)
    resume = tmp_path / "run" / "ckpt-500"
    for i, name in enumerate(AGENTS):
        ls.learners[i].update_count = 3 + i
        ls.save(i, str(resume / ckpt.AGENTS_DIR / name), json.dumps({"update": ls.version(i), "agent": name}))
    other = rl_ckpt(tmp_path / "other", 200, {"main": tiny_net(21)}, update=9)
    again = make_learners(monkeypatch, tmp_path / "b", bc, other, resume)
    assert same(again.learners[0].net, init_net), "resumed Main, not the new --init"
    assert [l.update_count for l in again.learners] == [3, 4, 5]
    assert saved_train(again, 0, tmp_path / "save" / "main")["start"] == rec("init", first, update=7, step=100)
    assert saved_train(again, 2, tmp_path / "save" / "lx")["start"] == rec("bc", bc)


def test_init_bc_dropout_and_compat_feature_version_042(monkeypatch, tmp_path):
    """The real pair (042, AMB-042-13): a BC checkpoint with entity dropout 0.1 and an older
    load-compatible feature schema (030-R5), and an RL `--init` at dropout 0 (032 §2) and the
    current schema. Same model, so `--init` loads and Main gets its weights."""
    from uranium import ckpt

    compat = ckpt.feature_load_compat()
    if not compat:
        pytest.skip("this build has no load-compatible feature schema (030-R5)")
    bc_net = tiny_net(5)
    bc_net.cfg.entity.dropout = 0.1
    bc = save_ckpt(tmp_path / "bc", bc_net, {})
    meta = json.loads((bc / ckpt.META_FILE).read_text(encoding="utf-8"))
    meta["feature_version"] = compat[0]
    (bc / ckpt.META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    init_net = tiny_net(9)
    init_net.cfg.entity.dropout = 0.0
    root = rl_ckpt(tmp_path / "ac1", 42, {"main": init_net}, update=3)
    assert ckpt.read_meta(root / ckpt.AGENTS_DIR / "main")["feature_version"] == ckpt.schema_versions()[0] != compat[0]
    ls = make_learners(monkeypatch, tmp_path, bc, root, None)
    assert same(ls.learners[0].net, init_net)
    assert same(ls.learners[1].net, bc_net) and same(ls.learners[0].bc, bc_net)
    assert all(l.net.cfg.entity.dropout == 0.0 for l in ls.learners)


def test_init_config_mismatch_is_a_start_error_042(monkeypatch, tmp_path):
    """042 / 032-R10: an `--init` model other than the BC model fails at start."""
    bc = save_ckpt(tmp_path / "bc", tiny_net(5), {})
    cfg = tiny_net().cfg
    cfg.core.kind = "Gtrxl" if cfg.core.kind != "Gtrxl" else "Lstm"
    torch.manual_seed(3)
    root = rl_ckpt(tmp_path / "ac1", 1, {"main": UraniumNet(cfg, action_columns())}, update=1)
    with pytest.raises(ValueError, match=r"model config differs from the --bc model \(core\.kind"):
        make_learners(monkeypatch, tmp_path, bc, root, None)


def test_init_without_main_agent_is_an_error_042(monkeypatch, tmp_path):
    """A BC checkpoint (no `agents/main`) is not an `--init` (042)."""
    bc = save_ckpt(tmp_path / "bc", tiny_net(5), {})
    with pytest.raises(FileNotFoundError, match="agents/main"):
        make_learners(monkeypatch, tmp_path, bc, bc, None)
