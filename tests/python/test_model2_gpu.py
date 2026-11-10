"""032-AC13 GPU half, model-2 integration (the device tests are pytest `gpu`; the config check is CPU): every 032 §2.7 switch (terrain, ctx_token,
attn_pool, outer 32) and every §2.8 switch (bypass, spatial_preact, core_skip, aux_priv,
aux_jepa [1, 8, 32]) on together, on the full-size network (032 §6 defaults, the main-run config
`config/bc_m2_lambda_base.toml` `[model]`).

- Inference graph (032-R19, AC13): `act` replayed from one captured graph equals eager `act` over
  70 decisions (two outer updates), with a row whose entity set is empty on every other decision;
  the zero-initialised bypass and core-skip projections are made non-zero first so both paths count.
  The production path (`URANIUM_INFER_COMPILE` inside the graph) captures one graph and compiles
  nothing after its first call.
- BC training (041-R23, 041-R25, 032-R19): `URANIUM_COMPILE=reduce-overhead` (CUDA graph replay)
  with the CUDA training defaults, aux losses and the dead-channel monitor every 2nd update: once the
  shapes settle a new batch compiles nothing, and the losses are finite."""

import json
import math
from pathlib import Path

import pytest
import torch

from uranium import _core
from uranium.agent.infer import GraphedAct, leaves, rebuild
from uranium.bc.trainer import BcTrainer, new_net_config
from uranium.bench.inputs import random_obs
from uranium.device import IMPL_DEFAULTS, TRAIN_DEFAULTS, autocast, compiled, resolve
from uranium.model import UraniumNet, action_columns
from uranium.model.config import VariantConfig
from uranium.model.monitor import SPEC_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
# 032 §2.7 four (main BC model) + every 032 §2.8 switch, as config/bc_m2_lambda_base.toml.
ALL_ON = VariantConfig(terrain=True, ctx_token=True, attn_pool=True, outer=32, bypass=True, spatial_preact=True, core_skip=True, aux_priv=True, aux_jepa=[1, 8, 32])


def test_all_on_matches_the_lambda_base_config_041_r24():
    from uranium.bc.run import model_variant

    assert model_variant(str(ROOT / "config" / "bc_m2_lambda_base.toml")) == ALL_ON


def copied(state):
    """The graph's outputs live in its memory: a copy to carry to the next decision."""
    return rebuild(state, iter([t.clone() for t in leaves(state)]))


def full_net(dev) -> UraniumNet:
    """The full-size all-on net; the zero-initialised bypass output and core-skip Linear made
    non-zero (as after training) so both change the outputs."""
    torch.manual_seed(0)
    net = UraniumNet(new_net_config("gtrxl", True, ALL_ON), action_columns())
    g = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for p in list(net.variant.bypass.out.parameters()) + list(net.variant.core_skip.parameters()):
            p.copy_(torch.randn(p.shape, generator=g) * 0.02)
    return net.to(dev).eval()


def run_steps(net, act, dev, prec, steps: int, exact: bool, check=None):
    """`steps` decisions of 4 rows, eager `net.act` against `act` (graphed), each carrying its own
    state; row 1's entity set empty on even decisions."""
    close = (lambda a, b: torch.equal(a, b)) if exact else (lambda a, b: torch.allclose(a.float(), b.float(), atol=1e-5, rtol=1e-5))
    s_eager, s_graph = net.init_state(4, dev), net.init_state(4, dev)
    for i in range(steps):
        obs = random_obs(4, 128, net.cfg.input, dev, torch.Generator().manual_seed(100 + i), valid=119)
        obs["entity_mask"][1] = obs["entity_mask"][1] & (i % 2 == 1)
        assert obs["entity_mask"][1].any() == (i % 2 == 1)
        with autocast(dev, prec):
            want = net.act(obs, s_eager, 0.0)
        got = act(obs, s_graph, 0.0)
        for k in want["action"]:
            assert torch.equal(got["action"][k], want["action"][k]), (i, k)
        assert torch.isfinite(got["log_prob"]).all() and torch.isfinite(got["value"]).all()
        assert close(got["log_prob"], want["log_prob"]), i
        assert close(got["value"], want["value"]), i
        for x, y in zip(leaves(got["state"]), leaves(want["state"])):
            assert close(x, y), i
        if check is not None:
            check(i)
        s_eager, s_graph = want["state"], copied(got["state"])


@pytest.mark.gpu
@torch.no_grad()
def test_graphed_act_matches_eager_all_switches_032_ac13():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA / ROCm device")
    _, dev, prec = resolve(None)
    net = full_net(dev)
    graphed = GraphedAct(net.act, dev, prec)
    exact = torch.version.hip is None  # MIOpen may pick another algorithm inside the capture
    run_steps(net, graphed, dev, prec, 70, exact, check=lambda i: (len(graphed.graphs) == 1) or pytest.fail(f"step {i}: {len(graphed.graphs)} graphs"))
    assert len(graphed.graphs) == 1


@pytest.mark.gpu
@torch.no_grad()
def test_compiled_graphed_act_no_recompile_032_ac13(monkeypatch):
    """The inference server's path on a CUDA host (`URANIUM_INFER_COMPILE` = default inside the
    graph, IMPL_DEFAULTS): one graph, no compile after the first decision, the same actions as
    eager (values within compile rounding)."""
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("no CUDA device (the compiled act is the CUDA host default)")
    import torch._dynamo.utils as du

    for k, v in IMPL_DEFAULTS["cuda"].items():
        monkeypatch.setenv(k, v)
    torch._dynamo.reset()
    _, dev, prec = resolve(None)
    net = full_net(dev)
    graphed = GraphedAct(compiled(net.act, "URANIUM_INFER_COMPILE"), dev, prec)
    assert graphed.act is not net.act
    s_eager, s_graph = net.init_state(4, dev), net.init_state(4, dev)
    frames, worst = [], 0.0
    for i in range(40):
        obs = random_obs(4, 128, net.cfg.input, dev, torch.Generator().manual_seed(100 + i), valid=119)
        obs["entity_mask"][1] = obs["entity_mask"][1] & (i % 2 == 1)
        with autocast(dev, prec):
            want = net.act(obs, s_eager, 0.0)
        got = graphed(obs, s_graph, 0.0)
        assert len(graphed.graphs) == 1
        assert torch.isfinite(got["value"]).all() and torch.isfinite(got["log_prob"]).all()
        worst = max(worst, (got["value"].float() - want["value"].float()).abs().max().item())
        frames.append(du.counters["frames"]["ok"])
        # Each side keeps its own state (the compiled path's rounding stays on its own trajectory).
        s_eager, s_graph = want["state"], copied(got["state"])
    print("compiled frames after each decision", frames, "max |value - eager|", worst)
    assert frames[0] > 0 and frames[-1] == frames[0], f"compiled frames after each decision {frames}: a recompile after the first"
    assert worst < 5e-2, worst


@pytest.fixture
def shards(tmp_path: Path) -> Path:
    d = tmp_path / "shards"
    _core.write_test_shards(str(d), "mixed", 2)
    return d


# The CUDA host's lane group (URANIUM_BC_MICRO_LANES = 2): a group of one lane compiles a
# selection-head backward kernel that ptxas takes over 10 minutes on (2026-11-10).
BC_FULL = """
lanes = 4
seq_len = 64
micro_lanes = 2
pad_entities = 64
lr = 3e-4
min_lr = 3e-6
warmup_steps = 10
total_steps = 100
weight_decay = 1e-4
grad_clip = 1.0
s2_lr_scale = 0.1

[heads]
action = 1.0
delay = 0.5
queued = 0.2
selected = 1.0
target_unit = 1.0
target_loc = 1.0

[value]
win = 0.5
win_open = 0.5

[data]
z_drop = 0.25
augment = true
seed = 1
checkpoint_every = 1000
checkpoint_minutes = 0
enemy_race_mask = 0.05

# 041-R23 aux loss weights (AMB-041-13b default proposal, as config/bc_m2_lambda_base.toml).
[aux]
priv = 0.2
jepa = 0.1

# 041-R25: the dead-channel monitor every 2nd update.
[monitor]
spatial_every = 2
"""


@pytest.mark.gpu
def test_bc_compiled_all_switches_no_recompile_per_batch_032_ac13(tmp_path, shards, monkeypatch):
    """The CUDA BC trainer (TRAIN_DEFAULTS: reduce-overhead graphs, channels-last, spatial bf16,
    stem pad, pinned H2D) on the full-size all-on net, T = 64 (two outer segments), aux losses on
    and the monitor every 2nd update."""
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("no CUDA device (reduce-overhead is the CUDA training default)")
    import torch._dynamo.utils as du

    for k, v in (IMPL_DEFAULTS["cuda"] | TRAIN_DEFAULTS["cuda"]).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("URANIUM_BC_MICRO_LANES", "2")  # the CUDA host's lane group (.env)
    assert TRAIN_DEFAULTS["cuda"]["URANIUM_COMPILE"] == "reduce-overhead"
    torch._dynamo.reset()
    config = tmp_path / "bc_full.toml"
    config.write_text(BC_FULL, encoding="utf-8")
    cfg = json.loads(_core.bc_config(str(config), "s1")[0])
    assert cfg["micro_lanes"] == 2 and cfg["aux"] == {"priv": 0.2, "jepa": 0.1} and cfg["monitor"] == {"spatial_every": 2}, cfg
    dev = torch.device("cuda")
    torch.manual_seed(1)
    tr = BcTrainer(cfg, UraniumNet(new_net_config("gtrxl", True, ALL_ON), action_columns()), dev, "bf16")
    assert tr.graphs

    def feeder():
        return _core.Feeder(str(config), [str(shards)], prefetch=0, synthetic_maps=True, check_maps=False)

    f = feeder()
    frames = []
    for s in range(10):
        b = f.next()
        if b is None:  # the synthetic epoch is short: start it again (the same shapes come back)
            f = feeder()
            b = f.next()
        out = json.loads(tr.train_batch(b, 3e-4, s))
        assert math.isfinite(out["total"]) and out["aux_priv"] is not None and out["aux_jepa"] is not None, out
        assert all(math.isfinite(out[k]) for k in ("aux_priv", "aux_jepa")), out
        assert (set(SPEC_COLUMNS) <= out.keys()) == ((s + 1) % 2 == 0)
        frames.append(du.counters["frames"]["ok"])
    print("compiled frames after each batch", frames)
    assert frames[-1] > 0 and frames[-1] == frames[-5], f"compiled frames after each batch {frames}: a recompile after the shapes settled"
