"""Model sessions through the inference broker (AMB-000-4) with a tiny PyTorch network on CPU:
Rust sessions (synthetic frames, feature extraction, executor) on Rust threads, the model on a
Python thread. 031-R4 masked actions, 031-R7 argmax determinism, 032-R14 batched play = single
play, 032-R16 slot state. Session plumbing with a mock model is in
crates/uranium-agent/tests/model_policy.rs."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from uranium import _core, ckpt
from uranium.agent.infer import InferenceServer
from uranium.model import UraniumNet, UraniumNetConfig, action_columns, input_dims

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "burn_parity"
CPU = torch.device("cpu")
DELAY_BUCKETS = json.loads(_core.feature_dims())["delay_buckets"]


def checkpoint(tmp_path: Path, kind: str) -> Path:
    cfg = UraniumNetConfig.from_dict(json.loads((FIX / "tiny-lstm" / "meta.json").read_text())["config"])
    cfg.input = input_dims()
    cfg.core.kind = kind
    torch.manual_seed(11)
    net = UraniumNet(cfg, action_columns())
    d = tmp_path / f"ckpt-{kind}"
    d.mkdir()
    ckpt.save_model(d, net)
    ckpt.save_meta(d, cfg, {"random_init": True}, "f32")
    return d


def server(d: Path, sample: str, max_batch: int):
    b = _core.Broker(str(d), sample, max_batch=max_batch, max_wait_ms=200, capacity=8)
    s = InferenceServer(b, CPU, "f32")
    s.start()
    return b, s


def stop(s: InferenceServer) -> None:
    s.stop.set()
    s.join(timeout=5)
    assert s.error is None


@pytest.mark.parametrize("kind", ["Lstm", "Gtrxl"])
def test_plays_valid_acts_031_r4(tmp_path, kind):
    b, s = server(checkpoint(tmp_path, kind), "stochastic", 4)
    [(acts, stats)] = _core.play_sessions(b, 1, 12)
    stats = json.loads(stats)
    for cmds, adv in acts:
        assert cmds <= 1, "031-R3: one command per action"
        assert adv in DELAY_BUCKETS, f"031-R9: advance {adv} is not a delay bucket"
    assert stats["decisions"] == 12
    # 031-AC2: the model samples within the masks, so only executor-side causes can reject.
    assert set(stats["rejected"]) <= {"no_executor_unit"}, stats["rejected"]
    assert stats["map_precompute"] == "missing"
    assert stats["checkpoint"] == b.checkpoint_id
    stop(s)


@pytest.mark.parametrize("kind", ["Lstm", "Gtrxl"])
def test_argmax_is_deterministic_031_r7(tmp_path, kind):
    b, s = server(checkpoint(tmp_path, kind), "argmax", 4)
    first = _core.play_sessions(b, 1, 8)[0][0]
    assert _core.play_sessions(b, 1, 8)[0][0] == first
    stop(s)


@pytest.mark.parametrize("kind", ["Lstm", "Gtrxl"])
def test_batched_sessions_match_single_032_r14(tmp_path, kind):
    """Three sessions batched together act like each session served alone, and their Core states
    do not mix."""
    d = checkpoint(tmp_path, kind)
    runs = {}
    for max_batch in (3, 1):
        b, s = server(d, "argmax", max_batch)
        runs[max_batch] = [(acts, json.loads(st)["mean_batch"]) for acts, st in _core.play_sessions(b, 3, 10)]
        stop(s)
    assert any(mb > 1.0 for _, mb in runs[3]), f"sessions were never batched: {[mb for _, mb in runs[3]]}"
    for k, ((a, _), (c, _)) in enumerate(zip(runs[3], runs[1])):
        assert a == c, f"session {k}: batched play differs from single play"


@pytest.mark.parametrize("kind", ["Lstm", "Gtrxl"])
def test_slot_state_032_r16(tmp_path, kind):
    b, s = server(checkpoint(tmp_path, kind), "argmax", 1)
    states = _core.slot_states(b)
    assert len(states) == 3
    fresh, one, two = ([np.asarray(v) for _, v in st] for st in states)
    assert all((v == 0).all() for v in fresh), "a fresh slot's state is zero (032-R5)"
    assert any((x != y).any() for x, y in zip(fresh, one)), "the state advanced after a decision"
    assert any((x != y).any() for x, y in zip(one, two))
    assert all(shape[0] == 1 for shape, _ in states[0]), "batch dimension 1 per slot"
    stop(s)


def test_random_checkpoint_command(tmp_path, python_cmd):
    """`random-checkpoint` writes a loadable checkpoint and prints its policy version (020-R5)."""
    import subprocess

    out = tmp_path / "rand"
    r = subprocess.run(python_cmd + ["random-checkpoint", "--out", str(out), "--core", "lstm", "--win-open"], check=True, capture_output=True, text=True)
    assert r.stdout.strip() == f"rand@{ckpt.policy_version(out)[:12]}"
    net, meta = ckpt.load_model(out)
    assert net.value.open is not None and meta["train"]["random_init"]
    assert _core.Broker(str(out)).checkpoint_id == r.stdout.strip()
