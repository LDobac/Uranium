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


def test_close_stops_the_thread_and_frees_the_model_042(tmp_path):
    """042 스냅샷 추론 서버 (AMB-042-14): `close` ends the model thread and drops the model (its
    device memory); a plain `stop` keeps the model."""
    import weakref

    d = checkpoint(tmp_path, "Lstm")
    b, s = server(d, "stochastic", 4)
    _core.play_sessions(b, 1, 3)
    net = weakref.ref(s.net)
    s.close()
    s.join(timeout=10)
    assert not s.is_alive() and s.error is None
    assert s.net is None and s.state is None and net() is None, "model freed"
    _, s2 = server(d, "stochastic", 4)
    stop(s2)
    assert s2.net is not None


CUDA_FATAL = "CUDA error: an illegal memory access was encountered\nCUDA kernel errors might be asynchronously reported"


def test_device_fatal_classification_042_r20():
    """042-R20: which errors end the process. `torch.AcceleratorError`, a message naming a sticky
    CUDA / HIP error (also through a cause, or as text in a Rust error carrying a traceback) are
    fatal; out of memory and ordinary errors are not."""
    from uranium.device import is_device_fatal

    assert is_device_fatal(RuntimeError(CUDA_FATAL))
    assert is_device_fatal(RuntimeError("Sticky error detected"))
    assert is_device_fatal(RuntimeError("HIP error: an illegal memory access was encountered"))
    assert is_device_fatal(RuntimeError("Traceback (most recent call last):\n...\nAcceleratorError: CUDA error: an illegal memory access"))
    if hasattr(torch, "AcceleratorError"):
        assert is_device_fatal(torch.AcceleratorError("unspecified launch failure"))
    try:
        try:
            raise RuntimeError(CUDA_FATAL)
        except RuntimeError as inner:
            raise ValueError("while copying the reply") from inner
    except ValueError as outer:
        assert is_device_fatal(outer), "a fatal error as the cause"
    assert not is_device_fatal(torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert not is_device_fatal(RuntimeError("warm-up failed"))
    assert not is_device_fatal(ValueError("bad observation"))
    assert not is_device_fatal(None)


class ExitCalled(BaseException):
    pass


def test_device_fatal_guard_exits_70_042_r20(monkeypatch, capsys):
    """042-R20: `device_fatal_guard` turns a fatal device error into one stderr line and
    `os._exit(70)`; another exception propagates unchanged and nothing exits."""
    import os

    from uranium import device

    codes = []

    def fake_exit(code):
        codes.append(code)
        raise ExitCalled

    monkeypatch.setattr(os, "_exit", fake_exit)
    with pytest.raises(ExitCalled):
        with device.device_fatal_guard("learner main update"):
            raise RuntimeError(CUDA_FATAL)
    assert codes == [device.EXIT_DEVICE_FATAL] == [70]
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("[device] learner main update: fatal device error") and "042-R20" in err[0]
    with pytest.raises(ValueError):
        with device.device_fatal_guard("learner main update"):
            raise ValueError("not a device error")
    assert codes == [70]


def test_server_exits_70_on_a_fatal_device_error_042_r20(tmp_path, monkeypatch):
    """042-R20: an inference server whose batch fails with a fatal device error exits the process
    (70) at once instead of failing every later batch; an ordinary error is still sent to the
    sessions and the server goes on."""
    import os
    import threading

    codes, called = [], threading.Event()

    def fake_exit(code):  # records and returns: the server then handles the error as before
        codes.append(code)
        called.set()

    monkeypatch.setattr(os, "_exit", fake_exit)
    d = checkpoint(tmp_path, "Lstm")
    b, s = server(d, "stochastic", 4)
    errors = iter([ValueError("bad batch"), RuntimeError(CUDA_FATAL)])

    def failing(w, t_py=None):
        if w["batch"] is not None:
            raise next(errors)

    monkeypatch.setattr(s, "handle", failing)
    with pytest.raises(Exception):
        _core.play_sessions(b, 1, 1)
    assert codes == [], "an ordinary error does not exit"
    assert s.is_alive()
    with pytest.raises(Exception):
        _core.play_sessions(b, 1, 1)
    assert called.wait(10) and codes == [70]
    stop(s)


def test_free_waits_for_the_device_then_drops_graphs_042_r20(tmp_path, monkeypatch):
    """042-R20 (6th advisory): closing a server frees its device memory only under
    `DEVICE_LOCK.exclusive()`, after `torch.cuda.synchronize()`, destroying the captured graphs
    before `empty_cache()`; nothing of it while a shared holder (learner update) still runs."""
    import contextlib
    import threading
    import types

    from uranium import device

    d = checkpoint(tmp_path, "Lstm")
    _, s = server(d, "stochastic", 4)
    stop(s)
    calls = []
    real = device.DEVICE_LOCK.exclusive

    @contextlib.contextmanager
    def exclusive():
        with real():
            calls.append("lock")
            yield
            calls.append("unlock")

    class Graph:
        def reset(self):
            calls.append("graph reset")

    monkeypatch.setattr(device.DEVICE_LOCK, "exclusive", exclusive)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: calls.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    s.device = torch.device("cuda")  # only the calls above touch it
    s.graphed = types.SimpleNamespace(graphs={"a": {"graph": Graph()}, "b": {"graph": Graph()}})
    learner_in, learner_out = threading.Event(), threading.Event()

    def learner():
        with device.DEVICE_LOCK.shared():
            learner_in.set()
            learner_out.wait(10)
            calls.append("learner done")

    t = threading.Thread(target=learner)
    t.start()
    assert learner_in.wait(10)
    f = threading.Thread(target=s._free)
    f.start()
    f.join(0.3)
    assert f.is_alive() and calls == [], "waits for the learner's shared hold"
    learner_out.set()
    t.join(10)
    f.join(10)
    assert calls == ["learner done", "lock", "synchronize", "graph reset", "graph reset", "empty_cache", "unlock"]
    assert s.net is None and s.graphed is None and s.state is None


def test_random_checkpoint_command(tmp_path, python_cmd):
    """`random-checkpoint` writes a loadable checkpoint and prints its policy version (020-R5)."""
    import subprocess

    out = tmp_path / "rand"
    r = subprocess.run(python_cmd + ["random-checkpoint", "--out", str(out), "--core", "lstm", "--win-open"], check=True, capture_output=True, text=True)
    assert r.stdout.strip() == f"rand@{ckpt.policy_version(out)[:12]}"
    net, meta = ckpt.load_model(out)
    assert net.value.open is not None and meta["train"]["random_init"]
    assert _core.Broker(str(out)).checkpoint_id == r.stdout.strip()
