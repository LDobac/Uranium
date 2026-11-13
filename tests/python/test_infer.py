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
from uranium.model.config import VariantConfig

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "burn_parity"
CPU = torch.device("cpu")
DELAY_BUCKETS = json.loads(_core.feature_dims())["delay_buckets"]


def checkpoint(tmp_path: Path, kind: str, model2: bool = False) -> Path:
    """A random tiny checkpoint; `model2`: with the outer LSTM (segment 4 decisions, 032 §2.7),
    the head bypass and core skip (032-R21, R23) on, their zero-initialised outputs made non-zero
    so they change the outputs."""
    cfg = UraniumNetConfig.from_dict(json.loads((FIX / "tiny-lstm" / "meta.json").read_text())["config"])
    cfg.input = input_dims()
    cfg.core.kind = kind
    if model2:
        cfg.variant = VariantConfig(outer=4, bypass=True, core_skip=True)
    torch.manual_seed(11)
    net = UraniumNet(cfg, action_columns())
    if model2:
        g = torch.Generator().manual_seed(5)
        with torch.no_grad():
            for p in [*net.variant.core_skip.parameters(), *net.variant.bypass.out.parameters()]:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    d = tmp_path / f"ckpt-{kind}{'-m2' if model2 else ''}"
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


def bits(state) -> list[tuple[list, bytes]]:
    return [(list(shape), np.asarray(v, dtype=np.float32).tobytes()) for shape, v in state]


@pytest.mark.parametrize("model2", [False, True], ids=["base", "model2"])
@pytest.mark.parametrize("kind", ["Lstm", "Gtrxl"])
def test_set_state_round_trip_and_resume_051_r4(tmp_path, kind, model2):
    """051-R4 (032-R16): a second slot restores the first slot's state from before decision 5 of
    12: `state()` -> `set_state()` -> `state()` is bit for bit, the other slot's state is untouched
    (032-R14), and the 7 decisions after the restore (action, log-probs, value) are bit for bit
    those of the uninterrupted slot, every batch one row. The GTrXL memory (4) and the outer
    segment (4) both roll over after the restore."""
    b, s = server(checkpoint(tmp_path, kind, model2), "argmax", 1)
    r = _core.restore_play(b, 5, 12)
    stop(s)
    assert bits(r["restored"]) == bits(r["saved"]), "state() -> set_state() -> state() round trip"
    assert any(np.asarray(v).any() for _, v in r["saved"]), "a non-trivial state"
    assert bits(r["other_after"]) == bits(r["other_before"]), "032-R14: the other slot is unchanged"
    assert len(r["straight"]) == 7
    assert r["resumed"] == r["straight"], "decisions after the restore = uninterrupted decisions"


@pytest.mark.gpu
@pytest.mark.parametrize("graph", ["off", "on"])
def test_set_state_on_the_device_051_r4(tmp_path, monkeypatch, graph):
    """051-R4 on a CUDA / ROCm device, eager and with captured graphs (URANIUM_INFER_GRAPH):
    the round trip is bit for bit (state values are f32 / bool), the decisions after the restore
    are those of the uninterrupted slot (same actions; log-probs and values within the 032-R19
    tolerance)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA / ROCm device")
    from uranium.device import resolve

    _, dev, prec = resolve(None)
    monkeypatch.setenv("URANIUM_INFER_GRAPH", graph)
    b = _core.Broker(str(checkpoint(tmp_path, "Gtrxl", True)), "argmax", max_batch=1, max_wait_ms=200, capacity=8)
    s = InferenceServer(b, dev, prec)
    assert (s.graphed is not None) == (graph == "on")
    s.start()
    r = _core.restore_play(b, 5, 12)
    stop(s)
    assert bits(r["restored"]) == bits(r["saved"])
    assert bits(r["other_after"]) == bits(r["other_before"])
    for (a1, x), (a2, y) in zip(r["straight"], r["resumed"]):
        assert a1 == a2
        x, y = np.array(x, dtype=np.uint32).view(np.float32), np.array(y, dtype=np.uint32).view(np.float32)
        assert np.allclose(x, y, atol=1e-4, rtol=1e-4), (x, y)


def test_set_state_refuses_a_mismatch_051_r4(tmp_path):
    """051-R4: a state of another Core kind or model (part count, shape, a mask part with values
    other than 0/1) is refused with an error and the slot keeps its state; a fitting state is
    written."""
    lstm, gtrxl = checkpoint(tmp_path, "Lstm"), checkpoint(tmp_path, "Gtrxl")
    b, s = server(lstm, "argmax", 1)
    bg, sg = server(gtrxl, "argmax", 1)
    try:
        lstm_state = [(shape, np.asarray(v).tolist()) for shape, v in _core.slot_states(b)[1]]
        g_state = [(shape, np.asarray(v).tolist()) for shape, v in _core.slot_states(bg)[2]]
        before, e, after = _core.set_slot_state(b, lstm_state)
        assert e is None and bits(after) == bits([(sh, np.asarray(v, dtype=np.float32)) for sh, v in lstm_state])
        bad = {
            "another core": g_state,
            "missing part": lstm_state[:1],
            "wider": [([1, sh[1] + 1], v + [0.0]) for sh, v in lstm_state],
        }
        mask = list(g_state)
        mask[-1] = (mask[-1][0], [0.5] * len(mask[-1][1]))
        for name, st in bad.items():
            before, e, after = _core.set_slot_state(b, st)
            assert e is not None and "051-R4" in e, (name, e)
            assert bits(after) == bits(before), f"{name}: the slot keeps its state"
        before, e, after = _core.set_slot_state(bg, mask)
        assert e is not None and "0 and 1" in e, e
        assert bits(after) == bits(before)
    finally:
        stop(s)
        stop(sg)


def test_restore_order_in_one_work_051_r4(tmp_path):
    """051-R4: one work with a reset, a restore and a state read of the same slot is handled as
    reset -> restore -> state read: the read gives the restored state."""
    from uranium.agent.infer import leaves, rebuild, state_to_host

    class FakeBroker:
        checkpoint_id, temp, capacity, pad_entities, warmup_entities = "fake@0", 0.0, 3, 64, 0
        restored = read = None

        def reply_restores(self, errors):
            self.restored = errors

        def reply_states(self, parts, shapes):
            self.read = [(sh, p) for p, sh in zip(parts[0], shapes[0])]

    net, _ = ckpt.load_model(checkpoint(tmp_path, "Gtrxl"), CPU)
    fb = FakeBroker()
    srv = InferenceServer(fb, CPU, "f32", net=net)
    g = torch.Generator().manual_seed(1)
    rand = lambda t: torch.rand(t.shape, generator=g) > 0.5 if t.dtype == torch.bool else torch.randn(t.shape, generator=g)  # noqa: E731
    srv.state = rebuild(srv.state, iter([rand(t) for t in leaves(srv.state)]))
    host = lambda r: bits(zip(*reversed(state_to_host(srv.state, r))))  # noqa: E731
    want, others = host(2), [host(0), host(2)]
    parts, shapes = state_to_host(srv.state, 2)
    srv.handle({"resets": [1], "restores": [(1, list(zip(shapes, parts)))], "states": [1], "batch": None})
    assert fb.restored == [None]
    assert bits(fb.read) == want, "the read sees the restore, not the reset"
    assert [host(0), host(2)] == others, "032-R14: other rows unchanged"


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
    # Out of memory in any form, and CUDA graph capture errors, are never fatal.
    assert not is_device_fatal(RuntimeError("CUDA error: out of memory"))
    assert not is_device_fatal(RuntimeError("HIP error: out of memory"))
    if hasattr(torch, "AcceleratorError"):
        assert not is_device_fatal(torch.AcceleratorError("CUDA error: out of memory"))
    assert not is_device_fatal(RuntimeError("CUDA error: operation not permitted when stream is capturing"))
    assert not is_device_fatal(RuntimeError("CUDA error: operation failed due to a previous error during capture (capture invalidated)"))

    class Coded(RuntimeError):
        def __init__(self, msg, code):
            super().__init__(msg)
            self.error_code = code

    assert not is_device_fatal(Coded("CUDA error: allocation failed", 2))
    assert not is_device_fatal(Coded("CUDA error: stream capture unsupported", 900))
    assert not is_device_fatal(Coded("CUDA error: stream capture wrong thread", 908))
    assert is_device_fatal(Coded("CUDA error: an illegal memory access was encountered", 700))
    try:
        try:
            raise RuntimeError("CUDA error: out of memory")
        except RuntimeError as inner:
            raise RuntimeError("CUDA error: while copying the reply") from inner
    except RuntimeError as outer:
        assert not is_device_fatal(outer), "out of memory anywhere in the chain"
    assert is_device_fatal(RuntimeError("CUDA error: an illegal memory access was encountered")), "still fatal"
    assert not is_device_fatal(RuntimeError("warm-up failed"))
    assert not is_device_fatal(ValueError("bad observation"))
    assert not is_device_fatal(None)


class ExitCalled(BaseException):
    pass


def test_device_fatal_guard_exits_70_042_r20(monkeypatch, capsys):
    """042-R20: `device_fatal_guard` turns a fatal device error into one stderr line and
    `os._exit(70)` after flushing and a 0.2 s pause (so a tee keeps the line); another exception
    propagates unchanged and nothing exits."""
    import os
    import time

    from uranium import device

    codes = []

    def fake_exit(code):
        codes.append(code)
        raise ExitCalled

    monkeypatch.setattr(os, "_exit", fake_exit)
    monkeypatch.setattr(time, "sleep", lambda s: codes.append(("sleep", s)))
    with pytest.raises(ExitCalled):
        with device.device_fatal_guard("learner main update"):
            raise RuntimeError(CUDA_FATAL)
    assert codes == [("sleep", 0.2), device.EXIT_DEVICE_FATAL]
    assert device.EXIT_DEVICE_FATAL == 70
    codes.clear()
    codes.append(70)
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
