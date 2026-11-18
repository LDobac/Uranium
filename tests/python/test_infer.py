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


@pytest.mark.parametrize("k", [2, 5])
@pytest.mark.parametrize("model2", [False, True], ids=["base", "model2"])
@pytest.mark.parametrize("kind", ["Lstm", "Gtrxl"])
def test_set_state_round_trip_and_resume_051_r4(tmp_path, kind, model2, k):
    b, s = server(checkpoint(tmp_path, kind, model2), "argmax", 1)
    r = _core.restore_play(b, k, 12)
    stop(s)
    assert bits(r["restored"]) == bits(r["saved"]), "state() -> set_state() -> state() round trip"
    assert any(np.asarray(v).any() for _, v in r["saved"]), "a non-trivial state"
    if kind == "Gtrxl":
        mask = np.asarray(r["saved"][2][1])
        assert mask.sum() == min(k, 4), "GTrXL valid mask: k of 4 memory rows"
    assert bits(r["other_after"]) == bits(r["other_before"]), "032-R14: the other slot is unchanged"
    assert len(r["straight"]) == 12 - k
    assert r["resumed"] == r["straight"], "decisions after the restore = uninterrupted decisions"


@pytest.mark.gpu
@pytest.mark.parametrize("graph", ["off", "on"])
def test_set_state_on_the_device_051_r4(tmp_path, monkeypatch, graph):
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
    from uranium.agent.infer import state_to_host
    from uranium.model.state import leaves, rebuild, select_rows
    from uranium.bench.inputs import random_obs

    class FakeBroker:
        checkpoint_id, temp, capacity, pad_entities, warmup_entities = "fake@0", 0.0, 3, 64, 0
        restored = read = out = None

        def reply_restores(self, errors):
            self.restored = errors

        def reply_states(self, parts, shapes):
            self.read = [(sh, p) for p, sh in zip(parts[0], shapes[0])]

        def reply(self, *arrays):
            self.out = arrays

    net, _ = ckpt.load_model(checkpoint(tmp_path, "Gtrxl", True), CPU)
    fb = FakeBroker()
    srv = InferenceServer(fb, CPU, "f32", net=net)
    g = torch.Generator().manual_seed(1)
    rand = lambda t: torch.rand(t.shape, generator=g) > 0.5 if t.dtype == torch.bool else torch.randn(t.shape, generator=g)  # noqa: E731
    srv.state = rebuild(srv.state, iter([rand(t) for t in leaves(srv.state)]))
    host = lambda r: bits(zip(*reversed(state_to_host(srv.state, r))))  # noqa: E731
    want, row2 = host(2), host(2)
    parts, shapes = state_to_host(srv.state, 2)
    obs = random_obs(2, 64, net.cfg.input, CPU, torch.Generator().manual_seed(2))
    start = select_rows(srv.state, torch.tensor([2, 0]))
    with torch.no_grad():
        exp = net.act(obs, start, 0.0)
    batch = {"slots": [1, 0], "n": 64, **{k: v.numpy() for k, v in obs.items()}}
    srv.handle({"resets": [1], "restores": [(1, list(zip(shapes, parts)))], "states": [1], "batch": batch})
    assert fb.restored == [None]
    assert bits(fb.read) == want, "the read sees the restore, not the reset"
    lp, value = fb.out[6], fb.out[8]
    assert lp.tobytes() == exp["log_prob"].numpy().tobytes() and value.tobytes() == exp["value"].reshape(-1).numpy().tobytes(), "the act sees the restore"
    got = select_rows(srv.state, torch.tensor([1, 0]))
    assert all(torch.equal(x, y.to(x.dtype)) for x, y in zip(leaves(got), leaves(exp["state"]))), "the act's state written to slots 1, 0"
    assert host(2) == row2, "032-R14: the row outside the batch is unchanged"


@pytest.mark.parametrize("k", [2, 5])
@pytest.mark.parametrize("model2", [False, True], ids=["base", "model2"])
@pytest.mark.parametrize("kind", ["Lstm", "Gtrxl"])
def test_explore_restore_then_act_is_bit_equal_054_ac2(tmp_path, kind, model2, k):
    b, s = server(checkpoint(tmp_path, kind, model2), "argmax", 1)
    r = _core.explore_restore(b, k, 12)
    stop(s)
    assert r["again"] == r["first"], "save -> act -> restore -> act: the second act equals the first"
    assert not r["again_type_logp"], "054-R5: no type_logp without a request"
    assert r["forced_fresh"] == r["forced"], "the forced pass from the restored state equals a fresh slot restored to it"
    assert r["forced_action"] == r["c"] and f"action: {r['c']}," in r["forced"][0], "the forced act samples c"
    mask = np.asarray(r["type_mask"])
    assert mask[r["forced_action"]]
    assert bits(r["other_after"]) == bits(r["other_before"]), "032-R14: the other slot is unchanged"
    lp = np.asarray(r["type_logp"])
    assert lp.shape == mask.shape and np.isneginf(lp[~mask]).all() and np.isfinite(lp[mask]).all()
    assert abs(np.logaddexp.reduce(lp[mask].astype(np.float64))) < 1e-5


def _fake_server(tmp_path):
    class FakeBroker:
        checkpoint_id, temp, capacity, pad_entities, warmup_entities = "fake@0", 1.0, 3, 64, 0
        out = None

        def reply_restores(self, errors):
            pass

        def reply_states(self, parts, shapes):
            pass

        def reply(self, *arrays):
            self.out = arrays

    net, _ = ckpt.load_model(checkpoint(tmp_path, "Gtrxl", True), CPU)
    fb = FakeBroker()
    return fb, InferenceServer(fb, CPU, "f32", net=net), net


def test_type_logp_is_the_heads_masked_log_prob_054_ac3(tmp_path):
    from uranium.bench.inputs import random_obs

    fb, srv, net = _fake_server(tmp_path)
    obs = random_obs(3, 64, net.cfg.input, CPU, torch.Generator().manual_seed(4))
    obs["type_mask"][:, 0] = True
    seen = []
    real = net.heads.action_logp

    def spy(*a, **kw):
        out = real(*a, **kw)
        seen.append(out.detach().clone())
        return out

    batch = {"slots": [0, 1, 2], "n": 64, **{k: v.numpy() for k, v in obs.items()}}
    outs = {}
    for rows in ([], [0, 2]):
        srv.state = net.init_state(3, CPU)
        net.heads.action_logp = spy
        try:
            torch.manual_seed(9)
            srv.handle({"resets": [], "restores": [], "states": [], "batch": {**batch, "logp_rows": rows}})
        finally:
            net.heads.action_logp = real
        outs[tuple(rows)] = fb.out
    plain, asked = outs[()], outs[(0, 2)]
    assert plain[10] is None, "no type_logp without a request"
    for x, y in zip(plain[:10], asked[:10]):
        assert (x is None and y is None) or x.tobytes() == y.tobytes(), "054-R5: the act outputs do not change"
    head = seen[-1]
    a = head.shape[1]
    got = asked[10].reshape(2, a)
    mask = obs["type_mask"].numpy()
    for i, row in enumerate((0, 2)):
        m = mask[row]
        assert got[i][m].tobytes() == head[row].numpy()[m].tobytes(), "bit-equal to heads.action_logp on open actions"
        assert np.isneginf(got[i][~m]).all(), "closed actions are -inf"


def test_server_seed_051_r8(tmp_path):
    from uranium.agent.serve import new_infer_seed, seed_inference

    rng = torch.get_rng_state()
    try:
        d = checkpoint(tmp_path, "Lstm")
        seed = 2**63 - 1
        plays = []
        for _ in range(2):
            b = _core.Broker(str(d), "stochastic", max_batch=1, max_wait_ms=200, capacity=8, seed=seed)
            assert b.seed == seed
            s = InferenceServer(b, CPU, "f32")
            s.start()
            seed_inference(b.seed)
            plays.append(_core.play_sessions(b, 1, 12)[0][0])
            stop(s)
            assert torch.initial_seed() == seed
        assert plays[0] == plays[1], "same seed, same draws"
        torch.manual_seed(77)
        b, s = server(d, "stochastic", 1)
        assert b.seed is None
        _core.play_sessions(b, 1, 4)
        stop(s)
        assert torch.initial_seed() == 77, "the model thread never reseeds"
        assert all(0 <= new_infer_seed() < 2**63 for _ in range(100))
    finally:
        torch.set_rng_state(rng)


def test_close_stops_the_thread_and_frees_the_model_042(tmp_path):
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
    import os
    import threading

    codes, called = [], threading.Event()

    def fake_exit(code):
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
    s.device = torch.device("cuda")
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
    import subprocess

    out = tmp_path / "rand"
    r = subprocess.run(python_cmd + ["random-checkpoint", "--out", str(out), "--core", "lstm", "--win-open"], check=True, capture_output=True, text=True)
    assert r.stdout.strip() == f"rand@{ckpt.policy_version(out)[:12]}"
    net, meta = ckpt.load_model(out)
    assert net.value.open is not None and meta["train"]["random_init"]
    assert _core.Broker(str(out)).checkpoint_id == r.stdout.strip()
