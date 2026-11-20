import json
from pathlib import Path

import pytest
import torch

from uranium import _core, ckpt, kernels
from uranium.device import IMPL_DEFAULTS
from uranium.kernels import ENTITY_ATTN, KernelError
from uranium.kernels.attention import available, require
from uranium.model import UraniumNet, action_columns

from testkit import CPU, bc_cfg, net_config, rl_cfg

REASON = "simulated: 'cp1' codec can't decode bytes in position 0--1"


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(kernels, "_LOADED", {})
    monkeypatch.setattr(kernels, "_ERRORS", {})
    monkeypatch.setattr(kernels, "_WARNED", set())
    calls = []

    def fail(name):
        calls.append(name)
        raise RuntimeError(REASON)

    monkeypatch.setattr(kernels, "build", fail)
    return calls


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(kernels, "on_gpu", lambda device: True)


def supported_net(heads: int = 1, buckets: int | None = None, seed: int = 5) -> UraniumNet:
    cfg = net_config("tiny-gtrxl")
    cfg.entity.heads = heads
    if buckets is not None:
        cfg.entity.distance_buckets = buckets
    torch.manual_seed(seed)
    return UraniumNet(cfg, action_columns())


def saved(tmp_path: Path, net: UraniumNet) -> Path:
    d = tmp_path / "ck"
    d.mkdir()
    ckpt.save_model(d, net)
    ckpt.save_meta(d, net.cfg, {"random_init": True}, "f32")
    return d


@pytest.mark.parametrize("backend", ["cuda", "rocm"])
def test_060_r1_requested_kernel_load_failure_is_start_error(monkeypatch, fresh, gpu, backend):
    monkeypatch.setenv(ENTITY_ATTN, IMPL_DEFAULTS[backend][ENTITY_ATTN])
    with pytest.raises(KernelError) as e:
        kernels.check(supported_net(), "cuda")
    assert REASON in str(e.value) and f"{ENTITY_ATTN}=torch" in str(e.value)
    assert fresh == ["attn_bias"]


def test_060_r1_reference_on_cpu_or_torch_switch(monkeypatch, fresh):
    net = supported_net()
    monkeypatch.setenv(ENTITY_ATTN, "fused")
    assert kernels.check(net, CPU) == "torch"
    monkeypatch.setattr(kernels, "on_gpu", lambda device: True)
    monkeypatch.setenv(ENTITY_ATTN, "torch")
    assert kernels.check(net, "cuda") == "torch"
    assert fresh == []


@pytest.mark.parametrize("heads,buckets,why", [(4, None, "head width 8"), (1, 33, "rows 66 > 64")])
def test_060_r1_unsupported_shape_warns_once_without_loading(monkeypatch, capsys, fresh, gpu, heads, buckets, why):
    monkeypatch.setenv(ENTITY_ATTN, "fused")
    net = supported_net(heads, buckets)
    assert kernels.entity_attn_unsupported(net) is not None
    assert kernels.check(net, "cuda") == "torch"
    assert kernels.check(net, "cuda") == "torch"
    err = capsys.readouterr().err
    assert err.count("[kernels] WARNING") == 1 and why in err
    assert fresh == []


def test_060_r1_supported_shape_matches_kernel_limits():
    assert kernels.entity_attn_unsupported(supported_net()) is None


def test_060_r1_available_does_not_raise(fresh):
    assert available("cuda") is False
    assert available("cpu") is False
    with pytest.raises(KernelError, match="not loaded"):
        require("cuda")


def test_041_r15_meta_records_used_implementation(monkeypatch, fresh):
    net = supported_net()
    monkeypatch.delenv(ENTITY_ATTN, raising=False)
    assert f"{ENTITY_ATTN}=torch" in kernels.meta(net, CPU)
    assert not any(s.startswith(kernels.REQUESTED) for s in kernels.meta(net, CPU))
    monkeypatch.setenv(ENTITY_ATTN, "fused")
    m = kernels.meta(net, CPU)
    assert f"{ENTITY_ATTN}=torch" in m and f"{kernels.REQUESTED}{ENTITY_ATTN}=fused" in m and f"{ENTITY_ATTN}=fused" not in m
    monkeypatch.setattr(kernels, "on_gpu", lambda device: True)
    monkeypatch.setattr(kernels, "_LOADED", {"attn_bias": object()})
    m = kernels.meta(net, "cuda")
    assert f"{ENTITY_ATTN}=fused" in m and not any(s.startswith(kernels.REQUESTED) for s in m)


def test_041_r15_bc_checkpoint_meta_has_used_kernel(monkeypatch, tmp_path, tiny_config, fresh):
    from uranium.bc.trainer import BcTrainer

    monkeypatch.setenv(ENTITY_ATTN, "fused")
    tr = BcTrainer(bc_cfg(tiny_config), supported_net(), CPU, "f32")
    tr.save(str(tmp_path / "out"), json.dumps({"update": 0}))
    meta = json.loads((tmp_path / "out" / ckpt.META_FILE).read_text(encoding="utf-8"))
    assert f"{ENTITY_ATTN}=torch" in meta["kernels"] and f"{kernels.REQUESTED}{ENTITY_ATTN}=fused" in meta["kernels"]


def test_060_ac8_entry_points_stop_on_requested_kernel_failure(monkeypatch, tmp_path, tiny_config, fresh, gpu):
    from uranium.agent.infer import InferenceServer
    from uranium.bc.eval import evaluate
    from uranium.bc.trainer import BcTrainer
    from uranium.rl.learner import Learner

    monkeypatch.setenv(ENTITY_ATTN, "fused")
    d = saved(tmp_path, supported_net())
    with pytest.raises(KernelError):
        BcTrainer(bc_cfg(tiny_config), supported_net(), CPU, "f32")
    with pytest.raises(KernelError):
        Learner(rl_cfg(), supported_net(), supported_net(), CPU, "f32", "vtrace")
    with pytest.raises(KernelError):
        ckpt.load_model(d, CPU)
    with pytest.raises(KernelError):
        InferenceServer(_core.Broker(str(d), "stochastic", max_batch=1, max_wait_ms=200, capacity=8), CPU, "f32")
    with pytest.raises(KernelError):
        InferenceServer(_core.Broker(str(d), "stochastic", max_batch=1, max_wait_ms=200, capacity=8), CPU, "f32", net=supported_net())
    with pytest.raises(KernelError):
        evaluate([], str(d), "val", None, 0, str(tiny_config), CPU, "f32")
    assert fresh == ["attn_bias"]


def test_060_ac8_entry_points_run_with_torch_switch(monkeypatch, tmp_path, tiny_config, fresh, gpu):
    from uranium.bc.trainer import BcTrainer

    monkeypatch.setenv(ENTITY_ATTN, "torch")
    d = saved(tmp_path, supported_net())
    BcTrainer(bc_cfg(tiny_config), supported_net(), CPU, "f32")
    ckpt.load_model(d, CPU)
    assert fresh == []
