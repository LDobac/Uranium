"""URANIUM_INFER_GRAPH (pytest `gpu`): `act` replayed from captured graphs gives the eager outputs,
including a batch padded to the next power of two, a second call on the same graph and a weight
swap after capture (AMB-000-4 step 2)."""

import pytest
import torch

from uranium.agent.infer import GraphedAct
from uranium.bc.trainer import new_net_config
from uranium.bench.inputs import random_obs
from uranium.device import autocast, resolve
from uranium.model import UraniumNet, action_columns

pytestmark = pytest.mark.gpu


def close(a, b, exact):
    return torch.equal(a, b) if exact else torch.allclose(a.float(), b.float(), atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_graphed_act_matches_eager():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA / ROCm device")
    _, dev, prec = resolve(None)
    torch.manual_seed(0)
    net = UraniumNet(new_net_config("gtrxl", True), action_columns()).to(dev).eval()
    graphed = GraphedAct(net.act, dev, prec)
    exact = torch.version.hip is None  # MIOpen may pick another algorithm inside the capture

    def eager(obs, state):
        with autocast(dev, prec):
            return net.act(obs, state, 0.0)

    for b, n, seed in ((3, 128, 1), (3, 128, 2), (1, 192, 3)):
        obs = random_obs(b, n, net.cfg.input, dev, torch.Generator().manual_seed(seed), valid=n - 9)
        state = net.init_state(b, dev)
        want, got = eager(obs, state), graphed(obs, state, 0.0)
        for k in want["action"]:
            assert torch.equal(got["action"][k], want["action"][k]), (b, n, k)
        assert close(got["value"], want["value"], exact) and close(got["log_prob"], want["log_prob"], exact)
        for x, y in zip(got["state"][0], want["state"][0]):
            assert x.shape == y.shape and close(x, y, exact)
    assert len(graphed.graphs) == 2  # batch 3 padded to 4 (captured once), batch 1
    # A weight swap after capture is seen by the graph (parameters copied in place).
    other = UraniumNet(new_net_config("gtrxl", True), action_columns()).to(dev).eval()
    net.load_state_dict(other.state_dict())
    obs = random_obs(3, 128, net.cfg.input, dev, torch.Generator().manual_seed(5), valid=100)
    state = net.init_state(3, dev)
    assert close(graphed(obs, state, 0.0)["value"], eager(obs, state)["value"], exact)


def test_capture_while_another_thread_trains():
    """RL runs a learner thread and inference threads in one process: capturing a new shape must
    not fail the learner's backward (or itself) when they overlap."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA / ROCm device")
    import threading

    _, dev, prec = resolve(None)
    torch.manual_seed(0)
    net = UraniumNet(new_net_config("gtrxl", True), action_columns()).to(dev).eval()
    graphed = GraphedAct(net.act, dev, prec)
    trainee = torch.nn.Sequential(torch.nn.Linear(512, 2048), torch.nn.ReLU(), torch.nn.Linear(2048, 512)).to(dev)
    stop, errors = threading.Event(), []

    from uranium.device import DEVICE_LOCK

    def train():
        try:
            x = torch.randn(4096, 512, device=dev)
            while not stop.is_set():
                with DEVICE_LOCK.shared():  # as the RL learner's update
                    trainee(x).square().mean().backward()
                    torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001 - reported below
            errors.append(e)

    t = threading.Thread(target=train)
    t.start()
    try:
        with torch.no_grad():
            for b, n in ((1, 64), (2, 128), (4, 192), (8, 64)):
                with DEVICE_LOCK.shared():  # as the server's request handling; the capture takes it over
                    obs = random_obs(b, n, net.cfg.input, dev, torch.Generator().manual_seed(b + n), valid=n - 5)
                    graphed(obs, net.init_state(b, dev), 1.0)["action"]["action"].cpu()
    finally:
        stop.set()
        t.join()
    assert not errors, errors
    assert len(graphed.graphs) == 4
