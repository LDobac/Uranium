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
    graphed = GraphedAct(net, dev, prec)
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
