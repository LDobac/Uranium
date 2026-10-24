"""060-R1 hand-written kernels against their PyTorch references (pytest `gpu`): the fused entity
attention with the distance bias (forward, dq, dk, dv, table gradient; masked keys and a fully
masked row), and the entity encoder with `URANIUM_ENTITY_ATTN=fused` against the PyTorch path."""

import json
import math
from pathlib import Path

import pytest
import torch

from uranium.model import UraniumNet, UraniumNetConfig, action_columns, input_dims

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "burn_parity"
pytestmark = pytest.mark.gpu


def kernel_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA / ROCm device")
    from uranium.kernels.attention import available

    if not available("cuda"):
        pytest.skip("attn_bias kernel not built on this host")


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-12)).item()


@pytest.mark.parametrize("n,valid,all_masked", [(100, 77, False), (130, 90, True), (64, 64, False)])
def test_fused_attention_matches_reference(n, valid, all_masked):
    kernel_or_skip()
    from uranium.kernels.attention import fused, reference

    g = torch.Generator(device="cuda").manual_seed(n)
    q, k, v, dout = (torch.randn(2, 8, n, 32, device="cuda", generator=g) for _ in range(4))
    bucket = torch.randint(0, 32, (2, n, n), device="cuda", generator=g, dtype=torch.uint8)
    table = torch.randn(32, 8, device="cuda", generator=g)
    key_ok = torch.zeros(2, n, dtype=torch.bool, device="cuda")
    key_ok[:, :valid] = True
    if all_masked:
        key_ok[0] = False
    res = {}
    for name in ("ref", "fused"):
        qq, kk, vv, tt = (x.detach().clone().requires_grad_() for x in (q, k, v, table))
        out = reference(qq, kk, vv, bucket, tt, key_ok, 1 / math.sqrt(32)) if name == "ref" else fused(qq, kk, vv, bucket, tt, key_ok)
        out.backward(dout)
        res[name] = (out, qq.grad, kk.grad, vv.grad, tt.grad)
    for label, a, b in zip(("out", "dq", "dk", "dv", "dtable"), res["fused"], res["ref"]):
        assert rel(a, b) < 1e-4, label


def test_entity_encoder_fused_matches_torch(monkeypatch):
    kernel_or_skip()
    cfg = UraniumNetConfig.from_dict(json.loads((FIX / "tiny-gtrxl" / "meta.json").read_text())["config"])
    cfg.input = input_dims()
    cfg.entity.dropout = 0.0
    if cfg.entity.d_model // cfg.entity.heads != 32:
        cfg.entity.d_model, cfg.entity.heads = 64, 2
    from uranium.bench.inputs import random_obs

    obs = random_obs(6, 40, cfg.input, "cuda", torch.Generator().manual_seed(3), valid=33)
    res = {}
    for mode in ("torch", "fused"):
        monkeypatch.setenv("URANIUM_ENTITY_ATTN", mode)
        torch.manual_seed(5)
        net = UraniumNet(cfg, action_columns()).cuda()
        assert net.entity.use_fused(torch.zeros(1, 1, cfg.entity.d_model, device="cuda"), net.entity.bias_table) == (mode == "fused")
        e, pool = net.entity(obs)
        (e.square().sum() + pool.sum()).backward()
        res[mode] = (e.detach(), {k: p.grad.clone() for k, p in net.entity.named_parameters() if p.grad is not None})
    assert rel(res["fused"][0], res["torch"][0]) < 1e-4
    assert res["fused"][1].keys() == res["torch"][1].keys()
    # Relative L2 with the denominator floored at 1e-3 of the largest gradient norm (as the
    # cross-device test): the key projection bias has a zero gradient up to rounding noise.
    floor = 1e-3 * max(float(g.norm()) for g in res["torch"][1].values())
    for k, g in res["torch"][1].items():
        err = float((res["fused"][1][k] - g).norm()) / max(float(g.norm()), floor)
        assert err < 1e-3, (k, err)
