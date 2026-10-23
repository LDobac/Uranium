"""032-AC1 / AC8 (pytest `gpu`): the same weights and inputs on CPU and on the GPU (CUDA or
ROCm) give the same forward outputs and gradients within the device tolerance. The f32 check runs
without TF32 on both backends (the port's math); the CUDA bf16 check (032-R8, training and
inference precision there) uses provisional tolerances from the 2026-10-24 measurement
(`reports/pytorch_port.md` 9: forward ≤ 0.028, gradient median 1.5%), TBD(M6)."""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from uranium.model import UraniumNet, UraniumNetConfig, action_columns, to_labels, to_obs
from uranium.model.burn import burn_to_torch

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "burn_parity"
HEADS = ("action", "delay", "queued", "selected", "target_unit", "target_loc")
ROCM = torch.version.hip is not None
# 032-AC1 forward, 032-AC8 gradient (relative L2), f32 without TF32 (measured: CUDA ≤ 4e-5 /
# 3e-4, ROCm ≤ 1e-4 / 1e-3).
FWD_TOL = 1e-4
GRAD_TOL = 1e-3
# CUDA bf16 autocast, provisional (TBD(M6)), about 2x the 2026-10-24 measurement.
BF16_FWD_TOL = 0.06
BF16_GRAD_MEDIAN_TOL = 0.03
BF16_LOSS_REL_TOL = 1e-3


def dumps():
    dirs = sorted(p for p in FIX.glob("*") if (p / "meta.json").is_file())
    return dirs + [Path(d) for d in os.environ.get("URANIUM_PARITY_DIRS", "").split(os.pathsep) if d]


def run(d: Path, device: str, bf16: bool = False):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    meta = json.loads((d / "meta.json").read_text())
    net = UraniumNet(UraniumNetConfig.from_dict(meta["config"]), action_columns())
    net.load_state_dict(burn_to_torch(net, load_file(d / "weights.safetensors")))
    net = net.to(device)
    inp = load_file(d / "inputs.safetensors")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        out = net.forward_seq(to_obs(inp, device), to_labels(inp, device), inp["privileged"].to(device), net.init_state(meta["b"], device), meta["t"])
    w = inp["weights"].to(device)
    loss = sum((-out["logp"][k].float() * w).sum() for k in HEADS) / meta["valid"] + out["value"][:, 0].float().square().mean()
    loss.backward()
    fwd = {f"logp.{k}": out["logp"][k].detach().float().cpu() for k in HEADS} | {"value": out["value"].detach().float().cpu()}
    grads = {k: p.grad.detach().float().cpu() for k, p in net.named_parameters() if p.grad is not None}
    return fwd, grads, float(loss)


def rel_errors(gpu_g: dict, cpu_g: dict) -> list:
    floor = 1e-3 * max(float(g.norm()) for g in cpu_g.values())
    return sorted((float((gpu_g[k] - g).norm()) / max(float(g.norm()), floor), k) for k, g in cpu_g.items())


@pytest.mark.gpu
@pytest.mark.parametrize("d", dumps(), ids=lambda p: p.name)
def test_gpu_matches_cpu_032_ac1_ac8(d):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA / ROCm device")
    cpu_f, cpu_g, _ = run(d, "cpu")
    gpu_f, gpu_g, _ = run(d, "cuda")
    for k, v in cpu_f.items():
        diff = (gpu_f[k] - v).abs().max().item()
        assert diff <= FWD_TOL, f"{k}: forward max diff {diff:.3g} > {FWD_TOL}"
    worst = rel_errors(gpu_g, cpu_g)[-1]
    assert worst[0] <= GRAD_TOL, f"gradient rel L2 {worst[0]:.3g} at {worst[1]} > {GRAD_TOL}"


@pytest.mark.gpu
@pytest.mark.parametrize("d", dumps(), ids=lambda p: p.name)
def test_cuda_bf16_close_to_cpu_032_r8(d):
    if not torch.cuda.is_available() or ROCM:
        pytest.skip("CUDA only (ROCm trains in f32, 032-R8)")
    cpu_f, cpu_g, cpu_loss = run(d, "cpu")
    gpu_f, gpu_g, loss = run(d, "cuda", bf16=True)
    assert abs(loss - cpu_loss) <= BF16_LOSS_REL_TOL * abs(cpu_loss), f"loss {loss} vs {cpu_loss}"
    for k, v in cpu_f.items():
        diff = (gpu_f[k] - v).abs().max().item()
        assert diff <= BF16_FWD_TOL, f"{k}: forward max diff {diff:.3g} > {BF16_FWD_TOL}"
    rel = [r for r, _ in rel_errors(gpu_g, cpu_g)]
    median = rel[len(rel) // 2]
    assert median <= BF16_GRAD_MEDIAN_TOL, f"gradient rel L2 median {median:.3g} > {BF16_GRAD_MEDIAN_TOL}"
