"""032-AC1 / AC8 (pytest `gpu`): the same weights and inputs on CPU and on the GPU (CUDA or
ROCm) give the same forward outputs and gradients within the device tolerance. The f32 check runs
without TF32 on both backends (the port's math); the CUDA bf16 check (032-R8, training and
inference precision there) uses the 2026-10-24 tolerances (user decision, about 2x the measured
values, `reports/pytorch_opt.md` 2.24): forward ≤ 0.15, per-head policy KL mean ≤ 1e-3, greedy
agreement ≥ 95%, gradient relative L2 median ≤ 3%."""

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
# CUDA bf16 autocast (032-AC1 / AC8, user decision 2026-10-24).
BF16_FWD_TOL = 0.15
BF16_KL_MEAN_TOL = 1e-3
BF16_GREEDY_MIN = 0.95
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
    # Per-head log-prob rows [rows, K] (the Selected head's steps are 3-D, left out).
    dists = {k: v.detach().float().cpu() for k, v in out["dists"].items() if v.dim() == 2}
    return fwd, grads, float(loss), dists


def rel_errors(gpu_g: dict, cpu_g: dict) -> list:
    floor = 1e-3 * max(float(g.norm()) for g in cpu_g.values())
    return sorted((float((gpu_g[k] - g).norm()) / max(float(g.norm()), floor), k) for k, g in cpu_g.items())


@pytest.mark.gpu
@pytest.mark.parametrize("d", dumps(), ids=lambda p: p.name)
def test_gpu_matches_cpu_032_ac1_ac8(d):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA / ROCm device")
    cpu_f, cpu_g, _, _ = run(d, "cpu")
    gpu_f, gpu_g, _, _ = run(d, "cuda")
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
    cpu_f, cpu_g, cpu_loss, cpu_d = run(d, "cpu")
    gpu_f, gpu_g, loss, gpu_d = run(d, "cuda", bf16=True)
    assert abs(loss - cpu_loss) <= BF16_LOSS_REL_TOL * abs(cpu_loss), f"loss {loss} vs {cpu_loss}"
    for k, v in cpu_f.items():
        diff = (gpu_f[k] - v).abs().max().item()
        assert diff <= BF16_FWD_TOL, f"{k}: forward max diff {diff:.3g} > {BF16_FWD_TOL}"
    rel = [r for r, _ in rel_errors(gpu_g, cpu_g)]
    median = rel[len(rel) // 2]
    assert median <= BF16_GRAD_MEDIAN_TOL, f"gradient rel L2 median {median:.3g} > {BF16_GRAD_MEDIAN_TOL}"
    for k, p in cpu_d.items():
        q, ok = gpu_d[k], p > -1e8
        kl = (p.exp() * (p - q)).masked_fill(~ok, 0).sum(1).mean().item()
        assert kl <= BF16_KL_MEAN_TOL, f"{k}: policy KL mean {kl:.3g} > {BF16_KL_MEAN_TOL}"
        agree = (p.argmax(1) == q.argmax(1)).float().mean().item()
        assert agree >= BF16_GREEDY_MIN, f"{k}: greedy agreement {agree:.3f} < {BF16_GREEDY_MIN}"
