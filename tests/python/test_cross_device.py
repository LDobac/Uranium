"""032-AC1 / AC8 (pytest `gpu`): the same weights and inputs on CPU and on the GPU (CUDA or
ROCm) give the same forward outputs and gradients within the device tolerance. f32 here (the
bf16 tolerances of 032-R8 are TBD(M6))."""

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
# 032-AC1 forward, 032-AC8 gradient (relative L2). CUDA f32 uses TF32 matmul (1e-2).
FWD_TOL = 1e-4 if ROCM else 1e-2
GRAD_TOL = 1e-3 if ROCM else 1e-2


def dumps():
    dirs = sorted(p for p in FIX.glob("*") if (p / "meta.json").is_file())
    return dirs + [Path(d) for d in os.environ.get("URANIUM_PARITY_DIRS", "").split(os.pathsep) if d]


def run(d: Path, device: str):
    meta = json.loads((d / "meta.json").read_text())
    net = UraniumNet(UraniumNetConfig.from_dict(meta["config"]), action_columns())
    net.load_state_dict(burn_to_torch(net, load_file(d / "weights.safetensors")))
    net = net.to(device)
    inp = load_file(d / "inputs.safetensors")
    out = net.forward_seq(to_obs(inp, device), to_labels(inp, device), inp["privileged"].to(device), net.init_state(meta["b"], device), meta["t"])
    w = inp["weights"].to(device)
    loss = sum((-out["logp"][k] * w).sum() for k in HEADS) / meta["valid"] + out["value"][:, 0].square().mean()
    loss.backward()
    fwd = {f"logp.{k}": out["logp"][k].detach().cpu() for k in HEADS} | {"value": out["value"].detach().cpu()}
    grads = {k: p.grad.detach().cpu() for k, p in net.named_parameters() if p.grad is not None}
    return fwd, grads


@pytest.mark.gpu
@pytest.mark.parametrize("d", dumps(), ids=lambda p: p.name)
def test_gpu_matches_cpu_032_ac1_ac8(d):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA / ROCm device")
    if ROCM:
        torch.backends.cudnn.benchmark = True  # MIOpen find (041-R15)
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
    cpu_f, cpu_g = run(d, "cpu")
    gpu_f, gpu_g = run(d, "cuda")
    for k, v in cpu_f.items():
        diff = (gpu_f[k] - v).abs().max().item()
        assert diff <= FWD_TOL, f"{k}: forward max diff {diff:.3g} > {FWD_TOL}"
    floor = 1e-3 * max(float(g.norm()) for g in cpu_g.values())
    worst = max((float((gpu_g[k] - g).norm()) / max(float(g.norm()), floor), k) for k, g in cpu_g.items())
    assert worst[0] <= GRAD_TOL, f"gradient rel L2 {worst[0]:.3g} at {worst[1]} > {GRAD_TOL}"
