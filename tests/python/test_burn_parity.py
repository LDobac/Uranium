"""032-AC10 (AMB-000-4): the PyTorch UraniumNet reproduces the Burn model on CPU — forward
outputs, BC loss, every gradient, the global clip (041-R3, R19) and one AdamW step — from the
dumps of `tools/burn-parity` (tiny configs in tests/fixtures/burn_parity, full-size dumps via
`URANIUM_PARITY_DIRS`). Burn's one-hot distance / relative-position bias is a gather here
(032-R13 2026-10-23): the same forward values, the gradient summed in another order."""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from uranium.bc.clip import clip_global_norm
from uranium.model import UraniumNet, UraniumNetConfig, action_columns, to_labels, to_obs
from uranium.model.burn import burn_to_torch, torch_to_burn

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "burn_parity"
HEADS = ("action", "delay", "queued", "selected", "target_unit", "target_loc")


def dumps() -> list[Path]:
    dirs = sorted(p for p in FIXTURES.glob("*") if (p / "meta.json").is_file())
    dirs += [Path(d) for d in os.environ.get("URANIUM_PARITY_DIRS", "").split(os.pathsep) if d]
    return dirs


def loss_of(out: dict, inp: dict, meta: dict) -> torch.Tensor:
    """The BC loss of the dump (041-R1, R7, R19): per-head CE over the batch-wide valid steps,
    win MSE over decided steps, win_open MSE added last."""
    w = inp["weights"]
    ce = lambda lp: (-lp * w).sum() / meta["valid"]  # noqa: E731
    target, vw = inp["results"], inp["value_w"]
    loss = ((out["value"][:, 0] - target) ** 2 * vw).sum() / meta["value_valid"] * meta["value_weight"]
    for k, hw in zip(HEADS, meta["head_weights"]):
        loss = loss + ce(out["logp"][k]) * hw
    if out["win_open"] is not None:
        loss = loss + ((out["win_open"] - target) ** 2 * vw).sum() / meta["value_valid"] * meta["win_open_weight"]
    return loss


def rel_l2(a: torch.Tensor, b: torch.Tensor, floor: float) -> float:
    return float((a - b).norm() / max(float(b.norm()), floor))


@pytest.mark.parametrize("d", dumps(), ids=lambda p: p.name)
def test_matches_burn_032_ac10(d: Path):
    torch.manual_seed(0)
    meta = json.loads((d / "meta.json").read_text())
    cfg = UraniumNetConfig.from_dict(meta["config"])
    net = UraniumNet(cfg, action_columns())
    assert sum(p.numel() for p in net.parameters()) == meta["num_params"]
    burn_w = load_file(d / "weights.safetensors")
    net.load_state_dict(burn_to_torch(net, burn_w))
    inp = load_file(d / "inputs.safetensors")
    obs, labels = to_obs(inp, "cpu"), to_labels(inp, "cpu")
    b, t = meta["b"], meta["t"]

    out = net.forward_seq(obs, labels, inp["privileged"], net.init_state(b, "cpu"), t)
    want = load_file(d / "outputs.safetensors")
    for k in HEADS:
        assert torch.allclose(out["logp"][k], want[f"logp.{k}"], atol=1e-5, rtol=1e-5), f"logp.{k}: max diff {(out['logp'][k] - want[f'logp.{k}']).abs().max():.3g}"
        assert torch.allclose(out["entropy"][k], want[f"entropy.{k}"], atol=1e-5, rtol=1e-5), f"entropy.{k}"
    assert torch.allclose(out["value"], want["value"], atol=1e-5, rtol=1e-5), "value"
    if "win_open" in want:
        assert torch.allclose(out["win_open"], want["win_open"], atol=1e-5, rtol=1e-5), "win_open"
    loss = loss_of(out, inp, meta)
    assert torch.allclose(loss, want["loss"].reshape(()), atol=1e-5, rtol=1e-5), f"loss {loss.item()} vs {want['loss'].item()}"

    loss.backward()
    grads = torch_to_burn(net, {k: p.grad for k, p in net.named_parameters()})
    burn_g = load_file(d / "grads.safetensors")
    floor = 1e-3 * max(float(g.norm()) for g in burn_g.values())
    worst = max((rel_l2(grads[k], burn_g[k], floor), k) for k in burn_g)
    assert worst[0] <= 1e-4, f"gradient rel L2 {worst[0]:.3g} at {worst[1]}"

    apart = net.win_open_params()
    norm = clip_global_norm(list(net.parameters()), meta["clip"], apart)
    if meta["clip_norm"] is not None:
        assert norm.item() == pytest.approx(meta["clip_norm"], rel=1e-4)
    clipped = torch_to_burn(net, {k: p.grad for k, p in net.named_parameters()})
    burn_c = load_file(d / "clipped.safetensors")
    worst = max((rel_l2(clipped[k], burn_c[k], floor * meta["clip"]), k) for k in burn_c)
    assert worst[0] <= 1e-4, f"clipped gradient rel L2 {worst[0]:.3g} at {worst[1]}"

    a = meta["adam"]
    opt = torch.optim.AdamW(net.parameters(), lr=meta["lr"], betas=(a["beta1"], a["beta2"]), eps=a["eps"], weight_decay=meta["weight_decay"])
    opt.step()
    stepped = torch_to_burn(net, dict(net.named_parameters()))
    burn_a = load_file(d / "adamw.safetensors")
    for k, v in burn_a.items():
        assert torch.allclose(stepped[k].detach(), v, atol=1e-6, rtol=1e-5), f"AdamW step {k}: max diff {(stepped[k].detach() - v).abs().max():.3g}"
