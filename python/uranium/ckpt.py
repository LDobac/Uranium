"""Checkpoint files (032-R6, 061-R10): `model.safetensors` (f32 weights, names sorted, so the same
weights give the same bytes), `optim-<name>.safetensors` (per-parameter AdamW state and step),
`meta.json` (net config, schema versions, normalisation stats, training summary, framework,
precision, hand-written kernels). `train_state.json`, `run_ident.json` and `config/` are written
by the Rust run loop (`uranium._core.run_bc`). The policy version (020-R5) is the SHA-256 of
`model.safetensors`."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from uranium.model import UraniumNet, UraniumNetConfig

WEIGHTS_FILE = "model.safetensors"
META_FILE = "meta.json"
BURN_WEIGHTS_FILE = "model.bpk"
OPTIM_MAIN = "main"
# RL checkpoints: one folder per learning agent, `agents/<name>/` (061-R10, 042).
AGENTS_DIR = "agents"


def schema_versions() -> tuple[int, int]:
    """(feature, action) schema versions of this build (030, 034)."""
    from uranium import _core

    d = json.loads(_core.feature_dims())
    return d["feature_version"], d["action_version"]


def optim_file(name: str) -> str:
    return f"optim-{name}.safetensors"


def save_model(dir: Path, net: torch.nn.Module) -> None:
    sd = net.state_dict()
    save_file({k: sd[k].detach().to("cpu", torch.float32).contiguous() for k in sorted(sd)}, str(Path(dir) / WEIGHTS_FILE))


def save_meta(dir: Path, cfg: UraniumNetConfig, train: dict, precision: str, kernels: list[str] | None = None) -> None:
    fv, av = schema_versions()
    meta = {
        "config": cfg.to_dict(),
        "feature_version": fv,
        "action_version": av,
        "norm": {"scalar_mean": [], "scalar_std": []},
        "train": train,
        "framework": "pytorch",
        "torch": torch.__version__,
        "precision": precision,
        "kernels": kernels or [],
    }
    (Path(dir) / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_meta(dir: Path) -> dict:
    dir = Path(dir)
    if not (dir / WEIGHTS_FILE).is_file():
        if (dir / BURN_WEIGHTS_FILE).is_file():
            raise ValueError(f"{dir} is a Burn checkpoint ({BURN_WEIGHTS_FILE}): Burn checkpoints are kept but not loaded (061-R10)")
        raise FileNotFoundError(f"{dir}: no {WEIGHTS_FILE}")
    meta = json.loads((dir / META_FILE).read_text(encoding="utf-8"))
    fv, av = schema_versions()
    if meta["feature_version"] != fv:
        raise ValueError(f"{dir}: feature schema {meta['feature_version']} != {fv}")
    if meta["action_version"] != av:
        raise ValueError(f"{dir}: action schema {meta['action_version']} != {av}")
    return meta


def load_model(dir: Path, device="cpu", adjust=None) -> tuple[UraniumNet, dict]:
    """The net of a checkpoint and its metadata; `adjust(cfg)` changes weightless settings
    (032 §2: dropout) before the net is built."""
    meta = read_meta(dir)
    cfg = UraniumNetConfig.from_dict(meta["config"])
    if adjust is not None:
        cfg = adjust(cfg)
    net = UraniumNet(cfg)
    sd = load_file(str(Path(dir) / WEIGHTS_FILE))
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise ValueError(f"{dir}: missing tensors {missing}, unexpected {unexpected}")
    return net.to(device), meta


def policy_version(dir: Path) -> str:
    """020-R5: SHA-256 of the weights file."""
    h = hashlib.sha256()
    with open(Path(dir) / WEIGHTS_FILE, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_optimizer(dir: Path, name: str, net: torch.nn.Module, opt: torch.optim.Optimizer) -> None:
    """AdamW state per parameter name: `<p>.exp_avg`, `<p>.exp_avg_sq`, `<p>.step`."""
    out = {}
    for pname, p in net.named_parameters():
        st = opt.state.get(p)
        if not st:
            continue
        out[f"{pname}.exp_avg"] = st["exp_avg"].detach().to("cpu", torch.float32).contiguous()
        out[f"{pname}.exp_avg_sq"] = st["exp_avg_sq"].detach().to("cpu", torch.float32).contiguous()
        out[f"{pname}.step"] = torch.as_tensor(st["step"], dtype=torch.float32).detach().reshape(1).cpu()
    save_file(dict(sorted(out.items())), str(Path(dir) / optim_file(name)))


def load_optimizer(dir: Path, name: str, net: torch.nn.Module, opt: torch.optim.Optimizer) -> None:
    """Puts the saved AdamW state back into `opt` (built over `net.parameters()`)."""
    t = load_file(str(Path(dir) / optim_file(name)))
    names = {k.rsplit(".", 1)[0] for k in t}
    params = dict(net.named_parameters())
    unknown = names - params.keys()
    if unknown:
        raise ValueError(f"{dir}: optimizer state for unknown parameters {sorted(unknown)[:5]}")
    fused = any(g.get("fused") or g.get("capturable") for g in opt.param_groups)
    for pname in names:
        p = params[pname]
        opt.state[p] = {
            "step": t[f"{pname}.step"].reshape(()).to(p.device if fused else "cpu"),
            # In the parameter's memory format (channels-last conv weights, 032-R19).
            "exp_avg": torch.empty_like(p).copy_(t[f"{pname}.exp_avg"]),
            "exp_avg_sq": torch.empty_like(p).copy_(t[f"{pname}.exp_avg_sq"]),
        }
