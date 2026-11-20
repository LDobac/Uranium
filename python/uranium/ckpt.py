import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from uranium import kernels
from uranium.model import UraniumNet, UraniumNetConfig

WEIGHTS_FILE = "model.safetensors"
META_FILE = "meta.json"
BURN_WEIGHTS_FILE = "model.bpk"
OPTIM_MAIN = "main"
AGENTS_DIR = "agents"
SPATIAL_NORM_PREFIX = "variant.spatial_norm."


def schema_versions() -> tuple[int, int]:
    from uranium import _core

    d = json.loads(_core.feature_dims())
    return d["feature_version"], d["action_version"]


def feature_load_compat() -> list[int]:
    from uranium import _core

    return list(json.loads(_core.feature_dims())["feature_load_compat"])


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
    compat = feature_load_compat()
    if meta["feature_version"] != fv and meta["feature_version"] not in compat:
        raise ValueError(f"{dir}: feature schema {meta['feature_version']} != {fv} (load-compatible: {compat})")
    if meta["action_version"] != av:
        raise ValueError(f"{dir}: action schema {meta['action_version']} != {av}")
    return meta


def no_dropout(cfg: UraniumNetConfig) -> UraniumNetConfig:
    cfg.entity.dropout = 0.0
    return cfg


def load_model(dir: Path, device="cpu", adjust=None, graft=None) -> tuple[UraniumNet, dict]:
    meta = read_meta(dir)
    cfg = UraniumNetConfig.from_dict(meta["config"])
    if adjust is not None:
        cfg = adjust(cfg)
    fresh_norm = False
    if graft is not None and graft.grafts_onto(cfg.variant):
        fresh_norm = cfg.variant.spatial_norm == 0
        cfg.variant = graft
    net = UraniumNet(cfg)
    sd = load_file(str(Path(dir) / WEIGHTS_FILE))
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if fresh_norm:
        missing = [k for k in missing if not k.startswith(SPATIAL_NORM_PREFIX)]
    if missing or unexpected:
        raise ValueError(f"{dir}: missing tensors {missing}, unexpected {unexpected}")
    net = net.to(device)
    kernels.check(net, device)
    return net, meta


def policy_version(dir: Path) -> str:
    h = hashlib.sha256()
    with open(Path(dir) / WEIGHTS_FILE, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_optimizer(dir: Path, name: str, net: torch.nn.Module, opt: torch.optim.Optimizer) -> None:
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
            "exp_avg": torch.empty_like(p).copy_(t[f"{pname}.exp_avg"]),
            "exp_avg_sq": torch.empty_like(p).copy_(t[f"{pname}.exp_avg_sq"]),
        }
