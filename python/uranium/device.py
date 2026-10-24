"""Backends (001-R10: device labels cuda / rocm / cpu) and their precision (032-R8: CUDA bf16
mixed, ROCm and CPU f32). ROCm runs through PyTorch's `cuda` device API."""

import contextlib
import os

import torch

BACKENDS = ("cuda", "rocm", "cpu")


def detect() -> str:
    if torch.cuda.is_available():
        return "rocm" if torch.version.hip else "cuda"
    return "cpu"


def resolve(backend: str | None) -> tuple[str, torch.device, str]:
    """(backend label, torch device, precision) for `backend` (None = this host's GPU, else CPU)."""
    backend = backend or detect()
    if backend not in BACKENDS:
        raise ValueError(f"backend {backend!r}: expected one of {BACKENDS}")
    if backend == "cpu":
        return backend, torch.device("cpu"), "f32"
    if not torch.cuda.is_available():
        raise RuntimeError(f"backend {backend}: no GPU visible to PyTorch")
    hip = torch.version.hip is not None
    if (backend == "rocm") != hip:
        raise RuntimeError(f"backend {backend}: this PyTorch is built for {'ROCm' if hip else 'CUDA'}")
    return backend, torch.device("cuda"), "bf16" if backend == "cuda" else "f32"


# AMB-000-4 step 2: per backend, the implementation each switch defaults to (same math, the faster
# one in the comparison on that backend, `reports/pytorch_opt.md` 2). A process variable or `.env`
# key wins.
IMPL_DEFAULTS = {
    "rocm": {"URANIUM_LOC_ROWS": "exact", "URANIUM_LOC_OUT": "mulsum", "URANIUM_BIAS_GRAD": "table_t", "URANIUM_CONV_WGRAD": "bmm"},
    "cuda": {},
    "cpu": {},
}


def configure(backend: str) -> None:
    """041-R15 kernel selection: cuDNN benchmark mode on CUDA hosts only. ROCm keeps MIOpen's
    immediate mode with its find-db (`MIOPEN_USER_DB_PATH`): benchmark mode there runs a MIOpen
    search for every new shape (4.6 s to 2 min per shape on the Linux ROCm host, 2026-10-24), and
    the location head's row count changes every call (032-R10). Then the backend's implementation
    defaults (`IMPL_DEFAULTS`); call before building the network."""
    torch.backends.cudnn.benchmark = backend == "cuda"
    for k, v in IMPL_DEFAULTS[backend].items():
        os.environ.setdefault(k, v)


def autocast(device: torch.device, precision: str):
    """032-R8: bf16 autocast for matmul / conv on CUDA; f32 elsewhere (a no-op context)."""
    if precision == "bf16":
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()
