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
    "rocm": {"URANIUM_LOC_ROWS": "exact", "URANIUM_LOC_OUT": "mulsum", "URANIUM_BIAS_GRAD": "table_t", "URANIUM_CONV_WGRAD": "bmm", "URANIUM_ENTITY_ATTN": "fused", "URANIUM_LINEAR_WGRAD": "splitk"},
    "cuda": {"URANIUM_LOC_ROWS": "pow2", "URANIUM_ENTITY_ATTN": "fused"},
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


COMPILE_MODES = ("off", "default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead")
# `URANIUM_COMPILE_DYNAMIC`: shapes (entity count, location rows) compiled as seen and made
# symbolic after a change (`auto`), symbolic from the start (`on`) or one graph per shape (`off`).
COMPILE_DYNAMIC = {"auto": None, "on": True, "off": False}
COMPILE_RECOMPILE_LIMIT = 64
# `URANIUM_COMPILE_RNG`: dropout draws from Inductor's own generator (`inductor`) or from the eager
# generator (`eager`: the draws of an uncompiled run, so hosts with and without compile agree).
COMPILE_RNG_MODES = ("inductor", "eager")


def compiled(fn):
    """`fn` through `torch.compile` when `URANIUM_COMPILE` names a mode (AMB-000-4 step 2: same
    math; Triton needed on GPU, `triton-windows` on Windows), else `fn` itself. The recompile
    limit is raised so a new shape recompiles instead of falling back to eager."""
    from uranium.model.layers import env_choice

    mode = env_choice("URANIUM_COMPILE", COMPILE_MODES, "off")
    if mode == "off":
        return fn
    dynamic = COMPILE_DYNAMIC[env_choice("URANIUM_COMPILE_DYNAMIC", tuple(COMPILE_DYNAMIC), "auto")]
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, COMPILE_RECOMPILE_LIMIT)
    torch._inductor.config.fallback_random = env_choice("URANIUM_COMPILE_RNG", COMPILE_RNG_MODES, "inductor") == "eager"
    return torch.compile(fn, mode=None if mode == "default" else mode, dynamic=dynamic)


# `URANIUM_ADAM`: the Adam / AdamW update as PyTorch picks (`default`: multi-tensor on GPU), one
# kernel per parameter (`single`), multi-tensor (`foreach`) or the fused kernel (`fused`). Same
# update up to float rounding.
ADAM_IMPLS = ("default", "foreach", "fused", "single")


def adam_impl() -> dict:
    """Keyword arguments of `torch.optim.Adam(W)` for `URANIUM_ADAM`."""
    from uranium.model.layers import env_choice

    mode = env_choice("URANIUM_ADAM", ADAM_IMPLS, "default")
    return {} if mode == "default" else {"fused": True} if mode == "fused" else {"foreach": mode == "foreach"}


def mark_dynamic(obs: dict, labels: dict) -> None:
    """For torch.compile: the entity count (dim 1 of the entity tensors) and the location row count
    are symbolic from the first graph, so a new size does not recompile."""
    n = obs["entity_mask"].shape[1]
    for v in obs.values():
        if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] == n:
            torch._dynamo.maybe_mark_dynamic(v, 1)
    for k in ("loc_rows", "loc_w"):
        if isinstance(labels.get(k), torch.Tensor):
            torch._dynamo.maybe_mark_dynamic(labels[k], 0)


def autocast(device: torch.device, precision: str):
    """032-R8: bf16 autocast for matmul / conv on CUDA; f32 elsewhere (a no-op context)."""
    if precision == "bf16":
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()
