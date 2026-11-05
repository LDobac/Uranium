"""Backends (001-R10: device labels cuda / rocm / cpu) and their precision (032-R8: CUDA bf16
mixed, ROCm and CPU f32). ROCm runs through PyTorch's `cuda` device API."""

import contextlib
import os
import sys
import threading
import time

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
    "rocm": {"URANIUM_LOC_ROWS": "exact", "URANIUM_LOC_OUT": "mulsum", "URANIUM_BIAS_GRAD": "table_t", "URANIUM_CONV_WGRAD": "bmm", "URANIUM_ENTITY_ATTN": "fused", "URANIUM_LINEAR_WGRAD": "splitk", "URANIUM_INFER_GRAPH": "on", "URANIUM_INFER_GRAPH_MAX_BATCH": "1"},
    "cuda": {"URANIUM_LOC_ROWS": "pow2", "URANIUM_ENTITY_ATTN": "fused", "URANIUM_COMPILE": "default", "URANIUM_INFER_GRAPH": "on", "URANIUM_INFER_COMPILE": "default"},
    "cpu": {},
}
# The BC trainer's extra defaults (`configure(train=True)`, `reports/pytorch_opt.md` 6, 2026-10-28):
# channels-last activations and weights, the spatial planes rounded to bf16 on the host, the stem
# input padded to 8-channel multiples, CUDA graph replay of the compiled forward / backward. Not
# for inference processes (eval jobs, SC:R, RL): channels-last makes torch.compile several times
# slower per shape, which a per-shape compiled `act` would pay at every warm-up.
TRAIN_DEFAULTS = {
    "rocm": {},
    "cuda": {"URANIUM_H2D": "pinned", "URANIUM_CHANNELS_LAST": "on", "URANIUM_SPATIAL_BF16": "on", "URANIUM_STEM_PAD": "on", "URANIUM_COMPILE": "reduce-overhead"},
    "cpu": {},
}


# Every implementation switch (same math, AMB-000-4 step 2), recorded in checkpoint metadata
# (041-R15) with the value in effect.
IMPL_SWITCHES = (
    "URANIUM_LOC_ROWS", "URANIUM_LOC_OUT", "URANIUM_H2D", "URANIUM_BIAS_GRAD", "URANIUM_CONV_WGRAD", "URANIUM_LINEAR_WGRAD",
    "URANIUM_ENTITY_ATTN", "URANIUM_ADAM", "URANIUM_COMPILE", "URANIUM_COMPILE_DYNAMIC", "URANIUM_COMPILE_RNG", "URANIUM_INFER_GRAPH",
    "URANIUM_INFER_COMPILE", "URANIUM_INFER_STREAM", "URANIUM_INFER_GRAPH_MAX_BATCH", "URANIUM_CHANNELS_LAST", "URANIUM_SPATIAL_BF16", "URANIUM_STEM_PAD",
)


def impl_settings() -> list[str]:
    """`KEY=value` of the implementation switches set in this process (after `configure`); unset
    ones run their built-in default."""
    return [f"{k}={os.environ[k]}" for k in IMPL_SWITCHES if os.environ.get(k)]


def configure(backend: str, train: bool = False) -> None:
    """041-R15 kernel selection: cuDNN benchmark mode on CUDA hosts only. ROCm keeps MIOpen's
    immediate mode with its find-db (`MIOPEN_USER_DB_PATH`): benchmark mode there runs a MIOpen
    search for every new shape (4.6 s to 2 min per shape on the Linux ROCm host, 2026-10-24), and
    the location head's row count changes every call (032-R10). Then the backend's implementation
    defaults (`IMPL_DEFAULTS`, and `TRAIN_DEFAULTS` first for the BC trainer); call before
    building the network."""
    torch.backends.cudnn.benchmark = backend == "cuda"
    for k, v in (IMPL_DEFAULTS[backend] | (TRAIN_DEFAULTS[backend] if train else {})).items():
        os.environ.setdefault(k, v)


COMPILE_MODES = ("off", "default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead")
# `URANIUM_COMPILE_DYNAMIC`: shapes (entity count, location rows) compiled as seen and made
# symbolic after a change (`auto`), symbolic from the start (`on`) or one graph per shape (`off`).
COMPILE_DYNAMIC = {"auto": None, "on": True, "off": False}
COMPILE_RECOMPILE_LIMIT = 64
# `URANIUM_COMPILE_RNG`: dropout draws from Inductor's own generator (`inductor`) or from the eager
# generator (`eager`: the draws of an uncompiled run, so hosts with and without compile agree).
COMPILE_RNG_MODES = ("eager", "inductor")


def compile_graphs(key: str = "URANIUM_COMPILE") -> bool:
    """`key` names a compile mode that replays CUDA graphs (`reduce-overhead`, `max-autotune`):
    the caller marks each new step (`torch.compiler.cudagraph_mark_step_begin`) and copies the
    outputs it keeps past the next call, which the next replay overwrites."""
    from uranium.model.layers import env_choice

    return env_choice(key, COMPILE_MODES, "off") in ("reduce-overhead", "max-autotune")


def compiled(fn, key: str = "URANIUM_COMPILE"):
    """`fn` through `torch.compile` when `key` (`URANIUM_COMPILE` for training,
    `URANIUM_INFER_COMPILE` for inference) names a mode (AMB-000-4 step 2: same math; Triton
    needed on GPU, `triton-windows` on Windows), else `fn` itself. The recompile limit is raised
    so a new shape recompiles instead of falling back to eager."""
    from uranium.model.layers import env_choice

    mode = env_choice(key, COMPILE_MODES, "off")
    if mode == "off":
        return fn
    import importlib.util

    if torch.cuda.is_available() and importlib.util.find_spec("triton") is None:
        print(f"[compile] {key}={mode} needs Triton (Windows: the triton-windows package); running eager", flush=True)
        return fn
    dynamic = COMPILE_DYNAMIC[env_choice("URANIUM_COMPILE_DYNAMIC", tuple(COMPILE_DYNAMIC), "auto")]
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, COMPILE_RECOMPILE_LIMIT)
    torch._inductor.config.fallback_random = env_choice("URANIUM_COMPILE_RNG", COMPILE_RNG_MODES, "eager") == "eager"
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


class DeviceLock:
    """CUDA / HIP graph capture against every other use of the device in this process (RL runs the
    learner and the inference servers as threads of one process): a capture fails, and fails the
    other thread's call, when that thread synchronises or frees device memory meanwhile.
    `shared()` around device work (any number at once), `exclusive()` around a capture (waits
    for the shared holders; new ones wait behind it). A thread inside `shared()` may call
    `exclusive()`: its shared holds are released for the capture and taken back after."""

    def __init__(self):
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting = 0
        self._mine = threading.local()

    def _held(self) -> int:
        return getattr(self._mine, "n", 0)

    @contextlib.contextmanager
    def shared(self):
        with self._cond:
            while self._writer or (self._waiting and not self._held()):
                self._cond.wait()
            self._readers += 1
        self._mine.n = self._held() + 1
        try:
            yield
        finally:
            self._mine.n -= 1
            with self._cond:
                self._readers -= 1
                self._cond.notify_all()

    @contextlib.contextmanager
    def exclusive(self):
        held = self._held()
        with self._cond:
            self._readers -= held
            self._waiting += 1
            self._cond.notify_all()
            while self._writer or self._readers:
                self._cond.wait()
            self._waiting -= 1
            self._writer = True
        self._mine.n = 0
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._readers += held
                self._cond.notify_all()
            self._mine.n = held


DEVICE_LOCK = DeviceLock()

# 042-R20 (AMB-042-20): a fatal device error (CUDA / HIP "sticky" errors such as an illegal memory
# access) leaves the process's device context unusable: every later call fails the same way. The
# process exits at once with this code, without a checkpoint save (it would fail or write garbage),
# so the 060-R6 watchdog restarts the run from LATEST.
EXIT_DEVICE_FATAL = 70
DEVICE_FATAL_TEXT = ("illegal memory access", "Sticky error", "CUDA error", "HIP error")
# Not fatal whatever else the message says (the context stays usable): out of memory (CUDA / HIP
# "out of memory", cudaErrorMemoryAllocation = 2) and CUDA graph capture errors (a capture that
# failed or was invalidated, cudaErrorStreamCapture* = 900-908).
DEVICE_NONFATAL_TEXT = ("out of memory", "operation not permitted when stream is capturing", "capture invalidated")
DEVICE_NONFATAL_CODES = frozenset([2, *range(900, 909)])


def _chain(e: BaseException | None):
    """`e` and the exceptions it was raised from (cause, else context), each once."""
    seen: set[int] = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        yield e
        e = e.__cause__ or e.__context__


def is_device_fatal(e: BaseException | None) -> bool:
    """`e` (or an exception it was raised from) is a fatal device error (042-R20):
    `torch.AcceleratorError`, or a message naming one (`DEVICE_FATAL_TEXT`, e.g. a Rust error
    carrying a Python traceback). Never fatal: out of memory (`torch.OutOfMemoryError`, a message
    with "out of memory", error code 2) and CUDA graph capture errors (their messages, error codes
    900-908) anywhere in the chain."""
    acc = getattr(torch, "AcceleratorError", None)
    oom = getattr(torch, "OutOfMemoryError", None)
    chain = list(_chain(e))
    for x in chain:
        if oom is not None and isinstance(x, oom):
            return False
        if getattr(x, "error_code", None) in DEVICE_NONFATAL_CODES:
            return False
        msg = str(x)
        if any(t in msg for t in DEVICE_NONFATAL_TEXT):
            return False
    for x in chain:
        if acc is not None and isinstance(x, acc):
            return True
        if any(t in str(x) for t in DEVICE_FATAL_TEXT):
            return True
    return False


def exit_if_device_fatal(e: BaseException, where: str) -> None:
    """042-R20: on a fatal device error, one line on stderr and `os._exit(70)` (no cleanup, no
    checkpoint save, any thread); otherwise returns and the caller handles `e` as before. The
    streams are flushed and the exit waits 0.2 s, so a tee or the watchdog's pipe still gets the
    line."""
    if not is_device_fatal(e):
        return
    lines = str(e).strip().splitlines()
    first = lines[0] if lines else ""
    try:
        print(f"[device] {where}: fatal device error ({type(e).__name__}: {first}); exiting {EXIT_DEVICE_FATAL} without saving, the watchdog restarts from LATEST (042-R20, 060-R6)", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - a closed stderr must not keep the process alive
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
    time.sleep(0.2)
    os._exit(EXIT_DEVICE_FATAL)


@contextlib.contextmanager
def device_fatal_guard(where: str):
    """042-R20 around device work: a fatal device error ends the process (`exit_if_device_fatal`),
    any other exception propagates unchanged."""
    try:
        yield
    except BaseException as e:
        exit_if_device_fatal(e, where)
        raise


def autocast(device: torch.device, precision: str):
    """032-R8: bf16 autocast for matmul / conv on CUDA; f32 elsewhere (a no-op context)."""
    if precision == "bf16":
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()
