import contextlib
import os
import sys
import threading
import time

import torch

from uranium.env import env_choice

BACKENDS = ("cuda", "rocm", "cpu")


def detect() -> str:
    if torch.cuda.is_available():
        return "rocm" if torch.version.hip else "cuda"
    return "cpu"


def resolve(backend: str | None) -> tuple[str, torch.device, str]:
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


IMPL_DEFAULTS = {
    "rocm": {"URANIUM_LOC_ROWS": "exact", "URANIUM_LOC_OUT": "mulsum", "URANIUM_BIAS_GRAD": "table_t", "URANIUM_CONV_WGRAD": "bmm", "URANIUM_ENTITY_ATTN": "fused", "URANIUM_LINEAR_WGRAD": "splitk", "URANIUM_INFER_GRAPH": "on", "URANIUM_INFER_GRAPH_MAX_BATCH": "4"},
    "cuda": {"URANIUM_LOC_ROWS": "pow2", "URANIUM_ENTITY_ATTN": "fused", "URANIUM_COMPILE": "default", "URANIUM_INFER_GRAPH": "on", "URANIUM_INFER_COMPILE": "default"},
    "cpu": {},
}
TRAIN_DEFAULTS = {
    "rocm": {},
    "cuda": {"URANIUM_H2D": "pinned", "URANIUM_CHANNELS_LAST": "on", "URANIUM_SPATIAL_BF16": "on", "URANIUM_STEM_PAD": "on", "URANIUM_COMPILE": "reduce-overhead"},
    "cpu": {},
}


IMPL_SWITCHES = (
    "URANIUM_LOC_ROWS", "URANIUM_LOC_OUT", "URANIUM_H2D", "URANIUM_BIAS_GRAD", "URANIUM_CONV_WGRAD", "URANIUM_LINEAR_WGRAD",
    "URANIUM_ENTITY_ATTN", "URANIUM_ADAM", "URANIUM_COMPILE", "URANIUM_COMPILE_DYNAMIC", "URANIUM_COMPILE_RNG", "URANIUM_INFER_GRAPH",
    "URANIUM_INFER_COMPILE", "URANIUM_INFER_STREAM", "URANIUM_INFER_GRAPH_MAX_BATCH", "URANIUM_CHANNELS_LAST", "URANIUM_SPATIAL_BF16", "URANIUM_STEM_PAD",
)


def impl_settings() -> list[str]:
    return [f"{k}={os.environ[k]}" for k in IMPL_SWITCHES if os.environ.get(k)]


def configure(backend: str, train: bool = False) -> None:
    torch.backends.cudnn.benchmark = backend == "cuda"
    for k, v in (IMPL_DEFAULTS[backend] | (TRAIN_DEFAULTS[backend] if train else {})).items():
        os.environ.setdefault(k, v)


COMPILE_MODES = ("off", "default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead")
COMPILE_DYNAMIC = {"auto": None, "on": True, "off": False}
COMPILE_RECOMPILE_LIMIT = 64
COMPILE_RNG_MODES = ("eager", "inductor")


def compile_graphs(key: str = "URANIUM_COMPILE") -> bool:
    return env_choice(key, COMPILE_MODES, "off") in ("reduce-overhead", "max-autotune")


def compiled(fn, key: str = "URANIUM_COMPILE"):
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


ADAM_IMPLS = ("default", "foreach", "fused", "single")


def adam_impl() -> dict:
    mode = env_choice("URANIUM_ADAM", ADAM_IMPLS, "default")
    return {} if mode == "default" else {"fused": True} if mode == "fused" else {"foreach": mode == "foreach"}


def mark_dynamic(obs: dict, labels: dict) -> None:
    n = obs["entity_mask"].shape[1]
    for v in obs.values():
        if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] == n:
            torch._dynamo.maybe_mark_dynamic(v, 1)
    for k in ("loc_rows", "loc_w"):
        if isinstance(labels.get(k), torch.Tensor):
            torch._dynamo.maybe_mark_dynamic(labels[k], 0)


class DeviceLock:
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

EXIT_DEVICE_FATAL = 70
DEVICE_FATAL_TEXT = ("illegal memory access", "Sticky error", "CUDA error", "HIP error")
DEVICE_NONFATAL_TEXT = ("out of memory", "operation not permitted when stream is capturing", "capture invalidated")
DEVICE_NONFATAL_CODES = frozenset([2, *range(900, 909)])


def _chain(e: BaseException | None):
    seen: set[int] = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        yield e
        e = e.__cause__ or e.__context__


def is_device_fatal(e: BaseException | None) -> bool:
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
    if not is_device_fatal(e):
        return
    lines = str(e).strip().splitlines()
    first = lines[0] if lines else ""
    try:
        print(f"[device] {where}: fatal device error ({type(e).__name__}: {first}); exiting {EXIT_DEVICE_FATAL} without saving, the watchdog restarts from LATEST (042-R20, 060-R6)", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
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
    try:
        yield
    except BaseException as e:
        exit_if_device_fatal(e, where)
        raise


def autocast(device: torch.device, precision: str):
    if precision == "bf16":
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()
