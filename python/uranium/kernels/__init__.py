import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from uranium.env import env_choice

ENTITY_ATTN = "URANIUM_ENTITY_ATTN"
ENTITY_ATTN_MODES = ("torch", "fused")
REQUESTED = "requested:"
MASKED = -1e9
WIN_BUILD_ENV = {"TORCH_DONT_CHECK_COMPILER_ABI": "1"}
_HERE = Path(__file__).resolve().parent
_LOADED: dict = {}
_ERRORS: dict = {}
_WARNED: set = set()


class KernelError(RuntimeError):
    pass


def build_dir(name: str) -> Path:
    root = Path(os.environ.get("URANIUM_KERNEL_DIR") or Path.home() / ".cache" / "uranium" / "kernels")
    tag = f"torch{torch.__version__}-{'hip' + str(torch.version.hip) if torch.version.hip else 'cuda' + str(torch.version.cuda)}".replace("+", "_")
    d = root / tag / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def msvc_env() -> None:
    if sys.platform != "win32" or shutil.which("cl"):
        return
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        return
    root = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], capture_output=True, text=True).stdout.strip()
    bat = Path(root) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not root or not bat.exists():
        return
    out = subprocess.run(f'"{bat}" >nul && set', shell=True, capture_output=True, text=True).stdout
    for line in out.splitlines():
        k, sep, v = line.partition("=")
        if sep and k:
            os.environ[k] = v


@contextlib.contextmanager
def build_env():
    if sys.platform != "win32":
        yield
        return
    old = {k: os.environ.get(k) for k in WIN_BUILD_ENV}
    os.environ.update(WIN_BUILD_ENV)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def build(name: str):
    from torch.utils.cpp_extension import load as cpp_load

    msvc_env()
    if torch.version.hip and not os.environ.get("PYTORCH_ROCM_ARCH") and torch.cuda.is_available():
        os.environ["PYTORCH_ROCM_ARCH"] = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    win = ["/Zc:preprocessor"] if sys.platform == "win32" else []
    with build_env():
        return cpp_load(name=f"uranium_{name}", sources=[str(_HERE / f"{name}.cu")], extra_cflags=["-O3", *win], extra_cuda_cflags=["-O3", *(f"-Xcompiler={f}" for f in win)], build_directory=str(build_dir(name)), verbose=False)


def load(name: str):
    if name in _LOADED:
        return _LOADED[name]
    mod = None
    try:
        mod = build(name)
    except Exception as e:  # noqa: BLE001
        lines = str(e).strip().splitlines()
        _ERRORS[name] = f"{type(e).__name__}: {lines[0][:300] if lines else ''}"
    _LOADED[name] = mod
    return mod


def load_error(name: str) -> str | None:
    return _ERRORS.get(name)


def entity_attn_requested() -> str:
    return env_choice(ENTITY_ATTN, ENTITY_ATTN_MODES, "torch")


def entity_attn_unsupported(net) -> str | None:
    from uranium.kernels.attention import HEAD_DIM, MAX_TABLE_ROWS

    c = net.cfg.entity
    width = c.d_model // c.heads
    rows = net.entity.bias_table.shape[0] + net.cfg.variant.tokens()
    if width != HEAD_DIM:
        return f"head width {width} != {HEAD_DIM}"
    if rows > MAX_TABLE_ROWS:
        return f"distance bias table rows {rows} > {MAX_TABLE_ROWS}"
    return None


def on_gpu(device) -> bool:
    return torch.device(device).type == "cuda"


def check(net, device) -> str:
    if entity_attn_requested() != "fused" or not on_gpu(device):
        return "torch"
    why = entity_attn_unsupported(net)
    if why is not None:
        if why not in _WARNED:
            _WARNED.add(why)
            print(f"[kernels] WARNING: attn_bias does not support this model ({why}): {ENTITY_ATTN}=fused runs the PyTorch reference (060-R1 ⑥)", file=sys.stderr, flush=True)
        return "torch"
    if load("attn_bias") is None:
        raise KernelError(f"[kernels] attn_bias: {ENTITY_ATTN}=fused is requested on {'ROCm' if torch.version.hip else 'CUDA'} but the kernel did not build or load ({load_error('attn_bias')}). Fix the build or set {ENTITY_ATTN}=torch to run the PyTorch reference (060-R1 ⑤)")
    return "fused"


def meta(net, device) -> list[str]:
    from uranium.device import impl_settings

    used = check(net, device)
    want = os.environ.get(ENTITY_ATTN, "")
    entry = f"{ENTITY_ATTN}={used}"
    out = [entry if s.startswith(f"{ENTITY_ATTN}=") else s for s in impl_settings()]
    if entry not in out:
        out.append(entry)
    if want and want != used:
        out.append(f"{REQUESTED}{ENTITY_ATTN}={want}")
    return out
