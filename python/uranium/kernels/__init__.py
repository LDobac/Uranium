"""Hand-written GPU kernels (060-R1): one `.cu` source per kernel, compiled at first use with
`torch.utils.cpp_extension` (CUDA hosts as is, ROCm hosts hipified), cached per PyTorch build in
`URANIUM_KERNEL_DIR` (default `~/.cache/uranium/kernels`). Every kernel has a pure PyTorch
reference and a switch that turns it off; a kernel that fails to build is reported once and the
reference runs instead."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_LOADED: dict = {}


def build_dir(name: str) -> Path:
    root = Path(os.environ.get("URANIUM_KERNEL_DIR") or Path.home() / ".cache" / "uranium" / "kernels")
    tag = f"torch{torch.__version__}-{'hip' + str(torch.version.hip) if torch.version.hip else 'cuda' + str(torch.version.cuda)}".replace("+", "_")
    d = root / tag / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def msvc_env() -> None:
    """Windows CUDA hosts: when `cl` is not on PATH, the environment of the newest Visual Studio's
    `vcvars64.bat` (found with vswhere), so the extension builds from a plain shell."""
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


def load(name: str):
    """The compiled module for `<name>.cu`, or None when it cannot be built (reason on stderr)."""
    if name in _LOADED:
        return _LOADED[name]
    mod = None
    try:
        from torch.utils.cpp_extension import load as cpp_load

        msvc_env()
        if torch.version.hip and not os.environ.get("PYTORCH_ROCM_ARCH") and torch.cuda.is_available():
            os.environ["PYTORCH_ROCM_ARCH"] = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        # CUDA 13's CCCL headers need MSVC's conforming preprocessor.
        win = ["/Zc:preprocessor"] if sys.platform == "win32" else []
        mod = cpp_load(name=f"uranium_{name}", sources=[str(_HERE / f"{name}.cu")], extra_cflags=["-O3", *win], extra_cuda_cflags=["-O3", *(f"-Xcompiler={f}" for f in win)], build_directory=str(build_dir(name)), verbose=False)
    except Exception as e:  # noqa: BLE001 - the reference path runs instead
        print(f"[kernels] {name}: not built ({str(e).splitlines()[0][:200]}), using the PyTorch reference", file=sys.stderr, flush=True)
    _LOADED[name] = mod
    return mod
