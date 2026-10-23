"""Host settings from `.env` (001-R9): the first `.env` in the working directory or a parent,
read like the Rust binaries' dotenvy. A variable already in the process environment wins."""

import os
from pathlib import Path


def _value(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] == "'":
        return raw[1:-1]  # single quotes: literal (Windows paths, 001-R9 note in docs)
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        return bytes(raw[1:-1], "utf-8").decode("unicode_escape")
    return raw.split(" #", 1)[0].strip()


def find(start: Path | None = None) -> Path | None:
    d = (start or Path.cwd()).resolve()
    for p in [d, *d.parents]:
        if (p / ".env").is_file():
            return p / ".env"
    return None


def load(path: Path | None = None) -> Path | None:
    """Loads `.env` into `os.environ` without overriding set variables; returns the file used."""
    path = path or find()
    if path is None:
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, raw = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = _value(raw)
    return path


# glibc mallopt parameters (malloc.h).
M_TRIM_THRESHOLD = -1
M_MMAP_THRESHOLD = -3


def apply_malloc() -> int | None:
    """`URANIUM_MALLOC_MMAP_THRESHOLD` (bytes, Linux glibc): blocks at least this large are mapped
    and given back to the OS when freed. glibc otherwise raises its threshold after such a free
    and keeps the BC batches' feature blocks (MBs each) in its heap, so a process holding one
    batch stays at the peak of two (Linux ROCm host: Feeder 8.2 -> 2.4 GB peak RSS at 1 MiB,
    AMB-000-4 step 2). Host memory only; returns the threshold set, or None."""
    import sys

    raw = os.environ.get("URANIUM_MALLOC_MMAP_THRESHOLD", "")
    if not raw or not sys.platform.startswith("linux"):
        return None
    import ctypes

    n = int(raw)
    libc = ctypes.CDLL("libc.so.6")
    if not (libc.mallopt(M_MMAP_THRESHOLD, n) and libc.mallopt(M_TRIM_THRESHOLD, n)):
        raise RuntimeError(f"URANIUM_MALLOC_MMAP_THRESHOLD={n}: mallopt refused")
    return n
