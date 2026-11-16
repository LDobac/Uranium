import os
from pathlib import Path


def _value(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] == "'":
        return raw[1:-1]
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


M_TRIM_THRESHOLD = -1
M_MMAP_THRESHOLD = -3
M_ARENA_MAX = -8


def apply_malloc() -> int | None:
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


def apply_malloc_arenas() -> int | None:
    import sys

    raw = os.environ.get("URANIUM_MALLOC_ARENA_MAX", "")
    if not raw or not sys.platform.startswith("linux"):
        return None
    import ctypes

    n = int(raw)
    if n < 1:
        raise ValueError(f"URANIUM_MALLOC_ARENA_MAX={n}: must be at least 1 (glibc ignores 0 and below)")
    if not ctypes.CDLL("libc.so.6").mallopt(M_ARENA_MAX, n):
        raise RuntimeError(f"URANIUM_MALLOC_ARENA_MAX={n}: mallopt refused")
    return n
