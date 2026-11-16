import os
import sys
from pathlib import Path

ENV_FILE_KEY = "URANIUM_ENV_FILE"
ESCAPES = {"\\": "\\", "'": "'", '"': '"', "$": "$", " ": " ", "n": "\n"}


def _comment(tail: str) -> bool:
    return not tail.strip() or (tail[0] in " \t" and tail.lstrip(" \t").startswith("#"))


def _weak(text: str, whole: bool) -> str | None:
    out = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 1
            if i == len(text) or text[i] not in ESCAPES:
                raise ValueError(f"escape {text[i - 1:i + 1]!r} in a double-quoted value (the Rust loader takes only {sorted(ESCAPES)} after a backslash)")
            out.append(ESCAPES[text[i]])
        elif c == '"' and not whole:
            return "".join(out) if _comment(text[i + 1:]) else None
        else:
            out.append(c)
        i += 1
    return "".join(out) if whole else None


def _value(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("#"):
        return ""
    if raw.startswith("'"):
        end = raw.find("'", 1)
        if end > 0 and _comment(raw[end + 1:]):
            return raw[1:end]
        if len(raw) >= 2 and raw.endswith("'"):
            return raw[1:-1]
    if raw.startswith('"'):
        v = _weak(raw[1:], False)
        if v is None and len(raw) >= 2 and raw.endswith('"'):
            v = _weak(raw[1:-1], True)
        if v is not None:
            return v
    return raw.split(" #", 1)[0].strip()


def find(start: Path | None = None) -> Path | None:
    d = (start or Path.cwd()).resolve()
    for p in [d, *d.parents]:
        if (p / ".env").is_file():
            return p / ".env"
    return None


def load(path: Path | None = None) -> Path | None:
    if path is None and ENV_FILE_KEY in os.environ:
        path = Path(os.environ[ENV_FILE_KEY])
        if not path.is_file():
            return None
    path = path or find()
    if path is None:
        return None
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, raw = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        try:
            value = _value(raw)
        except ValueError as e:
            print(f"warning: failed to load {path}: line {n}: {e}; the lines after it are not read", file=sys.stderr, flush=True)
            break
        if key and key not in os.environ:
            os.environ[key] = value
    return path


def env_choice(key: str, choices: tuple, default: str) -> str:
    v = os.environ.get(key, "") or default
    if v not in choices:
        raise ValueError(f"{key}={v!r}: expected one of {choices}")
    return v


M_TRIM_THRESHOLD = -1
M_MMAP_THRESHOLD = -3
M_ARENA_MAX = -8


def apply_malloc() -> int | None:
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
