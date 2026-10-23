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
