import os
import subprocess
from pathlib import Path


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def redact(path) -> str:
    p = Path(path).resolve()
    sd = os.environ.get("URANIUM_SHARED_DIR")
    if sd:
        try:
            return "<shared>/" + p.relative_to(Path(sd).resolve()).as_posix()
        except ValueError:
            pass
    return Path(path).as_posix()
