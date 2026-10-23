import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# `tiny_bc` of the Rust trainer tests (crates/uranium-train/tests/bc.rs), as a config file.
TINY_BC = """
lanes = 2
seq_len = 4
micro_lanes = 1
pad_entities = 0
lr = 3e-3
min_lr = 3e-3
warmup_steps = 1
total_steps = 100
weight_decay = 0.0
grad_clip = 1.0
s2_lr_scale = 0.1

[heads]
action = 1.0
delay = 0.5
queued = 0.2
selected = 1.0
target_unit = 1.0
target_loc = 1.0

[value]
win = 0.5

[data]
z_drop = 0.0
augment = false
seed = 1
checkpoint_every = 1000
checkpoint_minutes = 0
enemy_race_mask = 0.0
"""


@pytest.fixture
def tiny_config(tmp_path: Path) -> Path:
    p = tmp_path / "bc_tiny.toml"
    p.write_text(TINY_BC, encoding="utf-8")
    return p


@pytest.fixture
def train_bin() -> Path:
    """The Rust `uranium-train` binary: `URANIUM_TRAIN_BIN`, else the newer of target/release and
    target/debug (an old build may lack the command under test)."""
    if os.environ.get("URANIUM_TRAIN_BIN"):
        return Path(os.environ["URANIUM_TRAIN_BIN"])
    found = [Path(p) for prof in ("release", "debug") for p in (ROOT / "target" / prof / "uranium-train", ROOT / "target" / prof / "uranium-train.exe") if Path(p).is_file()]
    if not found:
        pytest.skip("uranium-train binary not built (set URANIUM_TRAIN_BIN)")
    return max(found, key=lambda p: p.stat().st_mtime)


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT).stdout


@pytest.fixture
def python_cmd() -> list[str]:
    import sys

    return [sys.executable, "-m", "uranium"]


def which(name: str) -> str | None:
    return shutil.which(name)
