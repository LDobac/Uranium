"""060-R2 bench suite on CPU: cases run and fill the report table, the real-data BC bench runs a
lane group sweep on synthetic shards, and two runs from the same seed log the same loss curve
(the parity comparison of AMB-000-4 step 2 is exact for identical runs)."""

import argparse
from pathlib import Path

import pytest
import torch

from uranium import _core
from uranium.bench.cases import Ctx, run_suites
from uranium.bench.realbc import compare_losses, run_group
from uranium.bench.report import bc_markdown, environment, markdown
from uranium.bench.timing import measure, peak_rss_gib

CPU = torch.device("cpu")


def test_measure_stops_when_steady():
    t = measure(lambda: sum(range(1000)), CPU, warmup=1, min_reps=10, budget_s=1.0)
    assert t.reps >= 10 and 0 < t.median_ms <= t.p95_ms
    rss = peak_rss_gib()
    assert rss is None or rss > 0


def test_suite_cases_fill_the_report():
    rows = run_suites(["kernels"], Ctx(CPU, "f32", True, 0.2), "gather_unit_cap_mask|softmax_masked|lstm_step", log=lambda m: None)
    assert [r.case for r in rows] == ["lstm_step_b16", "gather_unit_cap_mask", "softmax_masked_128sq"]
    assert all(r.status == "ok" and r.fwd is not None for r in rows), [(r.case, r.note) for r in rows]
    assert rows[2].fwd_bwd is not None
    md = markdown(rows, environment("cpu", "f32", CPU), "test")
    assert "| softmax_masked_128sq |" in md and "failed: none" in md


@pytest.fixture
def shards(tmp_path: Path) -> Path:
    d = tmp_path / "shards"
    _core.write_test_shards(str(d), "mixed", 2)
    return d


def bc_args(config: Path, shards: Path, updates: int) -> argparse.Namespace:
    return argparse.Namespace(config=str(config), stage="s1", core="lstm", dataset=[], shards=[str(shards)], synthetic_maps=True, seed=3, updates=updates, ops=0, trace=None)


def test_bc_bench_sweeps_groups_and_logs_parity(tiny_config, shards, monkeypatch):
    monkeypatch.setenv("URANIUM_BC_PREFETCH", "0")
    a = bc_args(tiny_config, shards, 2)
    runs = [run_group(a, g, CPU, "f32", log=lambda m: None) for g in (1, 0)]
    assert [r["group"] for r in runs] == [1, 0]
    assert all(len(r["updates"]) == 3 and r["median_s"] > 0 for r in runs)
    again = run_group(a, 1, CPU, "f32", log=lambda m: None)
    same = compare_losses(runs[0], again)
    assert same["updates"] == 3 and same["max_rel"] == 0.0 and same["finite"]
    # Another group size draws other dropout masks (041-R3), so the curves differ but stay finite;
    # the exact group equivalence at dropout 0 is test_bc's 041-R3 test.
    assert compare_losses(runs[0], runs[1])["finite"]
    md = bc_markdown(runs, environment("cpu", "f32", CPU), "test")
    assert md.count("\n| ") >= 3
