"""044-R6 / 044-AC6 distillation round verdict (tools/eval/distill_verdict.py, AMB-044-7 Resolved 2026-11-14): pass =
the whole 050-R10 gate + TvZ >= 16 / 32 + late held_8m >= 24 / 64 + held sum >= S - 6; d2 (round d1 only) = not
passed but T0 Protoss wins >= S + 4 with TvZ kept; stop otherwise (d2 is the last round); an unknown input that the
decision depends on gives `incomplete`. Boundaries inclusive. CPU, no game data."""

import json
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "eval"))

import distill_verdict as dv  # noqa: E402
import eval_common as ec  # noqa: E402
import protocol  # noqa: E402

from test_eval_protocol import LIN, build, dump, make_job, row  # noqa: E402
from test_regression_gate import val  # noqa: E402


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    monkeypatch.setattr(ec, "load_env", lambda: None)


def tvz(won, missing=0, n=32):
    return {"won": won, "n": n, "invalid": 0, "missing": missing, "skipped_lines": 0, "complete": missing == 0}


def rush(early, late, missing=0):
    return {"early": early, "late": late, "n_early": 64, "n_late": 64, "invalid": 0, "missing": missing, "skipped_lines": 0, "complete": missing == 0}


S_RUSH = rush(20, 36)   # S's held sum 56: the round needs >= 50


@pytest.mark.parametrize("args, want", [
    # pass: every condition at its boundary
    (("PASS", tvz(16), rush(26, 24), 5), "pass"),
    # one short of each pass condition, Protoss +4 over S (1) and TvZ kept: d2
    (("FAIL", tvz(16), rush(26, 24), 5), "d2"),
    (("PASS", tvz(16), rush(25, 24), 5), "d2"),          # held sum 49 < 50
    # TvZ not kept: stop whatever the rest
    (("PASS", tvz(15), rush(30, 30), 9), "stop"),
    (("PASS", tvz(20), rush(30, 23), 9), "stop"),
    # not passed and Protoss +3 only: stop
    (("FAIL", tvz(20), rush(30, 30), 4), "stop"),
    # gate MISSING: could still pass -> incomplete; a definite failure elsewhere decides
    (("MISSING", tvz(20), rush(30, 30), 9), "incomplete"),
    (("MISSING", tvz(15), rush(30, 30), 9), "stop"),
    (("MISSING", tvz(20), rush(25, 24), 9), "d2"),
    # unfinished TvZ job: unknown
    (("PASS", tvz(20, missing=1), rush(30, 30), 9), "incomplete"),
    (("FAIL", tvz(20), rush(30, 30), None), "incomplete"),  # Protoss wins unknown
])
def test_decide_d1_044_r6(args, want):
    gate, t, r, protoss = args
    word, cond = dv.decide("d1", gate, t, r, S_RUSH, protoss, 1)
    assert word == want, cond


def test_decide_d2_is_the_last_round_044_r6():
    assert dv.decide("d2", "PASS", tvz(16), rush(26, 24), S_RUSH, 1, 1)[0] == "pass"
    assert dv.decide("d2", "FAIL", tvz(30), rush(30, 30), S_RUSH, 20, 1)[0] == "stop", "no d3"
    with pytest.raises(SystemExit, match="d2 is the last round"):
        dv.decide("d3", "PASS", tvz(16), rush(26, 24), S_RUSH, 1, 1)
    _, cond = dv.decide("d1", "FAIL", tvz(16), rush(26, 24), S_RUSH, 5, 1)
    assert cond == {"gate": False, "tvz_wins": True, "late_held": True, "held_sum": True, "protoss_gain": True, "tvz_kept": True}


def games_job(tmp_path: Path, name: str, pid: str, rows: list[dict], point: str | None = None, sha: str = "a" * 64) -> Path:
    """An eval job folder `name` with one stage of protocol plan `pid` (its stage folder and plan name `pid`)."""
    job = make_job(tmp_path / "_", pid, rows).rename(tmp_path / name)
    if point is not None:
        (job / "protocol.json").write_text(json.dumps({"lineage": "m2", "point": point, "checkpoint": f"checkpoints/m2-distill-{point}/ckpt-9",
                                                        "checkpoint_sha256": sha}), encoding="utf-8")
    return job


def t0_rows(t_won: int, p_won: int) -> list[dict]:
    return [row(n, n <= t_won, bunker_s=None) for n in range(1, 33)] + [row(n, n - 32 <= p_won, bunker_s=None) for n in range(33, 65)]


def rush_rows(monkeypatch, early_held: int, late_held: int) -> list[dict]:
    """Games 1..64 early (first target hit), 65..128 late: start_condition stubbed (it needs uranium._core)."""
    seeds = tomllib.loads(protocol.render(LIN, "rush_check", "c", "n"))["seeds"]
    early_seeds = set(seeds[:64])
    monkeypatch.setattr(ec, "start_condition", lambda seed: "early" if seed in early_seeds else "late")
    early, late = range(1, 65), range(65, 129)
    return [row(n, False, scv_8m=8 if i < early_held else 7) for i, n in enumerate(early)] + \
           [row(n, False, scv_8m=8 if i < late_held else None) for i, n in enumerate(late)]


def test_verdict_end_to_end_044_ac6(tmp_path, monkeypatch):
    """main(): jobs in, verdict out (first line the verdict, then the conditions), JSON; the file is never
    overwritten; the round's jobs must be of the round's point and of one checkpoint."""
    cache = tmp_path / "cache"
    base = games_job(tmp_path, "base", "t0_tp", t0_rows(16, 10))
    cand = games_job(tmp_path, "cand", "t0_tp", t0_rows(16, 6), point="d1")      # Protoss 6 < 10: gate FAIL
    s_t0 = games_job(tmp_path, "s", "t0_tp", t0_rows(13, 1))
    for j in (base, cand, s_t0):
        for n in range(1, 65):
            dump(cache / j.name / "t0_tp" / f"g{n}.json", [[100, 0, 9, "0101"]] * (5 if n > 32 else 2000) + [build(500, 10, 10, 109)])
    tvz_job = games_job(tmp_path, "tvz", "tvz_bc", [{**row(n, n <= 20), "model": {**row(n, True)["model"], "zero_last_failed": True}} for n in range(1, 33)], point="d1")
    rush_job = games_job(tmp_path, "rush", "rush_check", rush_rows(monkeypatch, 20, 30), point="d1")
    s_rush = games_job(tmp_path, "srush", "rush_check", rush_rows(monkeypatch, 19, 37))
    t, nl, b = {"terran": 0.7, "protoss": 0.65}, {"terran": 0.9, "protoss": 1.0}, {"terran": 0.02, "protoss": 0.01}
    bv, cv = tmp_path / "b.json", tmp_path / "c.json"
    bv.write_text(json.dumps(val(t, nl, b)), encoding="utf-8")
    cv.write_text(json.dumps(val(t, nl, b)), encoding="utf-8")
    verdict = tmp_path / "m2-lineage" / "d1_first_test.verdict"
    args = ["--round", "d1", "--baseline-val", str(bv), "--candidate-val", str(cv), "--baseline-games", str(base),
            "--candidate-games", str(cand), "--candidate-tvz", str(tvz_job), "--candidate-rush", str(rush_job),
            "--s-games", str(s_t0), "--s-rush", str(s_rush), "--cache", str(cache), "--verdict", str(verdict), "--json", str(tmp_path / "v.json")]
    out = dv.main(args)
    # gate FAIL (Protoss 6 < 10); TvZ 20, late 30, held 50 >= 56 - 6; Protoss 6 >= 1 + 4 -> d2
    assert out["verdict"] == "d2" and out["gate"]["overall"] == "FAIL"
    assert (out["tvz"]["won"], out["rush"]["early"], out["rush"]["late"], out["s_rush"]["early"] + out["s_rush"]["late"]) == (20, 20, 30, 56)
    assert (out["protoss_wins"], out["s_protoss_wins"]) == (6, 1)
    assert out["teacher_mode"]["protoss"]["teacher"] == 32 and out["teacher_mode"]["terran"]["teacher"] == 0
    assert out["candidate"] == {"checkpoint": "checkpoints/m2-distill-d1/ckpt-9", "checkpoint_sha256": "a" * 64, "no_protocol_json": []}
    lines = verdict.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "d2" and all(x.startswith("# ") for x in lines[1:])
    assert any("TvZ 32: 20/32 won" in x for x in lines) and any("teacher mode" in x and "TvP 32/32" in x for x in lines)
    assert json.loads((tmp_path / "v.json").read_text(encoding="utf-8"))["verdict"] == "d2"
    with pytest.raises(SystemExit, match="is not overwritten"):
        dv.main(args)
    # a job of another point, or of another checkpoint, is refused
    games_job(tmp_path, "rush2", "rush_check", rush_rows(monkeypatch, 20, 30), point="d2")
    with pytest.raises(SystemExit, match="is of point 'd2', not round d1"):
        dv.main([x if x != str(rush_job) else str(tmp_path / "rush2") for x in args[:-4]])
    games_job(tmp_path, "rush3", "rush_check", rush_rows(monkeypatch, 20, 30), point="d1", sha="b" * 64)
    with pytest.raises(SystemExit, match="different checkpoints"):
        dv.main([x if x != str(rush_job) else str(tmp_path / "rush3") for x in args[:-4]])


def test_unfinished_rush_job_is_incomplete_044_r6(tmp_path, monkeypatch):
    rows = rush_rows(monkeypatch, 30, 30)
    job = games_job(tmp_path, "r", "rush_check", rows[:-1])
    c = dv.rush_counts(job, LIN)
    assert c["missing"] == 1 and not c["complete"]
    assert dv.decide("d1", "PASS", tvz(20), c, S_RUSH, 9, 1)[0] == "incomplete"
    with pytest.raises(SystemExit, match="0 stages of the protocol plan tvz_bc / tvz_rl"):
        dv.tvz_counts(job, LIN)
