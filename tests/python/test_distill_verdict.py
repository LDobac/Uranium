"""044-R6 / 044-AC6 distillation round verdict (tools/eval/distill_verdict.py, AMB-044-7 Resolved 2026-11-14): pass =
the whole 050-R10 gate + TvZ >= 16 / 32 + late held_8m >= 24 / 64 + held sum >= S - 6; d2 (round d1 only) = not
passed but T0 Protoss wins >= S + 4 with TvZ kept; stop otherwise (d2 is the last round). Only these three
(user decision 2026-11-14): a MISSING gate check or a missing cell fails; an unfinished or damaged input job is refused
with no verdict and no file. Boundaries inclusive. CPU, no game data."""

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
import regression_gate as rg  # noqa: E402

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


def cells(t: int | None, p: int | None, missing: str = "") -> dict:
    """teacher_cells-shaped TvT / TvP teacher-mode cells (None = no value)."""
    def one(x):
        why = missing if x is not None else (missing or "no games")
        return {"teacher": x, "n": 32, "ok": not why and x <= dv.TEACHER_MODE_MAX, "missing": why}
    return {"terran": one(t), "protoss": one(p)}


CELLS_OK = cells(3, 3)


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
    # user decision 2026-11-14: a MISSING gate check fails, a missing cell fails
    (("MISSING", tvz(20), rush(30, 30), 9), "d2"),
    (("MISSING", tvz(15), rush(30, 30), 9), "stop"),
    (("MISSING", tvz(20), rush(30, 30), 4), "stop"),
    (("PASS", tvz(20, missing=1), rush(30, 30), 9), "stop"),   # TvZ count with a game missing: not kept
    (("PASS", tvz(20), rush(30, 30, missing=1), 9), "stop"),
    (("FAIL", tvz(20), rush(30, 30), None), "stop"),           # no Protoss count
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
    # 044-R8 teacher round t1 (user confirmed 2026-11-14): pass or stop only, never "d2" (teacher-mode cells, 2026-11-15).
    assert dv.decide("t1", "PASS", tvz(16), rush(26, 24), S_RUSH, 1, 1, CELLS_OK)[0] == "pass"
    assert dv.decide("t1", "FAIL", tvz(16), rush(26, 24), S_RUSH, 5, 1, CELLS_OK)[0] == "stop", "the d1 'd2' case is a stop for t1"
    assert dv.decide("t1", "FAIL", tvz(30), rush(30, 30), S_RUSH, 20, 1, CELLS_OK)[0] == "stop"
    assert "t1" in dv.ROUNDS
    _, cond = dv.decide("d1", "FAIL", tvz(16), rush(26, 24), S_RUSH, 5, 1)
    assert cond == {"gate": False, "tvz_wins": True, "late_held": True, "held_sum": True, "protoss_gain": True, "tvz_kept": True}


def games_job(tmp_path: Path, name: str, pid: str, rows: list[dict], point: str | None = None, sha: str = "a" * 64,
              done: bool = True) -> Path:
    """A finished (DONE) eval job folder `name` with one stage of protocol plan `pid` (stage folder and plan name `pid`)."""
    job = make_job(tmp_path / "_", pid, rows).rename(tmp_path / name)
    if point is not None:
        (job / "protocol.json").write_text(json.dumps({"lineage": "m2", "point": point, "checkpoint": f"checkpoints/m2-{point}/ckpt-9",
                                                        "checkpoint_sha256": sha}), encoding="utf-8")
    if done:
        (job / "DONE").write_text("", encoding="utf-8")
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
    base = games_job(tmp_path, "base", "t0_tp", t0_rows(16, 10), point="s2e2", sha="0" * 64)
    cand = games_job(tmp_path, "cand", "t0_tp", t0_rows(16, 6), point="d1")      # Protoss 6 < 10: gate FAIL
    s_t0 = games_job(tmp_path, "s", "t0_tp", t0_rows(13, 1), point="r2", sha="2" * 64)
    for j in (base, cand, s_t0):
        for n in range(1, 65):
            dump(cache / j.name / "t0_tp" / f"g{n}.json", [[100, 0, 9, "0101"]] * (5 if n > 32 else 2000) + [build(500, 10, 10, 109)])
    tvz_job = games_job(tmp_path, "tvz", "tvz_bc", [{**row(n, n <= 20), "model": {**row(n, True)["model"], "zero_last_failed": True}} for n in range(1, 33)], point="d1")
    rush_job = games_job(tmp_path, "rush", "rush_check", rush_rows(monkeypatch, 20, 30), point="d1")
    s_rush = games_job(tmp_path, "srush", "rush_check", rush_rows(monkeypatch, 19, 37), point="r2", sha="2" * 64)
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
    assert out["candidate"] == {"checkpoint": "checkpoints/m2-d1/ckpt-9", "checkpoint_sha256": "a" * 64, "no_protocol_json": []}
    assert out["reference"] == {"baseline_checkpoint": "checkpoints/m2-s2e2/ckpt-9", "s_point": "r2", "s_checkpoint": "checkpoints/m2-r2/ckpt-9"}
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
    # the baseline must be s2e2's job, S's two jobs one selection candidate point and checkpoint
    with pytest.raises(SystemExit, match="baseline job s is of point 'r2', not s2e2"):
        dv.main([x if x != str(base) else str(s_t0) for x in args[:-4]])
    games_job(tmp_path, "srush1", "rush_check", rush_rows(monkeypatch, 19, 37), point="r1", sha="1" * 64)
    assert dv.S_POINTS == ("r1", "r2", "r3", "s2e2")
    with pytest.raises(SystemExit, match="S's jobs must be of one selection candidate point"):
        dv.main([x if x != str(s_rush) else str(tmp_path / "srush1") for x in args[:-4]])
    games_job(tmp_path, "srush4", "rush_check", rush_rows(monkeypatch, 19, 37))
    with pytest.raises(SystemExit, match="srush4 has no lineage-2 protocol.json"):
        dv.main([x if x != str(s_rush) else str(tmp_path / "srush4") for x in args[:-4]])


def test_unfinished_or_damaged_job_is_refused_044_r6(tmp_path, monkeypatch):
    """User decision 2026-11-14: no verdict and no file while an input job has no DONE or has undecodable lines."""
    rows = rush_rows(monkeypatch, 30, 30)
    run = games_job(tmp_path, "running", "rush_check", rows, done=False)
    bad = games_job(tmp_path, "bad", "rush_check", rows)
    (bad / "results" / "h.jsonl").write_bytes((bad / "results" / "h.jsonl").read_bytes() + b'{"game_id": 1\n')
    ok = games_job(tmp_path, "ok", "rush_check", rows)
    dv.check_finished([ok])
    with pytest.raises(SystemExit, match=r"running: not finished \(no DONE\)"):
        dv.check_finished([ok, run])
    with pytest.raises(SystemExit, match=r"bad: 1 undecodable line\(s\) \(results/h.jsonl:129\)"):
        dv.check_finished([bad])
    verdict = tmp_path / "v.verdict"
    v = tmp_path / "val.json"
    v.write_text("{}", encoding="utf-8")
    args = ["--round", "d1", "--baseline-games", str(ok), "--candidate-games", str(ok), "--candidate-tvz", str(ok),
            "--candidate-rush", str(run), "--s-games", str(ok), "--s-rush", str(ok), "--verdict", str(verdict)]
    with pytest.raises(SystemExit, match="no verdict while an input job is unfinished"):
        dv.main(args + ["--baseline-val", str(v), "--candidate-val", str(v)])
    # with --verdict both val outputs must be given and exist (a forgotten flag would fail checks 2 / 3a for good)
    with pytest.raises(SystemExit, match="--verdict needs --baseline-val"):
        dv.main(args + ["--candidate-val", str(v)])
    with pytest.raises(SystemExit, match="--verdict needs --candidate-val .*missing.json"):
        dv.main(args + ["--baseline-val", str(v), "--candidate-val", str(tmp_path / "missing.json")])
    assert not verdict.exists()


def test_rush_job_with_a_missing_game_fails_044_r6(tmp_path, monkeypatch):
    rows = rush_rows(monkeypatch, 30, 30)
    job = games_job(tmp_path, "r", "rush_check", rows[:-1])
    c = dv.rush_counts(job, LIN)
    assert c["missing"] == 1 and not c["complete"]
    word, cond = dv.decide("d1", "PASS", tvz(20), c, S_RUSH, 9, 1)
    assert word == "stop" and cond["late_held"] is False and cond["held_sum"] is False
    with pytest.raises(SystemExit, match="0 stages of the protocol plan tvz_bc / tvz_rl"):
        dv.tvz_counts(job, LIN)


# --- 044-R6 / 044-R7 / 050-R10 amended 2026-11-15 (user-delegated, Fable advice 2026-11-15): from t1 on the per-race
# NLL is recorded only, teacher-mode games <= 3 / 32 per matchup join pass, and the RL start D is t1 or d1.


def gate_of(nll_t: float, nll_p: float, top1_t: float = 0.7) -> dict:
    b = {"terran": 0.02, "protoss": 0.01}
    return rg.gate(val({"terran": 0.7, "protoss": 0.65}, {"terran": 0.9, "protoss": 1.0}, b),
                   val({"terran": top1_t, "protoss": 0.65}, {"terran": nll_t, "protoss": nll_p}, b), None, None)


def test_t1_nll_record_only_044_r6_050_r10():
    """The ② nll.action check never fails from t1 on: RECORD, WARN above baseline + 0.03 (the boundary is no warning);
    top-1 and the other checks stay."""
    g = gate_of(0.95, 1.03)   # terran +0.05, protoss +0.03
    assert {(c["race"], c["result"]) for c in g["checks"] if c["metric"] == "nll.action"} == {("terran", "FAIL"), ("protoss", "FAIL")}
    r = dv.nll_record_only(g)
    assert {(c["race"], c["result"]) for c in r["checks"] if c["metric"] == "nll.action"} == {("terran", "WARN"), ("protoss", "RECORD")}
    assert g["overall"] == "FAIL" and r["overall"] == "MISSING", "no T0 games given: checks 1 / 3b MISSING decide"
    only_val = lambda g: dv.nll_record_only({**g, "checks": [c for c in g["checks"] if c["check"] in ("2", "3a")]})  # noqa: E731
    assert only_val(gate_of(0.95, 1.03))["overall"] == "PASS"
    assert only_val(gate_of(0.90, 1.0, top1_t=0.694))["overall"] == "FAIL", "top-1 -0.6 %p still fails"
    assert {c["result"] for c in dv.nll_record_only(rg.gate(None, None, None, None))["checks"] if c["metric"] == "nll.action"} == {"RECORD"}


@pytest.mark.parametrize("tc, want", [
    (CELLS_OK, "pass"),                       # 3 / 32 each: the boundary passes
    (cells(4, 0), "stop"),
    (cells(0, 4), "stop"),
    (cells(None, 0), "stop"),                 # no TvT value: fails (as a MISSING cell)
    (cells(0, 0, missing="1 game(s) without a command stream"), "stop"),
    (None, "stop"),
])
def test_decide_t1_teacher_mode_044_r6(tc, want):
    word, cond = dv.decide("t1", "PASS", tvz(16), rush(26, 24), S_RUSH, 1, 1, tc)
    assert word == want and cond["teacher_mode"] is (want == "pass")
    # d1 / d2 do not judge teacher mode (their verdicts stand)
    for rnd in ("d1", "d2"):
        word, cond = dv.decide(rnd, "PASS", tvz(16), rush(26, 24), S_RUSH, 1, 1, cells(30, 30))
        assert word == "pass" and "teacher_mode" not in cond


def test_teacher_cells_missing_data_fails_044_r6():
    tm = {"terran": {"n": 32, "teacher": 3, "no_stream": 0}, "protoss": {"n": 32, "teacher": 4, "no_stream": 0}}
    c = {"skipped_lines": [], "terran": {"missing": 0}, "protoss": {"missing": 0}}
    got = dv.teacher_cells(tm, c)
    assert (got["terran"]["ok"], got["protoss"]["ok"], got["protoss"]["teacher"]) == (True, False, 4)
    assert not any(dv.teacher_cells({"error": "no binary"}, c)[r]["ok"] for r in ("terran", "protoss"))
    assert dv.teacher_cells({"protoss": tm["protoss"]}, c)["terran"]["missing"] == "no games"
    assert "command stream" in dv.teacher_cells({**tm, "terran": {**tm["terran"], "no_stream": 1}}, c)["terran"]["missing"]
    assert "without a result" in dv.teacher_cells(tm, {**c, "terran": {"missing": 1}})["terran"]["missing"]
    assert dv.teacher_cells(tm, {**c, "skipped_lines": [{"file": "x", "line": 1}]})["terran"]["missing"] == "undecodable job lines"
    assert not dv.teacher_cells(tm, None)["terran"]["ok"]


def test_d_choice_044_r7():
    """D = t1 when the verdict passes, or when every 044-R7 condition holds (boundaries inclusive); else d1."""
    ck = "checkpoints/m2-teacher-t1/ckpt-9"
    d = dv.d_choice("pass", tvz(0), rush(0, 0), 0, cells(9, 9), ck)
    assert (d["d"], d["checkpoint"], d["by"]) == ("t1", ck, "verdict pass")
    d = dv.d_choice("stop", tvz(16), rush(36, 24), 20, CELLS_OK, ck)
    assert (d["d"], d["checkpoint"], d["by"]) == ("t1", ck, "044-R7 conditions met")
    assert all(c["ok"] for c in d["conditions"].values()) and d["conditions"]["held_sum"]["value"] == 60
    for args in ((tvz(15), rush(36, 24), 20, CELLS_OK), (tvz(16), rush(37, 23), 20, CELLS_OK), (tvz(16), rush(35, 24), 20, CELLS_OK),
                 (tvz(16), rush(36, 24), 19, CELLS_OK), (tvz(16), rush(36, 24), 20, cells(4, 3)), (tvz(16), rush(36, 24), 20, cells(3, 4)),
                 (tvz(16, missing=1), rush(36, 24), 20, CELLS_OK), (tvz(16), rush(36, 24, missing=1), 20, CELLS_OK),
                 (tvz(16), rush(36, 24), None, CELLS_OK), (tvz(16), rush(36, 24), 20, None)):
        d = dv.d_choice("stop", *args, ck)
        assert (d["d"], d["checkpoint"]) == ("d1", dv.D1_CHECKPOINT), args
    assert dv.D1_CHECKPOINT == "checkpoints/m2-distill-d1/ckpt-2000"
    full = {"skipped_lines": [], "terran": {"n": 32, "won": 12, "missing": 0}, "protoss": {"n": 32, "won": 8, "missing": 0}}
    assert dv.t0_wins(full) == 20
    assert dv.t0_wins({**full, "terran": {"n": 31, "won": 12, "missing": 1}}) is None
    assert dv.t0_wins({**full, "skipped_lines": [{"file": "x", "line": 1}]}) is None
    assert dv.t0_wins({"skipped_lines": [], "terran": full["terran"]}) is None


def t0_dumps(cache: Path, job: Path, teacher: set[int]) -> None:
    """Command streams of a t0_tp job: games in `teacher` low-APM (teacher mode), the others 2,000 commands before 6:00."""
    for n in range(1, 65):
        dump(cache / job.name / "t0_tp" / f"g{n}.json", [[100, 0, 9, "0101"]] * (5 if n in teacher else 2000) + [build(500, 10, 10, 109)])


def test_verdict_t1_end_to_end_044_r6_r7(tmp_path, monkeypatch):
    """main() --round t1: NLL above +0.03 only warns, teacher mode TvP 4 / 32 stops, D by the 044-R7 rule (comment
    lines of the verdict file, whose first line stays the verdict, and `d_choice` in the JSON); S must be r2."""
    cache = tmp_path / "cache"
    base = games_job(tmp_path, "base", "t0_tp", t0_rows(16, 10), point="s2e2", sha="0" * 64)
    cand = games_job(tmp_path, "cand", "t0_tp", t0_rows(16, 10), point="t1")
    cand2 = games_job(tmp_path, "cand2", "t0_tp", t0_rows(16, 10), point="t1")
    s_t0 = games_job(tmp_path, "s", "t0_tp", t0_rows(13, 1), point="r2", sha="2" * 64)
    t0_dumps(cache, base, set())
    t0_dumps(cache, s_t0, set(range(33, 65)))
    t0_dumps(cache, cand, {1, 2, 33, 34, 35, 36})        # TvT 2, TvP 4
    t0_dumps(cache, cand2, {1, 2, 3, 33, 34, 35})        # TvT 3, TvP 3
    tvz_job = games_job(tmp_path, "tvz", "tvz_bc", [{**row(n, n <= 20), "model": {**row(n, True)["model"], "zero_last_failed": True}} for n in range(1, 33)], point="t1")
    rush_job = games_job(tmp_path, "rush", "rush_check", rush_rows(monkeypatch, 20, 30), point="t1")     # held 50
    rush60 = games_job(tmp_path, "rush60", "rush_check", rush_rows(monkeypatch, 30, 30), point="t1")     # held 60
    s_rush = games_job(tmp_path, "srush", "rush_check", rush_rows(monkeypatch, 19, 37), point="r2", sha="2" * 64)
    t, b = {"terran": 0.7, "protoss": 0.65}, {"terran": 0.02, "protoss": 0.01}
    bv, cv = tmp_path / "b.json", tmp_path / "c.json"
    bv.write_text(json.dumps(val(t, {"terran": 0.9, "protoss": 1.0}, b)), encoding="utf-8")
    cv.write_text(json.dumps(val(t, {"terran": 0.95, "protoss": 1.02}, b)), encoding="utf-8")   # +0.05 WARN, +0.02 RECORD
    verdict = tmp_path / "m2-lineage" / "t1_first_test.verdict"
    args = ["--round", "t1", "--baseline-val", str(bv), "--candidate-val", str(cv), "--baseline-games", str(base),
            "--candidate-games", str(cand), "--candidate-tvz", str(tvz_job), "--candidate-rush", str(rush_job),
            "--s-games", str(s_t0), "--s-rush", str(s_rush), "--cache", str(cache), "--verdict", str(verdict), "--json", str(tmp_path / "v.json")]
    out = dv.main(args)
    assert out["gate"]["overall"] == "PASS"
    assert {(c["race"], c["result"]) for c in out["gate"]["checks"] if c["metric"] == "nll.action"} == {("terran", "WARN"), ("protoss", "RECORD")}
    assert (out["teacher_cells"]["terran"]["teacher"], out["teacher_cells"]["protoss"]["teacher"]) == (2, 4)
    assert out["verdict"] == "stop" and out["conditions"]["teacher_mode"] is False and out["conditions"]["held_sum"]
    d = out["d_choice"]
    assert (d["d"], d["checkpoint"]) == ("d1", "checkpoints/m2-distill-d1/ckpt-2000")
    assert {k: (c["value"], c["ok"]) for k, c in d["conditions"].items()} == {
        "tvz_wins": (20, True), "late_held": (30, True), "held_sum": (50, False), "t0_wins": (26, True),
        "teacher_mode_terran": (2, True), "teacher_mode_protoss": (4, False)}
    lines = verdict.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "stop" and all(x.startswith("# ") for x in lines[1:])
    assert any(x.startswith("# 044-R7 RL start: D = d1 (checkpoints/m2-distill-d1/ckpt-2000") for x in lines)
    assert any("nll.action" in x and "terran 0.9000 -> 0.9500 WARN" in x for x in lines)
    assert json.loads((tmp_path / "v.json").read_text(encoding="utf-8"))["d_choice"]["d"] == "d1"
    # teacher mode 3 / 32 each and held 60: pass, D = t1 (the round's checkpoint)
    swap = {str(cand): str(cand2), str(rush_job): str(rush60)}
    out = dv.main([swap.get(x, x) for x in args[:-4]])
    assert out["verdict"] == "pass" and (out["d_choice"]["d"], out["d_choice"]["checkpoint"]) == ("t1", "checkpoints/m2-t1/ckpt-9")
    # top-1 down 0.6 %p: gate FAIL -> stop, but every 044-R7 condition holds -> D = t1
    cv.write_text(json.dumps(val({"terran": 0.694, "protoss": 0.65}, {"terran": 0.9, "protoss": 1.0}, b)), encoding="utf-8")
    out = dv.main([swap.get(x, x) for x in args[:-4]])
    assert out["verdict"] == "stop" and out["gate"]["overall"] == "FAIL"
    assert (out["d_choice"]["d"], out["d_choice"]["by"]) == ("t1", "044-R7 conditions met")
    # t1's S is r2 (044-R8)
    s1 = games_job(tmp_path, "s1", "t0_tp", t0_rows(13, 1), point="r1", sha="1" * 64)
    srush1 = games_job(tmp_path, "srush1", "rush_check", rush_rows(monkeypatch, 19, 37), point="r1", sha="1" * 64)
    swap = {str(s_t0): str(s1), str(s_rush): str(srush1)}
    with pytest.raises(SystemExit, match="round t1's S is r2"):
        dv.main([swap.get(x, x) for x in args[:-4]])


def t1_args(tmp_path: Path, monkeypatch, point: str | None = "t1", cand_top1: float = 0.7) -> tuple[list[str], Path, Path]:
    """--round t1 arguments (with --verdict) over finished jobs: TvT / TvP teacher mode 1 / 1, held 60, TvZ 20; returns
    (args, the verdict path, the round t0_tp job's command stream cache)."""
    cache = tmp_path / "cache"
    base = games_job(tmp_path, "base", "t0_tp", t0_rows(16, 10), point="s2e2", sha="0" * 64)
    cand = games_job(tmp_path, "cand", "t0_tp", t0_rows(16, 10), point=point)
    s_t0 = games_job(tmp_path, "s", "t0_tp", t0_rows(13, 1), point="r2", sha="2" * 64)
    t0_dumps(cache, base, set())
    t0_dumps(cache, s_t0, set())
    t0_dumps(cache, cand, {1, 33})
    tvz_job = games_job(tmp_path, "tvz", "tvz_bc", [{**row(n, n <= 20), "model": {**row(n, True)["model"], "zero_last_failed": True}} for n in range(1, 33)], point=point)
    rush_job = games_job(tmp_path, "rush", "rush_check", rush_rows(monkeypatch, 30, 30), point=point)
    s_rush = games_job(tmp_path, "srush", "rush_check", rush_rows(monkeypatch, 19, 37), point="r2", sha="2" * 64)
    b = {"terran": 0.02, "protoss": 0.01}
    bv, cv = tmp_path / "b.json", tmp_path / "c.json"
    bv.write_text(json.dumps(val({"terran": 0.7, "protoss": 0.65}, {"terran": 0.9, "protoss": 1.0}, b)), encoding="utf-8")
    cv.write_text(json.dumps(val({"terran": cand_top1, "protoss": 0.65}, {"terran": 0.9, "protoss": 1.0}, b)), encoding="utf-8")
    verdict = tmp_path / "t1_first_test.verdict"
    return (["--round", "t1", "--baseline-val", str(bv), "--candidate-val", str(cv), "--baseline-games", str(base),
             "--candidate-games", str(cand), "--candidate-tvz", str(tvz_job), "--candidate-rush", str(rush_job),
             "--s-games", str(s_t0), "--s-rush", str(s_rush), "--cache", str(cache), "--verdict", str(verdict)],
            verdict, cache / cand.name / "t0_tp")


def without_verdict(args: list[str]) -> list[str]:
    i = args.index("--verdict")
    return args[:i] + args[i + 2:]


def test_t1_verdict_needs_the_d_checkpoint_044_r7(tmp_path, monkeypatch):
    """A t1 verdict file names D's checkpoint: refused (no file) when no round job has a protocol.json; without
    --verdict the run still reports (D = t1 shown as '?')."""
    args, verdict, _ = t1_args(tmp_path, monkeypatch, point=None)
    with pytest.raises(SystemExit, match="needs the round's checkpoint"):
        dv.main(args)
    assert not verdict.exists()
    out = dv.main(without_verdict(args))
    assert out["verdict"] == "pass" and (out["d_choice"]["d"], out["d_choice"]["checkpoint"]) == ("t1", "?")


def test_t1_verdict_refused_while_a_command_stream_is_missing_044_r6(tmp_path, monkeypatch, capsys):
    """A valid T0 TvT / TvP game without a command stream refuses a t1 verdict file (a passing cmddump failure must not
    fix D = d1 for good); without --verdict the teacher-mode cell is MISSING (a fail) with the reason, also when every
    game of a race lacks its stream. The t1 report lists the failing gate checks."""
    args, verdict, streams = t1_args(tmp_path, monkeypatch, cand_top1=0.694)
    (streams / "g5.json").unlink()
    with pytest.raises(SystemExit, match=r"have no command stream \(TvT 1\)"):
        dv.main(args)
    assert not verdict.exists()
    capsys.readouterr()
    out = dv.main(without_verdict(args))
    assert out["teacher_cells"]["terran"] == {"teacher": 1, "n": 31, "ok": False, "missing": "1 game(s) without a command stream"}
    assert out["teacher_cells"]["protoss"]["ok"] and out["verdict"] == "stop" and out["d_choice"]["d"] == "d1"
    text = capsys.readouterr().out
    # gate: top-1 -0.6 %p FAIL; 3b bunker_order_4m MISSING (a stream gone); nll.action RECORD is not listed
    assert "050-R10 gate vs m2-s2e2 (2 nll.action record only): FAIL (2 terran action_top1 FAIL, 3b terran bunker_order_4m MISSING)" in text
    assert "teacher mode (050-R10, need <= 3 each): TvT - (1 game(s) without a command stream), TvP 1/32: NO" in text
    assert "teacher mode TvT 1, TvP 1 <= 3/32: NO, yes" in text
    for n in range(1, 33):
        (streams / f"g{n}.json").unlink(missing_ok=True)
    out = dv.main(without_verdict(args))
    assert out["teacher_cells"]["terran"]["missing"] == "32 game(s) without a command stream"
    # streams not counted at all (no cmddump binary): refused with --verdict, MISSING without
    monkeypatch.setattr(dv, "t0_teacher_mode", lambda job, cache, lin: {"error": "no uranium-replay"})
    with pytest.raises(SystemExit, match=r"command streams are not counted \(no uranium-replay\)"):
        dv.main(args)
    assert not verdict.exists()
    assert dv.main(without_verdict(args))["teacher_cells"]["protoss"]["missing"] == "command streams not counted"
