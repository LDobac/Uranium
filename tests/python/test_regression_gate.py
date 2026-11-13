"""050-R10 / 050-AC5 regression gate (tools/eval/regression_gate.py): every check splits at its AMB-050-3 threshold
(inclusive boundary passes, just past it fails), missing inputs are MISSING, the T0 t0_tp counts come from an eval job
like tables.py (opponent-crash games invalid, both bunker definitions); and bc-eval `--bunker-before` (050-R10 3,
the human TvT / TvP part) adds only the `bunker_before` field. CPU, no game data."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "eval"))

import eval_common as ec  # noqa: E402
import regression_gate as rg  # noqa: E402

from test_eval_protocol import LIN, build, dump, make_job, row  # noqa: E402


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    """The tools read the repository .env (eval_common.load_env); kept out so later tests in the process see
    the plain test environment."""
    monkeypatch.setattr(ec, "load_env", lambda: None)


def val(top1: dict, nll: dict, bunker: dict | None = None) -> dict:
    d = {"split": "val", "games": 500, "selection_seed": 1,
         "by_enemy_race": {r: {"steps": 100, "action_top1": top1[r], "nll": {"action": nll[r]}} for r in top1}}
    if bunker is not None:
        d["bunker_before"] = {"before_frame": 5760, "by_enemy_race": {r: {"steps": 40, "mean_p": p} for r, p in bunker.items()}}
    return d


def results(checks, check, race, metric):
    (c,) = [c for c in checks if (c["check"], c["race"], c["metric"]) == (check, race, metric)]
    return c["result"]


def test_val_top1_and_nll_boundaries_050_r10():
    """(2): top-1 within -0.5 %p and action NLL within +0.01 of the baseline, boundaries inclusive."""
    base = val({"terran": 0.700, "protoss": 0.650, "zerg": 0.6}, {"terran": 0.90, "protoss": 1.00, "zerg": 1.0})
    at = val({"terran": 0.695, "protoss": 0.645, "zerg": 0.1}, {"terran": 0.91, "protoss": 1.01, "zerg": 9.0})
    c = rg.judge_val(base, at)
    assert {x["result"] for x in c} == {"PASS"}, c  # zerg is not judged (Terran / Protoss gate)
    assert {x["race"] for x in c} == {"terran", "protoss"}
    past = val({"terran": 0.69499, "protoss": 0.645}, {"terran": 0.90, "protoss": 1.01001})
    c = rg.judge_val(base, past)
    assert results(c, "2", "terran", "action_top1") == "FAIL" and results(c, "2", "terran", rg.NLL_HEAD) == "PASS"
    assert results(c, "2", "protoss", "action_top1") == "PASS" and results(c, "2", "protoss", rg.NLL_HEAD) == "FAIL"
    assert {x["result"] for x in rg.judge_val(base, None)} == {"MISSING"}
    with pytest.raises(SystemExit, match="differ in games"):
        rg.judge_val(base, {**at, "games": 400})


def test_tf_bunker_does_not_increase_050_r10():
    """(3a): mean p(Build_Bunker) before 4:00 on human TvT / TvP val games <= the baseline."""
    t = {"terran": 0.9, "protoss": 0.9}
    base = val(t, t, {"terran": 0.020, "protoss": 0.010})
    c = rg.judge_tf_bunker(base, val(t, t, {"terran": 0.020, "protoss": 0.005}))
    assert [x["result"] for x in c] == ["PASS", "PASS"]
    c = rg.judge_tf_bunker(base, val(t, t, {"terran": 0.0201, "protoss": 0.010}))
    assert [x["result"] for x in c] == ["FAIL", "PASS"]
    assert {x["result"] for x in rg.judge_tf_bunker(base, val(t, t))} == {"MISSING"}
    other = val(t, t, {"terran": 0.0, "protoss": 0.0})
    other["bunker_before"]["before_frame"] = 4320
    with pytest.raises(SystemExit, match="4:00"):
        rg.judge_tf_bunker(base, other)


def counts(won, order, unit, n=32, dumped=None):
    return {"n": n, "won": won, "bunker_order_4m": order, "bunker_unit_4m": unit, "dumped": n if dumped is None else dumped}


def test_games_win_rate_and_bunker_boundaries_050_r10():
    """(1) win rate >= the baseline's on the same 32 games; (3b) bunker_order / bunker_unit shares <= the baseline's.
    Rates are compared exactly when invalid games make the counts differ."""
    base = {"terran": counts(12, 3, 2), "protoss": counts(5, 1, 1)}
    same = {"terran": counts(12, 3, 2), "protoss": counts(5, 1, 1)}
    assert {x["result"] for x in rg.judge_games(base, same)} == {"PASS"}
    worse = {"terran": counts(11, 4, 2), "protoss": counts(5, 1, 2)}
    c = rg.judge_games(base, worse)
    assert results(c, "1", "terran", "won") == "FAIL" and results(c, "3b", "terran", "bunker_order_4m") == "FAIL"
    assert results(c, "3b", "terran", "bunker_unit_4m") == "PASS" and results(c, "3b", "protoss", "bunker_unit_4m") == "FAIL"
    # 11 / 29 = 37.9% >= 12 / 32 = 37.5%; 10 / 27 = 37.04% < 37.5%; bunker 3 / 29 > 3 / 32
    c = rg.judge_games(base, {"terran": counts(11, 3, 2, n=29), "protoss": counts(5, 1, 1)})
    assert results(c, "1", "terran", "won") == "PASS" and results(c, "3b", "terran", "bunker_order_4m") == "FAIL"
    assert results(rg.judge_games(base, {"terran": counts(10, 0, 0, n=27), "protoss": counts(5, 1, 1)}), "1", "terran", "won") == "FAIL"
    # a game without its command stream: bunker_order is not judged
    c = rg.judge_games(base, {"terran": counts(12, 3, 2, dumped=31), "protoss": counts(5, 1, 1)})
    assert results(c, "3b", "terran", "bunker_order_4m") == "MISSING" and results(c, "3b", "terran", "bunker_unit_4m") == "PASS"
    assert {x["result"] for x in rg.judge_games(base, None)} == {"MISSING"}


def test_gate_overall_050_r10():
    t = {"terran": 0.7, "protoss": 0.65}
    n = {"terran": 0.9, "protoss": 1.0}
    b = {"terran": 0.02, "protoss": 0.01}
    games = {"terran": counts(12, 3, 2), "protoss": counts(5, 1, 1)}
    g = rg.gate(val(t, n, b), val(t, n, b), games, games)
    assert g["overall"] == "PASS" and [c["check"] for c in g["checks"]] == ["1"] * 2 + ["2"] * 4 + ["3a"] * 2 + ["3b"] * 4
    assert rg.gate(val(t, n, b), val(t, n, b), None, None)["overall"] == "MISSING"
    worse = {"terran": counts(11, 3, 2), "protoss": counts(5, 1, 1)}
    assert rg.gate(val(t, n, b), val(t, n, b), games, worse)["overall"] == "FAIL"
    assert rg.gate(None, None, games, worse)["overall"] == "FAIL", "a failure wins over missing inputs"


def t0_tp_job(tmp_path: Path, name: str, rows, ledger=None) -> Path:
    job = make_job(tmp_path / name, "t0_tp", rows, ledger)
    return job


def test_t0_tp_counts_from_a_job_050_r10(tmp_path):
    """Games 1..32 are T0 Terran, 33..64 T0 Protoss (t0_tp template); bunker_order_4m from the command stream
    (< 5,760), bunker_unit_4m from bunker_s (< 240), opponent-crash games invalid."""
    rows = [row(1, True, bunker_s=239.9), row(2, None, bunker_s=240.0), row(3, False), row(33, True), row(34, False, bunker_s=100.0)]
    ledger = [{"game_id": 100035, "outcome": "Aborted", "detail": "launcher exit codes [Some(0), None]"}]
    job = t0_tp_job(tmp_path, "cand", rows + [row(35, True)], ledger)
    cache = tmp_path / "cache" / job.name / "t0_tp"
    dump(cache / "g1.json", [build(500, 10, 10, 109), build(5759, 12, 12, ec.BUNKER)])
    dump(cache / "g2.json", [build(500, 10, 10, 109), build(5760, 12, 12, ec.BUNKER)])
    dump(cache / "g3.json", [build(500, 10, 10, 109)])
    dump(cache / "g33.json", [build(500, 10, 10, 109)])
    c = rg.t0_tp_counts(job, tmp_path / "cache", LIN)
    assert c["terran"] == {"n": 3, "won": 1, "bunker_order_4m": 1, "bunker_unit_4m": 1, "dumped": 3, "invalid": 0, "games": 32, "missing": 29}
    assert c["protoss"] == {"n": 2, "won": 1, "bunker_order_4m": 0, "bunker_unit_4m": 1, "dumped": 1, "invalid": 1, "games": 32, "missing": 29}
    assert c["skipped_lines"] == []
    # 050-R9: a NUL line is ignored, a truncated line is skipped and counted; the decodable rows are kept
    f = job / "results" / "h.jsonl"
    f.write_bytes(f.read_bytes() + b"\x00" * 64 + b"\n" + b'{"game_id": 100036, "won"\n')
    c2 = rg.t0_tp_counts(job, tmp_path / "cache", LIN)
    assert {r: c2[r] for r in ("terran", "protoss")} == {r: c[r] for r in ("terran", "protoss")}
    assert [(s["file"], s["line"]) for s in c2["skipped_lines"]] == [("results/h.jsonl", 8)]
    with pytest.raises(SystemExit, match="0 stages of the protocol plan t0_tp"):
        rg.t0_tp_counts(make_job(tmp_path / "m", "macro", []), tmp_path / "cache", LIN)


def test_unfinished_t0_job_is_missing_050_r10(tmp_path):
    """A job with planned games that have neither a result nor an invalid mark is not judged (1 and 3b MISSING),
    even when its finished games would pass; once every game has a result or is invalid it is judged."""
    def job(name, rows, ledger=None):
        j = t0_tp_job(tmp_path, name, rows, ledger)
        for n in range(1, 65):
            dump(tmp_path / "cache" / j.name / "t0_tp" / f"g{n}.json", [build(500, 10, 10, 109)])
        return rg.t0_tp_counts(j, tmp_path / "cache", LIN)

    base = job("base", [row(n, n % 2 == 1) for n in range(1, 65)])
    half = job("half", [row(n, True) for n in range(1, 65) if n != 40])   # Protoss game 40 not played yet
    assert base["protoss"]["missing"] == 0 and half["protoss"]["missing"] == 1 and half["terran"]["missing"] == 0
    c = rg.judge_games(base, half)
    assert {x["result"] for x in c if x["race"] == "protoss"} == {"MISSING"}
    assert {x["result"] for x in c if x["race"] == "terran"} == {"PASS"}
    done = job("done", [row(n, True) for n in range(1, 65) if n != 40],
               [{"game_id": 100040, "outcome": "Aborted", "detail": "launcher exit codes [Some(0), None]"}])
    assert done["protoss"] == {**half["protoss"], "invalid": 1, "missing": 0}
    assert {x["result"] for x in rg.judge_games(base, done)} == {"PASS"}


def test_gate_cli_end_to_end_050_ac5(tmp_path):
    """main(): bc-eval JSONs and t0_tp jobs in, one verdict per check and the JSON out."""
    base_rows = [row(n, n % 2 == 1, bunker_s=None) for n in range(1, 65)]
    cand_rows = [row(n, n % 2 == 1 and n != 1, bunker_s=None) for n in range(1, 65)]   # one Terran win fewer
    bj, cj = t0_tp_job(tmp_path, "base", base_rows), t0_tp_job(tmp_path, "cand", cand_rows)
    for j in (bj, cj):
        for n in range(1, 65):
            dump(tmp_path / "cache" / j.name / "t0_tp" / f"g{n}.json", [build(500, 10, 10, 109)])
    t, nl, b = {"terran": 0.7, "protoss": 0.65}, {"terran": 0.9, "protoss": 1.0}, {"terran": 0.02, "protoss": 0.01}
    bv, cv = tmp_path / "b.json", tmp_path / "c.json"
    bv.write_text(json.dumps(val(t, nl, b)), encoding="utf-8")
    cv.write_text(json.dumps(val(t, nl, b)), encoding="utf-8")
    out = tmp_path / "gate.json"
    g = rg.main(["--baseline-val", str(bv), "--candidate-val", str(cv), "--baseline-games", str(bj), "--candidate-games", str(cj),
                 "--cache", str(tmp_path / "cache"), "--json", str(out)])
    assert g["overall"] == "FAIL"
    assert results(g["checks"], "1", "terran", "won") == "FAIL" and results(g["checks"], "1", "protoss", "won") == "PASS"
    assert g["inputs"]["baseline_t0_tp"]["terran"]["won"] == 16 and g["inputs"]["candidate_t0_tp"]["terran"]["won"] == 15
    assert g["inputs"]["candidate_t0_tp"]["protoss"]["missing"] == 0
    assert json.loads(out.read_text(encoding="utf-8"))["overall"] == "FAIL"


def test_bc_eval_bunker_before_050_r10(tiny_config, tmp_path):
    """bc-eval --bunker-before F: per-race steps before F add up to the whole, every scored step counts when F
    is past the games, none at 0, p within [0, 1]; without the option the report has no such field."""
    from uranium import _core
    from uranium.bc.eval import evaluate

    from test_bc_eval_race import _tiny_make
    import functools

    shards = tmp_path / "races"
    _core.write_test_shards(str(shards), "races", 4)
    out = tmp_path / "out"
    assert _core.run_bc(functools.partial(_tiny_make, None, None), str(tiny_config), out=str(out), shards=[str(shards)], steps=1, synthetic_maps=True) == "done"
    from test_aux import CPU

    def ev(before):
        return evaluate([str(shards)], str(out / "last"), "train", None, 1, str(tiny_config), CPU, "f32", synthetic_maps=True, bunker_before=before)

    plain = ev(None)
    assert "bunker_before" not in plain
    full = ev(10**6)
    bb = full["bunker_before"]
    assert bb["all"]["steps"] == full["steps"] > 0 and sum(x["steps"] for x in bb["by_enemy_race"].values()) == full["steps"]
    assert all(x["steps"] == full["by_enemy_race"][r]["steps"] for r, x in bb["by_enemy_race"].items())
    assert 0.0 <= bb["all"]["mean_p"] <= 1.0
    assert {k: v for k, v in full.items() if k != "bunker_before"} == plain
    # synthetic steps are 8 frames apart from 0: frame < 64 keeps the first 8 steps of each game
    early = ev(64)["bunker_before"]
    assert 0 < early["all"]["steps"] < full["steps"]
    assert ev(0)["bunker_before"]["all"] == {"steps": 0, "mean_p": None}


def test_protocol_bc_eval_prints_bunker_before_050_r10(capsys, tmp_path, monkeypatch):
    """val500_race (and val500, the same output file) run bc-eval with --bunker-before 5760; check-bc-eval warns
    when an output lacks `bunker_before` (gate check 3a would be MISSING) and is quiet when it has it."""
    import protocol

    monkeypatch.setattr(protocol, "checkpoint_sha256", lambda c: "a" * 64)
    monkeypatch.setattr(protocol, "git_state", lambda: {"commit": "0" * 40})
    r1 = ["--lineage", "ck1", "--point", "r1", "--checkpoint", "checkpoints/bc-dagger-r1/ckpt-1137"]
    protocol.main(["bc-eval", "--metric", "val500_race"] + r1)
    cmd = capsys.readouterr().out.splitlines()[0]
    assert cmd.startswith("python -m uranium bc-eval ") and " --bunker-before 5760" in cmd
    assert LIN["metric"]["val500"]["extra_args"] == LIN["metric"]["val500_race"]["extra_args"] == ["--bunker-before", "5760"]
    f = tmp_path / "v.json"
    per = {r: {"steps": 10, "action_top1": 0.7, "nll": {"action": 0.9}} for r in ("terran", "protoss", "zerg")}
    base = {"split": "val", "games": 500, "selection_seed": 1, "action_top1": 0.72, "by_enemy_race": per}
    f.write_text(json.dumps(base), encoding="utf-8")
    protocol.main(["check-bc-eval", str(f), "--metric", "val500_race"])
    assert "no `bunker_before`" in capsys.readouterr().out
    f.write_text(json.dumps({**base, "bunker_before": {"before_frame": 5760}}), encoding="utf-8")
    protocol.main(["check-bc-eval", str(f), "--metric", "val500"])
    assert "WARNING" not in capsys.readouterr().out
