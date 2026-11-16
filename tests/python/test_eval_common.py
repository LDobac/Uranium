import json
import os
import subprocess

import pytest

from eval_helpers import build, cmds, dump, ops_value, order, rclick, write_ops

from uranium.measure import eval_common as ec


def test_game_ids_and_opponent_crashes_050_r9():
    assert ec.game_key(10100188) == 100188 and ec.stage_of(10100188) == 1 and ec.stage_of(300070) == 3
    ledger = [
        {"game_id": 300070, "outcome": "Aborted", "detail": "launcher exit codes [Some(0), Some(-1073741819)]; p0: ..."},
        {"game_id": 100260, "outcome": "Aborted", "detail": "launcher exit codes [Some(0), None]; p0: ..."},
        {"game_id": 100005, "outcome": "Aborted", "detail": "launcher exit codes [None, Some(0)]; p0: ..."},
        {"game_id": 100006, "outcome": "Aborted", "detail": "launcher exit codes [None, None]; p0: ..."},
        {"game_id": 10100007, "outcome": "Aborted", "detail": "launcher exit codes [Some(0), Some(1)]"},
        {"game_id": 100008, "outcome": "Finished", "detail": None},
    ]
    assert ec.opponent_crashes(ledger) == {300070, 100260, 100007}


def test_model_rows_highest_attempt_050_r9():
    rows = [
        {"game_id": 100001, "model": {"scv_8m": 3}},
        {"game_id": 100001, "model": None},
        {"game_id": 100001, "model": {"scv_8m": 9}},
        {"game_id": 200001, "model": {"scv_8m": 1}},
        {"game_id": 10100002, "model": {"scv_8m": 5}},
        {"game_id": 100002, "model": {"scv_8m": 4}},
    ]
    got = ec.model_rows(rows, 1)
    assert sorted(got) == [1, 2] and got[1]["model"]["scv_8m"] == 9 and got[2]["model"]["scv_8m"] == 5


def test_wilson_and_fisher_050_r2():
    lo, hi = ec.wilson(33, 300)
    assert (round(100 * lo, 1), round(100 * hi, 1)) == (7.9, 15.0)
    lo, hi = ec.wilson(158, 200)
    assert (round(100 * lo, 1), round(100 * hi, 1)) == (72.8, 84.1)
    assert ec.fisher_p(5, 10, 5, 10) == pytest.approx(1.0)
    assert ec.fisher_p(10, 10, 0, 10) == pytest.approx(1.0825088224469026e-05)


def test_analyze_first_build_and_scout_050_r6(tmp_path):
    f = tmp_path / "g1.json"
    far_move = "".join(f"{v & 0xFF:02x}{v >> 8:02x}" for v in (3000, 3000, 0xFFFF)) + "e400" + "06" + "00"
    dump(f, [
        build(500, 10, 10, 109),
        [600, 0, 9, "0101"],
        [700, 0, 21, far_move],
        build(4000, 12, 12, ec.BUNKER),
        build(5000, 14, 14, ec.BUNKER),
        build(7200, 20, 20, ec.CC),
        [800, 1, 12, build(0, 1, 1, ec.BUNKER)[3]],
        build(12000, 5, 5, 111),
    ])
    first, scout = ec.analyze(f)
    assert first == {109: 500, ec.BUNKER: 4000, ec.CC: 7200}
    assert scout == 700
    assert ec.analyze_both(f) == (first, 700, 700)


@pytest.mark.parametrize("ground", [0, 0xFFFF])
def test_scout_fixed_counts_ground_right_click_050_r9(tmp_path, ground):
    f = tmp_path / "g1.json"
    dump(f, [
        build(500, 10, 10, 109),
        [600, 0, 9, "0101"],
        rclick(650, 2000, 320, 5),
        rclick(700, 2000, 320, ground),
        rclick(4400, 3000, 3000, ground),
    ])
    assert ec.analyze_both(f)[1:] == (None, 700)
    assert ec.analyze(f)[1] is None, "analyze keeps the old reading (scout_3m)"


def test_scout_old_and_fixed_read_move_orders_differently_050_r9(tmp_path):
    f = tmp_path / "g1.json"
    dump(f, [
        build(500, 10, 10, 109),
        [600, 0, 9, "0201"],
        order(610, 3000, 3000, 0, ec.ORDER_MOVE),
        [650, 0, 9, "0101"],
        order(660, 3000, 3000, 0, 10),
        order(700, 330, 330, 0, ec.ORDER_MOVE),
        order(800, 330, 2000, 0, ec.ORDER_MOVE),
    ])
    assert ec.analyze_both(f)[1:] == (700, 800)


def test_teacher_mode_rule_boundaries_050_r10(tmp_path):
    f = tmp_path / "g.json"

    def tm(c):
        dump(f, c)
        return ec.teacher_mode(f, 0)

    t = tm(cmds(899) + [[8640, 0, 9, "0101"]] * 50 + cmds(500, player=1))
    assert t == {"cmds_0_6m": 899, "cmds_per_min_0_6m": 899 / 6, "refinery_order_4m": False, "teacher_mode": True}
    assert tm(cmds(900))["teacher_mode"] is False
    assert tm(cmds(10) + [build(5759, 20, 20, ec.REFINERY)]) == {"cmds_0_6m": 11, "cmds_per_min_0_6m": 11 / 6, "refinery_order_4m": True, "teacher_mode": False}
    assert tm(cmds(10) + [build(5760, 20, 20, ec.REFINERY)])["teacher_mode"] is True
    assert tm(cmds(10) + [build(100, 20, 20, 109)])["teacher_mode"] is True, "another building is not a refinery"
    assert tm(cmds(10) + [[100, 1, 12, build(0, 20, 20, ec.REFINERY)[3]]])["teacher_mode"] is True, "the opponent's refinery"


def test_teacher_mode_model_player_050_r10(tmp_path):
    f = tmp_path / "g.json"
    f.write_text(json.dumps({"players": [{"id": 2, "race": 1}, {"id": 3, "race": 1}], "cmds": cmds(1000, player=2) + cmds(5, player=3)}), encoding="utf-8")
    assert ec.teacher_mode(f, 0)["cmds_0_6m"] == 1000 and ec.teacher_mode(f, 1)["cmds_0_6m"] == 5
    assert ec.model_player({"players": [{"id": 4, "race": 2}, {"id": 5, "race": 1}]}, 0) == 5
    assert ec.model_player({"players": [{"id": 4, "race": 2}, {"id": 5, "race": 0}]}, 0) is None
    assert ec.model_player({"players": [{"id": 2, "race": 1}, {"id": 3, "race": 1}]}, 2) is None


def test_ops_stage_runs_opsdump_from_our_slot_052_r7(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    (stage / "replays").mkdir(parents=True)
    for n in (1, 2, 3, 5):
        (stage / "replays" / f"g{n}.rep").write_bytes(b"rep")
    binary = tmp_path / "uranium-replay"
    binary.write_bytes(b"")
    calls = []

    def fake(b, rep, slot):
        calls.append((rep.name, slot))
        return {"rep": rep.name, "error": "desync"} if rep.name == "g3.rep" else {"rep": rep.name, "perspective": slot, "frames": 1, **ops_value(8)}

    monkeypatch.setattr(ec, "_opsdump", fake)
    cache = tmp_path / "cache"
    write_ops(cache, 2, ops_value(16), slot=0)
    write_ops(cache, 5, ops_value(16), version=ec.OPS_CACHE_VERSION - 1)
    got, failed = ec.ops_stage(stage, {1: 0, 2: 1, 3: 0, 4: 1, 5: 0}, cache, binary=binary, jobs=2)
    assert sorted(calls) == [("g1.rep", 0), ("g2.rep", 1), ("g3.rep", 0), ("g5.rep", 0)]
    assert sorted(got) == [1, 2, 5] and got[2]["perspective"] == 1 and got[5]["ops"]["mined"] == 8 and failed == {3: "desync"}
    assert json.loads((cache / "g5.ops.json").read_text(encoding="utf-8"))["cache_version"] == ec.OPS_CACHE_VERSION
    assert not (cache / "g3.ops.json").exists() and json.loads((cache / "g2.ops.json").read_text(encoding="utf-8"))["perspective"] == 1
    calls.clear()
    got, failed = ec.ops_stage(stage, {1: 0, 2: 1, 3: 0}, cache, binary=binary)
    assert calls == [("g3.rep", 0)] and sorted(got) == [1, 2], "cached lines reused, the failure tried again"
    monkeypatch.undo()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="{broken\n", stderr=""))
    assert "unreadable opsdump line" in ec._opsdump(binary, stage / "replays" / "g1.rep", 0)["error"]
    with pytest.raises(SystemExit, match="cargo build"):
        ec.ops_stage(stage, {3: 0}, tmp_path / "c2", binary=tmp_path / "missing")


@pytest.mark.parametrize("package", [True], ids=["uranium"])
def test_load_env_honours_env_file_001_r9(tmp_path, monkeypatch, package):
    key = "URANIUM_TEST_TOOLS_ENV_FILE_VALUE"
    (tmp_path / ".env").write_text(f"{key}=repo\n", encoding="utf-8")
    other = tmp_path / "other.env"
    other.write_text(f"{key}=file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("URANIUM_ENV_FILE", str(tmp_path / "missing.env"))
    ec.load_env()
    assert key not in os.environ, "a missing URANIUM_ENV_FILE loads nothing"
    monkeypatch.setenv("URANIUM_ENV_FILE", str(other))
    ec.load_env()
    assert os.environ[key] == "file"
