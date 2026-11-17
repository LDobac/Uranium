import json
import tomllib

import pytest

from eval_helpers import LIN, build, cmds, dump, make_job, ops_value, rclick, row, rush_value, write_ops

from uranium.measure import eval_common as ec
from uranium.measure import protocol
from uranium.measure import tables


def test_subset_tables_t0_tp_inside_t0_300_050_r9(tmp_path):
    rows = [{**row(1, True), "model": {**row(1, True)["model"], "zero_last_failed": True}},
            {**row(17, True), "model": {**row(17, True)["model"], "zero_last_failed": True}},
            {**row(51, None), "model": {**row(51, None)["model"], "zero_last_failed": True}},
            {**row(116, False), "model": {**row(116, False)["model"], "zero_last_failed": True}}]
    ledger = [{"game_id": 100151, "outcome": "Aborted", "detail": "launcher exit codes [Some(0), None]"}]
    job = make_job(tmp_path, "t0_300", rows, ledger)
    res = tables.subset_tables(job, "t0_tp", LIN)
    (t,) = res.values()
    assert t["plan"] == "t0_tp" and t["games_planned"] == 64 and t["rows"] == 3 and t["invalid_games"] == [100049]
    tot = t["table"]["by_race"]
    assert tot["stochastic vs terran"]["won"] == 1 and tot["stochastic vs terran"]["draw"] == 1 and tot["stochastic vs terran"]["n"] == 2
    assert tot["stochastic vs protoss"]["lost"] == 1 and tot["stochastic vs protoss"]["invalid"] == 1
    bad = make_job(tmp_path / "b", "t0_300", rows, plan_text=protocol.render(LIN, "t0_300", "c", "t0_300").replace("zero_last_failed = true", "zero_last_failed = false"))
    with pytest.raises(protocol.Refused, match="zero_last_failed differs"):
        tables.subset_tables(bad, "t0_tp", LIN)
    with pytest.raises(protocol.Refused, match="no stage holds every game"):
        tables.subset_tables(make_job(tmp_path / "c", "macro", []), "t0_tp", LIN)


def test_job_tables_stage_filter_050_r9(tmp_path):
    job = make_job(tmp_path, "tvz_bc", [row(1, True)])
    assert tables.job_tables(job, tmp_path / "cache", lin=LIN, stages={"other"}) == {}
    assert list(tables.job_tables(job, tmp_path / "cache", lin=LIN, stages={"tvz_bc"})) == ["tvz_bc/tvz_bc"]


def test_corrupt_result_lines_skipped_and_counted_050_r9(tmp_path, capsys):
    rows = [row(1, True), row(2, False), row(3, None)]
    job = make_job(tmp_path, "tvz_bc", [], [{"game_id": 100009, "outcome": "Finished", "detail": None}])
    (job / "results" / "h.jsonl").write_bytes(
        (json.dumps(rows[0]) + "\n").encode() + b"\x00" * 300 + b"\n"
        + json.dumps(rows[1])[:30].encode() + b"\x00" * 50 + (json.dumps(rows[1]) + "\n").encode()
        + json.dumps(rows[2])[:40].encode() + b"\n"
        + (json.dumps(rows[2]) + "\r\n").encode())
    (job / "games" / "h.jsonl").write_bytes(b'{"game_id": 100009, "outco\n[1, 2]\n' + b"\x00" * 8)
    got = ec.read_results(job)
    assert list(got) == rows
    assert [(s["file"], s["line"]) for s in got.skipped] == [("h.jsonl", 3), ("h.jsonl", 4)]
    assert ec.read_ledger(job) == [] and len(ec.read_ledger(job).skipped) == 2
    err = capsys.readouterr().err
    assert err.count(f"{job / 'results' / 'h.jsonl'}:4: skipped an undecodable line") == 1, "warned once per line"
    assert f"{job / 'results' / 'h.jsonl'}:3: skipped an undecodable line" in err
    assert err.count(f"{job / 'games' / 'h.jsonl'}:2: skipped an undecodable line (not a JSON object (list))") == 1
    (t,) = tables.job_tables(job, tmp_path / "cache", lin=LIN).values()
    assert capsys.readouterr().err == "", "lines warned about already are not warned again"
    assert t["rows"] == 3
    assert [(s["file"], s["line"]) for s in t["skipped_lines"]] == [("results/h.jsonl", 3), ("results/h.jsonl", 4), ("games/h.jsonl", 1), ("games/h.jsonl", 2)]
    tables.print_tables({"tvz_bc/tvz_bc": t})
    assert "SKIPPED 4 undecodable line(s) of the job (counted as no row): results/h.jsonl:3, results/h.jsonl:4, games/h.jsonl:1, games/h.jsonl:2" in capsys.readouterr().out
    clean = make_job(tmp_path / "clean", "tvz_bc", rows)
    (t,) = tables.job_tables(clean, tmp_path / "cache", lin=LIN).values()
    assert t["skipped_lines"] == [] and t["rows"] == 3


def early_games(monkeypatch, k: int) -> list[int]:
    monkeypatch.setattr(ec, "start_condition", lambda seed: "early" if seed % 2 else "late")
    seeds = tomllib.loads(protocol.render(LIN, "rush_check", "c", "n"))["seeds"]
    return [n for n in range(1, 129) if seeds[n - 1] % 2][:k]


def test_rush_table_both_held_and_bunker_definitions_050_r7(tmp_path, monkeypatch):
    early = early_games(monkeypatch, 4)
    rows = [
        row(early[0], True, scv_8m=10, scv_min_1m=3, bunker_s=170.0, scv_4m=9),
        row(early[1], False, scv_8m=None, scv_min_1m=6, bunker_s=200.0, scv_4m=4),
        row(early[2], None, scv_8m=8, scv_min_1m=5, bunker_s=None, scv_4m=None),
        row(early[3], True, scv_8m=None, scv_min_1m=2, bunker_s=None, scv_4m=6),
    ]
    job = make_job(tmp_path, "rush_check", rows)
    cache = tmp_path / "cache" / job.name / "rush_check"
    dump(cache / f"g{early[0]}.json", [build(500, 10, 10, 109), build(4000, 12, 12, ec.BUNKER)])
    dump(cache / f"g{early[1]}.json", [build(500, 10, 10, 109), [600, 0, 9, "0101"], rclick(700, 3000, 320, 0),
                                       build(5000, 12, 12, ec.BUNKER)])
    res = tables.job_tables(job, tmp_path / "cache", lin=LIN)
    t = res[f"{job.name}/rush_check"]
    assert t["plan"] == "rush_check" and t["kind"] == "rush" and t["protocol"] is None
    e = t["table"]["early"]
    assert (e["n"], e["held_8m"], e["held_8m_winpass"], e["held_s"]) == (4, 2, 3, 2)
    assert (e["bunker_order_3m"], e["bunker_order_4m"], e["bunker_unit_3m"], e["bunker_unit_4m"]) == (1, 2, 1, 2)
    assert e["scv_4m_p50"] == 5 and e["scv_4m_p50_known"] == 6 and e["won"] == 2 and e["dumped"] == 2
    assert t["counts"]["early"]["held_s"] == [2, 4] and t["table"]["late"]["n"] == 0
    assert (e["scout_3m"], e["scout_3m_fixed"]) == (0, 1) and t["counts"]["early"]["scout_3m_fixed"] == [1, 4]


def test_bunker_unit_boundaries_050_r7(tmp_path, monkeypatch):
    early = early_games(monkeypatch, 5)
    rows = [row(n, None, bunker_s=b) for n, b in zip(early, (179.9, 180.0, 239.9, 240.0, None))]
    t = tables.job_tables(make_job(tmp_path, "rush_check", rows), tmp_path / "cache", lin=LIN)
    e = next(iter(t.values()))["table"]["early"]
    assert (e["bunker_unit_3m"], e["bunker_unit_4m"]) == (1, 3)
    m = tables.job_tables(make_job(tmp_path / "m", "macro", [row(n, None, bunker_s=b) for n, b in zip(range(1, 5), (239.9, 240.0, 179.9, None))]),
                          tmp_path / "cache", lin=LIN)
    assert next(iter(m.values()))["table"]["total"]["bunker_unit_4m"] == 2


def test_macro_table_050_r8(tmp_path):
    rows = [row(1, True, scv_8m=16, bunker_s=230.0), row(2, None, scv_8m=15, bunker_s=250.0), row(11, False, scv_8m=None, bunker_s=None),
            row(12, True, scv_8m=None, bunker_s=None)]
    job = make_job(tmp_path, "macro", rows)
    cache = tmp_path / "cache" / job.name / "macro"
    dump(cache / "g1.json", [build(500, 10, 10, 109), build(5000, 12, 12, ec.BUNKER), build(7200, 20, 20, ec.CC)])
    dump(cache / "g2.json", [build(500, 10, 10, 109), build(5760, 12, 12, ec.BUNKER), build(7201, 20, 20, ec.CC)])
    t = tables.job_tables(job, tmp_path / "cache", lin=LIN)[f"{job.name}/macro"]["table"]
    a = t["ualbertabot2hatchhydra"]
    assert (a["n"], a["bunker_order_4m"], a["bunker_unit_4m"], a["cc_order_5m"], a["scv_8m_16"]) == (2, 1, 1, 1, 1)
    assert (a["won"], a["draw"], a["lost"]) == (1, 1, 0)
    b = t["ualbertabot3hatchmuta"]
    assert (b["lost"], b["won"], b["scv_8m_16"], b["scv_8m_16_winpass"]) == (1, 1, 0, 1)
    assert t["total"]["n"] == 4 and t["total"]["scv_8m_16"] == 1 and t["total"]["scv_8m_16_winpass"] == 2


def test_t1_table_drops_opponent_crash_games_050_r9(tmp_path):
    rows = [row(1, True), row(2, False), {**row(19, True), "reason": "Surrender"}, {**row(20, True), "reason": "Surrender"}]
    ledger = [{"game_id": 100020, "outcome": "Aborted", "detail": "launcher exit codes [Some(0), None]"}]
    job = make_job(tmp_path, "t1_bc", rows, ledger)
    t = tables.job_tables(job, tmp_path / "cache", lin=LIN)[f"{job.name}/t1_bc"]
    assert t["invalid_games"] == [100020]
    uab_z, uab_p = t["table"]["blocks"][0], t["table"]["blocks"][1]
    assert (uab_z["bot"], uab_z["n"], uab_z["won"], uab_z["missing"]) == ("ualbertabot", 2, 1, 16)
    assert (uab_p["n"], uab_p["won"], uab_p["surrender_wins"], uab_p["invalid"]) == (1, 1, 1, 1)
    assert t["table"]["total"]["won"] == 2 and t["table"]["total"]["n"] == 3


def test_winrate_table_tvz_050_r9(tmp_path):
    rows = [row(1, True), row(2, None), row(17, False), row(18, True)]
    job = make_job(tmp_path, "tvz_bc", [{**r, "model": {**r["model"], "zero_last_failed": True}} for r in rows])
    t = tables.job_tables(job, tmp_path / "cache", lin=LIN)[f"{job.name}/tvz_bc"]
    assert t["plan"] == "tvz_bc" and t["zero_last_failed"] == [True]
    assert t["table"]["total"] == {"n": 4, "won": 2, "draw": 1, "lost": 1, "invalid": 0, "missing": 28}


def test_unknown_plan_needs_kind_050_r9(tmp_path):
    text = protocol.render(LIN, "tvz_bc", "checkpoints/m2-r1/ckpt-5", "odd").replace("save_replay = true", "save_replay = false")
    job = make_job(tmp_path, "tvz_bc", [row(1, True)], plan_text=text)
    with pytest.raises(tables.UnknownPlan, match="match no protocol plan"):
        tables.job_tables(job, tmp_path / "cache", lin=LIN)
    t = tables.job_tables(job, tmp_path / "cache", {"odd": "winrate"}, lin=LIN)["tvz_bc/odd"]
    assert t["plan"] is None and t["table"]["total"]["won"] == 1
    with pytest.raises(SystemExit, match="REFUSED: stage odd"):
        tables.main([str(job), "--cache", str(tmp_path / "cache")])


def test_teacher_mode_table_050_r10(tmp_path, monkeypatch, capsys):
    rows = [{**row(1, True), "end_frame": 30000}, {**row(2, False), "end_frame": 14399}, {**row(3, False), "end_frame": 14400},
            {**row(4, None), "end_frame": 86400}, row(5, True),
            {**row(33, False), "end_frame": 10000}, {**row(34, True), "end_frame": 20000}, row(35, False)]
    ledger = [{"game_id": 100035, "outcome": "Aborted", "detail": "launcher exit codes [Some(0), None]"}]
    job = make_job(tmp_path, "t0_tp", rows, ledger)
    cache = tmp_path / "cache" / job.name / "t0_tp"
    for n in (1, 2, 3, 33):
        dump(cache / f"g{n}.json", cmds(100))
    dump(cache / "g4.json", cmds(100) + [build(3000, 20, 20, ec.REFINERY)])
    dump(cache / "g34.json", cmds(2000))
    dump(cache / "g35.json", cmds(10))
    t = tables.job_tables(job, tmp_path / "cache", lin=LIN, ops="off")["t0_tp/t0_tp"]
    tm = t["teacher_mode"]
    zero = {"n": 0, "won": 0, "draw": 0, "lost": 0, "lost_10m": 0}
    assert tm["terran"] == {"n": 4, "teacher": 3, "no_stream": 1,
                            "teacher_mode": {"n": 3, "won": 1, "draw": 0, "lost": 2, "lost_10m": 1},
                            "other": {"n": 1, "won": 0, "draw": 1, "lost": 0, "lost_10m": 0},
                            "zerg_build": 3, "tp_style": 1,
                            "style": {"zerg_build": {"n": 3, "won": 1, "draw": 0, "lost": 2, "lost_10m": 1},
                                      "tp_style": {"n": 1, "won": 0, "draw": 1, "lost": 0, "lost_10m": 0}, "human": zero}}
    assert tm["protoss"] == {"n": 2, "teacher": 1, "no_stream": 0,
                             "teacher_mode": {"n": 1, "won": 0, "draw": 0, "lost": 1, "lost_10m": 1},
                             "other": {"n": 1, "won": 1, "draw": 0, "lost": 0, "lost_10m": 0},
                             "zerg_build": 1, "tp_style": 0,
                             "style": {"zerg_build": {"n": 1, "won": 0, "draw": 0, "lost": 1, "lost_10m": 1},
                                       "tp_style": zero, "human": {"n": 1, "won": 1, "draw": 0, "lost": 0, "lost_10m": 0}}}
    assert t["counts"]["teacher_mode vs terran"] == {"teacher_mode": [3, 4]} and t["counts"]["teacher_mode vs protoss"] == {"teacher_mode": [1, 2]}
    assert t["counts"]["zerg_build vs terran"] == {"zerg_build": [3, 4]} and t["counts"]["tp_style vs terran"] == {"tp_style": [1, 4]}
    assert t["counts"]["zerg_build vs protoss"] == {"zerg_build": [1, 2]} and t["counts"]["tp_style vs protoss"] == {"tp_style": [0, 2]}
    assert t["table"]["by_race"]["stochastic vs terran"]["n"] == 5, "the win-rate table is unchanged"
    tables.print_tables({"t0_tp/t0_tp": t})
    a = tmp_path / "a.json"
    a.write_text(json.dumps({"t0_tp/t0_tp": t}), encoding="utf-8")
    capsys.readouterr()
    protocol.main(["compare", str(a), str(a)])
    out = capsys.readouterr().out
    assert "t0_tp/t0_tp zerg_build vs terran zerg_build: 3/4 vs 3/4  Fisher p 1.000" in out
    assert "t0_tp/t0_tp tp_style vs protoss tp_style: 0/2 vs 0/2  Fisher p 1.000" in out
    assert "t0_tp/t0_tp teacher_mode vs terran teacher_mode: 3/4 vs 3/4" in out
    tvz = make_job(tmp_path / "z", "tvz_bc", [row(1, True)])
    assert "teacher_mode" not in tables.job_tables(tvz, tmp_path / "cache", lin=LIN, ops="off")["tvz_bc/tvz_bc"]
    (job / "t0_tp" / "replays").mkdir(parents=True)
    (job / "t0_tp" / "replays" / "g5.rep").write_bytes(b"rep")
    monkeypatch.setenv("URANIUM_REPLAY_BIN", str(tmp_path / "missing-uranium-replay"))
    t2 = tables.job_tables(job, tmp_path / "cache", lin=LIN, ops="off")["t0_tp/t0_tp"]
    assert "cargo build" in t2["teacher_mode"]["error"] and t2["table"] == t["table"]
    tables.print_tables({"t0_tp/t0_tp": t2})


def test_teacher_mode_in_subset_tables_050_r10(tmp_path):
    zl = {"zero_last_failed": True}
    rows = [{**row(1, True), "model": {**row(1, True)["model"], **zl}}, {**row(116, False), "model": {**row(116, False)["model"], **zl}}]
    job = make_job(tmp_path, "t0_300", rows)
    cache = tmp_path / "cache" / job.name / "t0_300"
    dump(cache / "g1.json", cmds(2000))
    dump(cache / "g116.json", cmds(5))
    (t,) = tables.subset_tables(job, "t0_tp", LIN, tmp_path / "cache", ops="off").values()
    assert t["teacher_mode"]["terran"]["teacher"] == 0 and t["teacher_mode"]["terran"]["n"] == 1
    assert t["teacher_mode"]["protoss"]["teacher"] == 1 and t["teacher_mode"]["protoss"]["teacher_mode"]["lost"] == 1


def test_ops_tables_from_replay_values_052_ac3(tmp_path):
    zerg, terran = {"self_race": "Terran", "enemy_race": "Zerg"}, {"self_race": "Terran", "enemy_race": "Terran"}
    rows = [{**row(1, True, scv_min_1m=6, scv_8m=12), **zerg}, {**row(2, False, scv_min_1m=2, scv_8m=3), **zerg},
            {**row(3, True, scv_min_1m=8, scv_8m=None), **zerg}, {**row(4, None, scv_min_1m=5, scv_8m=9), **terran},
            {**row(5, False), **terran}, {**row(6, True), **zerg}, {**row(7, True), **zerg}]
    ledger = [{"game_id": 100007, "outcome": "Aborted", "detail": "launcher exit codes [Some(0), None]"}]
    job = make_job(tmp_path, "tvz_bc", rows, ledger)
    cache = tmp_path / "cache" / job.name / "tvz_bc"
    write_ops(cache, 1, ops_value(4000, sblock_first_s=90.0, cc_lift_s=60.0, rush=rush_value(pull_s=4.0, scv_lost=2, kills=3, bunker_load_s=0.0, focus_hhi=1.0)))
    write_ops(cache, 2, ops_value(2000, end_s=300.0, rush=rush_value(scv_lost=9, kills=1, focus_hhi=0.5)))
    write_ops(cache, 3, ops_value(6000, cc_lift_s=30.0))
    write_ops(cache, 4, ops_value(5000, sblock_first_s=120.0, rush=rush_value(pull_s=10.0, bunker_load_s=12.0)))
    write_ops(cache, 6, ops_value(1, end_s=60.0), slot=1)
    write_ops(cache, 7, ops_value(1))
    res = tables.job_tables(job, tmp_path / "cache", lin=LIN)
    t = res[f"{job.name}/tvz_bc"]
    assert t["table"]["total"]["n"] == 6 and t["ops_source"] == "replay" and t["ops_failed"] == {}
    ops, rush = t["ops"], t["rush_combat"]
    assert sorted(ops) == ["TvT", "TvZ", "all"]
    a = ops["all"]
    assert (a["n"], a["missing"]) == (4, 2), "games 5 (no replay) and 6 (another slot) have no value; 7 is invalid"
    assert a["mined"] == {"n": 4, "median": 4500.0, "mean": 4250.0}
    assert a["sblock_first_s"] == {"n": 2, "median": 105.0, "mean": 105.0}, "empty values left out"
    assert a["inc_10m"]["n"] == 3 and a["cc_lift_s_per_s"]["mean"] == (0.1 + 0.0 + 0.05 + 0.0) / 4
    assert (ops["TvZ"]["n"], ops["TvZ"]["missing"], ops["TvT"]["n"], ops["TvT"]["missing"]) == (3, 1, 1, 1)
    r = rush["all"]["all"]
    assert (r["n"], r["entered"], r["entered_rate"]) == (4, 3, 0.75)
    assert (r["pull_n"], r["pull_rate"], r["pull_s_p50"]) == (2, 2 / 3, 7.0)
    assert (r["scv_lost_p50"], r["kills_p50"], r["bunker_load_n"], r["bunker_load_rate"], r["focus_hhi_p50"]) == (2, 1, 2, 2 / 3, 0.75)
    held = rush["all"]
    assert (held["held_s"]["n"], held["held_s"]["entered"], held["not_held_s"]["n"], held["not_held_s"]["entered"]) == (3, 2, 1, 1)
    assert (held["held_8m"]["n"], held["held_8m"]["entered"], held["not_held_8m"]["entered"]) == (2, 2, 1), "scv_8m 12 and 9 held; 3 and none not"
    assert rush["TvT"]["all"]["entered"] == 1 and rush["TvZ"]["not_held_8m"]["pull_rate"] == 0.0
    none = tables.ops_tables({1: rows[0]}, {})
    assert none[0]["all"]["n"] == 0 and none[1]["all"]["all"]["entered_rate"] is None and none[1]["all"]["held_s"]["pull_rate"] is None
    out = tmp_path / "t.json"
    tables.main([str(job), "--cache", str(tmp_path / "cache"), "--json", str(out)])
    j = json.loads(out.read_text(encoding="utf-8"))[f"{job.name}/tvz_bc"]
    assert j["ops"]["all"]["mined"]["median"] == 4500.0 and j["rush_combat"]["all"]["all"]["entered"] == 3
    assert "ops" not in tables.job_tables(job, tmp_path / "cache", lin=LIN, ops="off")[f"{job.name}/tvz_bc"]
    live_rows = [{**rows[0], "model": {**rows[0]["model"], **ops_value(777, rush=rush_value(pull_s=1.0))}}, rows[1]]
    live = tables.job_tables(make_job(tmp_path / "l", "tvz_bc", live_rows), tmp_path / "cache", lin=LIN, ops="live")
    (lt,) = live.values()
    assert lt["ops_source"] == "live" and lt["ops"]["all"]["n"] == 1 and lt["ops"]["all"]["missing"] == 1
    assert lt["ops"]["all"]["mined"]["median"] == 777 and lt["rush_combat"]["all"]["all"]["pull_s_p50"] == 1.0
