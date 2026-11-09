"""050-R6 / 050-R7 / 050-R8 measurement tools (tools/eval, docs/measurement_protocol.md): the moved table rules
on small job fixtures, and the protocol (config/protocol/lineage.toml) refusing anything that differs from the
pinned Checkpoint 1 conditions. CPU only, no game data."""

import copy
import json
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "eval"))

import eval_common as ec  # noqa: E402
import protocol  # noqa: E402
import tables  # noqa: E402

LIN = protocol.load_lineage()


def plan_file(name: str) -> dict:
    return tomllib.loads((ROOT / "config" / "plans" / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ protocol: templates and pinned values


def test_templates_match_pinned_conditions_050():
    for pid in LIN["plan"]:
        plan = protocol.check_template(LIN, pid)
        assert plan["name"] == protocol.NAME


@pytest.mark.parametrize("pid, source", [
    ("t1_bc", "t1_eval_ck1.toml"), ("t1_bc", "t1_eval_bc.toml"), ("t1_rl", "t1_eval_ac1.toml"), ("macro", "model_macro_check.toml"),
    ("t0_600", "bc-vs-t0.toml"), ("dagger_r1", "dagger_r1.toml"), ("dagger_r2", "dagger_r2.toml"), ("rush_check", "rush5_check.toml"),
])
def test_templates_are_the_checkpoint1_plans_050(pid, source):
    """A template rendered with any checkpoint has the conditions of the tracked Checkpoint 1 plan."""
    src = plan_file(source)
    assert protocol.identify(LIN, src) == pid
    text = protocol.render(LIN, pid, "checkpoints/m2-x/ckpt-1", "m2-x")
    assert protocol.conditions_sha256(tomllib.loads(text)) == protocol.conditions_sha256(src)


def test_zero_last_failed_pinned_per_plan_050():
    """Checkpoint 1 values: rush / macro false for every family, TvZ / T1 / T0 true for BC, RL false."""
    zlf = {pid: tomllib.loads(protocol.render(LIN, pid, "c", "n"))["zero_last_failed"] for pid in LIN["plan"]}
    assert zlf["rush_check"] is False and zlf["macro"] is False
    assert zlf["tvz_bc"] and zlf["t1_bc"] and zlf["t0_100"] and zlf["t0_300"] and zlf["t0_600"] and zlf["ac1_stoch_lf0"]
    assert not zlf["tvz_rl"] and not zlf["t1_rl"] and not zlf["ac1_stoch_real"] and not zlf["ac1_argmax_real"]


def test_edited_template_is_refused_050(tmp_path):
    lin = copy.deepcopy(LIN)
    text = (ROOT / lin["plan"]["tvz_bc"]["template"]).read_text(encoding="utf-8").replace("zero_last_failed = true", "zero_last_failed = false")
    f = tmp_path / "tvz.toml"
    f.write_text(text, encoding="utf-8")
    lin["plan"]["tvz_bc"]["template"] = str(f)
    with pytest.raises(protocol.Refused, match="conditions"):
        protocol.render(lin, "tvz_bc", "c", "n")


def test_pinned_value_mismatch_is_refused_050():
    lin = copy.deepcopy(LIN)
    lin["plan"]["macro"]["max_frames"] = 86400
    with pytest.raises(protocol.Refused, match="differs from lineage.toml"):
        protocol.check_template(lin, "macro")


def test_family_and_menu_refusals_050():
    bc, rl = protocol.point(LIN, "r1"), protocol.point(LIN, "rl-s1")
    assert protocol.metric_plans(LIN, bc, "tvz") == ["tvz_bc"]
    assert protocol.metric_plans(LIN, rl, "tvz") == ["tvz_rl"]
    with pytest.raises(protocol.Refused, match="no plan for family"):
        protocol.metric_plans(LIN, rl, "t0_100")
    with pytest.raises(protocol.Refused, match="not in the menu"):
        protocol.check_metric(LIN, protocol.point(LIN, "mix-e1"), "t1")
    with pytest.raises(protocol.Refused, match="placeholder"):
        protocol.check_metric(LIN, bc, "cue_diag")
    lin = copy.deepcopy(LIN)
    lin["plan"]["tvz_rl"]["families"] = ["bc"]
    with pytest.raises(protocol.Refused, match="not pinned for family"):
        protocol.metric_plans(lin, rl, "tvz")


def test_run_prefix_refused_050():
    pt = protocol.point(LIN, "s2e2")
    protocol.check_checkpoint(LIN, "m2", pt, "checkpoints/m2-s2e2/ckpt-11387")
    with pytest.raises(protocol.Refused, match="start with"):
        protocol.check_checkpoint(LIN, "m2", pt, "checkpoints/bc-s2e2-pt/ckpt-11387")
    with pytest.raises(protocol.Refused, match="no lineage"):
        protocol.check_checkpoint(LIN, "m3", pt, "checkpoints/m2-s2e2/ckpt-1")
    assert protocol.run_of("checkpoints/ac1-vtrace-s1/ckpt-3133569/agents/main") == "ac1-vtrace-s1"


def test_job_refuses_mismatch_before_writing_050(tmp_path):
    hosts = tmp_path / "hosts.toml"
    hosts.write_text('[[host]]\naddr = "h:7700"\n', encoding="utf-8")
    base = ["job", "--lineage", "m2", "--point", "rl-s1", "--checkpoint", "checkpoints/m2-rl-s1/ckpt-1", "--job", "j", "--hosts", str(hosts),
            "--out-root", str(tmp_path), "--allow-dirty"]
    with pytest.raises(SystemExit, match="REFUSED: metric t0_100 is not in the menu"):
        protocol.main(base + ["--metrics", "t0_100"])
    with pytest.raises(SystemExit, match="REFUSED: lineage m2 runs start with"):
        protocol.main(base[:6] + ["checkpoints/ac1-vtrace-s1/ckpt-1/agents/main"] + base[7:] + ["--metrics", "rush"])
    assert not (tmp_path / "j").exists()


def test_verify_reports_condition_mismatch_050(tmp_path):
    """An eval job whose rows ran with another zero_last_failed, or whose plan is not a protocol plan."""
    job = tmp_path / "job"
    (job / "plans").mkdir(parents=True)
    (job / "results").mkdir()
    (job / "plans" / "01.toml").write_text(protocol.render(LIN, "tvz_bc", "checkpoints/m2-r1/ckpt-5", "tvz"), encoding="utf-8")
    other = protocol.render(LIN, "macro", "checkpoints/m2-r1/ckpt-5", "macro").replace("max_frames = 14400", "max_frames = 20000")
    (job / "plans" / "02.toml").write_text(other, encoding="utf-8")
    (job / "job.toml").write_text('stages = ["plans/01.toml", "plans/02.toml"]\nhosts = "hosts.toml"\n', encoding="utf-8")
    row = {"game_id": 100001, "slot": 0, "won": True, "reason": "Elimination", "end_frame": 100,
           "model": {"checkpoint": "ckpt-5@000000000000", "zero_last_failed": False, "sample": "stochastic"}}
    (job / "results" / "h.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    problems = protocol.verify_job(LIN, job, "m2", protocol.point(LIN, "r1"))
    assert any("zero_last_failed [False], pinned True" in p for p in problems), problems
    assert any("stage 2" in p and "match no protocol plan" in p for p in problems), problems


# ------------------------------------------------------------------ eval_common rules


def test_game_ids_and_opponent_crashes_050():
    assert ec.game_key(10100188) == 100188 and ec.stage_of(10100188) == 1 and ec.stage_of(300070) == 3
    ledger = [
        {"game_id": 300070, "outcome": "Aborted", "detail": "launcher exit codes [Some(0), Some(-1073741819)]; p0: ..."},
        {"game_id": 100260, "outcome": "Aborted", "detail": "launcher exit codes [Some(0), None]; p0: ..."},
        {"game_id": 100005, "outcome": "Aborted", "detail": "launcher exit codes [None, Some(0)]; p0: ..."},
        {"game_id": 10100007, "outcome": "Aborted", "detail": "launcher exit codes [Some(0), Some(1)]"},
        {"game_id": 100008, "outcome": "Finished", "detail": None},
    ]
    assert ec.opponent_crashes(ledger) == {300070, 100260, 100007}


def test_model_rows_last_row_and_retries_050():
    rows = [
        {"game_id": 100001, "model": {"scv_8m": 3}},
        {"game_id": 100001, "model": None},
        {"game_id": 100001, "model": {"scv_8m": 9}},
        {"game_id": 200001, "model": {"scv_8m": 1}},
        {"game_id": 10100002, "model": {"scv_8m": 5}},
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


def dump(path: Path, cmds: list, frames: int = 20000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rep": "x", "frames": frames, "players": [{"id": 0, "race": 1}, {"id": 1, "race": 0}], "cmds": cmds}), encoding="utf-8")


def build(frame: int, x: int, y: int, ty: int) -> list:
    """Build command [frame, 0, 12, hex]: order byte, x, y, unit type (u16 little endian)."""
    h = "1e" + "".join(f"{v & 0xFF:02x}{v >> 8:02x}" for v in (x, y, ty))
    return [frame, 0, 12, h]


def test_analyze_first_build_and_scout_050_r6(tmp_path):
    f = tmp_path / "g1.json"
    far_move = "".join(f"{v & 0xFF:02x}{v >> 8:02x}" for v in (3000, 3000, 0xFFFF)) + "e400" + "06" + "00"
    dump(f, [
        build(500, 10, 10, 109),        # depot: base at (320, 320)
        [600, 0, 9, "0101"],            # select 1 unit
        [700, 0, 21, far_move],         # targeted Move far from the base: scout
        build(4000, 12, 12, ec.BUNKER),
        build(5000, 14, 14, ec.BUNKER),
        build(7200, 20, 20, ec.CC),
        [800, 1, 12, build(0, 1, 1, ec.BUNKER)[3]],  # the other player's build is ignored
        build(12000, 5, 5, 111),        # after 8:00: ignored
    ])
    first, scout = ec.analyze(f)
    assert first == {109: 500, ec.BUNKER: 4000, ec.CC: 7200}
    assert scout == 700


def make_job(tmp_path: Path, pid: str, rows: list[dict], ledger: list[dict] | None = None) -> Path:
    job = tmp_path / pid
    (job / "plans").mkdir(parents=True)
    (job / "results").mkdir()
    (job / "games").mkdir()
    (job / "plans" / "01.toml").write_text(protocol.render(LIN, pid, "checkpoints/m2-r1/ckpt-5", pid), encoding="utf-8")
    (job / "job.toml").write_text('stages = ["plans/01.toml"]\nhosts = "hosts.toml"\n', encoding="utf-8")
    (job / "results" / "h.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (job / "games" / "h.jsonl").write_text("".join(json.dumps(g) + "\n" for g in ledger or []), encoding="utf-8")
    return job


def row(n: int, won, **model) -> dict:
    return {"game_id": 100000 + n, "slot": 0, "won": won, "reason": "Elimination", "end_frame": 1440 * n,
            "model": {"checkpoint": "ckpt-5@0", "zero_last_failed": False, "sample": "stochastic", **model}}


def test_rush_table_both_held_and_bunker_definitions_050_r7(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "start_condition", lambda seed: "early" if seed % 2 else "late")
    seeds = tomllib.loads(protocol.render(LIN, "rush_check", "c", "n"))["seeds"]
    early = [n for n in range(1, 129) if seeds[n - 1] % 2][:3]
    rows = [
        row(early[0], True, scv_8m=10, scv_min_1m=3, bunker_s=170.0, scv_4m=9),   # held_8m only, unit < 3:00
        row(early[1], False, scv_8m=None, scv_min_1m=6, bunker_s=200.0, scv_4m=4),  # held_s only, unit < 4:00
        row(early[2], None, scv_8m=8, scv_min_1m=5, bunker_s=None, scv_4m=None),   # both held, no bunker
    ]
    job = make_job(tmp_path, "rush_check", rows)
    cache = tmp_path / "cache" / job.name / "rush_check"
    dump(cache / f"g{early[0]}.json", [build(500, 10, 10, 109), build(4000, 12, 12, ec.BUNKER)])   # order < 3:00
    dump(cache / f"g{early[1]}.json", [build(500, 10, 10, 109), build(5000, 12, 12, ec.BUNKER)])   # order < 4:00
    res = tables.job_tables(job, tmp_path / "cache", lin=LIN)
    t = res[f"{job.name}/rush_check"]
    assert t["plan"] == "rush_check" and t["kind"] == "rush"
    e = t["table"]["early"]
    assert (e["n"], e["held_8m"], e["held_s"]) == (3, 2, 2)
    assert (e["bunker_order_3m"], e["bunker_order_4m"], e["bunker_unit_3m"], e["bunker_unit_4m"]) == (1, 2, 1, 2)
    assert e["scv_4m_p50"] == 4 and e["won"] == 1 and e["dumped"] == 2
    assert t["counts"]["early"]["held_s"] == [2, 3] and t["table"]["late"]["n"] == 0


def test_macro_table_050_r8(tmp_path):
    rows = [row(1, True, scv_8m=16, bunker_s=230.0), row(2, None, scv_8m=15, bunker_s=250.0), row(11, False, scv_8m=None, bunker_s=None)]
    job = make_job(tmp_path, "macro", rows)
    cache = tmp_path / "cache" / job.name / "macro"
    dump(cache / "g1.json", [build(500, 10, 10, 109), build(5000, 12, 12, ec.BUNKER), build(7200, 20, 20, ec.CC)])
    dump(cache / "g2.json", [build(500, 10, 10, 109), build(5760, 12, 12, ec.BUNKER), build(7201, 20, 20, ec.CC)])
    t = tables.job_tables(job, tmp_path / "cache", lin=LIN)[f"{job.name}/macro"]["table"]
    a = t["ualbertabot2hatchhydra"]
    assert (a["n"], a["bunker_order_4m"], a["bunker_unit_4m"], a["cc_order_5m"], a["scv_8m_16"]) == (2, 1, 1, 1, 1)
    assert (a["won"], a["draw"], a["lost"]) == (1, 1, 0)
    assert t["ualbertabot3hatchmuta"]["lost"] == 1 and t["total"]["n"] == 3


def test_t1_table_drops_opponent_crash_games_050(tmp_path):
    rows = [row(1, True), row(2, False), {**row(19, True), "reason": "Surrender"}, {**row(20, True), "reason": "Surrender"}]
    ledger = [{"game_id": 100020, "outcome": "Aborted", "detail": "launcher exit codes [Some(0), None]"}]
    job = make_job(tmp_path, "t1_bc", rows, ledger)
    t = tables.job_tables(job, tmp_path / "cache", lin=LIN)[f"{job.name}/t1_bc"]
    assert t["invalid_games"] == [100020]
    uab_z, uab_p = t["table"]["blocks"][0], t["table"]["blocks"][1]
    assert (uab_z["bot"], uab_z["n"], uab_z["won"], uab_z["missing"]) == ("ualbertabot", 2, 1, 16)
    assert (uab_p["n"], uab_p["won"], uab_p["surrender_wins"], uab_p["invalid"]) == (1, 1, 1, 1)
    assert t["table"]["total"]["won"] == 2 and t["table"]["total"]["n"] == 3


def test_winrate_table_tvz_050(tmp_path):
    rows = [row(1, True), row(2, None), row(17, False), row(18, True)]
    job = make_job(tmp_path, "tvz_bc", [{**r, "model": {**r["model"], "zero_last_failed": True}} for r in rows])
    t = tables.job_tables(job, tmp_path / "cache", lin=LIN)[f"{job.name}/tvz_bc"]
    assert t["plan"] == "tvz_bc" and t["zero_last_failed"] == [True]
    assert t["table"]["total"] == {"n": 4, "won": 2, "draw": 1, "lost": 1, "invalid": 0, "missing": 28}
