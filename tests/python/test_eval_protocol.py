"""050-R9 / 050-AC5 measurement protocol (tools/eval, config/protocol/lineage.toml, docs/measurement_protocol.md):
the moved table rules (050-R6 / 050-R7 / 050-R8) on small job fixtures, and the protocol refusing anything that
differs from the pinned Checkpoint 1 conditions. CPU only, no game data (the shared-drive comparison skips
without the shared drive)."""

import copy
import json
import os
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

# 050-AC5: the conditions of every protocol plan, frozen. A change here is a protocol change (spec first).
FROZEN = {
    "rush_check": "ed79acba03b0cbfe09a2c5fba77384b82609e4bdabd419a505d95e711c996b2b",
    "macro": "8e17896f8f6dbd5ab2992a1a7cce0db82f2f1900ab22d921a9dd24021a91897d",
    "tvz_bc": "088e2bd81edc3549f31e0d186d871b9dc6f1083d8251fcbbab740ba6728c385a",
    "tvz_rl": "dad4a4ab104e18a9f09b86a73787d51bc939c2d7dc8a28c2b99e318b615d0b0e",
    "t0_100": "30f6dd8e691b6606b7b313a80d4a0878bc504220591fc431582bc2d98b4b0fc2",
    "t0_300": "54594ce9ccfaf84d22a4e6297c6324e62d7a9993b24d7474f6d9ef278e337f6d",
    "t0_600": "0b9c682411be9414b5afb9e461a379004987021ee1c1c57e3b5f2b5eecd70f0d",
    "t1_bc": "7d3f6fcd7f6eabc30ac3f37683aa28f5a916759575fb1daf15cedf4770117405",
    "t1_rl": "e625288dac81cc035158210b17adb66df6d3554a3ba0608b3a2dc5670d07f140",
    "ac1_stoch_lf0": "10c933d68c30d1d9537aa32814ddb81ce030933d1c51b0934a2e310e3cacc8f9",
    "ac1_stoch_real": "26b945e09943581885a45e4cde93f65fd8af45d48c830e0574a12fcf9076d849",
    "ac1_argmax_real": "d6419d130914f07aad97e99e480b0e72a4fab12180cfa9f99dada8902a688917",
}

# The Checkpoint 1 plans on the shared drive the templates were taken from (relative to URANIUM_SHARED_DIR).
SHARED_PLANS = [
    ("rush_check", "eval/bc-check-20261105/plans/01-rush5check.toml"),
    ("rush_check", "eval/rush5-20261105/c0-ac1/plans/01-rush5check.toml"),
    ("macro", "eval/dagger-test-r1-20261107/plans/02-macro.toml"),
    ("macro", "eval/ac1-ck-20261108/plans/01-macro.toml"),
    ("tvz_bc", "eval/dagger-tvz-r1-20261108/plans/01-tvz.toml"),
    ("tvz_rl", "eval/ac1-ck-20261108/plans/02-tvz.toml"),
    ("t0_100", "eval/bc-mix-t0-e10-20261107/plans/01-t0.toml"),
    ("t0_300", "eval/bc-e2-final/plans/01-e2-final-vs-t0.toml"),
    ("t0_600", "eval/bc-main/plans/01-bc-vs-t0.toml"),
    ("t1_bc", "eval/t1-ck1-20261108/plans/01-t1.toml"),
    ("t1_rl", "eval/ac1-ck-20261108/plans/03-t1.toml"),
    ("ac1_stoch_lf0", "eval/ac1-20261101/plans/01-bc-stoch-lf0.toml"),
    ("ac1_stoch_real", "eval/ac1-20261101/ac1-vtrace-s1-3m/plans/01-ac1-vtrace-s1-3m-stoch.toml"),
    ("ac1_argmax_real", "eval/ac1-20261101/ac1-vtrace-s1-3m/plans/02-ac1-vtrace-s1-3m-argmax.toml"),
]

CLEAN_GIT = {"commit": "0" * 40, "protocol_files_dirty": False, "dirty_paths": [], "dirty_diff_sha256": None}


def plan_file(name: str) -> dict:
    return tomllib.loads((ROOT / "config" / "plans" / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ protocol: templates and pinned values


def test_conditions_frozen_050_ac5():
    assert {p: s["conditions_sha256"] for p, s in LIN["plan"].items()} == FROZEN
    for pid in LIN["plan"]:
        plan = protocol.check_template(LIN, pid)
        assert plan["name"] == protocol.NAME and protocol.conditions_sha256(plan) == FROZEN[pid]


def test_shared_drive_plans_050_ac5():
    """Opt-in: runs where the shared drive with the Checkpoint 1 eval jobs is mounted."""
    ec.load_env()
    shared = os.environ.get("URANIUM_SHARED_DIR")
    found = [(pid, Path(shared) / rel) for pid, rel in SHARED_PLANS] if shared else []
    found = [(pid, f) for pid, f in found if f.is_file()]
    if not found:
        pytest.skip("Checkpoint 1 eval plans not on this host")
    for pid, f in found:
        assert protocol.identify(LIN, tomllib.loads(f.read_text(encoding="utf-8"))) == pid, f


@pytest.mark.parametrize("pid, source", [
    ("t1_bc", "t1_eval_ck1.toml"), ("t1_bc", "t1_eval_bc.toml"), ("t1_rl", "t1_eval_ac1.toml"), ("macro", "model_macro_check.toml"),
    ("t0_600", "bc-vs-t0.toml"), ("rush_check", "rush5_check.toml"),
])
def test_templates_are_the_checkpoint1_plans_050_r9(pid, source):
    """A template rendered with any checkpoint has the conditions of the tracked Checkpoint 1 plan."""
    src = plan_file(source)
    assert protocol.identify(LIN, src) == pid
    text = protocol.render(LIN, pid, "checkpoints/m2-x/ckpt-1", "m2-x")
    plan = tomllib.loads(text)
    assert plan["name"] == "m2-x" and {s["checkpoint"] for s in protocol.model_sides(plan)} == {"checkpoints/m2-x/ckpt-1"}
    assert protocol.conditions_sha256(plan) == protocol.conditions_sha256(src)


def test_zero_last_failed_pinned_per_plan_050_r9():
    """Checkpoint 1 values: rush / macro false for every family, TvZ / T1 / T0 true for BC, RL false."""
    zlf = {pid: tomllib.loads(protocol.render(LIN, pid, "c", "n"))["zero_last_failed"] for pid in LIN["plan"]}
    assert zlf["rush_check"] is False and zlf["macro"] is False
    assert zlf["tvz_bc"] and zlf["t1_bc"] and zlf["t0_100"] and zlf["t0_300"] and zlf["t0_600"] and zlf["ac1_stoch_lf0"]
    assert not zlf["tvz_rl"] and not zlf["t1_rl"] and not zlf["ac1_stoch_real"] and not zlf["ac1_argmax_real"]


def test_edited_template_is_refused_050_ac5(tmp_path):
    lin = copy.deepcopy(LIN)
    text = (ROOT / lin["plan"]["tvz_bc"]["template"]).read_text(encoding="utf-8").replace("zero_last_failed = true", "zero_last_failed = false")
    f = tmp_path / "tvz.toml"
    f.write_text(text, encoding="utf-8")
    lin["plan"]["tvz_bc"]["template"] = str(f)
    with pytest.raises(protocol.Refused, match="conditions"):
        protocol.render(lin, "tvz_bc", "c", "n")


def test_template_needs_placeholders_050_ac5(tmp_path):
    lin = copy.deepcopy(LIN)
    text = (ROOT / lin["plan"]["tvz_bc"]["template"]).read_text(encoding="utf-8").replace('"@CHECKPOINT@"', '"checkpoints/x/ckpt-1"', 1)
    f = tmp_path / "tvz.toml"
    f.write_text(text, encoding="utf-8")
    lin["plan"]["tvz_bc"]["template"] = str(f)
    with pytest.raises(protocol.Refused, match="placeholders"):
        protocol.check_template(lin, "tvz_bc")


def test_pinned_value_mismatch_is_refused_050_ac5():
    lin = copy.deepcopy(LIN)
    lin["plan"]["macro"]["max_frames"] = 86400
    with pytest.raises(protocol.Refused, match="differs from lineage.toml"):
        protocol.check_template(lin, "macro")


def test_family_and_menu_refusals_050_ac5():
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


def test_run_prefix_refused_050_ac5():
    pt = protocol.point(LIN, "s2e2")
    protocol.check_checkpoint(LIN, "m2", pt, "checkpoints/m2-s2e2/ckpt-11387")
    with pytest.raises(protocol.Refused, match="start with"):
        protocol.check_checkpoint(LIN, "m2", pt, "checkpoints/bc-s2e2-pt/ckpt-11387")
    with pytest.raises(protocol.Refused, match="no lineage"):
        protocol.check_checkpoint(LIN, "m3", pt, "checkpoints/m2-s2e2/ckpt-1")
    assert protocol.run_of("checkpoints/ac1-vtrace-s1/ckpt-3133569/agents/main") == "ac1-vtrace-s1"


def test_ck1_point_with_another_checkpoint_refused_050_ac5(monkeypatch):
    monkeypatch.setattr(protocol, "checkpoint_sha256", lambda c: "a" * 64 if "bc-dagger-r1" in c or "copy-of-r1" in c else "b" * 64)
    pt = protocol.point(LIN, "r1")
    protocol.check_checkpoint(LIN, "ck1", pt, "checkpoints/bc-dagger-r1/ckpt-1137")
    protocol.check_checkpoint(LIN, "ck1", pt, "eval/x/copy-of-r1/ckpt")  # a copy with the same weights
    with pytest.raises(protocol.Refused, match="Checkpoint 1 lineage is checkpoints/bc-dagger-r1/ckpt-1137"):
        protocol.check_checkpoint(LIN, "ck1", pt, "checkpoints/bc-dagger-r2/ckpt-1218")


def test_ck1_lineage_takes_no_new_measurements_050_r9(tmp_path):
    hosts = tmp_path / "hosts.toml"
    hosts.write_text('[[host]]\naddr = "h:7700"\n', encoding="utf-8")
    for metric in ("rush", "tvz", "t1"):
        with pytest.raises(SystemExit, match="REFUSED: lineage ck1 is measured already"):
            protocol.main(["job", "--lineage", "ck1", "--point", "r1", "--checkpoint", "checkpoints/bc-dagger-r1/ckpt-1137", "--metrics", metric,
                           "--job", "j", "--hosts", str(hosts), "--out-root", str(tmp_path)])
    with pytest.raises(SystemExit, match="REFUSED: lineage ck1 is measured already"):
        protocol.main(["bc-eval", "--metric", "val500", "--lineage", "ck1", "--point", "r1", "--checkpoint", "checkpoints/bc-dagger-r1/ckpt-1137"])
    protocol.check_new_measurement(LIN, "ck1", "cue_diag")
    protocol.check_new_measurement(LIN, "ck1", "dead_channels")
    protocol.check_new_measurement(LIN, "m2", "rush")
    assert not (tmp_path / "j").exists()


def fake_checkpoint(tmp_path: Path, run: str = "m2-s2e2") -> Path:
    d = tmp_path / "checkpoints" / run / "ckpt-1"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"weights")
    return d


def accepted(bot: str) -> str:
    return tomllib.loads(protocol.BOTS.read_text(encoding="utf-8"))["bot"][bot]["sha256"][0]


def job_args(tmp_path, ck, metrics, hosts_text='[[host]]\naddr = "h:7700"\nmax_games = 3\n', point="s2e2", bot_lines=None):
    hosts = tmp_path / "hosts.toml"
    hosts.write_text(hosts_text, encoding="utf-8")
    args = ["job", "--lineage", "m2", "--point", point, "--checkpoint", str(ck), "--metrics", metrics, "--job", "j",
            "--hosts", str(hosts), "--out-root", str(tmp_path / "eval")]
    if bot_lines is not None:
        f = tmp_path / "bots.txt"
        f.write_text("".join(f"{line}\n" for line in bot_lines), encoding="utf-8")
        args += ["--bot-hashes", str(f)]
    return args


def test_job_written_and_recorded_050_r9(tmp_path, monkeypatch):
    """--metrics rush,tvz on a BC point: two jobs (zero_last_failed false / true), each with protocol.json."""
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    protocol.checkpoint_sha256.cache_clear()
    ck = fake_checkpoint(tmp_path)
    protocol.main(job_args(tmp_path, ck, "rush,tvz", bot_lines=[f"h:7700 ualbertabot12poollings {accepted('ualbertabot12poollings')}"]))
    rush, tvz = tmp_path / "eval" / "j-rush", tmp_path / "eval" / "j-tvz"
    assert sorted(p.name for p in (tmp_path / "eval").iterdir()) == ["j-rush", "j-tvz"]
    sha = protocol.file_sha256(ck / "model.safetensors")
    for d, zlf, pid in ((rush, False, "rush_check"), (tvz, True, "tvz_bc")):
        rec = json.loads((d / "protocol.json").read_text(encoding="utf-8"))
        assert (rec["lineage"], rec["point"], rec["family"], rec["zero_last_failed"]) == ("m2", "s2e2", "bc", zlf)
        assert rec["checkpoint_sha256"] == sha and rec["checkpoint_id"] == f"ckpt-1@{sha[:12]}"
        assert rec["git"] == CLEAN_GIT and rec["hosts"] == [{"addr": "h:7700", "platform": None}] and rec["t1_platforms"] is None
        (p,) = rec["plans"]
        assert p["plan"] == pid and p["conditions_sha256"] == FROZEN[pid]
        assert protocol.file_sha256(d / p["file"]) == p["content_sha256"]
        plan = tomllib.loads((d / p["file"]).read_text(encoding="utf-8"))
        assert plan["zero_last_failed"] is zlf and protocol.identify(LIN, plan) == pid
        assert tomllib.loads((d / "job.toml").read_text(encoding="utf-8"))["stages"] == [p["file"]]
        assert tomllib.loads((d / "hosts.toml").read_text(encoding="utf-8"))["host"] == [{"addr": "h:7700", "max_games": 3}]
        assert protocol.verify_job(LIN, d, "m2", protocol.point(LIN, "s2e2")) == []
    assert json.loads((rush / "protocol.json").read_text(encoding="utf-8"))["bot_hashes"] == {"h:7700": {"ualbertabot12poollings": accepted("ualbertabot12poollings")}}
    assert json.loads((tvz / "protocol.json").read_text(encoding="utf-8"))["bot_hashes"] == {}
    # 050-AC5: an edited plan file and a dirty record are flagged
    f = rush / "plans" / "01-rush_check.toml"
    f.write_text(f.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    rec = json.loads((tvz / "protocol.json").read_text(encoding="utf-8"))
    rec["git"] = {**CLEAN_GIT, "protocol_files_dirty": True, "dirty_paths": ["tools/eval/tables.py"], "dirty_diff_sha256": "c" * 64}
    (tvz / "protocol.json").write_text(json.dumps(rec), encoding="utf-8")
    assert any("content_sha256" in p for p in protocol.verify_job(LIN, rush, "m2", protocol.point(LIN, "s2e2")))
    assert any("uncommitted protocol files" in p for p in protocol.verify_job(LIN, tvz, "m2", protocol.point(LIN, "s2e2")))


def test_dirty_protocol_files_refused_050_ac5(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "git_state", lambda: {**CLEAN_GIT, "protocol_files_dirty": True, "dirty_paths": ["tools/eval/tables.py"], "dirty_diff_sha256": "c" * 64})
    ck = fake_checkpoint(tmp_path)
    with pytest.raises(SystemExit, match=r"REFUSED: uncommitted protocol files \(tools/eval/tables.py\)"):
        protocol.main(job_args(tmp_path, ck, "tvz"))
    assert not (tmp_path / "eval").exists()


def test_git_state_outside_a_checkout_refused_050_ac5(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "ROOT", tmp_path)
    with pytest.raises(protocol.Refused, match="not a git checkout"):
        protocol.git_state()


def test_missing_weights_refused_050_ac5(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    ck = tmp_path / "checkpoints" / "m2-s2e2" / "ckpt-9"
    with pytest.raises(SystemExit, match="REFUSED: no .*model.safetensors"):
        protocol.main(job_args(tmp_path, ck, "tvz"))


def test_bot_hash_refused_offline_050_ac5(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    ck = fake_checkpoint(tmp_path)
    with pytest.raises(SystemExit, match="ualbertabot12poollings 000000000000 is not an accepted NDEBUG build"):
        protocol.main(job_args(tmp_path, ck, "rush", bot_lines=["h:7700 ualbertabot12poollings " + "0" * 64]))
    with pytest.raises(SystemExit, match="no way to read its bot hashes"):
        protocol.main(job_args(tmp_path, ck, "rush"))
    with pytest.raises(SystemExit, match="ssh host without oss_bots_dir"):
        protocol.main(job_args(tmp_path, ck, "rush", hosts_text='[[host]]\naddr = "h:7700"\nssh = "x"\n'))
    assert not (tmp_path / "eval").exists()


def test_t1_platform_refused_050_r9(tmp_path, monkeypatch):
    """T1 runs on the platform of the point's Checkpoint 1 counterpart (BC baseline: Linux ROCm hosts)."""
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    ck = fake_checkpoint(tmp_path)
    with pytest.raises(SystemExit, match=r"T1 of point s2e2 runs on linux hosts only; host list has h:7700 \(windows\)"):
        protocol.main(job_args(tmp_path, ck, "t1", hosts_text='[[host]]\naddr = "h:7700"\nplatform = "windows"\n'))
    with pytest.raises(SystemExit, match=r"h:7700 \(no platform\)"):
        protocol.main(job_args(tmp_path, ck, "t1"))
    assert protocol.check_t1_platform(protocol.point(LIN, "r1"), [{"addr": "a", "platform": "windows"}]) == ["windows"]
    with pytest.raises(protocol.Refused, match="no Checkpoint 1 T1 counterpart"):
        protocol.check_t1_platform(protocol.point(LIN, "mix-e1"), [{"addr": "a", "platform": "linux"}])


def test_job_refuses_mismatch_before_writing_050_ac5(tmp_path):
    hosts = tmp_path / "hosts.toml"
    hosts.write_text('[[host]]\naddr = "h:7700"\n', encoding="utf-8")
    base = ["job", "--lineage", "m2", "--point", "rl-s1", "--checkpoint", "checkpoints/m2-rl-s1/ckpt-1", "--job", "j", "--hosts", str(hosts),
            "--out-root", str(tmp_path), "--allow-dirty"]
    with pytest.raises(SystemExit, match="REFUSED: metric t0_100 is not in the menu"):
        protocol.main(base + ["--metrics", "t0_100"])
    with pytest.raises(SystemExit, match="REFUSED: lineage m2 runs start with"):
        protocol.main(base[:6] + ["checkpoints/ac1-vtrace-s1/ckpt-1/agents/main"] + base[7:] + ["--metrics", "rush"])
    assert not (tmp_path / "j").exists()


def test_check_bc_eval_refuses_other_metrics_050_r9(tmp_path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"split": "val", "games": 500, "selection_seed": 1, "action_top1": 0.7}), encoding="utf-8")
    protocol.main(["check-bc-eval", str(f), "--metric", "val500"])
    with pytest.raises(SystemExit, match="REFUSED: per_action is not a bc-eval metric"):
        protocol.main(["check-bc-eval", str(f), "--metric", "per_action"])
    with pytest.raises(SystemExit, match="MISMATCH"):
        protocol.main(["check-bc-eval", str(f), "--metric", "test"])


def test_verify_reports_condition_mismatch_050_ac5(tmp_path):
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
    assert any("no protocol.json" in p for p in problems), problems
    assert any("zero_last_failed [False], pinned True" in p for p in problems), problems
    assert any("stage 2" in p and "match no protocol plan" in p for p in problems), problems


# ------------------------------------------------------------------ eval_common rules


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
        {"game_id": 100002, "model": {"scv_8m": 4}},  # attempt 0 written after attempt 1: attempt 1 wins
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


def make_job(tmp_path: Path, pid: str, rows: list[dict], ledger: list[dict] | None = None, plan_text: str | None = None) -> Path:
    job = tmp_path / pid
    (job / "plans").mkdir(parents=True)
    (job / "results").mkdir()
    (job / "games").mkdir()
    (job / "plans" / "01.toml").write_text(plan_text or protocol.render(LIN, pid, "checkpoints/m2-r1/ckpt-5", pid), encoding="utf-8")
    (job / "job.toml").write_text('stages = ["plans/01.toml"]\nhosts = "hosts.toml"\n', encoding="utf-8")
    (job / "results" / "h.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (job / "games" / "h.jsonl").write_text("".join(json.dumps(g) + "\n" for g in ledger or []), encoding="utf-8")
    return job


def row(n: int, won, **model) -> dict:
    return {"game_id": 100000 + n, "slot": 0, "won": won, "reason": "Elimination", "end_frame": 1440 * n,
            "model": {"checkpoint": "ckpt-5@0", "zero_last_failed": False, "sample": "stochastic", **model}}


def early_games(monkeypatch, k: int) -> list[int]:
    monkeypatch.setattr(ec, "start_condition", lambda seed: "early" if seed % 2 else "late")
    seeds = tomllib.loads(protocol.render(LIN, "rush_check", "c", "n"))["seeds"]
    return [n for n in range(1, 129) if seeds[n - 1] % 2][:k]


def test_rush_table_both_held_and_bunker_definitions_050_r7(tmp_path, monkeypatch):
    early = early_games(monkeypatch, 4)
    rows = [
        row(early[0], True, scv_8m=10, scv_min_1m=3, bunker_s=170.0, scv_4m=9),   # held_8m only, unit < 3:00
        row(early[1], False, scv_8m=None, scv_min_1m=6, bunker_s=200.0, scv_4m=4),  # held_s only, unit < 4:00
        row(early[2], None, scv_8m=8, scv_min_1m=5, bunker_s=None, scv_4m=None),   # both held, no bunker
        row(early[3], True, scv_8m=None, scv_min_1m=2, bunker_s=None, scv_4m=6),   # won before 8:00: held_8m_winpass only
    ]
    job = make_job(tmp_path, "rush_check", rows)
    cache = tmp_path / "cache" / job.name / "rush_check"
    dump(cache / f"g{early[0]}.json", [build(500, 10, 10, 109), build(4000, 12, 12, ec.BUNKER)])   # order < 3:00
    dump(cache / f"g{early[1]}.json", [build(500, 10, 10, 109), build(5000, 12, 12, ec.BUNKER)])   # order < 4:00
    res = tables.job_tables(job, tmp_path / "cache", lin=LIN)
    t = res[f"{job.name}/rush_check"]
    assert t["plan"] == "rush_check" and t["kind"] == "rush" and t["protocol"] is None
    e = t["table"]["early"]
    assert (e["n"], e["held_8m"], e["held_8m_winpass"], e["held_s"]) == (4, 2, 3, 2)
    assert (e["bunker_order_3m"], e["bunker_order_4m"], e["bunker_unit_3m"], e["bunker_unit_4m"]) == (1, 2, 1, 2)
    assert e["scv_4m_p50"] == 5 and e["scv_4m_p50_known"] == 6 and e["won"] == 2 and e["dumped"] == 2
    assert t["counts"]["early"]["held_s"] == [2, 4] and t["table"]["late"]["n"] == 0


def test_bunker_unit_boundaries_050_r7(tmp_path, monkeypatch):
    """bunker_unit_3m is bunker_s < 180, bunker_unit_4m < 240 (strict)."""
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
