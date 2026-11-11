"""050-R9 / 050-AC5 measurement protocol (tools/eval, config/protocol/lineage.toml, docs/measurement_protocol.md):
the moved table rules (050-R6 / 050-R7 / 050-R8) on small job fixtures, and the protocol refusing anything that
differs from the pinned Checkpoint 1 conditions; the diagnostics cue_diag / dead_channels (commands, refusal of
outputs made with other settings, tables, the 041-R25 training-log summary) and the s1e2 per-checkpoint loop
(041-R20 ③). CPU only, no game data (the shared-drive comparison skips without the shared drive)."""

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "eval"))

import diag_table  # noqa: E402
import eval_common as ec  # noqa: E402
import protocol  # noqa: E402
import s1e2_loop  # noqa: E402
import spatial_logs  # noqa: E402
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
    "t0_tp": "0e033d973095d2fa73d140c1385c0daa74a94b7e1173dbf09c108c07f5a9d4a5",
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
    ("t0_tp", "eval/jepa-diag-tp-20261109/plans/02-t0.toml"),
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
    assert protocol.check_metric(LIN, bc, "cue_diag")["kind"] == "diag_cue"
    assert protocol.check_metric(LIN, bc, "dead_channels")["kind"] == "diag_train_log"
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
    for pt in LIN["point"]:
        protocol.check_new_measurement(LIN, "ck1", "cue_diag", pt)
        with pytest.raises(protocol.Refused, match="not measured"):  # AMB-050-5: no training-log statistics
            protocol.check_new_measurement(LIN, "ck1", "dead_channels", pt)
        protocol.check_new_measurement(LIN, "m2", "dead_channels", pt)
    protocol.check_new_measurement(LIN, "m2", "rush", protocol.point(LIN, "r1"))
    assert not (tmp_path / "j").exists()


def test_lineage_menus_consistent_050_r9():
    """Every Checkpoint 1 output and supplement is on the point's menu; a supplement is a cell Checkpoint 1 lacks."""
    for pt in LIN["point"]:
        menu = set(pt["metrics"]) | set(pt.get("optional", []))
        outs, sup = set(pt.get("ck1_outputs", {})), set(pt.get("ck1_supplement", []))
        assert outs <= menu and sup <= menu, pt["id"]
        assert not outs & sup, pt["id"]
        assert set(pt["metrics"]) - {"cue_diag", "dead_channels"} <= outs, pt["id"]
        if "t1" in menu:
            assert pt.get("t1_platforms"), pt["id"]
    assert protocol.point(LIN, "s2e2")["ck1_supplement"] == ["t0_600", "val500_race"]


def test_ck1_supplement_only_for_missing_cells_050_r9(tmp_path, monkeypatch):
    """050-R9: Checkpoint 1 is measured with the same plan only where the point table has no Checkpoint 1 value."""
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    monkeypatch.setattr(protocol, "checkpoint_sha256", lambda c: "a" * 64)
    hosts = tmp_path / "hosts.toml"
    hosts.write_text('[[host]]\naddr = "h:7700"\n', encoding="utf-8")
    base = ["job", "--lineage", "ck1", "--point", "s2e2", "--checkpoint", "checkpoints/bc-s2e2-pt/ckpt-11387", "--job", "j",
            "--hosts", str(hosts), "--out-root", str(tmp_path / "eval")]
    protocol.main(base + ["--metrics", "t0_600"])
    rec = json.loads((tmp_path / "eval" / "j" / "protocol.json").read_text(encoding="utf-8"))
    assert (rec["lineage"], rec["point"], rec["plans"][0]["plan"]) == ("ck1", "s2e2", "t0_600")
    for metric in ("rush", "t0_tp", "t0_300"):
        with pytest.raises(SystemExit, match=rf"no new {metric} measurement at point s2e2 \(it has {metric}: `protocol.py tables`\)"):
            protocol.main(base[:-1] + ["j2", "--metrics", metric])
    with pytest.raises(SystemExit, match="no new macro measurement at point s2e2; allowed there: cue_diag, t0_600, val500_race"):
        protocol.main(base[:-1] + ["j2", "--metrics", "macro"])
    r1 = ["--lineage", "ck1", "--point", "r1", "--checkpoint", "checkpoints/bc-dagger-r1/ckpt-1137"]
    protocol.main(["bc-eval", "--metric", "val500_race"] + r1)
    protocol.main(["bc-eval", "--metric", "test"] + r1)
    with pytest.raises(SystemExit, match="no new val500 measurement at point r1"):
        protocol.main(["bc-eval", "--metric", "val500"] + r1)
    protocol.check_new_measurement(LIN, "ck1", "t0_tp", protocol.point(LIN, "r2"))
    protocol.check_new_measurement(LIN, "ck1", "t0_tp", protocol.point(LIN, "mix-e10"))
    with pytest.raises(protocol.Refused):
        protocol.check_new_measurement(LIN, "ck1", "t0_tp", protocol.point(LIN, "r1"))


def test_t0_tp_template_050_r10():
    """T0 standard Terran / Protoss 32 + 32: the first 16 seeds of each map block of the T0 300 (seed_base 41300)."""
    plan = tomllib.loads(protocol.render(LIN, "t0_tp", "c", "n"))
    assert plan["seeds"] == [*range(41301, 41317), *range(41351, 41367), *range(41401, 41417), *range(41451, 41467)]
    assert [(m["map"], m["games"], m["p1"]) for m in plan["matchup"]] == [
        ("fighting_spirit", 16, {"kind": "t0", "race": "terran"}), ("circuit_breaker", 16, {"kind": "t0", "race": "terran"}),
        ("fighting_spirit", 16, {"kind": "t0", "race": "protoss"}), ("circuit_breaker", 16, {"kind": "t0", "race": "protoss"})]
    n = protocol.normalize(plan)
    assert n["zero_last_failed"] is True and n["max_frames"] == 86400 and n["wall_timeout_s"] == 7200
    assert {m["p0"]["sample"] for m in n["matchup"]} == {"stochastic"}
    # the same games as the T0 300's: seed_base + game number, same map and opponent
    t300 = tables.game_keys(tomllib.loads(protocol.render(LIN, "t0_300", "c", "n")))
    assert set(tables.game_keys(plan)) <= set(t300)


def test_subset_tables_t0_tp_inside_t0_300_050_r9(tmp_path):
    """The BC baseline's T0 T / P 32 + 32 is counted from its T0 300 job by seed / map / opponent."""
    rows = [{**row(1, True), "model": {**row(1, True)["model"], "zero_last_failed": True}},     # T FS seed 41301: in
            {**row(17, True), "model": {**row(17, True)["model"], "zero_last_failed": True}},   # T FS seed 41317: out
            {**row(51, None), "model": {**row(51, None)["model"], "zero_last_failed": True}},   # T CB seed 41351: in
            {**row(116, False), "model": {**row(116, False)["model"], "zero_last_failed": True}}]  # P FS seed 41416: in
    ledger = [{"game_id": 100151, "outcome": "Aborted", "detail": "launcher exit codes [Some(0), None]"}]  # P CB 41451: invalid
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


def test_val500_race_needs_per_race_output_050_r9(tmp_path):
    f = tmp_path / "v.json"
    base = {"split": "val", "games": 500, "selection_seed": 1, "action_top1": 0.72}
    f.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(SystemExit, match="no `by_enemy_race` with terran, protoss, zerg"):
        protocol.main(["check-bc-eval", str(f), "--metric", "val500_race"])
    per = {r: {"steps": 10, "action_top1": 0.7, "nll": {"action": 0.9}} for r in ("terran", "protoss", "zerg")}
    f.write_text(json.dumps({**base, "by_enemy_race": per}), encoding="utf-8")
    protocol.main(["check-bc-eval", str(f), "--metric", "val500_race"])
    v, r = LIN["metric"]["val500"], LIN["metric"]["val500_race"]
    assert {k: v[k] for k in ("datasets", "split", "games", "seed", "backend", "config")} == {k: r[k] for k in ("datasets", "split", "games", "seed", "backend", "config")}


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


# ------------------------------------------------------------------ diagnostics (050-R9 cue_diag, dead_channels)

CUE, DEAD = LIN["metric"]["cue_diag"], LIN["metric"]["dead_channels"]
TEACHER, HUMAN = CUE["teacher"], CUE["human"]
SHA = "f" * 64
TAILS = ["by_name/hold_orig", "by_name/hold_macro2"]


@pytest.fixture(autouse=True)
def _game_order(monkeypatch):
    """The outputs written below have no rows parts: their game order is the one a `FAKE_GAME_ROWS` file in the
    dump names (the pinned value unless a test writes another); `game_rows_sha256` itself is tested apart."""
    real = protocol.game_rows_sha256
    monkeypatch.setattr(protocol, "game_rows_sha256", lambda d: (Path(d) / "FAKE_GAME_ROWS").read_text() if (Path(d) / "FAKE_GAME_ROWS").is_file() else real(d))
HUMAN_DS = sorted(HUMAN["datasets"])


def test_diag_settings_are_the_tools_050_r9():
    """The pinned values are the diag tools' own (dump CF_STARTS, probes stages, monitor alarm layers / start),
    and the user decisions of 2026-11-10 (AMB-050-4 / -5 / -6)."""
    from uranium.diag import dump, probes
    from uranium.model import monitor

    assert {r: tuple(v) for r, v in TEACHER["dump"]["cf_starts"].items()} == dump.CF_STARTS
    assert ",".join(TEACHER["probes"]["stages"]) == ",".join(HUMAN["probes"]["stages"]) == probes.DEFAULT_STAGES
    assert DEAD["layers"] == list(monitor.ALARM_LAYERS) and DEAD["from_update"] == monitor.ALARM_FROM
    assert CUE["sets"] == ["teacher", "human"] and CUE["backend"] == "rocm"
    # AMB-050-4: the max-seen production-building count on human T / P games; constant (skipped) on the Zerg teacher rows
    assert HUMAN["probes"]["races"] == ["T", "P"] and "count|focus_rows" in HUMAN["report"]["required"]
    assert "count|focus_rows" in TEACHER["probes"]["expect_skipped"] and "intervene" in TEACHER and "intervene" not in HUMAN
    assert (HUMAN["games"], HUMAN["seed"], HUMAN["splits"]) == (500, 1, ["val", "test"]) and HUMAN["datasets"] == LIN["metric"]["val500"]["datasets"]
    # AMB-050-5: training-log records only, Checkpoint 1 not measured
    assert DEAD["kind"] == "diag_train_log" and "ck1" in DEAD["not_measured"] and LIN["lineage"]["ck1"]["new_metrics"] == ["cue_diag"]
    # AMB-050-6: s1e2 diagnostics at the final checkpoint only
    assert [p["id"] for p in LIN["point"] if p.get("diag_final_only")] == ["s1e2"]
    for pt in LIN["point"]:
        assert {"cue_diag", "dead_channels"} <= set(pt["metrics"]), pt["id"]
        for s in CUE["sets"]:
            assert set(CUE[s]["report"]["stages"]) <= set(CUE[s]["probes"]["stages"])


def test_diag_commands_pinned_050_r9(tmp_path):
    """The emitted commands carry every pinned setting (deterministic, explicit flags, not the tools' defaults)."""
    data = ["--shards", "A", "--shards", "B"]
    out = tmp_path / "out" / "teacher"
    dump, probes_c, inter = protocol.cue_commands(LIN, "teacher", "CK", out, data)
    assert dump == ["python", "-m", "uranium", "diag", "dump", "--checkpoint", "CK", "--shards", "A", "--shards", "B", "--splits", "val,test",
                    "--config", "config/bc.toml", "--backend", "rocm", "--lanes-per-group", "4", "--row-until", "8640",
                    "--act-until", "5760", "--act-every", "1", "--core-points", "gate", "--cf", "--cf-starts", "T:1280,2080;P:1720,3030;Z:1800,3240",
                    "--out", str(out / "dump")]
    assert probes_c == ["python", "-m", "uranium", "diag", "probes", str(out / "dump"), "--races", "Z", "--stages", "token,e_pool,sp_pool,c,x_in,L1.g1,L1.out,h,hc",
                        "--gain", "resid", "--boot", "500", "--cap", "0", "--seed", "7", "--device", "cpu", "--out", str(out / "probes.json")]
    assert inter == ["python", "-m", "uranium", "diag", "intervene", str(out / "dump"), "--races", "Z", "--window", "2160,5760", "--boot", "2000", "--seed", "3",
                     "--out", str(out / "intervene.json")]
    hd = [x for ds in HUMAN_DS for x in ("--dataset", ds)]
    ho = tmp_path / "out" / "human"
    hdump, hprobes = protocol.cue_commands(LIN, "human", "CK", ho, hd, dump=tmp_path / "local" / "dump")
    assert hdump == ["python", "-m", "uranium", "diag", "dump", "--checkpoint", "CK", *hd, "--splits", "val,test", "--games", "500", "--seed", "1",
                     "--config", "config/bc.toml", "--backend", "rocm", "--lanes-per-group", "4", "--row-until", "8640", "--act-until", "8640",
                     "--act-every", "4", "--core-points", "gate", "--read-until", "8640", "--out", str(tmp_path / "local" / "dump")]
    assert hprobes == ["python", "-m", "uranium", "diag", "probes", str(tmp_path / "local" / "dump"), "--races", "T,P", "--stages", "token,e_pool,sp_pool,c,x_in,L1.g1,L1.out,h,hc",
                       "--gain", "resid", "--boot", "300", "--cap", "60", "--seed", "7", "--device", "cpu", "--out", str(ho / "probes.json")]
    assert protocol.cue_commands(LIN, "teacher", "CK", out, data) == [dump, probes_c, inter]
    s1 = protocol.point(LIN, "s1e1")
    assert protocol.summary_command(LIN, s1, "teacher", out)[5:] == ["--point", "s1e1", "--probes", str(out / "probes.json"), "--intervene", str(out / "intervene.json"),
                                                                     "--out", str(out / "summary.json")]
    assert "--intervene" not in protocol.summary_command(LIN, s1, "human", ho)
    o = tmp_path / "out"
    assert protocol.train_log_command(LIN, s1, "checkpoints/m2-s1/ckpt-39096", "s1.log", "CK", o) == [
        "python", "tools/eval/spatial_logs.py", "s1.log", "--checkpoint", "CK", "--from-update", "600", "--until-update", "39096", "--out", str(o / "spatial_log.json")]
    assert "--until-update" not in protocol.train_log_command(LIN, protocol.point(LIN, "rl-s1"), "checkpoints/m2-rl/ckpt-5/agents/main", "train.csv", "CK", o)
    assert protocol.diag_out(LIN, "cue_diag", "checkpoints/m2-s1/ckpt-39096") == "artifacts/diag/m2-s1__ckpt-39096"
    assert protocol.diag_out(LIN, "dead_channels", "checkpoints/ac1-vtrace-s1/ckpt-3133569/agents/main") == "artifacts/diag/ac1-vtrace-s1__ckpt-3133569-agents-main"


def test_game_rows_sha256_is_the_game_order_050_r9(tmp_path):
    """The dump's rows per game id in id order (the probes' folds follow the ids): the same rows under other ids
    give another value; parts are concatenated in order; no rows -> None."""
    import hashlib

    import numpy as np

    d = tmp_path / "dump"
    d.mkdir()
    assert protocol.game_rows_sha256(d) is None
    np.savez_compressed(d / "rows_p000.npz", game=np.array([0, 0, 1, 2, 2, 2]), frame=np.zeros(6))
    np.savez_compressed(d / "rows_p001.npz", game=np.array([3, 1]), frame=np.zeros(2))
    assert protocol.game_rows_sha256(d) == hashlib.sha256(b"2,2,3,1").hexdigest()
    np.savez_compressed(d / "rows_p000.npz", game=np.array([1, 1, 0, 2, 2, 2]), frame=np.zeros(6))  # games 0 and 1 swapped ids
    assert protocol.game_rows_sha256(d) == hashlib.sha256(b"1,3,3,1").hexdigest()


def test_teacher_link_trees_order_games_as_the_lambda_runs_050_r9():
    """The hold-out folders are named after the entries (by_name/<name>): the shard reader orders by folder name,
    hold_macro2 before hold_orig, as the U1 parity and lambda runs (<data>/hold_orig, <data>/hold_macro2)."""
    names = [h["name"] for h in TEACHER["holdout"]]
    assert protocol.data_tails(TEACHER) == names and sorted(names) == ["hold_macro2", "hold_orig"]
    assert TEACHER["dump"]["expect_game_rows_sha256"] == "a29f81c1580c83db17e056b45b28de871675b30985735d0c74a67be25371bdf6"


def fake_source(root: Path, parts: dict[str, list[int]]) -> Path:
    for p, sizes in parts.items():
        (root / p).mkdir(parents=True, exist_ok=True)
        for i, n in enumerate(sizes):
            (root / p / f"shard_{i:05d}.bin.zst").write_bytes(b"x" * n)
    return root


def test_holdout_link_trees_and_fingerprint_050_r9(tmp_path):
    shared = tmp_path / "shared"
    src = fake_source(shared / "datasets/replays/shards/botgen", {"teacher-20261106-a": [3, 4], "teacher-20261107-macro2": [5]})
    lin = copy.deepcopy(LIN)
    lin["metric"]["cue_diag"]["teacher"]["holdout"] = [
        {"name": "hold_orig", "source": "datasets/replays/shards/botgen", "parts": ["teacher-20261106-a"],
         "files": 2, "files_sha256": protocol.files_fingerprint(src, ["teacher-20261106-a"])[1]},
        {"name": "hold_macro2", "source": "datasets/replays/shards/botgen", "parts": ["teacher-20261107-macro2"],
         "files": 1, "files_sha256": protocol.files_fingerprint(src, ["teacher-20261107-macro2"])[1]}]
    work = tmp_path / "work"
    assert protocol.set_data(lin, "teacher", shared, work, create=False) == ["--shards", str(work / TAILS[0]), "--shards", str(work / TAILS[1])]
    assert not work.exists()  # print mode writes nothing
    args = protocol.set_data(lin, "teacher", shared, work)
    assert args == ["--shards", str(work / TAILS[0]), "--shards", str(work / TAILS[1])]
    assert sorted(f.name for f in (work / TAILS[0] / "teacher-20261106-a").iterdir()) == ["shard_00000.bin.zst", "shard_00001.bin.zst"]
    (work / TAILS[1] / "teacher-20261107-macro2" / "shard_00009.bin.zst").write_bytes(b"z")  # a stale file in the link tree
    with pytest.raises(protocol.Refused, match="hold-out hold_macro2 \\(link tree"):
        protocol.set_data(lin, "teacher", shared, work)
    (src / "teacher-20261107-macro2" / "shard_00001.bin.zst").write_bytes(b"y")  # a shard added to the source
    with pytest.raises(protocol.Refused, match="hold-out hold_macro2 .*2 files"):
        protocol.set_data(lin, "teacher", shared, work)


def fake_datasets(shared: Path, extra: str | None = None) -> None:
    for i, ds in enumerate(HUMAN_DS):
        d = shared / "datasets" / ds
        d.mkdir(parents=True, exist_ok=True)
        files = {f"p0/shard_{k:05d}.bin.zst": {"sha256": f"{i}{k}" * 8, "bytes": 10 + k} for k in range(2)}
        if extra and ds == extra:
            files["p0/shard_00099.bin.zst"] = {"sha256": "e" * 64, "bytes": 1}
        (d / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")


def test_human_holdout_manifest_fingerprint_050_r9(tmp_path):
    """AMB-050-4: the human set is the four human datasets as their manifests list them (2,069 files when pinned)."""
    shared = tmp_path / "shared"
    fake_datasets(shared)
    lin = copy.deepcopy(LIN)
    n, sha = protocol.manifest_fingerprint(shared, HUMAN["datasets"])
    assert n == 8
    lin["metric"]["cue_diag"]["human"] |= {"files": n, "files_sha256": sha}
    assert protocol.set_data(lin, "human", shared) == [x for ds in HUMAN["datasets"] for x in ("--dataset", ds)]
    fake_datasets(shared, extra="replays/shards/scr_ladder")  # a newly collected ladder shard
    with pytest.raises(protocol.Refused, match="hold-out human \\(dataset manifests\\): 9 files"):
        protocol.set_data(lin, "human", shared)
    with pytest.raises(protocol.Refused, match="2069 files"):
        protocol.set_data(LIN, "human", shared)


def probe_metrics(stages) -> dict:
    m = {f"{s}|ridge": {"pt": 0.5, "lo": 0.4, "hi": 0.6} for s in stages} | {f"{s}+nuis|ridge": {"pt": 0.6, "gain": 0.1, "gain_lo": 0.05, "gain_hi": 0.15} for s in stages}
    return m | {"nuis|ridge": {"pt": 0.3, "lo": 0.2, "hi": 0.4}, "h|mlp": {"pt": 0.45}}


def set_outputs(d: Path, name: str, sha: str = SHA, **edit) -> Path:
    """One cue set's outputs (<out>/<set>) as `protocol.py diag --run` leaves them (the pinned settings and data)."""
    s = CUE[name]
    dm, p = s["dump"], s["probes"]
    data = [f"/h/diag_holdout/{t}" for t in TAILS] if name == "teacher" else [f"<shared>/datasets/{x}" for x in HUMAN_DS]
    man = {"checkpoint": "<shared>/checkpoints/m2-s1/ckpt-9", "checkpoint_sha256": sha, "data": data, "splits": ["val", "test"], "games_per_split": s.get("games"),
           "seed": 1, "config": "config/bc.toml", "lanes_per_group": 4, "row_until": 8640.0, "act_until": float(dm["act_until"]), "act_every": dm["act_every"],
           "read_until": float(dm["read_until"]) if dm.get("read_until") else None,
           "passes": ["real", "early", "late", "sc+1", "sc-1", "tok+1", "tok-1"] if dm["cf"] else ["real"],
           "cf_starts": {r: [float(x) for x in v] for r, v in dm["cf_starts"].items()} if dm["cf"] else None,
           "stages": ["token", "e_pool", "sp_pool", "c", "h", "x_in", "L1.a", "L1.g1", "L1.f", "L1.out"], "backend": "rocm", "precision": "f32",
           "games": dm.get("expect_games", 1000), "stats": {"rows": dm.get("expect_rows", 500000), "act_rows": dm.get("expect_act_rows", 90000)}} | edit.get("manifest", {})
    head = {"checkpoint": man["checkpoint"], "checkpoint_sha256": sha, "data": data, "splits": ["val", "test"], "backend": "rocm", "precision": "f32"}
    metric = probe_metrics(p["stages"])
    races = {}
    for race in p["races"]:
        tasks = {t: {"metric": "r2", "rows": 100, "games": 20, "metrics": metric} for t in s["report"]["tasks"]}
        tasks |= {t: {"metric": "r2", "rows": r, "games": g, "metrics": metric} for t, (r, g) in p["expect_tasks"].items()}
        tasks |= {t: {"rows": 36516, "skipped": "constant target 1"} for t in p["expect_skipped"]}
        races[race] = {"tasks": tasks}
    probes_o = {"dump": head, "gain_method": "resid", "boot": p["boot"], "cap": p["cap"], "mlp": True, "races": races} | edit.get("probes", {})
    d.mkdir(parents=True, exist_ok=True)
    (d / "dump").mkdir(exist_ok=True)
    (d / "dump" / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    (d / "dump" / "FAKE_GAME_ROWS").write_text(edit.get("game_rows", dm.get("expect_game_rows_sha256", "")), encoding="utf-8")
    (d / "probes.json").write_text(json.dumps(probes_o), encoding="utf-8")
    if "intervene" in s:
        eff = {"n": 100, "mean": 0.01, "lo": -0.01, "hi": 0.03, "share_ge_0.15": 0.0}
        allc = {e: {m: eff for m in ("bunker_any", "marine_sum", "rax2_any", "label_p")} for e in ("start", "start_unf", "count_sc", "count_tok", "real")}
        inter = {"dump": head, "window": [2160.0, 5760.0], "races": {"Z": {"games": 433, "edited": 300, "edited_unf": 120, "count_edited": 310, "by_class": {"all": allc}}}} | edit.get("intervene", {})
        (d / "intervene.json").write_text(json.dumps(inter), encoding="utf-8")
    return d


def train_log_output(out: Path, **edit) -> Path:
    r = {"log": "s1.log", "from_update": 600, "until_update": 39096, "sp_records_from": 25, "alarm_lines": [], "alarm_records": [], "crossing_records": [],
         "dead": {k: {"max": 3, "max_share": 0.05, "at": 700, "last": 1} for k in DEAD["layers"]} | {"r4": {"max": 51, "max_share": 0.398, "at": 3000, "last": 51}},
         "worst_dead_layer": "r4", "worst_dead_share": 0.398, "const": {"sp_const_s8": {"max": 0, "share": 0.0}, "sp_const_pool": {"max": 0, "share": 0.0}},
         "min_pos_any": 0.2} | edit
    out.mkdir(parents=True, exist_ok=True)
    (out / "spatial_log.json").write_text(json.dumps(r), encoding="utf-8")
    return out


def record(out: Path, metrics=("cue_diag", "dead_channels"), sha: str = SHA, **edit) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    rec = {"metrics": list(metrics), "checkpoint_sha256": sha, "git": dict(CLEAN_GIT), "finished": "2026-11-10T00:00:00+09:00"} | edit
    (out / "protocol.json").write_text(json.dumps(rec), encoding="utf-8")
    return out


def all_outputs(out: Path, sha: str = SHA) -> Path:
    for s in CUE["sets"]:
        set_outputs(out / s, s, sha)
    train_log_output(out)
    return record(out, sha=sha)


def test_check_diag_accepts_the_pinned_outputs_050_r9(tmp_path, capsys):
    out = all_outputs(tmp_path / "d")
    assert protocol.check_diag(LIN, "cue_diag", out, SHA) == []
    assert protocol.check_diag(LIN, "dead_channels", out, SHA) == []
    protocol.main(["check-diag", str(out)])
    text = capsys.readouterr().out
    assert "NOTE" not in text and "conform to the protocol" in text  # both sets' data fingerprints pinned


@pytest.mark.parametrize("name, edit, want", [
    ("teacher", {"manifest": {"passes": ["real"], "cf_starts": None}}, "made without --cf"),       # a lambda-run dump (no counterfactual)
    ("teacher", {"manifest": {"backend": "cuda", "precision": "bf16"}}, "teacher dump backend: 'cuda', pinned 'rocm'"),
    ("teacher", {"manifest": {"splits": ["val"]}}, "teacher dump splits"),
    ("teacher", {"manifest": {"lanes_per_group": 8}}, "teacher dump lanes_per_group"),
    ("teacher", {"manifest": {"games": 202, "stats": {"rows": 82148, "act_rows": 15553}}}, "teacher dump games: 202, pinned 433"),
    ("teacher", {"manifest": {"data": ["/h/some/other"]}}, "not the pinned hold-out"),
    # the earlier link trees <name>/botgen: the same games, hold_orig read first (other game ids)
    ("teacher", {"manifest": {"data": ["/h/diag_holdout/hold_orig/botgen", "/h/diag_holdout/hold_macro2/botgen"]}}, "orders the games otherwise"),
    ("teacher", {"game_rows": "210c" + "0" * 60}, "teacher dump game order (game_rows_sha256)"),
    ("teacher", {"manifest": {"cf_starts": {"T": [1280.0, 2080.0], "P": [1720.0, 3030.0], "Z": [1500.0, 3240.0]}}}, "teacher dump cf_starts"),
    ("teacher", {"manifest": {"stages": ["token", "e_pool", "h"]}}, "no activations at ['x_in', 'L1.out']"),
    ("teacher", {"probes": {"gain_method": "joint"}}, "teacher probes gain_method"),
    ("teacher", {"probes": {"boot": 300}}, "teacher probes boot: 300, pinned 500"),
    ("teacher", {"probes": {"cap": 60}}, "teacher probes cap"),
    ("teacher", {"probes": {"races": {"T": {}, "Z": {}}}}, "teacher probes races"),
    ("teacher", {"intervene": {"window": [2160.0, 8640.0]}}, "teacher intervene window"),
    ("human", {"manifest": {"games_per_split": None}}, "human dump games_per_split: None, pinned 500"),
    ("human", {"manifest": {"seed": 2}}, "human dump seed"),
    ("human", {"manifest": {"act_every": 1, "act_until": 5760.0}}, "human dump act_until"),
    ("human", {"manifest": {"read_until": None}}, "human dump read_until: None, pinned 8640"),
    ("teacher", {"manifest": {"read_until": 8640.0}}, "teacher dump read_until: 8640.0, pinned None"),  # another row order
    ("human", {"manifest": {"passes": ["real", "early"]}}, "counterfactual passes not pinned"),
    ("human", {"manifest": {"data": [f"<shared>/datasets/{x}" for x in HUMAN_DS[:3]]}}, "not the pinned hold-out"),
    ("human", {"probes": {"races": {"T": {}}}}, "human probes races"),
    ("human", {"probes": {"boot": 500, "cap": 0}}, "human probes boot: 500, pinned 300"),
])
def test_check_diag_refuses_other_settings_050_r9(tmp_path, name, edit, want):
    out = all_outputs(tmp_path / "d")
    set_outputs(out / name, name, **edit)
    problems = protocol.check_diag(LIN, "cue_diag", out, SHA)
    assert any(want in p for p in problems), problems
    with pytest.raises(SystemExit):
        protocol.main(["check-diag", str(out), "--metrics", "cue_diag"])


def test_check_diag_refuses_tasks_and_checkpoint_050_r9(tmp_path):
    out = all_outputs(tmp_path / "d")
    assert any("dump checkpoint_sha256" in p for p in protocol.check_diag(LIN, "cue_diag", out, "a" * 64))
    pr = json.loads((out / "teacher" / "probes.json").read_text(encoding="utf-8"))
    pr["races"]["Z"]["tasks"]["hp|unf_token"]["rows"] = 999
    pr["races"]["Z"]["tasks"]["count|focus_rows"] = {"rows": 1, "metric": "r2", "games": 1, "metrics": {}}
    pr["dump"]["checkpoint_sha256"] = "b" * 64
    (out / "teacher" / "probes.json").write_text(json.dumps(pr), encoding="utf-8")
    hp = json.loads((out / "human" / "probes.json").read_text(encoding="utf-8"))
    hp["races"]["P"]["tasks"]["count|focus_rows"] = {"rows": 9, "skipped": "constant target 1"}  # AMB-050-4: the P count must have values
    hp["races"]["T"]["tasks"]["count|focus_rows"] = {"rows": 9, "skipped": "constant target 1"}  # T: not measurable (2026-11-11), not required
    (out / "human" / "probes.json").write_text(json.dumps(hp), encoding="utf-8")
    problems = protocol.check_diag(LIN, "cue_diag", out)
    assert any("hp|unf_token rows / games" in p for p in problems), problems
    assert any("teacher probes Z count|focus_rows: expected skipped" in p for p in problems), problems
    assert any("teacher probes dump checkpoint_sha256" in p for p in problems), problems
    assert any("human probes P count|focus_rows: no values (constant target 1)" in p for p in problems), problems
    assert not any("human probes T count|focus_rows" in p for p in problems), problems
    (out / "teacher" / "intervene.json").unlink()
    assert any("no " in p and "intervene.json" in p for p in protocol.check_diag(LIN, "cue_diag", out))


@pytest.mark.parametrize("edit, want", [
    (None, "no "),                                                   # no protocol.json (e.g. made by hand / print mode)
    ({"finished": None}, "not finished"),                            # an attempt in progress or failed
    ({"metrics": ["dead_channels"]}, "cue_diag not among the measured metrics"),
    ({"git": {}}, "no git commit"),
    ({"checkpoint_sha256": "c" * 64}, "protocol.json checkpoint_sha256"),
])
def test_check_diag_needs_the_finished_record_050_r9(tmp_path, edit, want):
    """050-R9 record: every result carries its checkpoint sha256 and git commit (the finished protocol.json)."""
    out = all_outputs(tmp_path / "d")
    if edit is None:
        (out / "protocol.json").unlink()
    else:
        record(out, **edit)
    problems = protocol.check_diag(LIN, "cue_diag", out, SHA)
    assert any(want in p for p in problems), problems


def test_check_dead_channels_050_r9(tmp_path):
    out = all_outputs(tmp_path / "d")
    train_log_output(out, from_update=0)
    assert any("from_update 0, pinned 600" in p for p in protocol.check_diag(LIN, "dead_channels", out))
    train_log_output(out, sp_records_from=0)
    assert any("no sp_* record" in p for p in protocol.check_diag(LIN, "dead_channels", out))
    (out / "spatial_log.json").unlink()
    assert any("spatial_log.json" in p for p in protocol.check_diag(LIN, "dead_channels", out))


def test_diag_metric_refusals_050_r9(tmp_path, monkeypatch):
    """Diagnostics only through `diag`; the Checkpoint 1 lineage takes the cue diagnostics, its dead-channel cell is
    not measured; dead_channels needs the training log; s1e2 at the final checkpoint only."""
    monkeypatch.setattr(protocol, "checkpoint_sha256", lambda c: SHA)
    with pytest.raises(SystemExit, match="REFUSED: metric cue_diag is a diagnostic"):
        protocol.main(["bc-eval", "--metric", "cue_diag", "--lineage", "m2", "--point", "s1e1", "--checkpoint", "checkpoints/m2-s1/ckpt-1"])
    with pytest.raises(SystemExit, match="REFUSED: metric dead_channels is not a game metric"):
        protocol.main(job_args(tmp_path, fake_checkpoint(tmp_path), "dead_channels"))
    with pytest.raises(SystemExit, match="REFUSED: metric val500 is not a diagnostic"):
        protocol.main(["diag", "--lineage", "m2", "--point", "s1e1", "--checkpoint", "checkpoints/m2-s1/ckpt-1", "--metrics", "val500"])
    with pytest.raises(SystemExit, match="REFUSED: lineage m2 runs start with"):
        protocol.main(["diag", "--lineage", "m2", "--point", "s1e1", "--checkpoint", "checkpoints/bc-s1-pt/ckpt-39096", "--metrics", "cue_diag"])
    with pytest.raises(SystemExit, match=r"REFUSED: dead_channels of lineage ck1: not measured \(no training-log statistics"):
        protocol.main(["diag", "--lineage", "ck1", "--point", "r1", "--checkpoint", "checkpoints/bc-dagger-r1/ckpt-1137"])
    for pt in LIN["point"]:
        protocol.check_new_measurement(LIN, "ck1", "cue_diag", pt)
    with pytest.raises(SystemExit, match="REFUSED: dead_channels reads the run's training log: --train-log"):
        protocol.main(["diag", "--lineage", "m2", "--point", "s1e1", "--checkpoint", "checkpoints/m2-s1/ckpt-1"])
    monkeypatch.setattr(protocol, "checkpoint_sha256", lambda c: "a" * 64 if "r1" in c else "b" * 64)
    with pytest.raises(SystemExit, match="REFUSED: point r1 of the Checkpoint 1 lineage is checkpoints/bc-dagger-r1/ckpt-1137"):
        protocol.main(["diag", "--lineage", "ck1", "--point", "r1", "--checkpoint", "checkpoints/bc-dagger-r2/ckpt-1218", "--metrics", "cue_diag"])
    with pytest.raises(protocol.Refused, match="not a diagnostic metric"):
        protocol.check_diag(LIN, "val500", tmp_path)


def test_s1e2_diagnostics_final_only_050_r9(tmp_path, monkeypatch):
    """AMB-050-6: the s1e2 point takes the diagnostics at the run's final checkpoint only."""
    monkeypatch.setenv("URANIUM_SHARED_DIR", str(tmp_path / "shared"))
    run = tmp_path / "shared" / "checkpoints" / "m2-s1e2"
    for n in (1000, 2000):
        (run / f"ckpt-{n}").mkdir(parents=True)
    pt = protocol.point(LIN, "s1e2")
    with pytest.raises(protocol.Refused, match="final checkpoint only .*DONE missing"):
        protocol.check_final_only(pt, "checkpoints/m2-s1e2/ckpt-2000")
    (run / "DONE").write_text("", encoding="utf-8")
    with pytest.raises(protocol.Refused, match="last ckpt-2000"):
        protocol.check_final_only(pt, "checkpoints/m2-s1e2/ckpt-1000")
    protocol.check_final_only(pt, "checkpoints/m2-s1e2/ckpt-2000")
    protocol.check_final_only(protocol.point(LIN, "s1e1"), "checkpoints/m2-s1/ckpt-5")  # other points: any checkpoint


def fake_set_data(lin, name, shared=None, work=None, create=True):
    fake_set_data.calls.append((name, create))
    return ["--shards", "W/hold_orig/botgen", "--shards", "W/hold_macro2/botgen"] if name == "teacher" else [x for ds in HUMAN_DS for x in ("--dataset", ds)]


def test_diag_prints_the_commands_050_r9(tmp_path, monkeypatch, capsys):
    """`protocol.py diag` (no --run): the commands for this host, the dump copies, the summaries and the check;
    nothing written (the link trees are made by --run). Checkpoint 1 lineage: cue_diag."""
    monkeypatch.setenv("URANIUM_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setattr(protocol, "checkpoint_sha256", lambda c: SHA)
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    fake_set_data.calls = []
    monkeypatch.setattr(protocol, "set_data", fake_set_data)
    protocol.main(["diag", "--lineage", "ck1", "--point", "r1", "--checkpoint", "checkpoints/bc-dagger-r1/ckpt-1137", "--metrics", "cue_diag"])
    assert fake_set_data.calls == [("teacher", False), ("human", False)]
    lines = capsys.readouterr().out.splitlines()
    steps = [l.split()[4] if l.startswith("python -m") else l.split()[0] for l in lines if not l.startswith("#")]
    assert steps == ["dump", "probes", "intervene", "cp", "summary", "dump", "probes", "cp", "summary", "python"]
    out = tmp_path / "shared" / "artifacts" / "diag" / "bc-dagger-r1__ckpt-1137"
    cmds = [l for l in lines if l.startswith("python -m")]
    assert f"--checkpoint {tmp_path / 'shared' / 'checkpoints/bc-dagger-r1/ckpt-1137'}" in cmds[0] and "--cf-starts 'T:1280,2080;P:1720,3030;Z:1800,3240'" in cmds[0]
    assert "--games 500 --seed 1" in cmds[4] and "--races T,P" in cmds[5]
    assert lines[-3] == f"python tools/eval/protocol.py check-diag {out} --metrics cue_diag --checkpoint checkpoints/bc-dagger-r1/ckpt-1137"
    assert not out.exists()


def test_diag_run_records_retries_and_checks_050_r9(tmp_path, monkeypatch):
    """--run: running.json during the attempt, the dumps in the cache copied to the shared drive (.part, rename),
    protocol.json at the end (sha256, git, backend / precision, `python` and <shared> / <cache> in the commands),
    check-diag; a finished folder is refused, a failed attempt is kept as .failed-<k> and run again."""
    shared, cache = tmp_path / "shared", tmp_path / "cache"
    monkeypatch.setenv("URANIUM_SHARED_DIR", str(shared))
    monkeypatch.setenv("URANIUM_CACHE_DIR", str(cache))
    monkeypatch.setattr(protocol, "checkpoint_sha256", lambda c: SHA)
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    fake_set_data.calls = []
    monkeypatch.setattr(protocol, "set_data", fake_set_data)
    log = tmp_path / "s1.log"
    log.write_text("{}\n", encoding="utf-8")
    ran, fail = [], set()

    def fake_run(cmd, cwd=None, check=None):
        step = protocol.step_name(cmd)
        ran.append(step)
        o = Path(cmd[cmd.index("--out") + 1])
        if step in fail:
            fail.discard(step)
            raise subprocess.CalledProcessError(139, cmd)
        if step == "dump":
            name = o.parent.name
            assert str(o).startswith(str(cache))  # the dump goes to this host's cache first
            set_outputs(tmp_path / "staged" / name, name)
            shutil.copytree(tmp_path / "staged" / name / "dump", o)
        elif step == "probes":
            shutil.copy(tmp_path / "staged" / o.parent.name / "probes.json", o)
        elif step == "intervene":
            shutil.copy(tmp_path / "staged" / o.parent.name / "intervene.json", o)
        elif step == "spatial_logs":
            train_log_output(o.parent)

    monkeypatch.setattr(protocol.subprocess, "run", fake_run)
    base = ["diag", "--lineage", "m2", "--point", "s1e1", "--train-log", str(log), "--run", "--checkpoint"]
    protocol.main(base + ["checkpoints/m2-s1/ckpt-9"])
    assert ran == ["dump", "probes", "intervene", "summary", "dump", "probes", "summary", "spatial_logs"]
    assert fake_set_data.calls == [("teacher", True), ("human", True)]
    out = shared / "artifacts" / "diag" / "m2-s1__ckpt-9"
    assert (out / "teacher" / "dump" / "manifest.json").is_file() and not (out / "teacher" / "dump.part").exists() and not (out / "running.json").exists()
    rec = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    assert (rec["lineage"], rec["point"], rec["checkpoint_sha256"], rec["git"], rec["backend"], rec["precision"]) == ("m2", "s1e1", SHA, CLEAN_GIT, "rocm", "f32")
    assert rec["finished"] and rec["check"] == [] and rec["train_log"] == {"file": "s1.log", "sha256": protocol.file_sha256(log)}
    assert [s["step"] for s in rec["steps"]] == ["dump", "probes", "intervene", "copy", "summary", "dump", "probes", "copy", "summary", "spatial_logs"]
    assert rec["commands"][0][0] == "python" and rec["commands"][0][6] == "<shared>/checkpoints/m2-s1/ckpt-9"
    assert rec["commands"][0][-1] == "<cache>/diag_dump/m2-s1__ckpt-9/teacher/dump" and rec["commands"][3] == ["copy", "<cache>/diag_dump/m2-s1__ckpt-9/teacher/dump", "<shared>/artifacts/diag/m2-s1__ckpt-9/teacher/dump"]
    assert rec["sets"]["human"]["files"] == 2069 and rec["unpinned"] == []
    with pytest.raises(SystemExit, match="REFUSED: .*protocol.json exists"):
        protocol.main(base + ["checkpoints/m2-s1/ckpt-9"])
    # a failed attempt: kept (running.json with the error), then run again
    fail.add("probes")
    with pytest.raises(subprocess.CalledProcessError):
        protocol.main(base + ["checkpoints/m2-s1/ckpt-10"])
    out10 = shared / "artifacts" / "diag" / "m2-s1__ckpt-10"
    run10 = json.loads((out10 / "running.json").read_text(encoding="utf-8"))
    assert run10["finished"] is None and run10["steps"][-1]["step"] == "probes" and "139" in run10["steps"][-1]["error"]
    protocol.main(base + ["checkpoints/m2-s1/ckpt-10"])
    assert (out10.parent / "m2-s1__ckpt-10.failed-1" / "running.json").is_file() and (out10 / "protocol.json").is_file()
    assert (cache / "diag_dump" / "m2-s1__ckpt-10.failed-1" / "teacher" / "dump").is_dir()
    monkeypatch.setattr(protocol, "git_state", lambda: {**CLEAN_GIT, "protocol_files_dirty": True, "dirty_paths": ["tools/eval/protocol.py"]})
    with pytest.raises(SystemExit, match="REFUSED: uncommitted protocol files"):
        protocol.main(base + ["checkpoints/m2-s1/ckpt-11"])


def test_diag_run_split_over_two_hosts_050_r9(tmp_path, monkeypatch):
    """--part gpu (the ROCm host: dumps and their copies, running.json kept) then --part cpu on another host at the
    same commit (dumps fetched into its cache, probes / intervene / summaries / training log, protocol.json, check);
    every step records its host. The cpu part is refused before the gpu part finished, at another commit, with
    other uncommitted protocol files or for another checkpoint; protocol.json lists the gpu part's commands, then
    the cpu part's (with the fetches) once, also after a failed cpu attempt is run again."""
    shared = tmp_path / "shared"
    monkeypatch.setenv("URANIUM_SHARED_DIR", str(shared))
    monkeypatch.setattr(protocol, "checkpoint_sha256", lambda c: SHA)
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    fake_set_data.calls = []
    monkeypatch.setattr(protocol, "set_data", fake_set_data)
    log = tmp_path / "s1.log"
    log.write_text("{}\n", encoding="utf-8")
    ran, fail = [], set()

    def fake_run(cmd, cwd=None, check=None):
        step = protocol.step_name(cmd)
        ran.append((protocol.socket.gethostname(), step))
        o = Path(cmd[cmd.index("--out") + 1])
        if step in fail:
            fail.discard(step)
            raise subprocess.CalledProcessError(1, cmd)
        if step == "dump":
            set_outputs(tmp_path / "staged" / o.parent.name, o.parent.name)
            shutil.copytree(tmp_path / "staged" / o.parent.name / "dump", o)
        elif step in ("probes", "intervene"):
            assert Path(cmd[5]).is_dir() and str(cmd[5]).startswith(str(tmp_path / "cache_cpu"))  # the fetched copy
            shutil.copy(tmp_path / "staged" / o.parent.name / f"{step}.json", o)
        elif step == "spatial_logs":
            train_log_output(o.parent)

    monkeypatch.setattr(protocol.subprocess, "run", fake_run)
    base = ["diag", "--lineage", "m2", "--point", "s1e1", "--run", "--checkpoint", "checkpoints/m2-s1/ckpt-9"]
    out = shared / "artifacts" / "diag" / "m2-s1__ckpt-9"

    def as_host(name):
        monkeypatch.setattr(protocol.socket, "gethostname", lambda: name)
        monkeypatch.setenv("URANIUM_CACHE_DIR", str(tmp_path / f"cache_{name}"))

    as_host("cpu")
    with pytest.raises(SystemExit, match="REFUSED: no .*running.json: run `diag --run --part gpu` first"):
        protocol.main(base + ["--part", "cpu", "--train-log", str(log)])
    as_host("gpu")
    protocol.main(base + ["--part", "gpu"])  # no --train-log needed here
    assert ran == [("gpu", "dump"), ("gpu", "dump")] and fake_set_data.calls[-2:] == [("teacher", True), ("human", True)]
    rec = json.loads((out / "running.json").read_text(encoding="utf-8"))
    assert [(s["step"], s["host"]) for s in rec["steps"]] == [("dump", "gpu"), ("copy", "gpu"), ("dump", "gpu"), ("copy", "gpu")]
    assert rec["parts"]["gpu"]["finished"] and rec["finished"] is None and not (out / "protocol.json").exists()
    assert (out / "human" / "dump" / "manifest.json").is_file()
    with pytest.raises(SystemExit, match="an attempt in progress"):  # the dumps are kept: no new attempt over them
        protocol.main(base + ["--part", "gpu"])
    as_host("cpu")
    monkeypatch.setattr(protocol, "git_state", lambda: {**CLEAN_GIT, "commit": "1" * 40})
    with pytest.raises(SystemExit, match="REFUSED: .*same commit on both parts"):
        protocol.main(base + ["--part", "cpu", "--train-log", str(log)])
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    monkeypatch.setattr(protocol, "checkpoint_sha256", lambda c: "a" * 64)
    with pytest.raises(SystemExit, match="REFUSED: .*dumps of checkpoint sha256"):
        protocol.main(base + ["--part", "cpu", "--train-log", str(log)])
    monkeypatch.setattr(protocol, "checkpoint_sha256", lambda c: SHA)
    dirty = {**CLEAN_GIT, "protocol_files_dirty": True, "dirty_paths": ["tools/eval/protocol.py"], "dirty_diff_sha256": "d" * 64}
    monkeypatch.setattr(protocol, "git_state", lambda: dirty)
    with pytest.raises(SystemExit, match="REFUSED: .*uncommitted protocol files differ between the parts"):
        protocol.main(base + ["--part", "cpu", "--train-log", str(log), "--allow-dirty"])
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    fail.add("summary")  # a failed cpu attempt, then run again on the same host
    with pytest.raises(subprocess.CalledProcessError):
        protocol.main(base + ["--part", "cpu", "--train-log", str(log)])
    ran.clear()
    protocol.main(base + ["--part", "cpu", "--train-log", str(log)])
    assert ran == [("cpu", "probes"), ("cpu", "intervene"), ("cpu", "summary"), ("cpu", "probes"), ("cpu", "summary"), ("cpu", "spatial_logs")]
    assert fake_set_data.calls[-2:] == [("teacher", False), ("human", False)]  # no link trees on the cpu host
    rec = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    assert [(s["step"], s["host"]) for s in rec["steps"]][4:] == [("fetch", "cpu"), ("probes", "cpu"), ("intervene", "cpu"), ("summary", "cpu"),  # failed attempt
                                                                  ("fetch", "cpu"), ("probes", "cpu"), ("intervene", "cpu"), ("summary", "cpu"),
                                                                  ("fetch", "cpu"), ("probes", "cpu"), ("summary", "cpu"), ("spatial_logs", "cpu")]
    assert "error" in rec["steps"][7] and sum(c[4] == "probes" for c in rec["commands"] if len(c) > 4) == 2  # commands not repeated
    assert rec["finished"] and rec["check"] == [] and rec["parts"]["cpu"]["host"] == "cpu" and rec["train_log"]["file"] == "s1.log"
    assert ["fetch", "<shared>/artifacts/diag/m2-s1__ckpt-9/teacher/dump", "<cache>/diag_dump/m2-s1__ckpt-9/teacher/dump"] in rec["commands"]
    assert not (out / "running.json").exists() and (tmp_path / "cache_cpu" / "diag_dump" / "m2-s1__ckpt-9" / "teacher" / "dump" / "manifest.json").is_file()


def test_diag_run_in_progress_needs_takeover_050_r9(tmp_path, monkeypatch):
    """A folder whose running.json has no failed step may be in progress on another host: refused unless --takeover."""
    shared = tmp_path / "shared"
    monkeypatch.setenv("URANIUM_SHARED_DIR", str(shared))
    monkeypatch.setenv("URANIUM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(protocol, "checkpoint_sha256", lambda c: SHA)
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    monkeypatch.setattr(protocol, "set_data", fake_set_data)
    fake_set_data.calls = []
    out = shared / "artifacts" / "diag" / "m2-s1__ckpt-9"
    out.mkdir(parents=True)
    (out / "running.json").write_text(json.dumps({"host": "other-host", "started": "t", "steps": [{"step": "dump", "secs": 5}]}), encoding="utf-8")
    base = ["diag", "--lineage", "m2", "--point", "s1e1", "--checkpoint", "checkpoints/m2-s1/ckpt-9", "--metrics", "cue_diag", "--run"]
    with pytest.raises(SystemExit, match="an attempt in progress \\(host other-host.*--takeover"):
        protocol.main(base)
    ran = []
    monkeypatch.setattr(protocol.subprocess, "run", lambda cmd, cwd=None, check=None: ran.append(cmd) or (_ for _ in ()).throw(subprocess.CalledProcessError(1, cmd)))
    with pytest.raises(subprocess.CalledProcessError):
        protocol.main(base + ["--takeover"])
    assert (out.parent / "m2-s1__ckpt-9.failed-1" / "running.json").is_file() and len(ran) == 1
    # the attempt just failed (an errored step): no --takeover needed
    with pytest.raises(subprocess.CalledProcessError):
        protocol.main(base)
    assert (out.parent / "m2-s1__ckpt-9.failed-2").is_dir()


def test_s1e2_final_only_restricts_cue_diag_only_050_r9(tmp_path, monkeypatch, capsys):
    """AMB-050-6 restricts the cue diagnostics; dead_channels of a non-final s1e2 checkpoint is accepted."""
    shared = tmp_path / "shared"
    monkeypatch.setenv("URANIUM_SHARED_DIR", str(shared))
    monkeypatch.setattr(protocol, "checkpoint_sha256", lambda c: SHA)
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    (shared / "checkpoints" / "m2-s1e2" / "ckpt-1000").mkdir(parents=True)
    (shared / "checkpoints" / "m2-s1e2" / "ckpt-2000").mkdir(parents=True)
    log = tmp_path / "s1e2.log"
    log.write_text("", encoding="utf-8")
    base = ["diag", "--lineage", "m2", "--point", "s1e2", "--checkpoint", "checkpoints/m2-s1e2/ckpt-1000", "--train-log", str(log)]
    with pytest.raises(SystemExit, match="cue diagnostics at the run's final checkpoint only"):
        protocol.main(base)
    protocol.main(base + ["--metrics", "dead_channels"])
    assert "spatial_logs.py" in capsys.readouterr().out


def test_human_data_consistency_across_folders_050_r9(tmp_path):
    """check-diag compares a set's data fingerprint (and the probe tasks' rows / games) with the other finished
    folders (the same data for every checkpoint), besides the pinned values."""
    root = tmp_path / "diag"
    a = all_outputs(root / "m2-s1__ckpt-1")
    b = all_outputs(root / "bc-dagger-r1__ckpt-1137")
    assert protocol.check_diag(LIN, "cue_diag", b) == []
    set_outputs(b / "human", "human", manifest={"games": 999})
    problems = protocol.check_diag(LIN, "cue_diag", b)
    assert any("human data differs from the finished m2-s1__ckpt-1: games 999 vs 1000" in p for p in problems), problems
    assert any("human dump games: 999, pinned 1000" in p for p in problems), problems
    hp = json.loads((a / "human" / "probes.json").read_text(encoding="utf-8"))
    hp["races"]["P"]["tasks"]["count|focus_rows"]["rows"] = 7
    (a / "human" / "probes.json").write_text(json.dumps(hp), encoding="utf-8")
    set_outputs(b / "human", "human")
    assert any("P count|focus_rows [100, 20] vs [7, 20]" in p for p in protocol.check_diag(LIN, "cue_diag", b))
    record(a, finished=None)  # an unfinished folder is not compared
    assert protocol.check_diag(LIN, "cue_diag", b) == []
    shutil.copytree(a, root / "m2-s1__ckpt-1.failed-1")  # nor a failed attempt
    assert protocol.check_diag(LIN, "cue_diag", b) == []


def test_diag_table_050_r9(tmp_path):
    out = all_outputs(tmp_path / "d")
    res = diag_table.folder_table(LIN, out, ["cue_diag", "dead_channels"])
    hp = res["cue"]["teacher"]["Z"]["hp|unf_token"]
    assert (hp["rows"], hp["games"], hp["stages"]["h"]["pt"], hp["h_mlp"]) == (17097, 154, 0.5, 0.45)
    assert res["cue"]["human"]["P"]["count|focus_rows"]["stages"]["h"]["pt"] == 0.5
    assert res["intervention"]["teacher"]["Z"]["effects"]["start"]["bunker_any"]["mean"] == 0.01
    md = "\n".join(diag_table.markdown(LIN, {"m2-s1": res}))
    assert "### cue probe, human hold-out, T `count|focus_rows`" in md and "| m2-s1 | 17097 / 154 | 0.500 [0.400, 0.600]" in md
    # user decision 2026-11-11 (AMB-050-4): the TvT barracks count is labelled, not compared; the P count has values
    assert res["cue"]["human"]["T"]["count|focus_rows"]["not_measurable"].startswith("TvT") and "| m2-s1 | 측정 불가 (not measurable): TvT" in md
    assert "### counterfactual intervention, teacher hold-out, Z" in md and "r4 39.8% (51 @ 3000)" in md
    ck1 = all_outputs(tmp_path / "c")
    (ck1 / "spatial_log.json").unlink()
    r1 = diag_table.folder_table(LIN, record(ck1, metrics=["cue_diag"]), ["cue_diag"]) | {"train_log": None}
    assert "| ck1 | not measured (no training-log statistics) |" in "\n".join(diag_table.markdown(LIN, {"ck1": r1}))
    iv = json.loads((out / "teacher" / "intervene.json").read_text(encoding="utf-8"))
    (out / "teacher" / "intervene.json").write_text(json.dumps({**iv, "window": [0.0, 1.0]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="REFUSED: .*intervene window"):
        diag_table.main([f"x={out}"])


LAYERS = ("stem", "c0", "c1", "c2", "c3", "r0", "r1", "r2", "r3", "r4", "r5")


def bc_log(path: Path, records: list[tuple[int, dict]], warn: list[str] | None = None) -> Path:
    base = {f"sp_dead_{n}": 0 for n in LAYERS} | {f"sp_pos_{n}": 0.3 for n in LAYERS} | {"sp_const_s8": 0, "sp_const_pool": 0}
    lines = [json.dumps({"step": step, "loss": {"action": 1.0, "total": 2.0, **base, **sp}}) for step, sp in records]
    path.write_text("[bc] start\n" + "\n".join(lines + (warn or [])) + "\n", encoding="utf-8")
    return path


def test_spatial_logs_summary_041_r25(tmp_path):
    """dead_channels (AMB-050-5): records from update 600 to the checkpoint's, max dead share, const, the alarm rule."""
    ch = spatial_logs.channels()
    assert (ch["stem"], ch["c2"], ch["r4"], ch["r5"], ch["sp_const_s8"], ch["sp_const_pool"]) == (32, 128, 128, 128, 128, 256)
    assert spatial_logs.default_from_update() == DEAD["from_update"]
    log = bc_log(tmp_path / "s1.log", [(500, {"sp_dead_r4": 100}), (600, {"sp_dead_r4": 51, "sp_pos_c3": 0.1}), (700, {"sp_dead_c3": 64, "sp_const_s8": 3}),
                                       (800, {"sp_dead_r4": 10}), (900, {"sp_dead_stem": 32})],
                 warn=["[bc] WARNING: update 700: spatial encoder collapse alarm (041-R25): sp_dead_c3 = 64 of 128",
                       "[bc] WARNING: update 900: spatial encoder collapse alarm (041-R25): sp_dead_stem = 32 of 32"])
    r = spatial_logs.scan(log, ch, until_update=800)
    assert (r["sp_records"], r["sp_records_from"], r["last_step"], r["until_update"]) == (4, 3, 800, 800)
    assert r["dead"]["r4"]["max"] == 51 and r["dead"]["r4"]["at"] == 600 and r["dead"]["r4"]["last"] == 10  # update 500 is before the alarm window
    assert r["worst_dead_layer"] == "c3" and r["worst_dead_share"] == 0.5 and r["crossing_records"] == [700]
    assert r["const"]["sp_const_s8"]["max"] == 3 and r["min_pos_any"] == 0.1 and len(r["alarm_lines"]) == 1  # the update 900 alarm is after the checkpoint
    assert r["train_tail100_mean"]["action"] == 1.0
    assert spatial_logs.scan(log, ch)["crossing_records"] == [700, 900]
    cols = [f"sp_{k}_{n}" for k in ("pos", "dead") for n in LAYERS] + ["sp_const_s8", "sp_const_pool"]
    rows = ["time_s,agent,update,loss," + ",".join(cols)]
    for agent, up, dead in (("main", 600, "2"), ("exploiter", 600, "99"), ("main", 700, None), ("main", 800, "5")):
        vals = ["" if dead is None else "0.3" if c.startswith("sp_pos") else dead for c in cols]
        rows.append(f"1,{agent},{up},0.5," + ",".join(vals))
    csv_log = tmp_path / "train.csv"
    csv_log.write_text("\n".join(rows) + "\n", encoding="utf-8")
    rc = spatial_logs.scan(csv_log, ch)
    assert rc["sp_records_from"] == 2 and rc["dead"]["stem"]["max"] == 5 and rc["dead_last"]["step"] == 800 and rc["alarm_lines"] == []


# ------------------------------------------------------------------ s1e2 per-checkpoint loop (041-R20 ③)


def s1e2_shared(tmp_path: Path, monkeypatch, steps=(1000, 2000, 3000), done=False) -> Path:
    shared = tmp_path / "shared"
    monkeypatch.setenv("URANIUM_SHARED_DIR", str(shared))
    for run, ss in (("m2-s1e2", steps), ("m2-s1", (39096,))):
        for s in ss:
            d = shared / "checkpoints" / run / f"ckpt-{s}"
            d.mkdir(parents=True)
            (d / "model.safetensors").write_bytes(f"{run}{s}".encode())
    (shared / "checkpoints" / "m2-s1e2" / "ckpt-4000").mkdir()  # no weights yet: not a checkpoint
    if done:
        (shared / "checkpoints" / "m2-s1e2" / "DONE").write_text("", encoding="utf-8")
    protocol.checkpoint_sha256.cache_clear()
    return shared


def test_s1e2_pick_newest_reference_first_041_r20(tmp_path, monkeypatch):
    shared = s1e2_shared(tmp_path, monkeypatch)
    a = argparse.Namespace(run="m2-s1e2", reference="checkpoints/m2-s1/ckpt-39096")
    items = s1e2_loop.targets(a, shared)
    assert items == [("s1e1", "checkpoints/m2-s1/ckpt-39096"), ("s1e2", "checkpoints/m2-s1e2/ckpt-1000"),
                     ("s1e2", "checkpoints/m2-s1e2/ckpt-2000"), ("s1e2", "checkpoints/m2-s1e2/ckpt-3000")]
    taken = set()
    assert s1e2_loop.pick(items, taken.__contains__) == items[0]
    taken.add(items[0][1])
    assert s1e2_loop.pick(items, taken.__contains__) == items[3]  # the newest; 1000 / 2000 are skipped
    taken.add(items[3][1])
    assert s1e2_loop.pick(items, taken.__contains__) is None


def test_s1e2_val_commands_041_r20(tmp_path, monkeypatch, capsys):
    shared = s1e2_shared(tmp_path, monkeypatch)
    s1e2_loop.main(["val", "--run", "m2-s1e2", "--reference", "checkpoints/m2-s1/ckpt-39096", "--dry-run"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    val = shared / "artifacts" / "bc_e2_val" / "val500"
    assert lines[0] == " ".join(["python", "-m", "uranium", "bc-eval", "--dataset", "replays/shards/stardata", "--dataset", "replays/shards/community",
                                 "--dataset", "replays/shards/scr_ladder", "--dataset", "replays/shards/personal",
                                 "--checkpoint", str(shared / "checkpoints/m2-s1/ckpt-39096"), "--split", "val", "--games", "500", "--seed", "1",
                                 "--backend", "rocm", "--config", "config/bc.toml", "--bunker-before", "5760", "--out", str(val / "m2-s1__ckpt-39096.json")])
    assert lines[3].endswith(f"--out {val / 'm2-s1e2__ckpt-3000.json'}")
    with pytest.raises(SystemExit, match="REFUSED: lineage m2 runs start with"):
        s1e2_loop.main(["val", "--run", "m2-s1e2", "--reference", "checkpoints/bc-s1-pt/ckpt-39096", "--dry-run"])


def test_s1e2_val_scores_newest_and_records_041_r20(tmp_path, monkeypatch):
    shared = s1e2_shared(tmp_path, monkeypatch, done=True)
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    calls = []

    def fake_run(cmd, cwd=None, stdout=None, stderr=None, timeout=None):
        out = Path(cmd[cmd.index("--out") + 1])
        calls.append((out.name, timeout))
        out.write_text(json.dumps({"split": "val", "games": 500, "selection_seed": 1, "action_top1": 0.7, "steps": 5,
                                   "nll": {h: 1.0 for h in ("action", "delay", "selected", "target_unit", "target_loc")}}), encoding="utf-8")
        return s1e2_loop.subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(s1e2_loop.subprocess, "run", fake_run)
    s1e2_loop.main(["val", "--run", "m2-s1e2", "--reference", "checkpoints/m2-s1/ckpt-39096", "--watch", "--poll-s", "0"])
    assert calls == [("m2-s1__ckpt-39096.json", 7200), ("m2-s1e2__ckpt-3000.json", 7200)]  # reference, then the final; run done
    val = shared / "artifacts" / "bc_e2_val" / "val500"
    rec = json.loads((val / "m2-s1e2__ckpt-3000.protocol.json").read_text(encoding="utf-8"))
    assert (rec["point"], rec["metric"], rec["backend"], rec["precision"], rec["returncode"]) == ("s1e2", "val500", "rocm", "f32", 0)
    assert rec["checkpoint_sha256"] == protocol.file_sha256(shared / "checkpoints/m2-s1e2/ckpt-3000/model.safetensors")
    assert (rec["attempt"], rec["check"], rec["command"][0]) == (1, "ok", "python")
    assert (val / "m2-s1e2__ckpt-3000.claim").is_file() and not (val / "m2-s1e2__ckpt-2000.claim").exists()


def val_runner(calls: list, rcs: dict, games: int = 500):
    """A fake bc-eval: return codes per output name in order (default 0), a val 500 report on 0."""
    def fake_run(cmd, cwd=None, stdout=None, stderr=None, timeout=None):
        out = Path(cmd[cmd.index("--out") + 1])
        calls.append(out.name)
        rc = rcs.get(out.name, [0]).pop(0) if rcs.get(out.name) else 0
        if rc == "timeout":
            raise subprocess.TimeoutExpired(cmd, timeout)
        if rc == 0:
            out.write_text(json.dumps({"split": "val", "games": games, "selection_seed": 1, "action_top1": 0.7, "steps": 5,
                                       "nll": {h: 1.0 for h in ("action", "delay", "selected", "target_unit", "target_loc")}}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, rc)
    return fake_run


def test_s1e2_val_retries_a_failed_final_041_r20(tmp_path, monkeypatch):
    """041-R20 ③ "최종 체크포인트는 반드시 한다": a crashed / timed-out run is kept as .failed-<k> and run again
    (up to VAL_ATTEMPTS in all)."""
    shared = s1e2_shared(tmp_path, monkeypatch, done=True)
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    calls = []
    monkeypatch.setattr(s1e2_loop.subprocess, "run", val_runner(calls, {"m2-s1e2__ckpt-3000.json": [139, "timeout", 0]}))
    s1e2_loop.main(["val", "--run", "m2-s1e2", "--watch", "--poll-s", "0"])
    assert calls == ["m2-s1e2__ckpt-3000.json"] * 3
    val = shared / "artifacts" / "bc_e2_val" / "val500"
    assert [json.loads((val / f"m2-s1e2__ckpt-3000.failed-{k}.protocol.json").read_text(encoding="utf-8"))["returncode"] for k in (1, 2)] == [139, "timeout"]
    assert (val / "m2-s1e2__ckpt-3000.failed-1.claim").is_file() and (val / "m2-s1e2__ckpt-3000.failed-2.log").is_file()
    rec = json.loads((val / "m2-s1e2__ckpt-3000.protocol.json").read_text(encoding="utf-8"))
    assert (rec["returncode"], rec["attempt"]) == (0, 3) and s1e2_loop.val_state(val / "m2-s1e2__ckpt-3000.json") == "done"


def test_s1e2_val_gives_up_and_exits_nonzero_041_r20(tmp_path, monkeypatch):
    shared = s1e2_shared(tmp_path, monkeypatch, done=True)
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    calls = []
    monkeypatch.setattr(s1e2_loop.subprocess, "run", val_runner(calls, {"m2-s1e2__ckpt-3000.json": [134, 134, 134, 134]}))
    with pytest.raises(SystemExit, match=r"val500 without a return-code-0 record: checkpoints/m2-s1e2/ckpt-3000 \(gave_up\)"):
        s1e2_loop.main(["val", "--run", "m2-s1e2", "--reference", "checkpoints/m2-s1/ckpt-39096", "--watch", "--poll-s", "0"])
    assert calls == ["m2-s1__ckpt-39096.json"] + ["m2-s1e2__ckpt-3000.json"] * s1e2_loop.VAL_ATTEMPTS


def test_s1e2_val_never_overwrites_and_records_the_check_041_r29(tmp_path, monkeypatch):
    """041-R29 ⑨: an existing output is never overwritten (taken, and score_val refuses); a check mismatch is
    recorded and the loop goes on."""
    shared = s1e2_shared(tmp_path, monkeypatch, done=True)
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    val = shared / "artifacts" / "bc_e2_val" / "val500"
    val.mkdir(parents=True)
    (val / "m2-s1__ckpt-39096.json").write_text("{}", encoding="utf-8")  # made by hand
    calls = []
    monkeypatch.setattr(s1e2_loop.subprocess, "run", val_runner(calls, {}, games=400))
    s1e2_loop.main(["val", "--run", "m2-s1e2", "--reference", "checkpoints/m2-s1/ckpt-39096", "--watch", "--poll-s", "0"])
    assert calls == ["m2-s1e2__ckpt-3000.json"]
    rec = json.loads((val / "m2-s1e2__ckpt-3000.protocol.json").read_text(encoding="utf-8"))
    assert rec["returncode"] == 0 and "MISMATCH" in rec["check"]
    with pytest.raises(protocol.Refused, match="never overwritten"):
        s1e2_loop.score_val(LIN, shared, "s1e1", "checkpoints/m2-s1/ckpt-39096", False, 10)


def test_s1e2_games_job_made_elsewhere_041_r20(tmp_path, monkeypatch):
    """A job another host made first (FileExistsError from protocol.py job) is taken, not an error."""
    shared = s1e2_shared(tmp_path, monkeypatch, done=True)

    def raise_exists(args):
        raise FileExistsError(args)

    monkeypatch.setattr(s1e2_loop.protocol, "main", raise_exists)
    a = argparse.Namespace(games_root="eval/x", hosts="h.toml", allow_dirty=False, dry_run=False)
    assert s1e2_loop.make_job(LIN, shared, a, "s1e2", "checkpoints/m2-s1e2/ckpt-1000") is None


def test_s1e2_games_joins_an_existing_job_with_a_worker_041_r20(tmp_path, monkeypatch):
    shared = s1e2_shared(tmp_path, monkeypatch, steps=(1000,), done=True)
    job = shared / "eval" / "m2-s1e2-games" / "m2-s1e2__ckpt-1000"
    job.mkdir(parents=True)  # made by another host, not DONE
    played = []

    def fake_play(j, shared_, worker, poll_s):
        played.append((j.name, worker))
        (j / "DONE").write_text("", encoding="utf-8")

    monkeypatch.setattr(s1e2_loop, "play", fake_play)
    monkeypatch.setattr(s1e2_loop, "make_job", lambda *a: pytest.fail("the job exists"))
    s1e2_loop.main(["games", "--run", "m2-s1e2", "--worker", "--priority 2", "--watch", "--poll-s", "0"])
    assert played == [("m2-s1e2__ckpt-1000", "--priority 2")]


def test_s1e2_val_orphan_claim_on_the_final_041_r20(tmp_path, monkeypatch, capsys):
    """An orphan claim (its scorer crashed / rebooted / was stopped) does not stall the final: a claim older than
    the time limit + 15 min, or this host's claim whose process is gone, is recorded as "stale claim", kept as
    .failed-<k> (one attempt) and the checkpoint is run again."""
    shared = s1e2_shared(tmp_path, monkeypatch, done=True)
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    val = shared / "artifacts" / "bc_e2_val" / "val500"
    val.mkdir(parents=True)
    old = val / "m2-s1e2__ckpt-3000.claim"
    old.write_text("some-host 2026-11-10T01:00:00+09:00\n", encoding="utf-8")  # an older plain-text claim
    t = old.stat().st_mtime - (s1e2_loop.VAL_TIMEOUT_S + s1e2_loop.STALE_MARGIN_S + 60)
    os.utime(old, (t, t))
    out = val / "m2-s1e2__ckpt-3000.json"
    assert s1e2_loop.val_state(out) == "stale"
    calls = []
    monkeypatch.setattr(s1e2_loop.subprocess, "run", val_runner(calls, {}))
    s1e2_loop.main(["val", "--run", "m2-s1e2", "--watch", "--poll-s", "0"])
    assert calls == ["m2-s1e2__ckpt-3000.json"]
    stale = json.loads((val / "m2-s1e2__ckpt-3000.failed-1.protocol.json").read_text(encoding="utf-8"))
    assert stale["returncode"] == "stale claim" and "older than" in stale["reason"] and (val / "m2-s1e2__ckpt-3000.failed-1.claim").is_file()
    assert json.loads((val / "m2-s1e2__ckpt-3000.protocol.json").read_text(encoding="utf-8"))["attempt"] == 2
    # this host's fresh claim of a process that no longer runs
    p = subprocess.Popen([sys.executable, "-c", "pass"])  # subprocess.run is the fake here
    p.wait()
    out2 = val / "m2-s1e2__ckpt-2000.json"
    out2.with_suffix(".claim").write_text(json.dumps({"host": s1e2_loop.socket.gethostname(), "pid": p.pid, "timeout_s": 7200}), encoding="utf-8")
    assert "no longer runs" in s1e2_loop.claim_stale(out2, 7200) and s1e2_loop.val_state(out2) == "stale"
    out2.with_suffix(".claim").write_text(json.dumps({"host": s1e2_loop.socket.gethostname(), "pid": os.getpid() + 0, "timeout_s": 7200}), encoding="utf-8")
    assert s1e2_loop.val_state(out2) == "running"  # a live claim of this process is not stale
    out2.with_suffix(".claim").write_text(json.dumps({"host": "other-host", "pid": 1, "timeout_s": 7200}), encoding="utf-8")
    assert s1e2_loop.val_state(out2) == "running"  # another host's recent claim: wait for it


def test_s1e2_val_waits_for_a_running_final_and_says_where_041_r20(tmp_path, monkeypatch, capsys):
    shared = s1e2_shared(tmp_path, monkeypatch, done=True)
    val = shared / "artifacts" / "bc_e2_val" / "val500"
    val.mkdir(parents=True)
    claim = val / "m2-s1e2__ckpt-3000.claim"
    claim.write_text(json.dumps({"host": "other-host", "pid": 1, "timeout_s": 7200}), encoding="utf-8")
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) == 3:  # the other host finishes
            (val / "m2-s1e2__ckpt-3000.json").write_text("{}", encoding="utf-8")
            (val / "m2-s1e2__ckpt-3000.protocol.json").write_text(json.dumps({"returncode": 0}), encoding="utf-8")

    monkeypatch.setattr(s1e2_loop.time, "sleep", fake_sleep)
    monkeypatch.setattr(s1e2_loop.subprocess, "run", lambda *a, **k: pytest.fail("nothing to run here"))
    s1e2_loop.main(["val", "--run", "m2-s1e2", "--watch", "--poll-s", "5"])
    text = capsys.readouterr().out
    assert text.count("waiting for checkpoints/m2-s1e2/ckpt-3000 running on other-host") == 1 and len(sleeps) == 3


def test_s1e2_val_interrupted_and_leftover_failed_041_r20(tmp_path, monkeypatch):
    """Ctrl-C / an error between the claim and the record: recorded ("error: ...") and set aside, then re-raised.
    A leftover failed record (a scorer stopped before setting it aside) is set aside before picking."""
    shared = s1e2_shared(tmp_path, monkeypatch, done=True)
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))

    def interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(s1e2_loop.subprocess, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        s1e2_loop.main(["val", "--run", "m2-s1e2", "--watch", "--poll-s", "0"])
    val = shared / "artifacts" / "bc_e2_val" / "val500"
    rec = json.loads((val / "m2-s1e2__ckpt-3000.failed-1.protocol.json").read_text(encoding="utf-8"))
    assert rec["returncode"] == "error: KeyboardInterrupt()" and not (val / "m2-s1e2__ckpt-3000.claim").exists()
    (val / "m2-s1e2__ckpt-3000.claim").write_text("{}", encoding="utf-8")
    (val / "m2-s1e2__ckpt-3000.protocol.json").write_text(json.dumps({"returncode": 139}), encoding="utf-8")
    assert s1e2_loop.val_state(val / "m2-s1e2__ckpt-3000.json") == "failed"
    calls = []
    monkeypatch.setattr(s1e2_loop.subprocess, "run", val_runner(calls, {}))
    s1e2_loop.main(["val", "--run", "m2-s1e2", "--watch", "--poll-s", "0"])
    assert calls == ["m2-s1e2__ckpt-3000.json"] and (val / "m2-s1e2__ckpt-3000.failed-2.protocol.json").is_file()
    assert json.loads((val / "m2-s1e2__ckpt-3000.protocol.json").read_text(encoding="utf-8"))["attempt"] == 3


def test_s1e2_games_jobs_and_table_041_r20(tmp_path, monkeypatch, capsys):
    shared = s1e2_shared(tmp_path, monkeypatch, steps=(1000,), done=True)
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    hosts = tmp_path / "hosts.toml"
    hosts.write_text('[[host]]\naddr = "h:7700"\n', encoding="utf-8")
    played = []

    def fake_play(job, shared_, worker, poll_s):
        played.append(job.name)
        rows = [{"game_id": 100000 + n, "slot": 0, "won": n % 2 == 0, "reason": "Elimination", "end_frame": 100,
                 "model": {"checkpoint": f"ckpt-x@{'0' * 12}", "zero_last_failed": True, "sample": "stochastic"}} for n in range(1, 101)]
        (job / "results").mkdir()
        (job / "results" / "h.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (job / "DONE").write_text("", encoding="utf-8")

    monkeypatch.setattr(s1e2_loop, "play", fake_play)
    s1e2_loop.main(["games", "--run", "m2-s1e2", "--reference", "checkpoints/m2-s1/ckpt-39096", "--hosts", str(hosts), "--watch", "--poll-s", "0"])
    assert played == ["m2-s1__ckpt-39096", "m2-s1e2__ckpt-1000"]
    job = shared / "eval" / "m2-s1e2-games" / "m2-s1e2__ckpt-1000"
    rec = json.loads((job / "protocol.json").read_text(encoding="utf-8"))
    assert (rec["point"], rec["plans"][0]["plan"], rec["zero_last_failed"]) == ("s1e2", "t0_100", True)
    assert json.loads((shared / "eval" / "m2-s1e2-games" / "m2-s1__ckpt-39096" / "protocol.json").read_text(encoding="utf-8"))["point"] == "s1e1"
    val = shared / "artifacts" / "bc_e2_val" / "val500"
    val.mkdir(parents=True)
    (val / "m2-s1e2__ckpt-1000.json").write_text(json.dumps({"split": "val", "games": 500, "selection_seed": 1, "action_top1": 0.7242, "steps": 5,
                                                             "win_open_mse": 0.9, "nll": {h: 0.862 for h in ("action", "delay", "selected", "target_unit", "target_loc")}}),
                                                 encoding="utf-8")
    capsys.readouterr()
    s1e2_loop.main(["table", "--run", "m2-s1e2", "--reference", "checkpoints/m2-s1/ckpt-39096", "--json", str(tmp_path / "t.json")])
    text = capsys.readouterr().out
    assert "| reference `checkpoints/m2-s1/ckpt-39096` | not scored |" in text
    assert "| `checkpoints/m2-s1e2/ckpt-1000` | 72.42% | 0.862 |" in text
    assert "| `checkpoints/m2-s1e2/ckpt-1000` (MISMATCH" in text  # the fixture rows ran another checkpoint id than the job's weights
    assert "50 / 50 / 0 | 50%" in text
    rows = json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))
    assert rows[1]["t0_100"]["table"]["total"]["n"] == 100
