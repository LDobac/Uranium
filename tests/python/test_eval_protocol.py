import copy
import hashlib
import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from eval_helpers import (
    CLEAN_GIT,
    CUE,
    DEAD,
    HUMAN,
    HUMAN_DS,
    LIN,
    ROOT,
    SHA,
    TAILS,
    TEACHER,
    all_outputs,
    fake_game_order,
    record,
    set_outputs,
    train_log_output,
)

import eval_common as ec
import protocol
import tables


@pytest.fixture(autouse=True)
def _game_order(monkeypatch):
    fake_game_order(monkeypatch)


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
    "t0_tp_rl": "a176734c88b906a0852d474f1a2ca1fe2c92ccbee078ed4ea93daae68ff4bf98",
}

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

CK1_POINTS = [p for p in LIN["point"] if p.get("ck1_checkpoint")]


def plan_file(name: str) -> dict:
    return tomllib.loads((ROOT / "config" / "plans" / name).read_text(encoding="utf-8"))


def test_conditions_frozen_050_ac5():
    assert {p: s["conditions_sha256"] for p, s in LIN["plan"].items()} == FROZEN
    for pid in LIN["plan"]:
        plan = protocol.check_template(LIN, pid)
        assert plan["name"] == protocol.NAME and protocol.conditions_sha256(plan) == FROZEN[pid]


def test_shared_drive_plans_050_ac5():
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
    src = plan_file(source)
    assert protocol.identify(LIN, src) == pid
    text = protocol.render(LIN, pid, "checkpoints/m2-x/ckpt-1", "m2-x")
    plan = tomllib.loads(text)
    assert plan["name"] == "m2-x" and {s["checkpoint"] for s in protocol.model_sides(plan)} == {"checkpoints/m2-x/ckpt-1"}
    assert protocol.conditions_sha256(plan) == protocol.conditions_sha256(src)


def test_zero_last_failed_pinned_per_plan_050_r9():
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
    protocol.check_checkpoint(LIN, "ck1", pt, "eval/x/copy-of-r1/ckpt")
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
    for pt in CK1_POINTS:
        protocol.check_new_measurement(LIN, "ck1", "cue_diag", pt)
        with pytest.raises(protocol.Refused, match="not measured"):
            protocol.check_new_measurement(LIN, "ck1", "dead_channels", pt)
        protocol.check_new_measurement(LIN, "m2", "dead_channels", pt)
    protocol.check_new_measurement(LIN, "m2", "rush", protocol.point(LIN, "r1"))
    assert not (tmp_path / "j").exists()


def test_lineage_menus_consistent_050_r9():
    for pt in LIN["point"]:
        menu = set(pt["metrics"]) | set(pt.get("optional", []))
        assert menu <= set(LIN["metric"]) and not set(pt["metrics"]) & set(pt.get("optional", [])), pt["id"]
        outs, sup = set(pt.get("ck1_outputs", {})), set(pt.get("ck1_supplement", []))
        assert outs <= menu and sup <= menu, pt["id"]
        assert not outs & sup, pt["id"]
        if pt.get("ck1_checkpoint"):
            assert set(pt["metrics"]) - {"cue_diag", "dead_channels"} <= outs, pt["id"]
        else:
            assert not outs and not sup, pt["id"]
        if "t1" in menu:
            assert pt.get("t1_platforms") and set(pt["t1_platforms"]) <= set(protocol.PLATFORMS), pt["id"]
        else:
            assert not pt.get("t1_platforms"), pt["id"]
    assert protocol.point(LIN, "s2e2")["ck1_supplement"] == ["t0_600", "val500_race"]


def test_ck1_supplement_only_for_missing_cells_050_r9(tmp_path, monkeypatch):
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
    with pytest.raises(SystemExit, match="no new macro measurement at point s2e2; allowed there: cue_diag, ops, rush_combat, t0_600, val500_race"):
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
    plan = tomllib.loads(protocol.render(LIN, "t0_tp", "c", "n"))
    assert plan["seeds"] == [*range(41301, 41317), *range(41351, 41367), *range(41401, 41417), *range(41451, 41467)]
    assert [(m["map"], m["games"], m["p1"]) for m in plan["matchup"]] == [
        ("fighting_spirit", 16, {"kind": "t0", "race": "terran"}), ("circuit_breaker", 16, {"kind": "t0", "race": "terran"}),
        ("fighting_spirit", 16, {"kind": "t0", "race": "protoss"}), ("circuit_breaker", 16, {"kind": "t0", "race": "protoss"})]
    n = protocol.normalize(plan)
    assert n["zero_last_failed"] is True and n["max_frames"] == 86400 and n["wall_timeout_s"] == 7200
    assert {m["p0"]["sample"] for m in n["matchup"]} == {"stochastic"}
    t300 = tables.game_keys(tomllib.loads(protocol.render(LIN, "t0_300", "c", "n")))
    assert set(tables.game_keys(plan)) <= set(t300)


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
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    protocol.checkpoint_sha256.cache_clear()
    ck = fake_checkpoint(tmp_path)
    protocol.main(job_args(tmp_path, ck, "rush,tvz", bot_lines=[f"h:7700 ualbertabot12poollings {accepted('ualbertabot12poollings')}"]))
    rush, tvz = tmp_path / "eval" / "j-rush", tmp_path / "eval" / "j-tvz"
    assert sorted(p.name for p in (tmp_path / "eval").iterdir()) == ["j-rush", "j-tvz"]
    sha = ec.file_sha256(ck / "model.safetensors")
    for d, zlf, pid in ((rush, False, "rush_check"), (tvz, True, "tvz_bc")):
        rec = json.loads((d / "protocol.json").read_text(encoding="utf-8"))
        assert (rec["lineage"], rec["point"], rec["family"], rec["zero_last_failed"]) == ("m2", "s2e2", "bc", zlf)
        assert rec["checkpoint_sha256"] == sha and rec["checkpoint_id"] == f"ckpt-1@{sha[:12]}"
        assert rec["git"] == CLEAN_GIT and rec["hosts"] == [{"addr": "h:7700", "platform": None}] and rec["t1_platforms"] is None
        (p,) = rec["plans"]
        assert p["plan"] == pid and p["conditions_sha256"] == FROZEN[pid]
        assert ec.file_sha256(d / p["file"]) == p["content_sha256"]
        plan = tomllib.loads((d / p["file"]).read_text(encoding="utf-8"))
        assert plan["zero_last_failed"] is zlf and protocol.identify(LIN, plan) == pid
        assert tomllib.loads((d / "job.toml").read_text(encoding="utf-8"))["stages"] == [p["file"]]
        assert tomllib.loads((d / "hosts.toml").read_text(encoding="utf-8"))["host"] == [{"addr": "h:7700", "max_games": 3}]
        assert protocol.verify_job(LIN, d, "m2", protocol.point(LIN, "s2e2")) == []
    assert json.loads((rush / "protocol.json").read_text(encoding="utf-8"))["bot_hashes"] == {"h:7700": {"ualbertabot12poollings": accepted("ualbertabot12poollings")}}
    assert json.loads((tvz / "protocol.json").read_text(encoding="utf-8"))["bot_hashes"] == {}
    f = rush / "plans" / "01-rush_check.toml"
    f.write_text(f.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    rec = json.loads((tvz / "protocol.json").read_text(encoding="utf-8"))
    rec["git"] = {**CLEAN_GIT, "protocol_files_dirty": True, "dirty_paths": ["tools/eval/tables.py"], "dirty_diff_sha256": "c" * 64}
    (tvz / "protocol.json").write_text(json.dumps(rec), encoding="utf-8")
    assert any("content_sha256" in p for p in protocol.verify_job(LIN, rush, "m2", protocol.point(LIN, "s2e2")))
    assert any("uncommitted protocol files" in p for p in protocol.verify_job(LIN, tvz, "m2", protocol.point(LIN, "s2e2")))


def test_job_ac1_alone_gets_distinct_names_050_r9(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    protocol.checkpoint_sha256.cache_clear()
    ck = fake_checkpoint(tmp_path)
    protocol.main(job_args(tmp_path, ck, "ac1"))
    assert sorted(p.name for p in (tmp_path / "eval").iterdir()) == ["j-ac1-lf0", "j-ac1-real"]
    for name, zlf, plans in (("j-ac1-lf0", True, ["ac1_stoch_lf0"]), ("j-ac1-real", False, ["ac1_stoch_real", "ac1_argmax_real"])):
        rec = json.loads((tmp_path / "eval" / name / "protocol.json").read_text(encoding="utf-8"))
        assert rec["zero_last_failed"] is zlf and [p["plan"] for p in rec["plans"]] == plans
    g = {True: [("t0_300", "t0_300", ""), ("ac1", "ac1_stoch_lf0", "")], False: [("ac1", "ac1_stoch_real", "")]}
    assert protocol.job_names("j", g) == {True: "j-t0_300-ac1", False: "j-ac1"}
    assert protocol.job_names("j", {True: g[True]}) == {True: "j"}


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


def test_distill_fix_points_menus_050_r9():
    d1, d2, D, rl = (protocol.point(LIN, p) for p in ("d1", "d2", "D", "rl-fix"))
    r2, r3 = protocol.point(LIN, "r2"), protocol.point(LIN, "r3")
    assert {k: r3.get(k) for k in ("family", "metrics", "optional", "t1_platforms")} == {k: r2.get(k) for k in ("family", "metrics", "optional", "t1_platforms")}
    assert not r3.get("ck1_checkpoint")
    with pytest.raises(protocol.Refused, match="point r3 has no Checkpoint 1 counterpart"):
        protocol.check_new_measurement(LIN, "ck1", "cue_diag", r3)
    for pt in (d1, d2):
        assert pt["family"] == "bc"
        assert {"val500", "val500_race", "per_action", "t0_tp", "tvz", "rush", "macro", "teacher_mode"} <= set(pt["metrics"])
        assert {"cue_diag", "dead_channels"} <= set(pt["metrics"]), "050-R9: the diagnostics at every point"
    assert D["family"] == "bc" and {"test", "val500", "val500_race", "t1", "t0_tp", "teacher_mode"} <= set(D["metrics"])
    assert rl["family"] == "rl" and {"t0_tp", "tvz", "rush", "macro", "teacher_mode"} <= set(rl["metrics"]) and "t1" in rl["optional"]
    assert protocol.metric_plans(LIN, rl, "t0_tp") == ["t0_tp_rl"] and protocol.metric_plans(LIN, rl, "tvz") == ["tvz_rl"]
    assert protocol.metric_plans(LIN, rl, "t1") == ["t1_rl"] and protocol.metric_plans(LIN, d1, "t0_tp") == ["t0_tp"]
    assert LIN["metric"]["teacher_mode"]["kind"] == "games_table"
    for pt in (d1, d2, D, rl):
        assert not pt.get("ck1_checkpoint")
        for m in ("cue_diag", "t0_tp"):
            with pytest.raises(protocol.Refused, match=f"point {pt['id']} has no Checkpoint 1 counterpart"):
                protocol.check_new_measurement(LIN, "ck1", m, pt)
        with pytest.raises(protocol.Refused, match="no Checkpoint 1 counterpart"):
            protocol.check_checkpoint(LIN, "ck1", pt, "checkpoints/bc-dagger-r1/ckpt-1137")
        protocol.check_new_measurement(LIN, "m2", "teacher_mode", pt)
    with pytest.raises(SystemExit, match="REFUSED: point d1 has no Checkpoint 1 counterpart"):
        protocol.main(["tables", "--lineage", "ck1", "--point", "d1"])
    t1 = protocol.point(LIN, "t1")
    assert {k: t1.get(k) for k in ("family", "metrics", "optional")} == {k: d1.get(k) for k in ("family", "metrics", "optional")}
    assert not t1.get("ck1_checkpoint")
    protocol.check_new_measurement(LIN, "m2", "teacher_mode", t1)
    protocol.check_checkpoint(LIN, "m2", t1, "checkpoints/m2-teacher-t1/ckpt-1500")
    with pytest.raises(protocol.Refused, match=r"point t1 takes runs m2-<"):
        protocol.check_checkpoint(LIN, "m2", t1, "checkpoints/m2-distill-d1/ckpt-1500")


def test_distill_points_run_names_050_r9():
    for pid, ok, bad in (("d1", "m2-distill-d1", "m2-distill-d2"), ("d2", "m2-distill-d2", "m2-dagger-r2"), ("D", "m2-distill-d2", "m2-distill-dx"),
                         ("D", "m2-teacher-t1", "m2-teacher-t2"), ("t1", "m2-teacher-t1", "m2-distill-d1")):
        pt = protocol.point(LIN, pid)
        protocol.check_checkpoint(LIN, "m2", pt, f"checkpoints/{ok}/ckpt-1850")
        with pytest.raises(protocol.Refused, match=rf"point {pid} takes runs m2-<"):
            protocol.check_checkpoint(LIN, "m2", pt, f"checkpoints/{bad}/ckpt-1850")
    protocol.check_checkpoint(LIN, "m2", protocol.point(LIN, "rl-fix"), "checkpoints/m2-rl-fix-1/ckpt-1500000/agents/main")
    with pytest.raises(protocol.Refused, match=r"point rl-fix takes runs m2-<rl-fix-\[0-9\]\+>"):
        protocol.check_checkpoint(LIN, "m2", protocol.point(LIN, "rl-fix"), "checkpoints/m2-rl-race-s1/ckpt-1500000/agents/main")
    with pytest.raises(protocol.Refused, match="start with"):
        protocol.check_checkpoint(LIN, "m2", protocol.point(LIN, "d1"), "checkpoints/distill-d1/ckpt-1")


def test_t1_platform_of_new_points_050_r9(tmp_path, monkeypatch):
    r2 = protocol.point(LIN, "r2")
    assert {"t1", "test", "val500_race"} <= set(r2["optional"]) and r2["t1_platforms"] == ["windows"]
    assert protocol.check_t1_platform(r2, [{"addr": "a", "platform": "windows"}]) == ["windows"]
    hosts = [{"addr": "a", "platform": "windows"}, {"addr": "b", "platform": "linux"}]
    for pid in ("D", "rl-fix"):
        assert protocol.check_t1_platform(protocol.point(LIN, pid), hosts) == ["linux", "windows"]
        with pytest.raises(protocol.Refused, match=r"b \(no platform\)"):
            protocol.check_t1_platform(protocol.point(LIN, pid), [hosts[0], {"addr": "b"}])
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    protocol.checkpoint_sha256.cache_clear()
    ck = fake_checkpoint(tmp_path, "m2-distill-d1")
    bots = [f"h:7700 {b} {accepted(b)}" for b in sorted(tomllib.loads(protocol.BOTS.read_text(encoding="utf-8"))["bot"])]
    protocol.main(job_args(tmp_path, ck, "t1", hosts_text='[[host]]\naddr = "h:7700"\nplatform = "linux"\n', point="D", bot_lines=bots))
    rec = json.loads((tmp_path / "eval" / "j" / "protocol.json").read_text(encoding="utf-8"))
    assert rec["point"] == "D" and rec["t1_platforms"] == ["linux"] and rec["hosts"] == [{"addr": "h:7700", "platform": "linux"}]
    shutil.rmtree(tmp_path / "eval")
    with pytest.raises(SystemExit, match="REFUSED: metric teacher_mode is a table of the point's game jobs"):
        protocol.main(job_args(tmp_path, ck, "teacher_mode", point="d1"))
    assert not (tmp_path / "eval").exists()


def test_rl_fix_diagnostics_final_only_050_r9(tmp_path, monkeypatch):
    monkeypatch.setenv("URANIUM_SHARED_DIR", str(tmp_path / "shared"))
    run = tmp_path / "shared" / "checkpoints" / "m2-rl-fix-1"
    for n in (1500000, 3000000):
        (run / f"ckpt-{n}" / "agents" / "main").mkdir(parents=True)
    pt = protocol.point(LIN, "rl-fix")
    with pytest.raises(protocol.Refused, match="final checkpoint only"):
        protocol.check_final_only(pt, "checkpoints/m2-rl-fix-1/ckpt-3000000/agents/main")
    (run / "DONE").write_text("", encoding="utf-8")
    protocol.check_final_only(pt, "checkpoints/m2-rl-fix-1/ckpt-3000000/agents/main")
    with pytest.raises(protocol.Refused, match=r"diagnostics \(cue_diag, dead_channels\) at the run's final checkpoint only \(user decision 2026-11-14\)"):
        protocol.check_final_only(pt, "checkpoints/m2-rl-fix-1/ckpt-1500000/agents/main")
    assert protocol.final_only_metrics(pt) == ["cue_diag", "dead_channels"]
    assert protocol.final_only_metrics(protocol.point(LIN, "s1e2")) == ["cue_diag"]
    assert protocol.final_only_metrics(protocol.point(LIN, "s1e1")) == []
    with pytest.raises(SystemExit, match=r"REFUSED: point rl-fix: diagnostics \(cue_diag, dead_channels\)"):
        protocol.main(["diag", "--lineage", "m2", "--point", "rl-fix", "--checkpoint", "checkpoints/m2-rl-fix-1/ckpt-1500000/agents/main",
                       "--metrics", "dead_channels", "--train-log", "train.csv"])


def test_compare_marks_backend_and_t1_platform_differences_050_r9(capsys, tmp_path):
    def stage(backend, prec, platforms=None, kind="t1"):
        return {"kind": kind, "inference": {f"{backend}@h": {"games": 300, "precision": prec}},
                "protocol": {"t1_platforms": platforms}, "counts": {"total": {"won": [10, 300]}}}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"x": stage("cuda", "bf16-mixed", ["windows"]), "y": stage("rocm", "f32", kind="winrate")}), encoding="utf-8")
    b.write_text(json.dumps({"x": stage("rocm", "f32", ["linux"]), "y": stage("rocm", "f32", kind="winrate")}), encoding="utf-8")
    protocol.main(["compare", str(a), str(b)])
    out = capsys.readouterr().out
    assert "NOTE x: inference backend differs: cuda (bf16-mixed) vs rocm (f32)" in out
    assert "NOTE x: T1 platform differs: ['windows'] vs ['linux']" in out
    assert "NOTE y" not in out and "x total won: 10/300 vs 10/300" in out
    z = stage("rocm", "f32", kind="winrate")
    assert protocol.backend_notes({**z, "inference": {}}, z) == ["inference backend not recorded on one side: not recorded vs rocm (f32)"]
    assert protocol.backend_notes({**z, "inference": {}}, {**z, "inference": {}}) == []


def test_t0_tp_rl_template_050_r9():
    bc = tomllib.loads(protocol.render(LIN, "t0_tp", "c", "n"))
    rl = tomllib.loads(protocol.render(LIN, "t0_tp_rl", "c", "n"))
    assert rl["zero_last_failed"] is False and bc["zero_last_failed"] is True
    assert {k: v for k, v in protocol.normalize(rl).items() if k != "zero_last_failed"} == {k: v for k, v in protocol.normalize(bc).items() if k != "zero_last_failed"}
    assert tables.game_keys(rl) == tables.game_keys(bc)
    assert protocol.identify(LIN, rl) == "t0_tp_rl" and protocol.identify(LIN, bc) == "t0_tp"
    assert LIN["plan"]["t0_tp_rl"]["families"] == ["rl"]


def test_rl_fix_job_uses_t0_tp_rl_050_r9(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    protocol.checkpoint_sha256.cache_clear()
    ck = fake_checkpoint(tmp_path, "m2-rl-fix-1")
    protocol.main(job_args(tmp_path, ck, "t0_tp,tvz", point="rl-fix"))
    rec = json.loads((tmp_path / "eval" / "j" / "protocol.json").read_text(encoding="utf-8"))
    assert (rec["point"], rec["family"], rec["zero_last_failed"]) == ("rl-fix", "rl", False)
    assert [p["plan"] for p in rec["plans"]] == ["t0_tp_rl", "tvz_rl"]
    assert protocol.verify_job(LIN, tmp_path / "eval" / "j", "m2", protocol.point(LIN, "rl-fix")) == []


def test_ops_tables_on_every_menu_052_r8(tmp_path):
    for m in ("ops", "rush_combat"):
        assert LIN["metric"][m]["kind"] == "games_table" and LIN["metric"][m]["source"] == "replay"
    for pt in LIN["point"]:
        assert {"ops", "rush_combat"} <= set(pt["optional"]) and not {"ops", "rush_combat"} & set(pt["metrics"]), pt["id"]
        protocol.check_metric(LIN, pt, "ops")
    for pt in CK1_POINTS:
        protocol.check_new_measurement(LIN, "ck1", "rush_combat", pt)
    for pid, h in FROZEN.items():
        assert LIN["plan"][pid]["conditions_sha256"] == h
    ck = fake_checkpoint(tmp_path)
    with pytest.raises(SystemExit, match="REFUSED: metric ops is a table of the point's game jobs"):
        protocol.main(job_args(tmp_path, ck, "ops"))
    assert not (tmp_path / "eval").exists()


def test_diag_settings_are_the_tools_050_r9():
    from uranium.diag import dump, probes
    from uranium.model import monitor

    assert {r: tuple(v) for r, v in TEACHER["dump"]["cf_starts"].items()} == dump.CF_STARTS
    assert ",".join(TEACHER["probes"]["stages"]) == ",".join(HUMAN["probes"]["stages"]) == probes.DEFAULT_STAGES
    assert DEAD["layers"] == list(monitor.ALARM_LAYERS) and DEAD["from_update"] == monitor.ALARM_FROM
    assert CUE["sets"] == ["teacher", "human"] and CUE["backend"] == "rocm"
    assert HUMAN["probes"]["races"] == ["T", "P"] and "count|focus_rows" in HUMAN["report"]["required"]
    assert "count|focus_rows" in TEACHER["probes"]["expect_skipped"] and "intervene" in TEACHER and "intervene" not in HUMAN
    assert (HUMAN["games"], HUMAN["seed"], HUMAN["splits"]) == (500, 1, ["val", "test"]) and HUMAN["datasets"] == LIN["metric"]["val500"]["datasets"]
    assert DEAD["kind"] == "diag_train_log" and "ck1" in DEAD["not_measured"] and LIN["lineage"]["ck1"]["new_metrics"] == ["cue_diag", "ops", "rush_combat"]
    assert [p["id"] for p in LIN["point"] if p.get("diag_final_only")] == ["s1e2", "rl-fix"]
    for pt in LIN["point"]:
        assert {"cue_diag", "dead_channels"} <= set(pt["metrics"]), pt["id"]
        for s in CUE["sets"]:
            assert set(CUE[s]["report"]["stages"]) <= set(CUE[s]["probes"]["stages"])


def test_diag_commands_pinned_050_r9(tmp_path):
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
    import hashlib

    import numpy as np

    d = tmp_path / "dump"
    d.mkdir()
    assert protocol.game_rows_sha256(d) is None
    np.savez_compressed(d / "rows_p000.npz", game=np.array([0, 0, 1, 2, 2, 2]), frame=np.zeros(6))
    np.savez_compressed(d / "rows_p001.npz", game=np.array([3, 1]), frame=np.zeros(2))
    assert protocol.game_rows_sha256(d) == hashlib.sha256(b"2,2,3,1").hexdigest()
    np.savez_compressed(d / "rows_p000.npz", game=np.array([1, 1, 0, 2, 2, 2]), frame=np.zeros(6))
    assert protocol.game_rows_sha256(d) == hashlib.sha256(b"1,3,3,1").hexdigest()


def test_teacher_link_trees_order_games_as_the_lambda_runs_050_r9():
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
    assert not work.exists()
    args = protocol.set_data(lin, "teacher", shared, work)
    assert args == ["--shards", str(work / TAILS[0]), "--shards", str(work / TAILS[1])]
    assert sorted(f.name for f in (work / TAILS[0] / "teacher-20261106-a").iterdir()) == ["shard_00000.bin.zst", "shard_00001.bin.zst"]
    (work / TAILS[1] / "teacher-20261107-macro2" / "shard_00009.bin.zst").write_bytes(b"z")
    with pytest.raises(protocol.Refused, match="hold-out hold_macro2 \\(link tree"):
        protocol.set_data(lin, "teacher", shared, work)
    (src / "teacher-20261107-macro2" / "shard_00001.bin.zst").write_bytes(b"y")
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
    shared = tmp_path / "shared"
    fake_datasets(shared)
    lin = copy.deepcopy(LIN)
    n, sha = protocol.manifest_fingerprint(shared, HUMAN["datasets"])
    assert n == 8
    lin["metric"]["cue_diag"]["human"] |= {"files": n, "files_sha256": sha}
    assert protocol.set_data(lin, "human", shared) == [x for ds in HUMAN["datasets"] for x in ("--dataset", ds)]
    fake_datasets(shared, extra="replays/shards/scr_ladder")
    with pytest.raises(protocol.Refused, match="hold-out human \\(dataset manifests\\): 9 files"):
        protocol.set_data(lin, "human", shared)
    with pytest.raises(protocol.Refused, match="2069 files"):
        protocol.set_data(LIN, "human", shared)


def test_check_diag_accepts_the_pinned_outputs_050_r9(tmp_path, capsys):
    out = all_outputs(tmp_path / "d")
    assert protocol.check_diag(LIN, "cue_diag", out, SHA) == []
    assert protocol.check_diag(LIN, "dead_channels", out, SHA) == []
    protocol.main(["check-diag", str(out)])
    text = capsys.readouterr().out
    assert "NOTE" not in text and "conform to the protocol" in text


@pytest.mark.parametrize("name, edit, want", [
    ("teacher", {"manifest": {"passes": ["real"], "cf_starts": None}}, "made without --cf"),
    ("teacher", {"manifest": {"backend": "cuda", "precision": "bf16"}}, "teacher dump backend: 'cuda', pinned 'rocm'"),
    ("teacher", {"manifest": {"splits": ["val"]}}, "teacher dump splits"),
    ("teacher", {"manifest": {"lanes_per_group": 8}}, "teacher dump lanes_per_group"),
    ("teacher", {"manifest": {"games": 202, "stats": {"rows": 82148, "act_rows": 15553}}}, "teacher dump games: 202, pinned 433"),
    ("teacher", {"manifest": {"data": ["/h/some/other"]}}, "not the pinned hold-out"),
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
    ("teacher", {"manifest": {"read_until": 8640.0}}, "teacher dump read_until: 8640.0, pinned None"),
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
    hp["races"]["P"]["tasks"]["count|focus_rows"] = {"rows": 9, "skipped": "constant target 1"}
    hp["races"]["T"]["tasks"]["count|focus_rows"] = {"rows": 9, "skipped": "constant target 1"}
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
    (None, "no "),
    ({"finished": None}, "not finished"),
    ({"metrics": ["dead_channels"]}, "cue_diag not among the measured metrics"),
    ({"git": {}}, "no git commit"),
    ({"checkpoint_sha256": "c" * 64}, "protocol.json checkpoint_sha256"),
])
def test_check_diag_needs_the_finished_record_050_r9(tmp_path, edit, want):
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
    for pt in CK1_POINTS:
        protocol.check_new_measurement(LIN, "ck1", "cue_diag", pt)
    with pytest.raises(SystemExit, match="REFUSED: dead_channels reads the run's training log: --train-log"):
        protocol.main(["diag", "--lineage", "m2", "--point", "s1e1", "--checkpoint", "checkpoints/m2-s1/ckpt-1"])
    monkeypatch.setattr(protocol, "checkpoint_sha256", lambda c: "a" * 64 if "r1" in c else "b" * 64)
    with pytest.raises(SystemExit, match="REFUSED: point r1 of the Checkpoint 1 lineage is checkpoints/bc-dagger-r1/ckpt-1137"):
        protocol.main(["diag", "--lineage", "ck1", "--point", "r1", "--checkpoint", "checkpoints/bc-dagger-r2/ckpt-1218", "--metrics", "cue_diag"])
    with pytest.raises(protocol.Refused, match="not a diagnostic metric"):
        protocol.check_diag(LIN, "val500", tmp_path)


def test_s1e2_diagnostics_final_only_050_r9(tmp_path, monkeypatch):
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
    protocol.check_final_only(protocol.point(LIN, "s1e1"), "checkpoints/m2-s1/ckpt-5")


def fake_set_data(lin, name, shared=None, work=None, create=True):
    fake_set_data.calls.append((name, create))
    return ["--shards", "W/hold_orig/botgen", "--shards", "W/hold_macro2/botgen"] if name == "teacher" else [x for ds in HUMAN_DS for x in ("--dataset", ds)]


def test_diag_prints_the_commands_050_r9(tmp_path, monkeypatch, capsys):
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
            assert str(o).startswith(str(cache))
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
    assert rec["finished"] and rec["check"] == [] and rec["train_log"] == {"file": "s1.log", "sha256": ec.file_sha256(log)}
    assert [s["step"] for s in rec["steps"]] == ["dump", "probes", "intervene", "copy", "summary", "dump", "probes", "copy", "summary", "spatial_logs"]
    assert rec["commands"][0][0] == "python" and rec["commands"][0][6] == "<shared>/checkpoints/m2-s1/ckpt-9"
    assert rec["commands"][0][-1] == "<cache>/diag_dump/m2-s1__ckpt-9/teacher/dump" and rec["commands"][3] == ["copy", "<cache>/diag_dump/m2-s1__ckpt-9/teacher/dump", "<shared>/artifacts/diag/m2-s1__ckpt-9/teacher/dump"]
    assert rec["sets"]["human"]["files"] == 2069 and rec["unpinned"] == []
    with pytest.raises(SystemExit, match="REFUSED: .*protocol.json exists"):
        protocol.main(base + ["checkpoints/m2-s1/ckpt-9"])
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
            assert Path(cmd[5]).is_dir() and str(cmd[5]).startswith(str(tmp_path / "cache_cpu"))
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
    protocol.main(base + ["--part", "gpu"])
    assert ran == [("gpu", "dump"), ("gpu", "dump")] and fake_set_data.calls[-2:] == [("teacher", True), ("human", True)]
    rec = json.loads((out / "running.json").read_text(encoding="utf-8"))
    assert [(s["step"], s["host"]) for s in rec["steps"]] == [("dump", "gpu"), ("copy", "gpu"), ("dump", "gpu"), ("copy", "gpu")]
    assert rec["parts"]["gpu"]["finished"] and rec["finished"] is None and not (out / "protocol.json").exists()
    assert (out / "human" / "dump" / "manifest.json").is_file()
    with pytest.raises(SystemExit, match="an attempt in progress"):
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
    fail.add("summary")
    with pytest.raises(subprocess.CalledProcessError):
        protocol.main(base + ["--part", "cpu", "--train-log", str(log)])
    ran.clear()
    protocol.main(base + ["--part", "cpu", "--train-log", str(log)])
    assert ran == [("cpu", "probes"), ("cpu", "intervene"), ("cpu", "summary"), ("cpu", "probes"), ("cpu", "summary"), ("cpu", "spatial_logs")]
    assert fake_set_data.calls[-2:] == [("teacher", False), ("human", False)]
    rec = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    assert [(s["step"], s["host"]) for s in rec["steps"]][4:] == [("fetch", "cpu"), ("probes", "cpu"), ("intervene", "cpu"), ("summary", "cpu"),
                                                                  ("fetch", "cpu"), ("probes", "cpu"), ("intervene", "cpu"), ("summary", "cpu"),
                                                                  ("fetch", "cpu"), ("probes", "cpu"), ("summary", "cpu"), ("spatial_logs", "cpu")]
    assert "error" in rec["steps"][7] and sum(c[4] == "probes" for c in rec["commands"] if len(c) > 4) == 2
    assert rec["finished"] and rec["check"] == [] and rec["parts"]["cpu"]["host"] == "cpu" and rec["train_log"]["file"] == "s1.log"
    assert ["fetch", "<shared>/artifacts/diag/m2-s1__ckpt-9/teacher/dump", "<cache>/diag_dump/m2-s1__ckpt-9/teacher/dump"] in rec["commands"]
    assert not (out / "running.json").exists() and (tmp_path / "cache_cpu" / "diag_dump" / "m2-s1__ckpt-9" / "teacher" / "dump" / "manifest.json").is_file()


def test_diag_run_in_progress_needs_takeover_050_r9(tmp_path, monkeypatch):
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
    with pytest.raises(subprocess.CalledProcessError):
        protocol.main(base)
    assert (out.parent / "m2-s1__ckpt-9.failed-2").is_dir()


def test_s1e2_final_only_restricts_cue_diag_only_050_r9(tmp_path, monkeypatch, capsys):
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
    record(a, finished=None)
    assert protocol.check_diag(LIN, "cue_diag", b) == []
    shutil.copytree(a, root / "m2-s1__ckpt-1.failed-1")
    assert protocol.check_diag(LIN, "cue_diag", b) == []
