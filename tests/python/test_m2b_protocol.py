import json
from pathlib import Path

import pytest

from eval_helpers import CLEAN_GIT, CUE, HUMAN_DS, LIN, SHA, fake_game_order, record, set_outputs

from uranium.measure import diag_table, protocol, spatial_logs

M2B = ("s2e2b", "t1b", "t2b")


def patch_outputs(out: Path) -> Path:
    for name in CUE["sets"]:
        d = set_outputs(out / name, name)
        man = json.loads((d / "dump" / "manifest.json").read_text(encoding="utf-8"))
        man["stages"] = [x for x in man["stages"] if x != "sp_pool"]
        (d / "dump" / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
        pr = json.loads((d / "probes.json").read_text(encoding="utf-8"))
        for r in pr["races"].values():
            for t in r["tasks"].values():
                if "metrics" in t:
                    t["metrics"] = {k: v for k, v in t["metrics"].items() if not k.startswith("sp_pool")}
        (d / "probes.json").write_text(json.dumps(pr), encoding="utf-8")
    return record(out, metrics=("cue_diag",), lineage="m2b")


def test_patch_model_diag_outputs_accepted_045_r6(tmp_path, monkeypatch):
    fake_game_order(monkeypatch)
    for name in CUE["sets"]:
        assert "sp_pool" not in CUE[name]["report"]["stages"]
    out = patch_outputs(tmp_path / "d")
    assert protocol.check_diag(LIN, "cue_diag", out) == []
    assert protocol.check_diag(LIN, "dead_channels", out) == []
    md = "\n".join(diag_table.markdown(LIN, {"m2b-t2": diag_table.folder_table(LIN, out, ["cue_diag", "dead_channels"])}))
    assert "| m2b-t2 | 측정 대상 없음 (no targets): no statistics on the patch spatial path (032-R25 scope, 045-R6 5) |" in md
    assert "none from update" not in md and "not measured (no training-log statistics)" not in md
    log = spatial_logs.not_measurable(Path("m2b-teacher-t2.log"), 8, LIN["metric"]["dead_channels"]["from_update"], 3258)
    (out / protocol.TRAIN_LOG).write_text(json.dumps(log), encoding="utf-8")
    assert protocol.check_diag(LIN, "dead_channels", out) == []
    md = "\n".join(diag_table.markdown(LIN, {"m2b-t2": diag_table.folder_table(LIN, out, ["cue_diag", "dead_channels"])}))
    assert "| m2b-t2 | 측정 대상 없음 (no targets): the patch spatial path (spatial_patch = 8) has no dead-channel targets" in md
    for edit, want in (({"spatial_patch": 0}, "not_measurable without a patch spatial path"), ({"from_update": 1}, "from_update 1, pinned")):
        (out / protocol.TRAIN_LOG).write_text(json.dumps({**log, **edit}), encoding="utf-8")
        assert any(want in p for p in protocol.check_diag(LIN, "dead_channels", out))
    (out / protocol.TRAIN_LOG).write_text(json.dumps(log), encoding="utf-8")
    for lineage in ("m2", "ck1"):
        record(out, lineage=lineage)
        assert any("not_measurable" in p and f"lineage {lineage!r}, which measures dead channels" in p for p in protocol.check_diag(LIN, "dead_channels", out))
    (out / protocol.TRAIN_LOG).unlink()
    record(out, metrics=("cue_diag",), lineage="ck1")
    assert any("dead_channels not among the measured metrics" in p for p in protocol.check_diag(LIN, "dead_channels", out))
    res = diag_table.folder_table(LIN, out, ["cue_diag"]) | {"train_log": diag_table._load(out / protocol.TRAIN_LOG)}
    assert "| ck1 | not measured (no training-log statistics) |" in "\n".join(diag_table.markdown(LIN, {"ck1": res}))


def fake_set_data(lin, name, shared=None, work=None, create=True):
    return ["--shards", "W/hold_orig/botgen", "--shards", "W/hold_macro2/botgen"] if name == "teacher" else [x for ds in HUMAN_DS for x in ("--dataset", ds)]


def test_m2b_lineage_points_045_r6():
    assert LIN["lineage"]["m2b"]["run_prefix"] == "m2b-" and LIN["lineage"]["m2b"]["points"] == list(M2B)
    s2e2b, t1b, t2b, t2 = (protocol.point(LIN, p) for p in (*M2B, "t2"))
    assert [p["id"] for p in LIN["point"] if p.get("lineages")] == list(M2B)
    assert set(s2e2b["metrics"]) == {"test", "val500", "val500_race", "val_loc", "cue_diag"}
    for pt in (t1b, t2b):
        assert set(pt["metrics"]) == set(t2["metrics"]) - {"dead_channels"} | {"val_loc"} and pt["t1_platforms"] == t2["t1_platforms"]
        assert set(pt["optional"]) == set(t2["optional"])
    for pt in (s2e2b, t1b, t2b):
        assert pt["family"] == "bc" and pt["lineages"] == ["m2b"] and not pt.get("ck1_checkpoint")
        assert "dead_channels" not in set(pt["metrics"]) | set(pt["optional"])
        for m in pt["metrics"]:
            protocol.check_new_measurement(LIN, "m2b", m, pt)
            if LIN["metric"][m]["kind"] == "games":
                assert protocol.metric_plans(LIN, pt, m)
        with pytest.raises(protocol.Refused, match="dead_channels of lineage m2b: no statistics on the patch spatial path"):
            protocol.check_new_measurement(LIN, "m2b", "dead_channels", pt)
        for lineage in ("m2", "ck1"):
            with pytest.raises(protocol.Refused, match=f"point {pt['id']} is a point of lineage m2b only"):
                protocol.check_new_measurement(LIN, lineage, "val500", pt)
        for ref in pt["compare"]:
            lineage, pid = ref.split(":")
            ref_pt = protocol.point(LIN, pid)
            protocol.check_new_measurement(LIN, lineage, "cue_diag", ref_pt)
            assert not ref_pt.get("lineages")
    assert s2e2b["compare"] == ["m2:s2e2", "ck1:r1"] and t1b["compare"] == ["m2:t1", "m2:D", "ck1:r1"] and t2b["compare"] == ["m2:t2", "ck1:r1"]
    for pid in ("s2e2", "t2", "r1"):
        with pytest.raises(protocol.Refused, match="lineage m2b has the points s2e2b, t1b, t2b only"):
            protocol.check_new_measurement(LIN, "m2b", "val500", protocol.point(LIN, pid))


def test_m2b_run_names_045_r6():
    for pid, ok, bad in (("s2e2b", "m2b-s2e2", "m2b-s2e2e2"), ("t1b", "m2b-teacher-t1", "m2b-teacher-t2"), ("t2b", "m2b-teacher-t2", "m2b-gn-t1e2")):
        pt = protocol.point(LIN, pid)
        protocol.check_checkpoint(LIN, "m2b", pt, f"checkpoints/{ok}/ckpt-3258")
        with pytest.raises(protocol.Refused, match=rf"point {pid} takes runs m2b-<"):
            protocol.check_checkpoint(LIN, "m2b", pt, f"checkpoints/{bad}/ckpt-3258")
        with pytest.raises(protocol.Refused, match="lineage m2b runs start with 'm2b-'"):
            protocol.check_checkpoint(LIN, "m2b", pt, f"checkpoints/{ok.replace('m2b-', 'm2-')}/ckpt-3258")
        with pytest.raises(protocol.Refused, match="a point of lineage m2b only"):
            protocol.check_checkpoint(LIN, "m2", pt, f"checkpoints/{ok}/ckpt-3258")
    with pytest.raises(protocol.Refused, match="lineage m2b has the points"):
        protocol.check_checkpoint(LIN, "m2b", protocol.point(LIN, "t2"), "checkpoints/m2b-teacher-t2/ckpt-3258")


def test_m2b_cue_probes_without_sp_pool_045_r6(tmp_path, monkeypatch, capsys):
    data = ["--shards", "A"]
    for name in CUE["sets"]:
        full = protocol.cue_commands(LIN, name, "CK", tmp_path, data)
        for lineage in (None, "m2", "ck1"):
            assert protocol.cue_commands(LIN, name, "CK", tmp_path, data, lineage=lineage) == full
        b = protocol.cue_commands(LIN, name, "CK", tmp_path, data, lineage="m2b")
        stages = CUE[name]["probes"]["stages"]
        assert "sp_pool" in stages and b[1][b[1].index("--stages") + 1] == ",".join(x for x in stages if x != "sp_pool")
        assert [c for c in b if c[4] != "probes"] == [c for c in full if c[4] != "probes"]
        assert set(CUE[name]["report"]["stages"]) <= set(protocol.probe_stages(LIN, name, "m2b"))
    monkeypatch.setenv("URANIUM_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setattr(protocol, "checkpoint_sha256", lambda c: SHA)
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    monkeypatch.setattr(protocol, "set_data", fake_set_data)
    protocol.main(["diag", "--lineage", "m2b", "--point", "t2b", "--checkpoint", "checkpoints/m2b-teacher-t2/ckpt-3258", "--metrics", "cue_diag"])
    probes = [l for l in capsys.readouterr().out.splitlines() if l.startswith("python -m uranium diag probes")]
    assert len(probes) == 2 and all("--stages token,e_pool,c,x_in,L1.g1,L1.out,h,hc " in l for l in probes)
    with pytest.raises(SystemExit, match="REFUSED: dead_channels of lineage m2b"):
        protocol.main(["diag", "--lineage", "m2b", "--point", "t2b", "--checkpoint", "checkpoints/m2b-teacher-t2/ckpt-3258", "--metrics", "dead_channels",
                       "--train-log", "x.log"])


def test_val_loc_metric_045_r7(tmp_path, capsys, monkeypatch):
    m = LIN["metric"]["val_loc"]
    assert (m["kind"], m["split"], m["expect_games"], m["backend"], m["precision"]) == ("bc_eval", "val", 2395, "cuda", "f32")
    assert m["datasets"] == LIN["metric"]["val500"]["datasets"] and "games" not in m and "seed" not in m
    assert protocol.bc_eval_args(m, "checkpoints/m2b-s2e2/ckpt-1", "o.json") == [
        "python", "-m", "uranium", "bc-eval", *[x for d in m["datasets"] for x in ("--dataset", d)], "--checkpoint", "checkpoints/m2b-s2e2/ckpt-1",
        "--split", "val", "--backend", "cuda", "--config", "config/bc.toml", "--precision", "f32", "--out", "o.json"]
    assert "--precision" not in protocol.bc_eval_args(LIN["metric"]["val500"], "c")
    monkeypatch.setattr(protocol, "checkpoint_sha256", lambda c: SHA)
    monkeypatch.setattr(protocol, "git_state", lambda: dict(CLEAN_GIT))
    protocol.main(["bc-eval", "--metric", "val_loc", "--lineage", "m2b", "--point", "s2e2b", "--checkpoint", "checkpoints/m2b-s2e2/ckpt-1"])
    assert "# pinned: backend cuda (f32)" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="REFUSED: metric val_loc is not in the menu of point t2"):
        protocol.main(["bc-eval", "--metric", "val_loc", "--lineage", "m2", "--point", "t2", "--checkpoint", "checkpoints/m2-teacher-t2/ckpt-3258"])
    grp = {k: {"steps": 10, "nll": 4.0, "within4": 0.5, "exact": 0.1, "err_mean": 9.0} for k in ("all", "build", "move", "rally", "other")}
    good = {"split": "val", "games": 2395, "selection_seed": None, "backend": "cuda", "precision": "f32", "action_top1": 0.72, "target_loc_groups": grp}
    f = tmp_path / "v.json"
    f.write_text(json.dumps(good), encoding="utf-8")
    protocol.main(["check-bc-eval", str(f), "--metric", "val_loc"])
    assert "location NLL 4.0000, within 4 cells 0.5000, build exact 0.1000, rally distance 9.000" in capsys.readouterr().out
    for edit, want in (({"precision": "bf16"}, "'cuda' / 'bf16' != pinned 'cuda' / 'f32'"), ({"backend": "rocm"}, "'rocm' / 'f32' != pinned"), ({"games": 500, "selection_seed": 1}, "!= pinned"),
                       ({"target_loc_groups": None}, "no `target_loc_groups`"),
                       ({"target_loc_groups": {**grp, "rally": {"steps": 0}}}, r"target_loc_groups.rally` has no scored step")):
        f.write_text(json.dumps({**good, **edit}), encoding="utf-8")
        with pytest.raises(SystemExit, match=want):
            protocol.main(["check-bc-eval", str(f), "--metric", "val_loc"])
