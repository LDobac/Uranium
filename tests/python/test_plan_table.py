import json
import math

import numpy as np
import pytest

from uranium.measure import branch_common as bc
from uranium.measure import plan_table as pt

CELLS = pt.CELLS
F_T = 4800
H = 17280
BAN = [{"op": "ban", "action": "Build_Barracks", "from_s": 0, "to_s": 90}]
FORCE = [{"op": "force", "action": "Build_Factory", "from_s": 0, "to_s": 30}]
RESULT = {1.0: "win", 0.5: "draw", 0.0: "loss"}


def pid(c: int, t: str, k: int) -> str:
    return f"trunks/s/g{c}{t}#p{k}"


def row(point: str, alt: str, score: float, rules=(), iv=(), dil=(), repeat: int = 1, valid: bool = True) -> dict:
    return {"point": point, "alt": alt, "rules": list(rules), "repeat": repeat, "valid": valid,
            "invalid": None if valid else {"reason": "digest"}, "intervention": list(iv), "dilution": list(dil),
            "end": {"result": RESULT[score]}, "labels": {"held_8m": int(score > 0.4), "held_s": 1, "alive_10m": 1, "win": None},
            "ops": {"inc_10m": 800.0 + 100 * score}, "score": score, "samples": {}}


def ban_score(c: int, t: str, k: int) -> float:
    if c == 0:
        return 1.0 if t == "a" or k == 0 else 0.5
    return 0.0 if (t, k) == ("b", 0) else 0.5


def dataset(force_score=lambda c, t, k: 0.0):
    points, rows = [], []
    for c, cell in enumerate(CELLS):
        for t in "ab":
            for k in (0, 1):
                p = pid(c, t, k)
                points.append({"point": p, "valid": True, "opponent": cell, "f_t": F_T})
                none_dil = [{"action": "Build_Barracks", "first_frame": F_T + (240 if t == "a" else 2400), "picks_30s": 0},
                            {"action": "Build_Factory", "first_frame": None if t == "a" else F_T + 12, "picks_30s": 0}]
                rows.append(row(p, "none", 0.5, dil=none_dil))
                rows.append(row(p, "ban_x", ban_score(c, t, k), BAN, [{**BAN[0], "applied": 3, "force_frame": None, "missed": False, "rejected": None}],
                                [{"action": "Build_Barracks", "first_frame": F_T + 24 * 95, "picks_30s": 0}]))
                missed = (t, k) == ("b", 1)
                rec = {**FORCE[0], "applied": 0 if missed else 1, "force_frame": None if missed else F_T + 24, "missed": missed,
                       "rejected": None if missed else (t, k) == ("a", 1)}
                rows.append(row(p, "force_y", force_score(c, t, k), FORCE, [rec], [{"action": "Build_Factory", "first_frame": None, "picks_30s": 0}]))
    rows.append(row(pid(0, "a", 0), "ban_x", 0.0, BAN, repeat=2, valid=False))
    points.append({"point": "trunks/s/g9z#p0", "valid": False, "opponent": CELLS[0], "f_t": F_T})
    return points, rows


def write(tmp_path, name, points, rows):
    job = tmp_path / name
    bc.write_jsonl(job / "points.jsonl", points)
    bc.write_jsonl(job / "branches.jsonl", rows)
    return job


def test_score_of_055_r6():
    assert pt.score_of({"score": 0.5}) == 0.5
    assert math.isnan(pt.score_of({"score": None}))
    assert pt.score_of({"end": {"result": "win", "before_horizon": True}, "labels": {"held_8m": 1}}) == 1.0
    assert pt.score_of({"end": {"result": "loss", "before_horizon": True}, "labels": {"held_8m": 0}}) == 0.0
    assert pt.score_of({"end": {"result": "win", "before_horizon": False}, "labels": {"held_8m": 1}}) == 0.5, "decided after 12:00"
    assert pt.score_of({"end": {"result": "draw", "before_horizon": False}, "labels": {"held_8m": 1}}) == 0.5
    assert math.isnan(pt.score_of({"end": {"result": "loss", "before_horizon": True}, "labels": {"held_8m": None, "win": None}})), "Aborted (053-R2 8)"


def test_chi2_sf_and_interaction_055_r7():
    for x, k in ((3.841458820694124, 1), (5.991464547107979, 2), (7.814727903251178, 3), (9.487729036781154, 4), (11.070497693516351, 5)):
        assert pt.chi2_sf(x, k) == pytest.approx(0.05, abs=1e-9)
    assert pt.chi2_sf(0.0, 3) == 1.0
    cells = {"a": {"D": {"pt": 0.2}, "se": 0.1}, "b": {"D": {"pt": 0.0}, "se": 0.1}, "c": {"D": None, "se": None}}
    it = pt.interaction(cells)
    assert it["df"] == 1 and it["pooled"] == pytest.approx(0.1) and it["chi2"] == pytest.approx(2.0)
    assert it["p"] == pytest.approx(math.erfc(1.0))
    assert pt.interaction({"a": cells["a"]})["chi2"] is None
    assert it["cells"] == ["a", "b"], "cells without a spread are left out and named"


def test_plan_effects_strata_and_noise_055_r7(tmp_path):
    job = write(tmp_path, "b3", *dataset())
    rep = pt.analyse(job, None, n=200, seed=7, splits=20, split_seed=3, horizon=H)
    assert (rep["games"], rep["valid"], rep["invalid"], rep["points"], rep["trunks"]) == (49, 48, 1, 16, 8)
    assert rep["invalid_by_alt"] == {"ban_x": {"digest": 1}} and rep["cells"] == dict.fromkeys(CELLS, 4)
    ban = rep["effects"]["score"]["plans"]["ban_x"]
    assert ban["D"]["pt"] == pytest.approx((1.5 - 3 * 0.5) / 16, abs=1e-12) and (ban["points"], ban["trunks"]) == (16, 8)
    assert ban["D"]["lo"] <= ban["D"]["pt"] <= ban["D"]["hi"] and ban["mde"] == pytest.approx(2.80 * ban["se"])
    assert rep["effects"]["score"]["plans"]["force_y"]["D"]["pt"] == pytest.approx(-0.5)
    assert rep["effects"]["inc_10m"]["plans"]["force_y"]["D"]["pt"] == pytest.approx(-50.0)
    assert rep["effects"]["score"]["none"] == 0.5
    st = rep["strata"]["score"]["ban_x"]["cells"]
    assert st["t0-zerg"]["D"]["pt"] == pytest.approx(0.375) and st["t0-protoss"]["D"]["pt"] == pytest.approx(-0.125)
    assert st["t0-zerg"]["points"] == 4 and st["t0-zerg"]["trunks"] == 2
    it = rep["strata"]["score"]["ban_x"]["interaction"]
    w = {c: 1 / st[c]["se"] ** 2 for c in CELLS}
    pooled = sum(w[c] * st[c]["D"]["pt"] for c in CELLS) / sum(w.values())
    assert it["df"] == 3 and it["cells"] == list(CELLS)
    assert it["chi2"] == pytest.approx(sum(w[c] * (st[c]["D"]["pt"] - pooled) ** 2 for c in CELLS)) and it["p"] == pytest.approx(pt.chi2_sf(it["chi2"], 3))
    assert rep["strata"]["score"]["force_y"]["interaction"]["chi2"] is None, "no spread inside the cells"
    assert set(rep["own"]) == {f"own@{s}:{c}" for s in pt.OWN_SAMPLES for c in pt.OWN_COLS}
    again = pt.analyse(job, None, n=200, seed=7, splits=20, split_seed=3, horizon=H)
    assert json.dumps(pt.clean(rep), sort_keys=True) == json.dumps(pt.clean(again), sort_keys=True), "same input, same output"
    md = pt.markdown(rep)
    assert "## 2. 층 표" in md and "ban_x" in md and "t0-zerg-rush" in md


def test_split_half_matches_a_repeatable_table_055_r7(tmp_path):
    pts, rows = dataset(lambda c, t, k: 0.0 if c == 1 else 0.5)
    rows = [r for r in rows if r["alt"] != "ban_x"]
    for r in list(rows):
        if r["alt"] == "none":
            c = int(r["point"].split("/g")[1][0])
            rows.append({**row(r["point"], "ban_x", 1.0 if c == 0 else 0.5, BAN), "dilution": []})
    rep = pt.analyse(write(tmp_path, "b3", pts, rows), None, n=50, seed=7, splits=10, split_seed=1, horizon=H)
    sh = rep["split_half"]["score"]
    assert sh["table"]["r"] == pytest.approx(1.0) and sh["interaction"]["r"] == pytest.approx(1.0)
    assert sh["table"]["spearman_brown"] == pytest.approx(1.0) and sh["splits"] == 10


def test_bite_missed_rejected_055_r7(tmp_path):
    rep = pt.analyse(write(tmp_path, "b3", *dataset()), None, n=100, seed=7, splits=5, split_seed=1, horizon=H)
    b = rep["bite"]["ban_x"]
    assert b["rate"] == pytest.approx(0.5) and b["points"] == 16
    r = b["rules"][0]
    assert r["bite"] == pytest.approx(0.5) and r["first_frame_diff_s"] == pytest.approx((85 - 5) / 2)
    assert r["missed"] is None and r["applied_mean"] == 3
    assert b["bitten"]["score"]["points"] == 8 and b["bitten"]["score"]["D"]["pt"] == pytest.approx((0.5 * 2) / 8)
    f = rep["bite"]["force_y"]
    fr = f["rules"][0]
    assert f["rate"] == pytest.approx(0.5) and fr["missed"] == pytest.approx(0.25) and fr["rejected"] == pytest.approx(4 / 12)
    assert fr["first_frame_diff_s"] == pytest.approx((8 * (F_T + 24 - H) + 4 * 12 + 4 * (H - F_T - 12)) / 24 / 16)
    md = pt.markdown(rep)
    assert fr["forced_at_t"] == 0.0 and math.isnan(fr["pi_at_t"])
    assert "## 5. 물림" in md and "force Build_Factory [0, 30)" in md


def test_drift_against_earlier_none_055_r7(tmp_path):
    pts, rows = dataset()
    job = write(tmp_path, "b3", pts, rows)
    old = [{**r, "score": 0.0, "end": {"result": "loss"}, "repeat": rep} for r in rows if r["alt"] == "none" for rep in (1, 2)]
    old += [{**old[0], "point": "trunks/s/gzz#p9"}]
    old += [{**r, "alt": "top1", "rules": [{"op": "force", "action": a, "from_s": 0, "to_s": 0}]}
            for r, a in zip(old[:2], ("Train_SCV", "Build_Barracks"))]
    cmp_ = write(tmp_path, "b2", pts + [{"point": "trunks/s/gzz#p9", "valid": True, "opponent": CELLS[0], "f_t": F_T}], old)
    rep = pt.analyse(job, cmp_, n=100, seed=7, splits=5, split_seed=1, horizon=H)
    d = rep["drift"]
    assert d["compare"] == "b2" and d["points"] == 16
    assert d["score"]["D"]["pt"] == pytest.approx(0.5) and d["score"]["mean_b3"] == 0.5 and d["score"]["mean_compare"] == 0.0
    assert "## 6. 환경 드리프트" in pt.markdown(rep)


def test_score_at_frame_055_r13(tmp_path):
    row = lambda f, r="win": {"score": 1.0, "end": {"result": r, "end_frame": f, "before_horizon": True}, "labels": {"held_8m": 1}}
    assert pt.score_of(row(17000), 17280) == 1.0
    assert pt.score_of(row(17280), 17280) == 0.5, "not over by 12:00"
    assert pt.score_of(row(9000, "loss"), 17280) == 0.0
    assert math.isnan(pt.score_of({"score": None, "end": {"result": "win", "end_frame": 100}, "labels": {"held_8m": None}}, 17280)), "Aborted stays none"
    assert math.isnan(pt.score_of({"score": 1.0, "end": {"result": "win"}, "labels": {"held_8m": 1}}, 17280)), "no end frame"
    assert pt.score_of(row(17280)) == 1.0
    job = write(tmp_path, "b3h", *dataset())
    pt.main(["--job", str(job), "--boot", "50", "--splits", "5", "--score-at", "17280"])
    out = json.loads((job / "analysis" / "plan-table.json").read_text(encoding="utf-8"))
    assert out["settings"]["score_at"] == 17280 and "17280 프레임 시점" in (job / "analysis" / "plan-table.md").read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="score-at"):
        pt.main(["--job", str(job), "--boot", "50", "--splits", "5", "--score-at", "20000"])


def test_cli_writes_table_055_r7(tmp_path):
    job = write(tmp_path, "b3", *dataset())
    pt.main(["--job", str(job), "--boot", "50", "--splits", "5"])
    out = json.loads((job / "analysis" / "plan-table.json").read_text(encoding="utf-8"))
    assert out["kind"] == "plan-table" and (job / "analysis" / "plan-table.md").is_file()


def test_half_corr_by_hand_055_r7(tmp_path):
    pts, rows = dataset()
    T = pt.Table(pts, rows)
    ds = {a: T.d(a, "score") for a in T.plans}
    in_a = T.trunk == np.array([p.split("#")[0][:-1] + "a" for p in T.points])
    a_tab = np.array([[0.5, 0.0, 0.0, 0.0], [-0.5] * 4])
    b_tab = np.array([[0.25, -0.25, -0.25, -0.25], [-0.5] * 4])
    r_table, r_inter = pt.half_corr(T, ds, in_a)
    assert r_table == pytest.approx(np.corrcoef(a_tab.ravel(), b_tab.ravel())[0, 1])
    ra, rb = a_tab - a_tab.mean(1, keepdims=True), b_tab - b_tab.mean(1, keepdims=True)
    assert r_inter == pytest.approx(np.corrcoef(ra.ravel(), rb.ravel())[0, 1])


def test_repeats_average_and_one_rule_list_055_r7(tmp_path):
    pts, rows = dataset()
    extra = [{**r, "repeat": 2, "score": 1.0} for r in rows if r["alt"] == "force_y" and r["valid"]]
    T = pt.Table(pts, rows + extra)
    assert T.d("force_y", "score") == pytest.approx(np.full(16, 0.0)), "mean of 0 and 1 against none 0.5"
    bad = rows + [{**rows[2], "rules": BAN}]
    with pytest.raises(bc.Refused, match="rules differ"):
        pt.Table(pts, bad)


def test_force_at_t_share_and_policy_055_r7(tmp_path):
    pts, rows = dataset()
    fac = bc.action_ids()["Build_Factory"]
    pts = [{**p, "policy_top": [[0, 0.5], [fac, 0.2]]} for p in pts]
    for r in rows:
        if r["alt"] == "force_y" and not r["intervention"][0]["missed"]:
            r["intervention"][0]["force_frame"] = F_T
    rep = pt.analyse(write(tmp_path, "b3", pts, rows), None, n=50, seed=7, splits=2, split_seed=1, horizon=H)
    fr = rep["bite"]["force_y"]["rules"][0]
    assert fr["forced_at_t"] == 1.0 and fr["pi_at_t"] == pytest.approx(0.2)
    assert rep["none_repeats_max"] == 1
