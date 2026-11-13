"""051-R13 `diag plan-branch` on synthetic branch records (CPU): paired effects recover a known effect (all
branches and executed-only), 0/1 counts, split-half, trunk bootstrap determinism, the dump row of a point
(041-R28 ② lag, missing rows counted), M3u against the S0b predictor A's ΔQ̂ and M2c against its ô change."""

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from test_plan_s0 import fake_dump

from uranium.diag import plan_branch as pb
from uranium.diag import plan_pred as pp
from uranium.diag import plan_s0
from uranium.diag import tables as tb
from uranium.diag.plan_dump import own_columns

BUNKER = "Build_Bunker"
ALTS = {
    "none": [],
    "force_bunker": [{"op": "force", "action": BUNKER, "from_s": 0, "to_s": 0}],
    "ban_bunker": [{"op": "ban", "action": BUNKER, "from_s": 0, "to_s": 90}],
    "force_marine": [{"op": "force", "action": "Train_Marine", "from_s": 0, "to_s": 0}],
}
EFFECT = {"none": 0.0, "force_bunker": 0.3, "ban_bunker": -0.2, "force_marine": 0.1}


def scalar_with(counts: dict) -> list:
    """A 030 §4 scalar vector whose own-state columns (`own_columns`) hold `counts` {column: n}."""
    names, types, _ = own_columns()
    sc = np.zeros(3000, np.float32)
    for col, n in counts.items():
        d, _ = tb.sc_own_cols(types[names.index(col)][0])
        sc[d] = tb.encode_count(n)
    return sc.tolist()


def args(d: Path, **kw) -> Namespace:
    base = dict(dir=str(d), points=None, branches=None, out=None, dump=None, s0=None, root=None, outcomes="held_8m,held_s,end.before_horizon",
                own_samples="", own_cols="bunker,marine", m3u_outcome="held_8m", baseline="none", ks="8,32", sample_k8="d8", sample_k32="d32", row_lag=2, boot=200, boot_seed=7)
    return Namespace(**{**base, **kw})


def write_records(d: Path, points: list, branches: list) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "points.jsonl").write_text("".join(json.dumps(p) + "\n" for p in points), encoding="utf-8")
    (d / "branches.jsonl").write_text("".join(json.dumps(b) + "\n" for b in branches), encoding="utf-8")


def synthetic(d: Path, n_trunks=10, per_trunk=2, repeats=4, noise=0.05, seed=0, missed_every=0):
    """Points with a per-point base level and per-point effect variation; every cell's repeat noise sums to 0,
    so the repeat means are exact: d_p(a) = EFFECT[a] + v_p(a). `missed_every`: every n-th force branch did
    not fire (`missed`) and its held_s is the baseline's (no effect)."""
    rng = np.random.default_rng(seed)
    points, branches, truth = [], [], {a: [] for a in ALTS if a != "none"}
    n = 0
    for t in range(n_trunks):
        for j in range(per_trunk):
            pid = f"job/s/g{t}#b1-p{j + 1}"
            points.append({"point": pid, "trunk": {"replay": f"job/s/replays/g{t}.rep"}, "f_t": 3242 + 24 * j, "valid": True})
            base = rng.uniform(0.2, 0.6)
            for a in ALTS:
                v = 0.0 if a == "none" else rng.normal(0, 0.05)
                eps = rng.normal(0, noise, repeats // 2)
                errs = np.stack([eps, -eps], 1).reshape(-1)  # repeats 2i+1 and 2i+2 cancel: exact cell means, noisy halves
                eff_exec = []
                for r in range(repeats):
                    n += 1
                    missed = bool(missed_every) and a.startswith("force") and n % missed_every == 0
                    eff = 0.0 if missed else EFFECT[a] + v
                    # `uranium_env::branch::RuleLog` per rule (051-R7 ⑤)
                    iv = [{**rule, "in_window": 1, "applied": 0 if missed else 1, "force_frame": None if missed else 3242, "missed": missed,
                           "rejected": None if missed else False} for rule in ALTS[a]]
                    if not missed:
                        eff_exec.append((r, eff))
                    branches.append({"point": pid, "alt": a, "rules": ALTS[a], "repeat": r + 1, "valid": True, "intervention": iv,
                                     "metrics": {"held_s": base + eff + errs[r], "held_8m": int(r % 2 == 0 or a == "force_bunker")},
                                     "dilution": [{"action": BUNKER, "first_frame": 3300 + r, "picks_30s": r}],
                                     "end": {"result": "draw", "end_frame": 11568, "reason": "TimeLimitDraw", "before_horizon": False}})
                if a != "none":
                    truth[a].append(EFFECT[a] + v)
    # one invalid branch and one skipped point
    branches.append({"point": points[0]["point"], "alt": "force_bunker", "rules": ALTS["force_bunker"], "repeat": 9, "valid": False,
                     "invalid": {"reason": "digest", "obs": 12, "frame": 97}})
    points.append({"point": "job/s/g99#b1-p1", "trunk": {"replay": "x"}, "f_t": 0, "valid": False, "skip_reason": "state_sha256"})
    write_records(d, points, branches)
    return points, branches, truth


def test_paired_effects_recover_known_effect_and_are_deterministic(tmp_path):
    """051-R13 ①: D(a) is the point mean of the repeat-mean differences (exact here: the repeat noise cancels);
    the trunk bootstrap interval holds it and is the same for the same seed; 0/1 outcomes get counts; the
    invalid branch and the skipped point are counted (⑥)."""
    _, _, truth = synthetic(tmp_path / "b")
    rep = pb.analyse(args(tmp_path / "b"))
    for a, t in truth.items():
        D = rep["paired"]["held_s"]["alts"][a]["all"]
        assert D["points"] == 20 and D["trunks"] == 10
        assert D["D"]["pt"] == pytest.approx(np.mean(t), abs=1e-9)
        assert D["D"]["lo"] <= D["D"]["pt"] <= D["D"]["hi"]
        # every intervention ran: executed-only = all
        assert rep["paired"]["held_s"]["alts"][a]["executed"]["D"] == D["D"]
    held = rep["paired"]["held_8m"]
    assert held["binary"] and held["counts"]["force_bunker"] == {"n": 80, "k": 80, "rate": 1.0}
    assert held["counts"]["none"]["n"] == 80 and held["counts"]["none"]["k"] == 40
    dg = rep["diagnostics"]
    assert dg["invalid"] == 1 and dg["invalid_by_alt"] == {"force_bunker": {"digest": 1}} and dg["skipped_points"] == {"job/s/g99#b1-p1": "state_sha256"}
    assert dg["alts"]["force_bunker"]["missed_rate"] == 0.0 and dg["alts"]["force_bunker"]["dilution"][f"{BUNKER}.picks_30s"]["mean"] == 1.5
    assert rep["paired"]["end.before_horizon"]["counts"]["none"] == {"n": 80, "k": 0, "rate": 0.0}
    assert rep["M3u"] == {"skipped": "no --dump / --s0"}
    # determinism: same seed same numbers; another seed other intervals, same points
    again = pb.analyse(args(tmp_path / "b"))
    assert json.dumps(again, sort_keys=True, default=float) == json.dumps(rep, sort_keys=True, default=float)
    other = pb.analyse(args(tmp_path / "b", boot_seed=8))
    a1, a2 = rep["paired"]["held_s"]["alts"]["force_bunker"]["all"]["D"], other["paired"]["held_s"]["alts"]["force_bunker"]["all"]["D"]
    assert a1["pt"] == a2["pt"] and (a1["lo"], a1["hi"]) != (a2["lo"], a2["hi"])


def test_executed_only_and_split_half(tmp_path):
    """051-R13 ①: the secondary analysis keeps only branches whose force fired (a missed force carries no
    effect here, so executed-only recovers the effect); without repeat noise the split halves agree
    (Spearman-Brown 1), with noise larger than the point spread they agree less."""
    _, _, truth = synthetic(tmp_path / "b", noise=0.0, missed_every=3)
    rep = pb.analyse(args(tmp_path / "b"))
    r = rep["paired"]["held_s"]["alts"]["force_bunker"]
    assert r["executed"]["D"]["pt"] == pytest.approx(np.mean(truth["force_bunker"]), abs=1e-9)
    assert r["all"]["D"]["pt"] < r["executed"]["D"]["pt"]  # missed branches dilute the primary analysis
    dg = rep["diagnostics"]["alts"]["force_bunker"]
    assert dg["missed_rate"] == pytest.approx(1 - dg["executed"] / dg["branches"]) and 0 < dg["missed_rate"] < 1
    # the ban alternative is not affected by missed forces: split halves agree exactly without noise
    sh = rep["paired"]["held_s"]["alts"]["ban_bunker"]["split_half"]
    assert sh["pairs"] == 20 and sh["r"] == pytest.approx(1.0) and sh["spearman_brown"] == pytest.approx(1.0)
    # repeat noise far above the point spread: the halves no longer agree
    synthetic(tmp_path / "n", noise=0.5, seed=1)
    noisy = pb.analyse(args(tmp_path / "n"))
    assert noisy["paired"]["held_s"]["alts"]["ban_bunker"]["split_half"]["spearman_brown"] < 0.9


def test_outcome_fields_and_executed():
    names = own_columns()[0]
    b = {"metrics": {"held_8m": True, "x": 2}, "end": {"frame": 900, "before_horizon": False}, "samples": {"d8": {"frame": 5, "scalar": scalar_with({"bunker": 2, "marine": 3})}, "d32": None}}
    assert pb.outcome(b, "held_8m") == 1.0 and pb.outcome(b, "frame") == 900 and pb.outcome(b, "end.before_horizon") == 0.0
    assert np.isnan(pb.outcome(b, "nothing")) and np.isnan(pb.outcome(b, "own@d32:bunker"))
    assert pb.outcome(b, "own@d8:bunker") == 2 and pb.outcome(b, "own@d8:marine") == 3 and pb.outcome(b, "own@d8:army") == 3
    with pytest.raises(SystemExit, match="own column"):
        pb.outcome(b, "own@d8:nope")
    assert "bunker" in names
    # list-form samples
    assert pb.outcome({"samples": [{"name": "d8", "scalar": scalar_with({"bunker": 1})}]}, "own@d8:bunker") == 1
    f = ALTS["force_bunker"]
    assert pb.executed({"rules": f, "intervention": [{"applied": 1, "frame": 10, "rejected": False}]})
    assert not pb.executed({"rules": f, "intervention": [{"applied": 1, "frame": 10, "rejected": True}]})
    assert not pb.executed({"rules": f, "intervention": [{"applied": 0, "frame": "missed"}]})
    assert not pb.executed({"rules": f, "intervention": [{"missed": True}]})
    assert not pb.executed({"rules": ALTS["ban_bunker"], "intervention": [{"applied": 0}]})
    assert pb.executed({"rules": ALTS["ban_bunker"], "intervention": [{"applied": 3}]}) and pb.executed({"rules": []})
    assert pb.one_decision_force(f) == tb.action_id(BUNKER) and pb.one_decision_force(ALTS["ban_bunker"]) is None
    assert pb.one_decision_force([{"op": "force", "action": 5, "from_s": 0, "to_s": 30}]) is None


def predictors(dump: Path, s0: Path) -> pp.Data:
    """S0b-style predictor A at k = 8 and 32 trained briefly on the fake dump, saved as plan-s0 models."""
    design = pp.Design()
    data = pp.Data([pp.Source("a", "teacher", "train", str(dump))], ks=(8, 32), design=design)
    for k in (8, 32):
        rows = data.pair_rows(k, "train")
        st = data.stats(rows, k)
        net, _ = pp.train_one(data, "A", k, rows, st, steps=30, batch=64, lr=1e-3, seed=3, log_every=10)
        pp.save(net, st, s0 / "models" / f"A_k{k}.pt", {"variant": "A", "k": k, "data": {"checkpoint_sha256": "s", "design": design.as_dict()}})
    (s0 / "s0.json").write_text(json.dumps({"M3b": {"8": {"A": {"both": {"scv8": {"rush_mean": {"pt": 0.02, "lo": 0.0, "hi": 0.04}}}}}}}))
    return data


def test_m3u_and_m2c_against_the_predictor(tmp_path):
    """051-R13 ② ③ ④: the point's dump row is the playback frame f_t - 2 of its trunk's game; M3u's ΔQ̂ is
    `plan_s0.delta_q` on the scv8 head there, so measured effects equal to it give sign agreement 1 and Spearman
    1; M2c's predicted change is ô(a) - Σ π ô in counts, and measured changes of its sign agree 1. A point whose
    frame has no row, and one whose replay is not in the dump, are left out and counted."""
    dump = fake_dump(tmp_path / "dump", n_games=6, rows=60)
    s0 = tmp_path / "s0"
    predictors(dump, s0)
    games = json.loads((dump / "games.json").read_text())
    data = pp.Data([pp.Source("trunk", "model", "eval", str(dump))], ks=(8, 32), keep_bot=True, design=pp.Design())
    nets = {k: pb.load_predictor(s0, k, data) for k in (8, 32)}
    names, _, of_action = own_columns()
    bunker, marine = tb.action_id(BUNKER), tb.action_id("Train_Marine")
    pts, brs = [], []
    # 6 trunks x 2 points at dump rows 10 and 30 (frames 2880 + 24 i -> live f_t = frame + 2)
    rows = {}
    for g in range(6):
        for j, i in enumerate((10, 30)):
            pid = f"job/s/g{g}#p{j}"
            f_t = int(2880 + 24 * i + 2)
            pts.append({"point": pid, "trunk": {"replay": "absent.rep", "replay_sha256": games[str(g)]["sha"]}, "f_t": f_t, "checkpoint_sha256": "s", "valid": True})
            r = int(np.nonzero((data.game == g) & (np.rint(data.frame) == f_t - 2))[0][0])
            rows[pid] = r
    # the fake dump's mask opens Move / SCV / Gather / Bunker only: open Train_Marine too for the test rows
    m = data.mask_rows(np.arange(data.N))
    m[:, marine] = True
    data.mask = np.packbits(m, axis=1)
    want_dq = {}
    for a, act in (("force_bunker", bunker), ("force_marine", marine)):
        idx = np.array([rows[p["point"]] for p in pts])
        want_dq[a] = {8: plan_s0.delta_q(nets[8][0], data, idx, act, nets[8][1])[:, pb.SCV8]}
    pred = {}
    for k in (8, 32):
        net, st, _ = nets[k]
        idx = np.array([rows[p["point"]] for p in pts])
        o_all, o_pi = pb.own_change_pred(net, st, data, idx, k)
        pred[k] = {a: o_all[:, act] - o_pi for a, act in (("force_bunker", bunker), ("force_marine", marine))}
    for pi, p in enumerate(pts):
        for a in ("none", "force_bunker", "force_marine"):
            for r in (1, 2):
                score = 0.0 if a == "none" else float(want_dq[a][8][pi]) * (1 + 0.1 * r)
                samples = {}
                for k, name in ((8, "d8"), (32, "d32")):
                    cnt = {"bunker": 5, "marine": 5}
                    if a != "none":
                        col = names[of_action[bunker if a == "force_bunker" else marine]]
                        s = np.sign(pred[k][a][pi, names.index(col)])  # NaN: a column constant in the predictor's data
                        cnt[col] += int(s) if np.isfinite(s) else 0
                    samples[name] = {"frame": p["f_t"] + 100, "scalar": scalar_with(cnt)}
                brs.append({"point": p["point"], "alt": a, "rules": ALTS[a], "repeat": r, "valid": True,
                            "intervention": [{"applied": 1, "frame": p["f_t"], "rejected": False}] if ALTS[a] else [],
                            "metrics": {"score": score, "held_8m": 1}, "samples": samples})
    # a point between rows (no row at f_t - 2) and one of a replay not in the dump
    pts.append({"point": "job/s/g0#late", "trunk": {"replay_sha256": games["0"]["sha"]}, "f_t": 2880 + 24 * 5 + 14, "valid": True})
    pts.append({"point": "job/s/gx#p0", "trunk": {"replay_sha256": "f" * 64}, "f_t": 2882, "valid": True})
    for p in pts[-2:]:
        for a in ("none", "force_bunker"):
            brs.append({"point": p["point"], "alt": a, "rules": ALTS[a], "repeat": 1, "valid": True, "metrics": {"score": 0.0}})
    write_records(tmp_path / "b", pts, brs)
    # the analysis builds its own Data: patch the mask through the dump file (Train_Marine open)
    mm = np.memmap(dump / "rows" / "mask.bin", dtype=np.uint8, mode="r+", shape=data.mask.shape)
    mm[:] = data.mask
    mm.flush()
    del mm
    rep = pb.analyse(args(tmp_path / "b", dump=str(dump), s0=str(s0), outcomes="score", m3u_outcome="score", own_samples="d8"))
    # 051-R11 ④ own-state outcome: the bunker count at the 8th decision moved by the sign of the k = 8 prediction
    ob = rep["paired"]["own@d8:bunker"]["alts"]["force_bunker"]["all"]
    assert ob["points"] == 12 and ob["D"]["pt"] == pytest.approx(np.sign(pred[8]["force_bunker"][:, names.index("bunker")]).mean())
    miss = rep["rows_missing"]
    assert miss["points"] == 2 and miss["by_reason"] == {"no_row": 1, "no_game": 1}
    assert miss["detail"]["job/s/g0#late"]["nearest_frames"] == 12
    assert rep["checks"]["points_match_dump"] is True
    for k in (8, 32):
        r = rep["M3u"][f"k{k}"]
        assert r["pairs"] == 24
        if k == 8:
            assert r["sign_agree"]["pt"] == 1.0 and r["spearman"]["pt"] == pytest.approx(1.0)
            assert r["alts"]["force_bunker"]["dq_mean"] == pytest.approx(float(want_dq["force_bunker"][8].mean()), rel=1e-6)
            assert r["s0b_m3b"]["rush_mean"]["pt"] == 0.02
        c = rep["M2c"][f"k{k}"]
        assert c["direction"]["pt"] == 1.0
        assert c["alts"]["force_bunker"]["column"] == "bunker" and c["alts"]["force_marine"]["column"] == "marine"
    assert rep["M3u"]["split_half"]["pairs"] == 24
    assert set(rep["bunker"]) == {"force_bunker"}
    assert rep["bunker"]["force_bunker"]["dq_mean"]["8"] == rep["M3u"]["k8"]["alts"]["force_bunker"]["dq_mean"]
    assert rep["bunker"]["force_bunker"]["s0b_m3b_rush"]["8"]["pt"] == 0.02
    md = pb.markdown(rep)
    assert "M3u" in md and "bunker" in md and "S0b M3b" in md


def test_cli_writes_json_and_markdown(tmp_path):
    from uranium.__main__ import main

    synthetic(tmp_path / "b", n_trunks=4)
    assert main(["diag", "plan-branch", "--dir", str(tmp_path / "b"), "--boot", "50"]) == 0
    out = tmp_path / "b" / "analysis"
    rep = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert rep["kind"] == "plan-branch" and rep["settings"]["boot"] == 50
    md = (out / "plan-branch.md").read_text(encoding="utf-8")
    assert "짝 효과" in md and "force_bunker" in md and "M3u" in md
