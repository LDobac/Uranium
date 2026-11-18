import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from testkit import fake_dump

from uranium.diag.planning import branch as pb
from uranium.diag.planning import labels as pl
from uranium.diag.planning import pred as pp


def _row(gid, side="model", slot=0, won=None, end=14401, scv_8m=None, scv_min_1m=6, reason="Elimination", **kw):
    m = {"scv_8m": scv_8m, "scv_min_1m": scv_min_1m}
    r = {"game_id": gid, "slot": slot, "won": won, "end_frame": end, "reason": reason, "model": None, "scripted": None, "error": None}
    r[side] = m
    return r | kw


def _opp(gid, slot=1):
    return {"game_id": gid, "slot": slot, "won": None, "end_frame": 9000, "reason": "Elimination", "model": None, "scripted": None, "error": None}


def _replay(folder: Path, n: int) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    data = f"replay {folder.as_posix()} {n}".encode()
    (folder / f"g{n}.rep").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _games(path: Path, shas: list, extra: dict | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    games = {str(i): {"sha": s, "start": 0, "result": 0, "rows": 1, "last_frame": 14399.0, "scv8": 0, **(extra or {}).get(i, {})} for i, s in enumerate(shas)}
    (path / "games.json").write_text(json.dumps(games))
    return path


def test_053_R2_bot_labels_join_rules_and_coverage(tmp_path):
    plan = tmp_path / "plan"
    shas = [_replay(plan / "replays", n) for n in range(1, 8)]
    rows = [
        _opp(1), _row(1, won=False, end=13000, scv_8m=9, scv_min_1m=6),
        _row(2, won=True, end=9000, scv_8m=None, scv_min_1m=3), _opp(2),
        _row(3, won=False, end=7000, scv_8m=None),
        _row(4, won=None, end=14401, scv_8m=5, reason="TimeLimitDraw"),
        _row(5, won=True, end=9000, scv_8m=None), _row(5, won=False, end=6000, scv_8m=None),
        _row(6, won=None, end=5000, scv_8m=None, scv_min_1m=None, error="crash"),
        _row(7, won=True, end=12000, scv_8m=3),
    ]
    (plan / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    dump = _games(tmp_path / "dump", shas[:6] + ["f" * 64, shas[6]], {7: {"skipped": True}})
    js = pl.build(dump, "model", [plan / "results.jsonl"])
    g = {int(k): {h: v[h] for h in (*pl.HEADS, "from")} for k, v in js["games"].items()}
    assert g[0] == {"held_8m": 1, "held_s": 1, "alive_10m": 0, "win": 0, "from": "results"}
    assert g[1] == {"held_8m": 1, "held_s": 0, "alive_10m": 1, "win": 1, "from": "results"}
    assert g[2] == {"held_8m": 0, "held_s": 1, "alive_10m": 0, "win": 0, "from": "results"}
    assert g[3] == {"held_8m": 0, "held_s": 1, "alive_10m": 1, "win": None, "from": "results"}
    assert g[4]["held_8m"] == 0 and g[4]["win"] == 0 and g[4]["alive_10m"] == 0
    assert g[5] == {"held_8m": None, "held_s": None, "alive_10m": None, "win": None, "from": "error"}
    assert g[6] == {"held_8m": None, "held_s": None, "alive_10m": None, "win": None, "from": None}
    assert 7 not in g and js["games"]["6"]["sha"] == "f" * 64 and js["games"]["0"]["start"] == 0
    assert js["join"] == {"results": 5, "dump": 0, "unmatched": 1, "skipped_view": 1, "other_view": 0, "ambiguous_view": 0, "error": 1, "aborted": 0}
    st = js["results_stats"]["files"][next(iter(js["results_stats"]["files"]))]
    assert st["bot_rows"] == 8 and st["other_rows"] == 2 and st["games"] == 7 and st["error_rows"] == 1
    cov = js["coverage"]
    assert cov["games"] == 7 and cov["error"] == 1
    assert cov["held_8m"] == {"labeled": 5, "rate": 5 / 7, "positive": 2}
    assert cov["win"]["labeled"] == 4 and cov["alive_10m"]["labeled"] == 5 and cov["held_s"]["labeled"] == 5
    (tmp_path / "labels.json").write_text(json.dumps(js))
    assert pl.read(tmp_path / "labels.json", dump)[1]["win"] == 1
    other = _games(tmp_path / "other", shas[:5] + [shas[6], "f" * 64, shas[5]], {7: {"skipped": True}})
    with pytest.raises(SystemExit, match="another dump"):
        pl.read(tmp_path / "labels.json", other)
    moved = _games(tmp_path / "moved", shas[:6] + ["f" * 64, shas[6]], {7: {"skipped": True}, 2: {"start": 3}})
    with pytest.raises(SystemExit, match="another dump"):
        pl.read(tmp_path / "labels.json", moved)


def _jsonl(path: Path, rows: list) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_053_R2_aborted_games_in_the_ledger_get_no_labels(tmp_path):
    plan = tmp_path / "plan"
    shas = [_replay(plan / "replays", n) for n in range(1, 4)]
    rows = [_row(1, won=True, end=7000, reason="Surrender"), _row(2, won=None, end=14401, scv_8m=9, reason="TimeLimitDraw"), _row(3, won=True, end=7000)]
    _jsonl(plan / "results.jsonl", rows)
    ledger = [{"game_id": 1, "outcome": "Aborted"}, {"game_id": 2, "outcome": "Aborted"}, {"game_id": 3, "outcome": "Finished"}]
    dump = _games(tmp_path / "dump", shas)
    before = pl.build(dump, "model", [plan / "results.jsonl"])
    assert before["games"]["0"]["win"] == 1 and before["games"]["1"]["held_8m"] == 1 and before["join"]["aborted"] == 0
    assert next(iter(before["results_stats"]["files"].values()))["ledger_files"] == 0
    _jsonl(plan / "games.jsonl", ledger)
    js = pl.build(dump, "model", [plan / "results.jsonl"])
    none = {h: None for h in pl.HEADS}
    for i in ("0", "1"):
        assert {h: js["games"][i][h] for h in pl.HEADS} == none and js["games"][i]["from"] == "aborted"
    assert js["games"]["2"]["win"] == 1 and js["games"]["2"]["from"] == "results"
    assert js["join"]["aborted"] == 2 and js["join"]["results"] == 1 and js["join"]["error"] == 0
    assert js["coverage"]["aborted"] == 2 and js["coverage"]["held_8m"]["labeled"] == 1 and js["coverage"]["games"] == 3
    st = next(iter(js["results_stats"]["files"].values()))
    assert st["ledger_files"] == 1 and st["ledger_rows"] == 3 and st["aborted"] == 2
    job = tmp_path / "job"
    (job / "plans").mkdir(parents=True)
    (job / "plans" / "a.toml").write_text('name = "stage"\nseeds = [1]\n')
    (job / "job.toml").write_text('stages = ["plans/a.toml"]\n')
    s5, s6 = _replay(job / "stage" / "replays", 5), _replay(job / "stage" / "replays", 6)
    (job / "results").mkdir()
    (job / "games").mkdir()
    _jsonl(job / "results" / "h.jsonl", [_row(100005, won=True, end=7000), _row(10_100_005, won=False, end=7000), _row(100006, won=True, end=7000, reason="Surrender")])
    _jsonl(job / "games" / "h.jsonl", [{"game_id": 100005, "outcome": "Aborted"}, {"game_id": 10_100_005, "outcome": "Finished"}, {"game_id": 100006, "outcome": "Aborted"}])
    jj = pl.build(_games(tmp_path / "jd", [s5, s6]), "model", [job / "results" / "h.jsonl"])
    assert jj["games"]["0"]["win"] == 0 and jj["games"]["0"]["from"] == "results"
    assert jj["games"]["1"]["win"] is None and jj["games"]["1"]["from"] == "aborted" and jj["join"]["aborted"] == 1


def test_053_R2_two_views_only_the_model_view_is_labeled(tmp_path):
    plan = tmp_path / "plan"
    a, b = _replay(plan / "replays", 1), _replay(plan / "replays", 2)
    (plan / "results.jsonl").write_text("\n".join(json.dumps(r) for r in [_row(1, won=True, end=9000), _row(2, won=False, end=9000)]))
    dump = _games(tmp_path / "dump", [a, a, b], {1: {"start": 2}})
    views = {"bot": [[a, 2]], "model": [[a, 0], [b, 0]]}
    js = pl.build(dump, "model", [plan / "results.jsonl"], views=views)
    assert js["games"]["0"]["win"] == 1 and js["games"]["1"]["win"] is None and js["games"]["1"]["from"] == "other_view"
    assert js["games"]["2"]["win"] == 0 and js["join"]["other_view"] == 1
    blind = pl.build(dump, "model", [plan / "results.jsonl"])
    assert blind["games"]["0"]["from"] == blind["games"]["1"]["from"] == "ambiguous_view" and blind["games"]["0"]["win"] is None
    assert blind["games"]["2"]["win"] == 0 and blind["join"]["ambiguous_view"] == 2


def test_053_R2_retries_take_the_last_attempt_and_scripted_side(tmp_path):
    job = tmp_path / "job"
    (job / "plans").mkdir(parents=True)
    (job / "plans" / "a.toml").write_text('name = "first"\nseeds = [1]\n')
    (job / "plans" / "b.toml").write_text('name = "second"\nseeds = [1]\n')
    (job / "job.toml").write_text('stages = ["plans/a.toml", "plans/b.toml"]\n')
    s1 = _replay(job / "second" / "replays", 3)
    s2 = _replay(job / "first" / "replays", 3)
    (job / "results").mkdir()
    rows = [_row(200003, won=False, end=5000), _row(10_200_003, won=True, end=8000), _opp(10_200_003), _row(100003, won=None, end=14401, scv_8m=8)]
    (job / "results" / "r.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    dump = _games(tmp_path / "dump", [s1, s2])
    js = pl.build(dump, "model", [job / "results" / "r.jsonl"])
    assert js["games"]["0"]["win"] == 1 and js["games"]["0"]["held_8m"] == 1
    assert js["games"]["1"]["held_8m"] == 1 and js["games"]["1"]["win"] is None
    st = next(iter(js["results_stats"]["files"].values()))
    assert st["retries_dropped"] == 1 and st["games"] == 2
    plan = tmp_path / "teach"
    sha = _replay(plan / "replays", 1)
    rows = [_row(1, side="scripted", won=True, end=9500), _row(1, side="model", slot=1, won=False, end=9500)]
    rows[1]["scripted"] = None
    (plan / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    d2 = _games(tmp_path / "d2", [sha])
    assert pl.build(d2, "teacher", [plan / "results.jsonl"])["games"]["0"]["win"] == 1
    assert pl.build(d2, "teacher", [plan / "results.jsonl"], side="model")["games"]["0"]["win"] == 0
    with pytest.raises(SystemExit, match="session results"):
        pl.build(d2, "teacher", [])


def test_053_R2_job_attempts_on_two_host_files_and_broken_lines(tmp_path):
    job = tmp_path / "job"
    (job / "plans").mkdir(parents=True)
    (job / "plans" / "a.toml").write_text('name = "stage"\nseeds = [1]\n')
    (job / "job.toml").write_text('stages = ["plans/a.toml"]\n')
    s3, s4 = _replay(job / "stage" / "replays", 3), _replay(job / "stage" / "replays", 4)
    (job / "results").mkdir()
    (job / "results" / "host1.jsonl").write_text(json.dumps(_row(10_100_003, won=True, end=8000)) + "\n{broken\n" + json.dumps(_row(100004, won=False, end=7000)))
    (job / "results" / "host2.jsonl").write_bytes((json.dumps(_row(100003, won=False, end=5000)) + "\n\x00\x00" + json.dumps(_opp(100003))).encode())
    dump = _games(tmp_path / "dump", [s3, s4])
    js = pl.build(dump, "model", [job / "results" / "host2.jsonl"])
    assert js["games"]["0"]["win"] == 1 and js["games"]["1"]["win"] == 0
    st = next(iter(js["results_stats"]["files"].values()))
    assert st["files"] == 2 and st["bad_lines"] == 1 and st["retries_dropped"] == 1 and st["games"] == 2
    both = pl.build(dump, "model", [job / "results" / "host1.jsonl", job / "results" / "host2.jsonl"])
    assert len(both["results_stats"]["files"]) == 1 and both["games"] == js["games"]


def test_053_R2_teacher_tvt_row_is_the_players0_build(tmp_path):
    plan = tmp_path / "teach"
    sha = _replay(plan / "replays", 1)
    (plan / "plan.toml").write_text('name = "t"\nseeds = [1]\n[[matchup]]\nmap = "fighting_spirit"\ngames = 1\n'
                                    'p0 = { kind = "t0", race = "terran", build = "terran_tp" }\np1 = { kind = "t0", race = "terran" }\n')
    (plan / "games.jsonl").write_text(json.dumps({"game_id": 1, "matchup": 0}) + "\n")
    rows = [_row(1, side="scripted", slot=0, won=False, end=9500, t0_build="standard"), _row(1, side="scripted", slot=1, won=True, end=9500, t0_build="terran_tp")]
    (plan / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    js = pl.build(_games(tmp_path / "d", [sha]), "teacher", [plan / "results.jsonl"])
    assert js["games"]["0"]["win"] == 1
    st = next(iter(js["results_stats"]["files"].values()))
    assert st["p0_build_known"] and st["ambiguous_slot"] == 0 and st["other_rows"] == 1
    (plan / "plan.toml").unlink()
    js = pl.build(_games(tmp_path / "d2", [sha]), "teacher", [plan / "results.jsonl"])
    assert js["games"]["0"]["win"] is None and next(iter(js["results_stats"]["files"].values()))["ambiguous_slot"] == 1


def test_053_R2_human_labels_from_the_dump():
    assert pl.human_heads({"result": 1, "last_frame": 9000.0, "scv8": 0}) == {"held_8m": 1, "held_s": None, "alive_10m": None, "win": 1}
    assert pl.human_heads({"result": -1, "last_frame": 9000.0, "scv8": 0})["held_8m"] == 0
    assert pl.human_heads({"result": 0, "last_frame": 9000.0, "scv8": 0}) == {"held_8m": None, "held_s": None, "alive_10m": None, "win": None}
    assert pl.human_heads({"result": -1, "last_frame": 14000.0, "scv8": 1})["held_8m"] == 1
    assert pl.human_heads({"result": 1, "last_frame": 5000.0, "scv8": None})["held_8m"] is None
    assert pl.held_8m(None, None, 14401) is None and pl.alive_10m(None, 9000) is None and pl.alive_10m(False, 14400) == 1
    assert pl.bot_heads({"won": True, "end_frame": 9000, "error": "crash", "model": {"scv_8m": 9, "scv_min_1m": 9}}, "model") == pl.NONE


def test_053_R2_branch_outcome_recomputes_held_8m():
    from uranium.diag.planning import branch as pbr

    br = {"metrics": {"held_8m": False, "scv_8m": None}, "end": {"result": "win", "end_frame": 9000}}
    assert pbr.outcome(br, "held_8m") == 1.0
    assert pbr.outcome({**br, "end": {"result": "loss", "end_frame": 9000}}, "held_8m") == 0.0
    assert pbr.outcome({"metrics": {"held_8m": True, "scv_8m": 9}, "end": {"result": "draw", "end_frame": 11568}}, "held_8m") == 1.0
    assert np.isnan(pbr.outcome({"metrics": {"held_8m": False, "scv_8m": None}, "end": {"result": "draw", "end_frame": 11568}}, "held_8m"))
    assert pbr.outcome({"metrics": {"held_8m": 1}}, "held_8m") == 1.0




def _labels_for(path: Path, seed: int = 0) -> dict:
    games = json.loads((path / "games.json").read_text())
    rng = np.random.default_rng(seed)
    out = {}
    for k in games:
        v = [None if x < 0.15 else int(x > 0.55) for x in rng.random(4)]
        out[int(k)] = dict(zip(pl.HEADS, v))
    return out


def test_053_R2_q_heads_by_name_in_data_predictor_and_branch(tmp_path):
    p = fake_dump(tmp_path / "a")
    lab = _labels_for(p)
    qn = ("held_8m", "win", "scv8")
    data = pp.Data([pp.Source("a", "teacher", "split", str(p))], ks=(8,), q_names=qn, labels={"a": lab})
    for e, q in zip(data.game_meta, data.g_q):
        want = [lab[e["gid"]]["held_8m"], lab[e["gid"]]["win"], e["scv8"]]
        assert np.array_equal(q, np.array([np.nan if x is None else x for x in want], float), equal_nan=True)
    old = pp.Data([pp.Source("a", "teacher", "split", str(p))], ks=(8,))
    assert old.q_names == pp.Q_NAMES and old.g_q.shape[1] == 2
    unmatched = pp.Data([pp.Source("a", "teacher", "split", str(p))], ks=(8,), q_names=("win",), labels={"a": {}})
    assert np.isnan(unmatched.g_q).all()
    rows = data.pair_rows(8, "train")
    st = data.stats(rows, 8)
    net, _ = pp.train_one(data, "A", 8, rows, st, steps=3, batch=32, lr=1e-3, seed=0)
    assert net.q.out_features == 3 and net.q_names == qn
    pp.save(net, st, tmp_path / "m.pt", {})
    n2, _, _ = pp.load(tmp_path / "m.pt", data)
    assert n2.q_names == qn and all(torch.equal(x, y) for x, y in zip(net.state_dict().values(), n2.state_dict().values()))
    assert torch.load(tmp_path / "m.pt", weights_only=False)["q_names"] == list(qn)
    nets = {8: (n2, st, {})}
    assert pb.q_head_of(nets, None) == "held_8m" and pb.q_head_of(nets, "win") == "win"
    old_net, _ = pp.train_one(old, "A", 8, old.pair_rows(8, "train"), old.stats(old.pair_rows(8, "train"), 8), steps=2, batch=16, lr=1e-3, seed=0)
    assert pb.q_head_of({8: (old_net, st, {})}, None) == "scv8"
    with pytest.raises(SystemExit, match="q-head"):
        pb.q_head_of({8: (old_net, st, {})}, "held_8m")
