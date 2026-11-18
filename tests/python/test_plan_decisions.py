import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from uranium import tables as tb
from uranium.diag.planning import decisions as dr
from uranium.diag.planning import pred as pp

from testkit import fake_dump

C_SHA = "abcdef012345" + "0" * 52
ROWS = 30
TRAIN = tb.action_id("Train_SCV")
BUNKER = tb.action_id("Build_Bunker")


def _frames(rows: int) -> np.ndarray:
    return (1 + 4 * np.arange(rows)).astype(np.float32)


def _decision_log(decisions, player=0) -> str:
    lines = [json.dumps({"frame": int(f), "action": int(a), "delay_bucket": 2, "units": []}) for f, a in decisions]
    lines.append(json.dumps({"decisions": len(decisions), "end_frame": 9000, "player": player}))
    return "\n".join(lines) + "\n"


def _act_log(decisions, policy) -> str:
    lines = [json.dumps({"i": i, "frame": int(f), "digest": "0" * 16, "act": {}, "decision": {"action": int(a), "delay_bucket": 1, "units": [], "rejected": False},
                         "state": None, "forced": False}) for i, (f, a) in enumerate(decisions)]
    lines.append(json.dumps({"end": {"obs": len(decisions), "decisions": len(decisions), "end_frame": 9000, "reason": "TimeLimitDraw", "player": 0,
                                     "players_index": 0, "policy": policy}}))
    return "\n".join(lines) + "\n"


def _game_decisions(g: int) -> list:
    acts = [0, TRAIN, 0, TRAIN, BUNKER, TRAIN, 0, TRAIN, TRAIN, BUNKER, 0, TRAIN, TRAIN, 0, TRAIN, TRAIN, BUNKER, TRAIN, TRAIN, 0]
    lives = [1, 3] + [3 + 4 * i for i in range(1, 18) if i != 6] + [3 + 4 * 6 - 2]
    lives = sorted(lives)
    return [(L, acts[(i + g) % len(acts)]) for i, L in enumerate(lives)]


@pytest.fixture
def setup(tmp_path):
    n_games = 6
    dump = fake_dump(tmp_path / "dump", n_games=n_games, rows=ROWS, race="P")
    fr = np.tile(_frames(ROWS), n_games)
    fr.tofile(dump / "rows" / "frame.bin")
    job = tmp_path / "job"
    (job / "replays").mkdir(parents=True)
    (job / "memory").mkdir()
    games = json.loads((dump / "games.json").read_text())
    for g in range(n_games):
        data = f"replay {g}".encode()
        (job / "replays" / f"g{g + 1}.rep").write_bytes(data)
        games[str(g)]["sha"] = hashlib.sha256(data).hexdigest()
    games["5"]["sha"] = games["4"]["sha"]
    (dump / "games.json").write_text(json.dumps(games))
    man = json.loads((dump / "manifest.json").read_text())
    man.update({"checkpoint_sha256": C_SHA, "row_until": 14400})
    (dump / "manifest.json").write_text(json.dumps(man))
    logs = {g: _game_decisions(g) for g in range(n_games)}
    (job / "memory" / "g1.jsonl").write_text(_decision_log(logs[0]))
    (job / "memory" / "g2.jsonl").write_text(_act_log(logs[1], f"model:ckpt-1@{C_SHA[:12]}:stochastic"))
    (job / "memory" / "g3.jsonl").write_text(_decision_log(logs[2])[:-60])
    (job / "memory" / "g5.jsonl").write_text(_decision_log(logs[4]))
    (job / "results.jsonl").write_text("\n".join(json.dumps({"game_id": i, "model": {}, "policy": "model:ckpt-9@123456789abc:stochastic"}) for i in (1, 5)) + "\n")
    return dump, job, logs


def _expected_rows(decisions):
    pb = np.maximum(np.array([L for L, _ in decisions]) - 2, 1)
    have = set((1 + 4 * np.arange(ROWS)).tolist())
    out, seen = [], set()
    for (L, a), p in zip(decisions, pb):
        ok = int(p) in have and int(p) not in seen
        seen.add(int(p))
        out.append((int(p), a, ok))
    return out


def test_decision_rows_join_and_status_053_r5(setup, tmp_path):
    dump, job, logs = setup
    s = dr.build(dump, [job], [], dump / dr.SIDECAR)
    meta = json.loads((dump / dr.SIDECAR / "games.json").read_text())
    st = {int(k): v["status"] for k, v in meta.items()}
    assert st[0] == "ok" and st[1] == "ok" and st[2] == "incomplete" and st[3] == "no_log"
    assert sorted([st[4], st[5]]) == ["ok", "other_view"]
    exp = _expected_rows(logs[0])
    m0 = meta["0"]
    assert m0["matched"] == sum(ok for *_, ok in exp) and m0["unmatched"] == sum(not ok for *_, ok in exp)
    assert m0["collisions"] == 1 and m0["unmatched"] == 2
    assert m0["policy_sha"] == "123456789abc" and meta["1"]["policy_sha"] == C_SHA[:12]
    assert s["played_by_dump_checkpoint"] == 1 and s["status"]["ok"] == 3
    arr, _ = dr.load(dump)
    g0 = arr["game"] == 0
    rows0 = arr["row"][g0]
    assert np.array_equal(arr["action"][g0], [a for _, a in logs[0]])
    assert np.array_equal(rows0 >= 0, [ok for *_, ok in exp])
    fr = np.fromfile(dump / "rows" / "frame.bin", np.float32)
    assert np.array_equal(fr[rows0[rows0 >= 0]], [p for p, _, ok in exp if ok])


def test_decision_records_reject_bad_memory_041_r28_053_r5(setup):
    dump, job, _ = setup
    rec = dump.parent / "branch_build.jsonl"
    sha0 = json.loads((dump / "games.json").read_text())["0"]["sha"]
    rec.write_text(json.dumps({"replay_sha256": sha0, "outcome": "written", "memory": {"status": "unit_mismatch"}}) + "\n")
    dr.build(dump, [job], [rec], dump / dr.SIDECAR)
    meta = json.loads((dump / dr.SIDECAR / "games.json").read_text())
    assert meta["0"]["status"] == "record_unit_mismatch" and meta["1"]["status"] == "ok"


def test_data_decision_rows_examples_and_k_pairs_053_r5(setup):
    dump, job, logs = setup
    dr.build(dump, [job], [], dump / dr.SIDECAR)
    src = pp.parse_src(f"m:model:split:decision={dump}")
    assert src.rows == "decision"
    d = pp.Data([src], (2, 3), eval_frac=0.0)
    meta = json.loads((dump / dr.SIDECAR / "games.json").read_text())
    ok = sorted(int(k) for k, v in meta.items() if v["status"] == "ok")
    assert sorted(e["gid"] for e in d.game_meta) == ok
    assert d.N == sum(meta[str(g)]["matched"] for g in ok)
    assert d.dropped["m"]["decision_status"] == {"incomplete": 1, "no_log": 1, "other_view": 1}
    assert d.valid.all() and (d.action == 0).any(), "NoOp decision rows are examples"
    fr = np.rint(d.frame).astype(int)
    for gi, e in enumerate(d.game_meta):
        exp = _expected_rows(logs[e["gid"]])
        m = d.game == gi
        assert sorted(fr[m].tolist()) == sorted(p for p, _, k in exp if k), "only decision rows; command rows are no examples"
        for k in (1, 2, 3):
            lives = [(p, a, okk) for p, a, okk in exp]
            for j, (p, a, okk) in enumerate(lives):
                if not okk:
                    continue
                here = np.nonzero(m & (fr == p))[0]
                assert len(here) == 1
                cnt, want = 0, None
                for q in range(j + 1, len(lives)):
                    if lives[q][1] != 0:
                        cnt += 1
                        if cnt == k:
                            want = lives[q]
                            break
                got = d.nxt[k][here[0]]
                if want is None or not want[2]:
                    assert got == -1, (e["gid"], k, j)
                else:
                    assert fr[got] == want[0] and d.game[got] == gi and d.action[got] == want[1], (e["gid"], k, j)
    played = [gi for gi, e in enumerate(d.game_meta) if e["gid"] == 1]
    assert d.prop_ok[d.game == played[0]].all() and not d.prop_ok[d.game != played[0]].any()
    idx = np.arange(d.N)
    p = d.propensity(idx)
    assert np.isnan(p[~d.prop_ok]).all()
    assert np.allclose(p[d.prop_ok], d.probs(idx[d.prop_ok])[np.arange(int(d.prop_ok.sum())), d.action[d.prop_ok]])
    info = pp.describe(d)["sources"][0]["decision_rows"]
    assert info["rows"] == d.N and info["played_by_checkpoint_games"] == 1 and info["noop_rows"] == int((d.action == 0).sum())


def test_decision_rows_with_mmap_and_h_window_053_r5_053_r6(setup, tmp_path):
    dump, job, _ = setup
    dr.build(dump, [job], [], dump / dr.SIDECAR)
    src = pp.parse_src(f"m:model:split:decision={dump}")
    a, b = pp.Data([src], (2,), eval_frac=0.0), pp.Data([src], (2,), eval_frac=0.0, mmap=tmp_path / "local")
    for f in ("h", "action", "dec", "frame", "logp"):
        assert np.array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f))), f
    assert np.array_equal(a.nxt[2], b.nxt[2])
    rows = a.pair_rows(2, "train")
    st = a.stats(rows, 2)
    hist = a.inputs(rows, 2, "H", st)["hist"].numpy()
    span = a.pos[a.nxt[2][rows]] - a.pos[rows]
    for i, r in enumerate(rows[:20]):
        win = a.order[a.pos[r]:a.pos[r] + span[i]]
        h = np.bincount(a.action[win].astype(int), minlength=a.n_act).astype(np.float32)
        assert np.allclose(hist[i], h / h.sum())


def test_parse_logs_053_r5():
    assert dr.parse_log(_act_log([(1, 0)], "t0@3:zerg_rush")) is None
    full = dr.parse_log(_decision_log([(1, 0), (9, 5)]))
    assert full["complete"] and [x[:2] for x in full["decisions"]] == [(1, 0), (9, 5)]
    assert not dr.parse_log(_decision_log([(9, 0), (1, 5)]))["complete"]
    assert not dr.parse_log(_decision_log([(1, 0)]).replace('"decisions": 1', '"decisions": 2'))["complete"]
    assert dr.policy_sha("model:ckpt-2844@e9efcca93e39:stochastic") == "e9efcca93e39" and dr.policy_sha(None) is None
    with pytest.raises(SystemExit):
        pp.parse_src("m:model:split:foo=x")
    assert pp.parse_src("m:model:split=x").rows == "all"
    assert Path(dr.SIDECAR).name == "decisions"
