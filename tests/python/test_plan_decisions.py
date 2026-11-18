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


def _rejected(decisions) -> list:
    return [i % 4 == 3 and a != 0 for i, (_, a) in enumerate(decisions)]


def _act_log(decisions, policy, rejected=None) -> str:
    rejected = rejected or [False] * len(decisions)
    lines = [json.dumps({"i": i, "frame": int(f), "digest": "0" * 16, "act": {}, "decision": {"action": int(a), "delay_bucket": 1, "units": [], "rejected": r},
                         "state": None, "forced": False}) for i, ((f, a), r) in enumerate(zip(decisions, rejected))]
    lines.append(json.dumps({"end": {"obs": len(decisions), "decisions": len(decisions), "end_frame": 9000, "reason": "TimeLimitDraw", "player": 0,
                                     "players_index": 0, "policy": policy}}))
    return "\n".join(lines) + "\n"


def _game_decisions(g: int) -> list:
    acts = [0, TRAIN, 0, TRAIN, BUNKER, TRAIN, 0, TRAIN, TRAIN, BUNKER, 0, TRAIN, TRAIN, 0, TRAIN, TRAIN, BUNKER, TRAIN, TRAIN, 0]
    lives = [1, 3] + [3 + 4 * i for i in range(1, 18) if i != 6] + [3 + 4 * 6 - 2]
    lives = sorted(lives)
    return [(L, acts[(i + g) % len(acts)]) for i, L in enumerate(lives)]


def _records(path: Path, shas, status="ok") -> Path:
    path.write_text("".join(json.dumps({"replay_sha256": s, "outcome": "written", "memory": {"status": status}}) + "\n" for s in shas))
    return path


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
    rej = {g: [False] * len(logs[g]) for g in logs}
    rej[1] = _rejected(logs[1])
    (job / "memory" / "g1.jsonl").write_text(_decision_log(logs[0]))
    (job / "memory" / "g2.jsonl").write_text(_act_log(logs[1], f"model:ckpt-1@{C_SHA[:12]}:stochastic", rej[1]))
    (job / "memory" / "g3.jsonl").write_text(_decision_log(logs[2])[:-60])
    (job / "memory" / "g5.jsonl").write_text(_decision_log(logs[4]))
    (job / "results.jsonl").write_text("\n".join(json.dumps({"game_id": i, "model": {}, "policy": "model:ckpt-9@123456789abc:stochastic"}) for i in (1, 5)) + "\n")
    rec = _records(tmp_path / "branch_build.jsonl", sorted({e["sha"] for e in games.values()}))
    return dump, job, logs, rej, rec


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
    dump, job, logs, rej, rec = setup
    s = dr.build(dump, [job], [rec], dump / dr.SIDECAR)
    meta = json.loads((dump / dr.SIDECAR / "games.json").read_text())
    st = {int(k): v["status"] for k, v in meta.items()}
    assert st[0] == "ok" and st[1] == "ok" and st[2] == "incomplete" and st[3] == "no_log"
    assert sorted([st[4], st[5]]) == ["ok", "other_view"]
    exp = _expected_rows(logs[0])
    m0 = meta["0"]
    assert m0["matched"] == sum(ok for *_, ok in exp) and m0["unmatched"] == sum(not ok for *_, ok in exp)
    assert m0["collisions"] == 1 and m0["unmatched"] == 2
    assert m0["policy_sha"] == "123456789abc" and meta["1"]["policy_sha"] == C_SHA[:12]
    assert m0["rejected"] == 0 and meta["1"]["rejected"] == sum(rej[1])
    assert s["played_by_dump_checkpoint"] == 1 and s["status"]["ok"] == 3
    arr, _ = dr.load(dump)
    g0 = arr["game"] == 0
    rows0 = arr["row"][g0]
    assert np.array_equal(arr["action"][g0], [a for _, a in logs[0]])
    assert np.array_equal(rows0 >= 0, [ok for *_, ok in exp])
    assert (arr["rejected"][g0] == -1).all() and np.array_equal(arr["rejected"][arr["game"] == 1] == 1, rej[1])
    fr = np.fromfile(dump / "rows" / "frame.bin", np.float32)
    assert np.array_equal(fr[rows0[rows0 >= 0]], [p for p, _, ok in exp if ok])


def test_decision_records_required_and_bad_memory_041_r28_053_r5(setup, tmp_path):
    dump, job, _, _, rec = setup
    games = json.loads((dump / "games.json").read_text())
    bad = _records(tmp_path / "branch_bad.jsonl", [games["0"]["sha"]], "unit_mismatch")
    part = _records(tmp_path / "branch_part.jsonl", [games["1"]["sha"]])
    dr.build(dump, [job], [bad, part], dump / dr.SIDECAR)
    meta = json.loads((dump / dr.SIDECAR / "games.json").read_text())
    assert meta["0"]["status"] == "record_unit_mismatch" and meta["1"]["status"] == "ok"
    assert meta["4"]["status"] == meta["5"]["status"] == "no_record", "a stream without a build record is left out"
    p = __import__("argparse").ArgumentParser()
    dr.add_args(p)
    with pytest.raises(SystemExit):
        p.parse_args(["--dump", str(dump), "--job", str(job)])


def _count_target(exp, rejected, j, k):
    cnt = 0
    for q in range(j + 1, len(exp)):
        if exp[q][1] != 0 and not rejected[q]:
            cnt += 1
            if cnt == k:
                return exp[q]
    return None


def test_data_decision_rows_examples_and_k_pairs_053_r5(setup):
    dump, job, logs, rej, rec = setup
    dr.build(dump, [job], [rec], dump / dr.SIDECAR)
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
            for j, (p, a, okk) in enumerate(exp):
                if not okk:
                    continue
                here = np.nonzero(m & (fr == p))[0]
                assert len(here) == 1
                want = _count_target(exp, rej[e["gid"]], j, k)
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
    from uranium.diag.planning import s0
    assert s0.fingerprint(d)["sources"] == [["m", "model", "split", "decision"]]
    info = pp.describe(d)["sources"][0]["decision_rows"]
    assert info["rows"] == d.N and info["played_by_checkpoint_games"] == 1 and info["noop_rows"] == int((d.action == 0).sum())


def test_rejected_decisions_not_counted_in_k_051_r9_053_r5():
    game = np.zeros(6, np.int64)
    ord_ = np.arange(6)
    action = np.array([TRAIN, TRAIN, 0, TRAIN, TRAIN, TRAIN])
    rejected = np.array([-1, 1, -1, 0, 1, 0])
    nx = dr.next_decision(game, ord_, dr.counted(action, rejected), 1)
    assert nx.tolist() == [3, 3, 3, 5, 5, -1]
    nx_mem = dr.next_decision(game, ord_, dr.counted(action, np.full(6, -1)), 1)
    assert nx_mem.tolist() == [1, 3, 3, 4, 5, -1], "memory logs (no rejected field) count every non-NoOp decision"


def test_decision_rows_keep_chains_and_need_decision_k_053_r5(setup):
    dump, job, logs, rej, rec = setup
    n_games = 6
    fr = np.tile(np.arange(1, ROWS + 1, dtype=np.float32), n_games)
    fr.tofile(dump / "rows" / "frame.bin")
    for g in (0, 4):
        dec = [(int(f) + 2, TRAIN) for f in range(1, ROWS + 1)]
        (job / "memory" / f"g{g + 1}.jsonl").write_text(_decision_log(dec))
    (job / "memory" / "g2.jsonl").unlink()
    dr.build(dump, [job], [rec], dump / dr.SIDECAR)
    dec_src = pp.parse_src(f"m:model:split:decision={dump}")
    d = pp.Data([dec_src], (2,), eval_frac=0.0)
    assert d.N > 0 and d.chained.sum() > d.N // 2, "consecutive decisions are chained"
    assert not d.chain_drop.any() and d.usable.all(), "decision-row sources keep chain rows (user decision 2026-11-18)"
    old = pp.Data([pp.Source("m", "model", "split", str(dump))], (2,), eval_frac=0.0, keep_bot=True)
    assert old.chain_drop.any(), "other sources keep the old rule"
    with pytest.raises(SystemExit):
        pp.Data([dec_src], (144,), eval_frac=0.0, design=pp.Design(k_unit="frame"))


def test_decision_rows_with_mmap_and_h_window_053_r5_053_r6(setup, tmp_path):
    dump, job, _, _, rec = setup
    dr.build(dump, [job], [rec], dump / dr.SIDECAR)
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
