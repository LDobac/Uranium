import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from testkit import fake_dump

from uranium import tables as tb
from uranium.diag.planning import decisions as dr
from uranium.diag.planning import pairs as pr
from uranium.diag.planning import pred as pp
from uranium.diag.planning import s0 as plan_s0

C_SHA = "abcdef012345" + "0" * 52
N_GAMES, ROWS = 20, 40
EXPLORE_AT = {3: ("forced", 4), 12: ("control", 2), 25: ("forced", 20)}
BUNKER, MARINE = tb.action_id("Build_Bunker"), tb.action_id("Train_Marine")


def _act_log(decisions, explore: dict) -> str:
    lines = []
    for i, (f, a) in enumerate(decisions):
        r = {"i": i, "frame": int(f), "digest": "0" * 16, "act": {}, "decision": {"action": int(a), "delay_bucket": 1, "units": [], "rejected": False},
             "state": None, "forced": False}
        if i in explore:
            coin, n_c = explore[i]
            prop = 0.5 / n_c if coin == "forced" else 0.5
            r["explore"] = {"point": 1, "tau": int(f) - 5, "frame": int(f), "epsilon": 0.5, "k": 4, "cands": [[c, 0.01] for c in range(1, n_c + 1)],
                            "a0": 0, "coin": coin, "c": int(a) if coin == "forced" else None, "rejected": False, "propensity": prop}
        lines.append(json.dumps(r))
    lines.append(json.dumps({"end": {"obs": len(decisions), "decisions": len(decisions), "end_frame": 9000, "reason": "Elimination", "player": 0,
                                     "players_index": 0, "policy": f"model:ckpt-1@{C_SHA[:12]}:stochastic"}}))
    return "\n".join(lines) + "\n"


def _t0_log() -> str:
    return json.dumps({"end": {"obs": 0, "decisions": 0, "end_frame": 9000, "policy": "t0@0.1.0"}}) + "\n"


def explore_setup(tmp_path: Path):
    dump = fake_dump(tmp_path / "dump", n_games=N_GAMES, rows=ROWS, race="P")
    job = tmp_path / "explore-pilot"
    for d in ("replays", "memory", "acts"):
        (job / d).mkdir(parents=True)
    games = json.loads((dump / "games.json").read_text())
    act = np.fromfile(dump / "rows" / "action.bin", np.int16).reshape(N_GAMES, ROWS)
    fr = np.fromfile(dump / "rows" / "frame.bin", np.float32).reshape(N_GAMES, ROWS)
    for g in range(N_GAMES):
        data = f"explore replay {g}".encode()
        (job / "replays" / f"g{g + 1}.rep").write_bytes(data)
        games[str(g)]["sha"] = hashlib.sha256(data).hexdigest()
        decisions = [(int(fr[g, i]) + 2, int(act[g, i])) for i in range(ROWS)]
        mem = [json.dumps({"frame": f, "action": a, "delay_bucket": 1, "units": []}) for f, a in decisions]
        mem.append(json.dumps({"decisions": ROWS, "end_frame": 9000, "player": 0}))
        (job / "memory" / f"g{g + 1}.jsonl").write_text("\n".join(mem) + "\n")
        (job / "acts" / f"g{g + 1}-p0.jsonl").write_text(_act_log(decisions, EXPLORE_AT))
        (job / "acts" / f"g{g + 1}-p1.jsonl").write_text(_t0_log())
    (dump / "games.json").write_text(json.dumps(games))
    man = json.loads((dump / "manifest.json").read_text())
    man.update({"checkpoint_sha256": C_SHA, "row_until": 14400})
    (dump / "manifest.json").write_text(json.dumps(man))
    rec = tmp_path / "records.jsonl"
    rec.write_text("".join(json.dumps({"replay_sha256": e["sha"], "outcome": "written", "memory": {"status": "ok"}}) + "\n" for e in games.values()))
    dr.build(dump, [job], [rec], dump / dr.SIDECAR)
    return dump, job


def explore_data(dump: Path, job: Path, eval_frac=0.5, val_frac=0.0, other=None):
    srcs = [pp.Source("explore", "model", "split", str(dump), rows="decision")] + (other or [])
    pts = dr.read_explore(job, dr.load(dump)[1])
    return pp.Data(srcs, ks=(8,), eval_frac=eval_frac, keep_bot=True, val_frac=val_frac, explore={"explore": pts}, hold={"explore": pp.EXPLORE_HOLDOUT})


def test_053_AC12_R8_explore_weight_is_inverse_propensity_capped_at_10():
    w = pp.explore_weight([0.5, 0.125, 1 / 6, 0.25, 0.1, 0.05, 0.001])
    assert np.allclose(w, [2.0, 8.0, 6.0, 4.0, 10.0, 10.0, 10.0])
    assert pp.EXPLORE_CAP == 10.0


def test_053_AC12_R8_explore_rows_weighted_only_at_points(tmp_path):
    dump, job = explore_setup(tmp_path)
    pts, st = dr.read_explore(job, dr.load(dump)[1])
    assert st == {"ok": N_GAMES} and all(len(v) == len(EXPLORE_AT) for v in pts.values())
    data = explore_data(dump, job)
    e = data.explore_pts
    assert len(e["row"]) == N_GAMES * len(EXPLORE_AT)
    assert sorted(set(data.dec[e["row"]].tolist())) == sorted(EXPLORE_AT)
    want = {3: 8.0, 12: 2.0, 25: 10.0}
    assert all(data.row_w[r] == want[int(data.dec[r])] for r in e["row"])
    rest = np.setdiff1d(np.arange(data.N), e["row"])
    assert (data.row_w[rest] == 1.0).all(), "explore rows that are not forced / control points weigh 1 (053-R8 1)"
    info = data.explore_info["explore"]
    assert info["forced"] == 2 * N_GAMES and info["control"] == N_GAMES and info["no_row"] == 0


def test_053_AC12_R8_weighted_loss_scales_rows(tmp_path):
    dump, job = explore_setup(tmp_path)
    data = explore_data(dump, job)
    rows = data.pair_rows(8, "train")
    st = data.stats(rows, 8)
    idx = np.sort(np.concatenate([data.explore_pts["row"][:4], rows[:8]]))
    idx = idx[data.nxt[8][idx] >= 0]
    net = pp.new_net(data, "A", st)
    b = data.batch(idx, 8, "A", st)
    assert torch.equal(b["w"], torch.from_numpy(data.row_w[idx]))
    o_on = torch.ones(data.own.shape[1])
    p_on = torch.ones(data.priv.shape[1])
    out = net(b["h"], b.get("a"), None, None, b.get("own"))
    lw = pp.loss_terms(out, b, o_on, p_on)
    lu = pp.loss_terms(out, {k: v for k, v in b.items() if k != "w"}, o_on, p_on)
    for k in ("y", "o", "p"):
        assert torch.allclose(lw[k], lu[k] * b["w"])
    assert torch.allclose(lw["q"], lu["q"] * b["w"][:, None])


def test_053_AC12_R8_held_out_explore_games_never_in_training_batches(tmp_path, monkeypatch):
    dump, job = explore_setup(tmp_path)
    data = explore_data(dump, job, eval_frac=0.5, val_frac=0.1)
    roles = [e["role"] for e in data.game_meta]
    assert roles.count("eval") == round(pp.EXPLORE_HOLDOUT * N_GAMES), "20% of the explore games whatever --eval-frac (053-R8 2)"
    assert roles.count("val") >= 1
    seen, in_val = [], [False]
    orig = pp.Data.batch

    def spy(self, idx, *a, **kw):
        if not in_val[0]:
            seen.append(np.asarray(idx).copy())
        return orig(self, idx, *a, **kw)

    rows = data.pair_rows(8, "train")
    st = data.stats(rows, 8)
    w, _ = pp.mix_weights(data, rows, {"human": 0.2})
    val = data.pair_rows(8, "val")
    monkeypatch.setattr(pp.Data, "batch", spy)
    seen_val = []
    orig_val = pp.val_loss

    def spy_val(net, d, vrows, *a, **kw):
        seen_val.append(np.asarray(vrows).copy())
        in_val[0] = True
        try:
            return orig_val(net, d, vrows, *a, **kw)
        finally:
            in_val[0] = False

    monkeypatch.setattr(pp, "val_loss", spy_val)
    pp.train_one(data, "A", 8, rows, st, steps=40, batch=64, lr=1e-3, seed=1, weights=w, val_rows=val, val_every=10, patience=10, split_nets=True)
    assert seen_val and len(seen) == 40
    drawn = np.concatenate(seen)
    assert not data.row_eval[drawn].any(), "held-out explore games are evaluation only"
    assert not data.row_val[drawn].any()
    assert data.row_val[np.concatenate(seen_val)].all()
    held_pts = data.explore_pts["row"][data.row_eval[data.explore_pts["row"]]]
    assert len(held_pts) == round(pp.EXPLORE_HOLDOUT * N_GAMES) * len(EXPLORE_AT) and not np.isin(held_pts, drawn).any()


def trunk_dump(tmp_path: Path, n=8, sha=C_SHA):
    dump = fake_dump(tmp_path / "trunks", n_games=n, rows=60)
    man = json.loads((dump / "manifest.json").read_text())
    man["checkpoint_sha256"] = sha
    (dump / "manifest.json").write_text(json.dumps(man))
    m = np.fromfile(dump / "rows" / "mask.bin", np.uint8).reshape(n * 60, -1)
    full = np.unpackbits(m, axis=1, count=101).astype(bool)
    full[:, MARINE] = True
    np.packbits(full, axis=1).tofile(dump / "rows" / "mask.bin")
    return dump


FORCE = {"none": [], "force_bunker": [{"op": "force", "action": "Build_Bunker", "from_s": 0, "to_s": 0}],
         "force_marine": [{"op": "force", "action": "Train_Marine", "from_s": 0, "to_s": 0}],
         "ban_bunker": [{"op": "ban", "action": "Build_Bunker", "from_s": 0, "to_s": 90}]}


def branch_set(d: Path, dump: Path, n=8, sha=C_SHA, ys=None):
    games = json.loads((dump / "games.json").read_text())
    pts, brs = [], []
    ys = ys or {}
    for g in range(n):
        for j, i in enumerate((10, 30)):
            pid = f"trunks/b2/g{g}#p{j}"
            pts.append({"point": pid, "trunk": {"replay": "absent.rep", "replay_sha256": games[str(g)]["sha"]}, "f_t": int(2880 + 24 * i + 2),
                        "checkpoint_sha256": sha, "valid": True})
            for a in FORCE:
                for r in (1, 2):
                    y = ys.get((pid, a), (g + j + (a == "force_bunker")) % 2)
                    brs.append({"point": pid, "alt": a, "rules": FORCE[a], "repeat": r, "valid": True,
                                "metrics": {"held_8m": int(y) if r == 1 else int(ys.get((pid, a, 2), y)), "held_s": 0.5}})
    d.mkdir(parents=True, exist_ok=True)
    (d / "points.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pts))
    (d / "branches.jsonl").write_text("".join(json.dumps(b) + "\n" for b in brs))
    return pts


def main_data(tmp_path: Path):
    p = fake_dump(tmp_path / "main", n_games=12, sha_prefix="e")
    man = json.loads((p / "manifest.json").read_text())
    man["checkpoint_sha256"] = C_SHA
    (p / "manifest.json").write_text(json.dumps(man))
    return pp.Data([pp.Source("m", "teacher", "split", str(p))], ks=(8,), q_names=("held_8m", "held_s"), labels={"m": {}})


def test_053_AC12_R8_held_out_trunks_never_in_pair_batches(tmp_path, monkeypatch):
    data = main_data(tmp_path)
    dump = trunk_dump(tmp_path)
    branch_set(tmp_path / "b2", dump)
    ps_ = pr.PairSet("b2", tmp_path / "b2", dump, data)
    assert ps_.info["trunks"] == 8 and ps_.info["held_trunks"] == 2, "25% of the trunks (053-R8 2)"
    assert ps_.info["alts"] == {"force_bunker": BUNKER, "force_marine": MARINE} and ps_.info["skipped"]["not_one_decision_force"] == 1
    trunk_of = np.array([ps_.points[i].split("#")[0] for i in ps_.point])
    for t in set(trunk_of):
        h = ps_.held[trunk_of == t]
        assert h.all() or not h.any(), "a trunk's branches are all held out or all in training"
    assert set(trunk_of[ps_.held]) == ps_.held_trunks
    seen = []
    orig = pr.PairSet.tensors

    def spy(self, idx, st):
        seen.append(np.asarray(idx).copy())
        return orig(self, idx, st)

    monkeypatch.setattr(pr.PairSet, "tensors", spy)
    rows = data.pair_rows(8, "train")
    st = data.stats(rows, 8)
    pp.train_one(data, "A", 8, rows, st, steps=20, batch=32, lr=1e-3, seed=2, split_nets=True, pairs=pr.PairUnion([ps_]), pair_batch=8)
    picked = np.concatenate(seen)
    assert len(picked) == 20 * 8 and not ps_.held[picked].any()


def test_053_AC12_R8_b1_and_other_checkpoint_trunks_refused(tmp_path):
    data = main_data(tmp_path)
    dump = trunk_dump(tmp_path)
    branch_set(tmp_path / "b1", dump, sha="e9efcca9" + "0" * 56)
    with pytest.raises(SystemExit, match="B1"):
        pr.PairSet("b1", tmp_path / "b1", dump, data)
    other = trunk_dump(tmp_path / "x", sha="1" * 64)
    branch_set(tmp_path / "bx", other)
    with pytest.raises(SystemExit, match="053-R1"):
        pr.PairSet("bx", tmp_path / "bx", other, data)
    pts = branch_set(tmp_path / "bn", dump)
    for q in pts:
        q["trunk"].pop("replay_sha256")
    (tmp_path / "bn" / "points.jsonl").write_text("".join(json.dumps(q) + "\n" for q in pts))
    with pytest.raises(SystemExit, match="no replay"):
        pr.PairSet("bn", tmp_path / "bn", dump, data)


def test_053_AC12_R8_held_out_trunk_as_training_source_refused(tmp_path):
    dump = trunk_dump(tmp_path)
    data = pp.Data([pp.Source("trunk-rows", "model", "train", str(dump))], ks=(8,), keep_bot=True, q_names=("held_8m",), labels={"trunk-rows": {}})
    branch_set(tmp_path / "b2", dump)
    a = Namespace(pairs=[f"b2={tmp_path / 'b2'}"], pair_dump=[f"b2={dump}"], pair_root=None, split_seed=1, split_nets=True, variants="U,A")
    with pytest.raises(SystemExit, match="held-out trunks"):
        plan_s0.pair_sets(a, data)


def test_053_AC12_R8_pair_loss_only_within_a_point(tmp_path):
    data = main_data(tmp_path)
    dump = trunk_dump(tmp_path)
    p0, p1 = "trunks/b2/g0#p0", "trunks/b2/g0#p1"
    ys = {(p0, "none"): 0, (p0, "none", 2): 0, (p0, "force_bunker"): 1, (p0, "force_bunker", 2): 1,
          (p1, "none"): 1, (p1, "none", 2): 1, (p1, "force_bunker"): 0, (p1, "force_bunker", 2): 1}
    branch_set(tmp_path / "b2", dump, ys=ys)
    ps_ = pr.PairSet("b2", tmp_path / "b2", dump, data)
    j = data.q_names.index("held_8m")
    pid = {p: i for i, p in enumerate(ps_.points)}
    a0 = np.nonzero((ps_.point == pid[p0]) & (ps_.c == BUNKER))[0][0]
    a1 = np.nonzero((ps_.point == pid[p1]) & (ps_.c == BUNKER))[0][0]
    assert ps_.d[a0, j] == pytest.approx(1.0) and ps_.d[a1, j] == pytest.approx(-0.5)
    assert ps_.ok[a0, data.q_names.index("held_s")] == 1 and ps_.d[a0, data.q_names.index("held_s")] == 0.0
    assert ps_.row[a0] != ps_.row[a1]
    ys2 = {**ys, (p1, "none"): 0, (p1, "none", 2): 0}
    branch_set(tmp_path / "b2x", dump, ys=ys2)
    ps2 = pr.PairSet("b2x", tmp_path / "b2x", dump, data)
    assert ps2.d[a0, j] == ps_.d[a0, j] and ps2.d[a1, j] == pytest.approx(0.5), "a point's pair target uses that point's branches only"
    rows = data.pair_rows(8, "train")
    st = data.stats(rows, 8)
    torch.manual_seed(0)
    net = pp.new_net(data, "A", st)
    t = ps_.tensors(np.array([a0, a1]), st)
    got = pp.pair_loss(net, t)
    with torch.no_grad():
        want = []
        for i, r in enumerate(ps_.row[[a0, a1]]):
            own = ps_.data.own_in(np.array([r]), st)
            h = torch.from_numpy(ps_.data.h[[r]].astype(np.float32))
            q = torch.sigmoid(torch.stack([net(h, torch.tensor([c]), own=own)["q"][0] for c in range(data.n_act)]))
            pi = torch.from_numpy(ps_.data.probs(np.array([r]))[0].astype(np.float32))
            e = (q[ps_.c[[a0, a1]][i]] - (pi[:, None] * q).sum(0) - t["d"][i]) ** 2 * t["ok"][i]
            want.append(e)
        want = torch.stack(want).sum() / t["ok"].sum()
    assert torch.allclose(got, want, atol=1e-6)


def test_053_R8_without_explore_and_pairs_the_old_path_is_kept(tmp_path):
    data = main_data(tmp_path)
    rows = data.pair_rows(8, "train")
    st = data.stats(rows, 8)
    assert not data.weighted and "w" not in data.batch(rows[:4], 8, "A", st)
    assert set(plan_s0.fingerprint(data)) == {"checkpoint", "checkpoint_sha256", "rows", "games", "sources", "split_digest", "design", "q_heads", "q_digest"}
    a = Namespace(mix="", val_frac=0.0, split_nets=False, explore=[], pairs=[])
    assert plan_s0.training(a) == {}


def test_053_R8_plan_s0_cli_with_explore(tmp_path):
    dump, job = explore_setup(tmp_path)
    labels = tmp_path / "labels.json"
    dg = json.loads((dump / "games.json").read_text())
    games = {str(g): {"sha": dg[str(g)]["sha"], "start": dg[str(g)]["start"], "held_8m": g % 2, "held_s": 1, "alive_10m": 1, "win": g % 2} for g in range(N_GAMES)}
    labels.write_text(json.dumps({"kind": "plan-label", "games": games, "join": {"aborted": 0}}))
    from uranium.__main__ import main

    out = tmp_path / "out"
    rc = main(["diag", "plan-s0", "--src", f"explore:model:split:decision={dump}", "--explore", f"explore={job}", "--labels", f"explore={labels}",
               "--keep-bot-perspective", "--q-heads", "held_8m,win", "--ks", "8", "--variants", "U,A", "--no-m3c", "--steps", "20", "--batch", "32",
               "--boot", "20", "--out", str(out), "--th-own-k", "8"])
    assert rc == 0
    rep = json.loads((out / "s0.json").read_text(encoding="utf-8"))
    assert rep["explore"]["weights"] == {"2": N_GAMES, "8": N_GAMES, "10": N_GAMES}
    assert rep["settings"]["training"]["explore"] == {"cap": 10.0, "hold": 0.2}
    assert "탐색 행 가중치" in (out / "s0.md").read_text(encoding="utf-8")


def test_053_R8_plan_s0_cli_with_pairs(tmp_path):
    from uranium.__main__ import main

    dump = trunk_dump(tmp_path)
    src = fake_dump(tmp_path / "main", n_games=12, sha_prefix="e")
    man = json.loads((src / "manifest.json").read_text())
    man["checkpoint_sha256"] = C_SHA
    (src / "manifest.json").write_text(json.dumps(man))
    branch_set(tmp_path / "b2", dump)
    base = ["diag", "plan-s0", "--src", f"m:human:split={src}", "--q-heads", "held_8m,win", "--ks", "8", "--variants", "U,A", "--no-m3c", "--steps", "10", "--batch", "32",
            "--boot", "20", "--pairs", f"b2={tmp_path / 'b2'}", "--pair-dump", f"b2={dump}", "--pair-batch", "4"]
    with pytest.raises(SystemExit, match="split-nets"):
        main([*base, "--out", str(tmp_path / "o0")])
    out = tmp_path / "out"
    assert main([*base, "--split-nets", "--out", str(out)]) == 0
    rep = json.loads((out / "s0.json").read_text(encoding="utf-8"))
    assert rep["pairs"]["sets"]["b2"]["held_trunks"] == 2 and rep["settings"]["training"]["pairs"]["hold"] == pr.PAIR_HOLDOUT
    assert {"train", "held"} <= set(rep["pairs"]["8"]["b2"]) and rep["pairs"]["8"]["b2"]["held"]["pairs"] > 0
    assert "분기 짝 손실" in (out / "s0.md").read_text(encoding="utf-8")
    meta_a = torch.load(out / "models" / "A_k8.pt", weights_only=False)["meta"]["data"]
    meta_u = torch.load(out / "models" / "U_k8.pt", weights_only=False)["meta"]["data"]
    assert "pairs" in meta_a and "pairs" not in meta_u
