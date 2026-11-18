import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from testkit import fake_dump

from uranium import tables as tb
from uranium.diag.planning import pred as pp


def legacy_train_one(data, variant, k, rows, st, *, steps, batch, lr, seed, log_every=500):
    torch.manual_seed(seed)
    net = pp.Predictor(variant, data.h.shape[1], data.y.shape[1], data.priv.shape[1], data.own.shape[1], data.n_act, data.n_delay,
                       own_in=data.own.shape[1] if data.design.own_input else 0)
    net.h_mu.copy_(torch.from_numpy(st["h_mu"]))
    net.h_sd.copy_(torch.from_numpy(st["h_sd"]))
    o_on = torch.from_numpy(np.isfinite(st["d_sd" if data.design.own_target == "delta" else "o_sd"]).astype(np.float32))
    p_on = torch.from_numpy(np.isfinite(st["p_sd"]).astype(np.float32))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    log, acc = [], {}
    for step in range(1, steps + 1):
        idx = np.sort(rows[rng.integers(0, len(rows), batch)])
        b = data.batch(idx, k, variant, st)
        out = net(b["h"], b.get("a"), b.get("d"), b.get("hist"), b.get("own"))
        lt = pp.loss_terms(out, b, o_on, p_on)
        parts = {"y": lt["y"].mean(), "p": lt["p"].sum() / b["p_ok"].sum().clamp_min(1),
                 "o": lt["o"].mean(), "q": lt["q"].sum(0).div(b["q_ok"].sum(0).clamp_min(1)).sum()}
        loss = sum(parts.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        for kk, v in parts.items():
            acc[kk] = acc.get(kk, 0.0) + float(v.detach())
        if step % log_every == 0 or step == steps:
            n = log_every if step % log_every == 0 else step % log_every
            log.append({"step": step, **{kk: v / n for kk, v in acc.items()}})
            acc = {}
    net.eval()
    return net, log


def _same(a, b) -> bool:
    return list(a.state_dict()) == list(b.state_dict()) and all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


@pytest.mark.parametrize("variant", ["A", "H", "S"])
def test_053_AC2_old_flags_train_bit_identical_to_s0b(tmp_path, variant):
    torch.set_num_threads(1)
    p = fake_dump(tmp_path / "a")
    data = pp.Data([pp.Source("a", "teacher", "split", str(p))], ks=(8,))
    rows = data.pair_rows(8, "train")
    st = data.stats(rows, 8)
    old, lo = legacy_train_one(data, variant, 8, rows, st, steps=25, batch=64, lr=1e-3, seed=3, log_every=10)
    new, ln = pp.train_one(data, variant, 8, rows, st, steps=25, batch=64, lr=1e-3, seed=3, log_every=10)
    assert lo == ln and _same(old, new)
    d0 = pp.Data([pp.Source("a", "teacher", "split", str(p))], ks=(8,), q_names=pp.Q_NAMES, labels=None, val_frac=0.0)
    assert d0.split_digest == data.split_digest and np.array_equal(d0.g_q, data.g_q, equal_nan=True)


def _three(tmp_path):
    t = fake_dump(tmp_path / "t", n_games=24, seed=1)
    m = fake_dump(tmp_path / "m", n_games=12, seed=2, sha_prefix="b")
    h = fake_dump(tmp_path / "h", n_games=40, seed=3, sha_prefix="c")
    return [pp.Source("t", "teacher", "split", str(t)), pp.Source("m", "model", "split", str(m)), pp.Source("h", "human", "train", str(h))]


def test_053_AC2_R3_realized_shares_within_one_point(tmp_path):
    torch.set_num_threads(1)
    data = pp.Data(_three(tmp_path), ks=(8,), keep_bot=True)
    rows = data.pair_rows(8, "train")
    st = data.stats(rows, 8)
    bunker, scv = tb.action_id("Build_Bunker"), tb.action_id("Train_SCV")
    w, plan = pp.mix_weights(data, rows, {"human": 0.2})
    n_k = {k: int((data.row_kind[rows] == pp.KINDS.index(k)).sum()) for k in pp.KINDS}
    assert plan["kinds"]["human"]["target"] == pytest.approx(0.2)
    assert plan["kinds"]["teacher"]["target"] == pytest.approx(0.8 * n_k["teacher"] / (n_k["teacher"] + n_k["model"]))
    assert w.sum() == pytest.approx(1.0) and plan["op_actions"][bunker]["multiplier"] == 5.0
    t_rows = rows[data.row_kind[rows] == pp.KINDS.index("teacher")]
    op_t = data.is_op[t_rows]
    assert w[np.isin(rows, t_rows[op_t])][0] == pytest.approx(5 * w[np.isin(rows, t_rows[~op_t])][0])
    info = {}
    pp.train_one(data, "U", 8, rows, st, steps=200, batch=512, lr=1e-3, seed=0, weights=w, info=info)
    for k, e in plan["kinds"].items():
        assert abs(info["drawn"]["kinds"][k] - e["target"]) < 0.01, (k, info["drawn"]["kinds"], plan["kinds"])
    for a, e in plan["op_actions"].items():
        assert abs(info["drawn"]["op_actions"][a] - e["expected_share"]) < 0.01
    few = rows[(data.action[rows] != bunker) | (np.arange(len(rows)) % 50 == 0)]
    _, rare = pp.mix_weights(data, few, {"human": 0.2}, op_x=5.0, rare_share=0.05, rare_x=20.0)
    sh = rare["op_actions"][bunker]["row_share"]
    assert sh < 0.05 and min(20.0, 0.05 / sh) > 5.0 and rare["op_actions"][bunker]["multiplier"] == pytest.approx(min(20.0, 0.05 / sh))
    assert rare["op_actions"][scv]["row_share"] >= 0.05 and rare["op_actions"][scv]["multiplier"] == 5.0
    _, near = pp.mix_weights(data, few, {"human": 0.2}, op_x=5.0, rare_share=2 * sh, rare_x=20.0)
    assert 2 * sh / 5 < sh < 2 * sh and near["op_actions"][bunker]["multiplier"] == 5.0
    _, nohum = pp.mix_weights(data, rows[data.row_kind[rows] != pp.KINDS.index("human")], {"human": 0.2})
    assert "human" not in nohum["kinds"] and sum(e["target"] for e in nohum["kinds"].values()) == pytest.approx(1.0)
    with pytest.raises(SystemExit):
        pp.parse_mix("human=0.7,teacher=0.6")
    assert pp.parse_mix("") is None and pp.parse_mix("human=0.35") == {"human": 0.35}


def test_053_AC3_R4_split_is_game_level_with_views_together(tmp_path):
    p = fake_dump(tmp_path / "a", n_games=40)
    games = json.loads((p / "games.json").read_text())
    for k, e in games.items():
        e["sha"] = f"a{int(k) // 2:063x}"
    (p / "games.json").write_text(json.dumps(games))
    src = [pp.Source("a", "teacher", "split", str(p))]
    base = pp.Data(src, ks=(8,))
    data = pp.Data(src, ks=(8,), val_frac=0.25)
    role = {}
    for e in data.game_meta:
        role.setdefault(e["sha"], set()).add(e["role"])
    assert all(len(r) == 1 for r in role.values())
    keys = {r: {s for s, x in role.items() if x == {r}} for r in ("train", "val", "eval")}
    assert keys["val"] and keys["train"] and keys["eval"]
    assert len(keys["val"]) == round(0.25 * (len(keys["val"]) + len(keys["train"])))
    assert np.array_equal(base.g_role_eval, data.g_role_eval) and base.split_digest != data.split_digest
    tr, va, ev = (set(data.pair_rows(8, r).tolist()) for r in ("train", "val", "eval"))
    assert tr and va and ev and not (tr & va) and not (tr & ev) and not (va & ev)
    assert set(base.pair_rows(8, "train").tolist()) == tr | va and set(base.pair_rows(8, "eval").tolist()) == ev
    assert not data.row_eval[list(va)].any() and data.row_val[list(va)].all() and not data.row_val[list(ev)].any()
    with pytest.raises(ValueError):
        data.pair_rows(8, "test")


def _val_loss(net, data, rows, st):
    o_on = torch.from_numpy(np.isfinite(st["d_sd"]).astype(np.float32))
    p_on = torch.from_numpy(np.isfinite(st["p_sd"]).astype(np.float32))
    return pp.val_loss(net, data, rows, 8, net.variant, st, o_on, p_on)


def test_053_AC3_R4_early_stopping_keeps_the_best_step_per_net(tmp_path):
    torch.set_num_threads(1)
    p = fake_dump(tmp_path / "a", n_games=12)
    data = pp.Data([pp.Source("a", "teacher", "split", str(p))], ks=(8,), eval_frac=0.25, val_frac=0.34)
    rows, val = data.pair_rows(8, "train"), data.pair_rows(8, "val")
    st = data.stats(rows, 8)
    info = {}
    net, _ = pp.train_one(data, "A", 8, rows, st, steps=300, batch=64, lr=3e-2, seed=1, val_rows=val, val_every=10, patience=3,
                          split_nets=True, score_wd=0.01, info=info)
    v = info["val"]
    assert set(v["best"]) == {"world", "score"}
    for g in ("world", "score"):
        seen = []
        for c in v["curve"]:
            seen.append((c[g], c["step"]))
            if c["step"] == v["best"][g]["step"] + 3 * 10 and v["best"][g]["stopped"]:
                break
        assert min(seen)[1] == v["best"][g]["step"] and min(seen)[0] == v["best"][g]["loss"]
    assert any(e["stopped"] for e in v["best"].values()), v["best"]
    got = _val_loss(net, data, val, st)
    assert got["world"] == pytest.approx(v["best"]["world"]["loss"], rel=1e-6) and got["score"] == pytest.approx(v["best"]["score"]["loss"], rel=1e-6)
    assert v["last_step"] <= 300 and isinstance(net, pp.Split)
    one, _ = pp.train_one(data, "A", 8, rows, st, steps=40, batch=64, lr=1e-3, seed=1, val_rows=val, val_every=10, patience=4, split_nets=True, score_wd=0.0)
    two, _ = pp.train_one(data, "A", 8, rows, st, steps=40, batch=64, lr=1e-3, seed=1, val_rows=val, val_every=10, patience=4, split_nets=True, score_wd=0.5)
    assert _same(one.world, two.world) and not _same(one.score, two.score)
    single = {}
    pp.train_one(data, "A", 8, rows, st, steps=60, batch=64, lr=3e-2, seed=1, val_rows=val, val_every=10, patience=2, info=single)
    curve = [(c["world"] + c["score"], c["step"]) for c in single["val"]["curve"]]
    assert set(single["val"]["best"]) == {"all"} and min(curve)[1] == single["val"]["best"]["all"]["step"]
    pp.save(net, st, tmp_path / "s.pt", {})
    back, _, _ = pp.load(tmp_path / "s.pt", data)
    assert isinstance(back, pp.Split) and _same(back, net)


def _bot_results(dump: Path, plan: Path) -> Path:
    games = json.loads((dump / "games.json").read_text())
    rows = []
    (plan / "replays").mkdir(parents=True)
    for k, e in games.items():
        n = int(k) + 1
        data = f"{plan.name} {n}".encode()
        (plan / "replays" / f"g{n}.rep").write_bytes(data)
        e["sha"] = hashlib.sha256(data).hexdigest()
        early = n % 3 == 0
        rows.append({"game_id": n, "slot": 0, "won": e["result"] > 0 if n % 5 else None, "end_frame": 9000 if early else 14401, "reason": "x", "error": None,
                     "model": {"scv_8m": None if early else (10 if n % 2 else 4), "scv_min_1m": 2 + n % 6}})
    (dump / "games.json").write_text(json.dumps(games))
    (plan / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return plan / "results.jsonl"


def test_053_plan_s0_with_labels_mix_validation_and_split_nets(tmp_path):
    from uranium.__main__ import main

    a = fake_dump(tmp_path / "t", n_games=30, seed=1)
    m = fake_dump(tmp_path / "m", n_games=30, seed=2, sha_prefix="b")
    h = fake_dump(tmp_path / "h", n_games=30, seed=3, sha_prefix="c")
    ra, rm = _bot_results(a, tmp_path / "pa"), _bot_results(m, tmp_path / "pm")
    heads = "held_8m,held_s,alive_10m,win"
    base = ["diag", "plan-s0", "--src", f"teach:teacher:split={a}", "--src", f"mod:model:split={m}", "--src", f"hum:human:train={h}", "--ks", "8",
            "--variants", "U,A", "--no-m3c", "--batch", "64", "--lr", "3e-3", "--boot", "20", "--threads", "1", "--m3b-window", "0,1e9", "--q-heads", heads]
    with pytest.raises(SystemExit, match="session results"):
        main([*base, "--out", str(tmp_path / "x")])
    args = [*base, "--results", f"teach={ra}", "--side", "teach=model", "--results", f"mod={rm}", "--mix", "human=0.2", "--val-frac", "0.25",
            "--val-every", "10", "--patience", "2", "--max-steps", "40", "--split-nets"]
    out = tmp_path / "o"
    assert main([*args, "--out", str(out)]) == 0
    cov = json.loads((out / "coverage.json").read_text())
    assert cov["teach"]["games"] == 30 and cov["teach"]["held_8m"]["labeled"] == 30 - 2 and cov["teach"]["win"]["labeled"] == 30 - 6
    assert cov["hum"]["held_s"]["labeled"] == 0 and cov["hum"]["win"]["labeled"] == 30 and cov["teach"]["source"]["join"]["results"] == 30
    assert cov["teach"]["val_games"] > 0 and cov["hum"]["val_games"] > 0
    mix = json.loads((out / "mix.json").read_text())
    assert set(mix) == {"U_k8", "A_k8"} and abs(mix["A_k8"]["kinds"]["human"]["drawn"] - 0.2) < 0.03
    assert "Build_Bunker" in mix["A_k8"]["op_actions"]
    v = json.loads((out / "val" / "A_k8.json").read_text())
    assert set(v["best"]) == {"world", "score"} and v["curve"][0]["step"] == 10
    rep = json.loads((out / "s0.json").read_text())
    assert rep["settings"]["q_heads"] == heads.split(",") and set(rep["verdict"]) == set(heads.split(","))
    assert rep["settings"]["training"]["split_nets"] == {"score_wd": 0.01} and rep["coverage"]["mod"]["games"] == 30
    assert set(rep["M1"]["8"]["pooled"]["all"]) >= set(heads.split(",")) and "val" in rep["models"]["A_k8"]
    srcs = [pp.Source("teach", "teacher", "split", str(a)), pp.Source("mod", "model", "split", str(m)), pp.Source("hum", "human", "train", str(h))]
    d = pp.Data(srcs, ks=(8,), val_frac=0.25)
    ev = d.pair_rows(8, "eval")
    assert rep["M1"]["8"]["pooled"]["all"]["_rows"] == len(ev) and not d.row_val[ev].any() and d.row_eval[ev].all()
    assert len(d.pair_rows(8, "val")) > 0 and rep["M2"]["8"]["rows"] == int(d.is_op[ev].sum())
    ck = torch.load(out / "models" / "A_k8.pt", weights_only=False)
    assert ck["split"] and ck["q_names"] == heads.split(",") and ck["meta"]["training"]["mix"]["shares"] == {"human": 0.2}
    md = (out / "s0.md").read_text(encoding="utf-8")
    assert "coverage" in md and "행 섞기" in md and "검증·조기 멈춤" in md and "held_8m" in md
    assert main([*args, "--out", str(out), "--stage", "report"]) == 0
    rep2 = json.loads((out / "s0.json").read_text())
    rep.pop("secs"), rep2.pop("secs")
    assert rep == rep2
    with pytest.raises(SystemExit, match="mix / validation"):
        main([*args[:-1], "--out", str(out), "--stage", "report"])
    lab = tmp_path / "teach_labels.json"
    assert main(["diag", "plan-label", "--dump", str(a), "--kind", "teacher", "--side", "model", "--results", str(ra), "--out", str(lab)]) == 0
    args3 = [x if x != f"teach={ra}" else f"teach={lab}" for x in args]
    i = args3.index(f"teach={lab}")
    args3[i - 1] = "--labels"
    assert main([*args3, "--out", str(tmp_path / "o3"), "--stage", "train"]) == 0
    for f in ("U_k8.pt", "A_k8.pt"):
        x, y = torch.load(out / "models" / f, weights_only=False), torch.load(tmp_path / "o3" / "models" / f, weights_only=False)
        assert all(torch.equal(p, q) for p, q in zip(x["state"].values(), y["state"].values()))
