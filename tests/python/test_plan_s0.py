"""jepa_planning S0 feasibility tooling (reports/jepa_planning.md 6; 050-R9 diagnostics, 032-R24 aux outputs) on
CPU: `diag plan-dump` row schema, its agreement with the model's own functions and determinism, `diag plan-s0`
predictor training determinism and the metric math (game bootstrap, weighted AUC, R², Spearman, M2b direction)
on known inputs, and the chain on synthetic shards with a tiny model 2 net."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from uranium import _core, ckpt
from uranium.diag import plan_pred as pp
from uranium.diag import plan_stats as ps
from uranium.diag import probe
from uranium.diag import tables as tb
from uranium.diag.plan_dump import load, load_rows, model_rows, op_actions, own_columns, own_state
from uranium.model import UraniumNet, UraniumNetConfig, action_columns, input_dims
from uranium.model.config import VariantConfig

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "burn_parity"


def tiny_model2(path: Path, outer: int = 2) -> Path:
    """The tiny fixture net with the model 2 switches (outer LSTM, bypass, core_skip, aux_priv, aux_jepa 1/8/32),
    the bypass output and core_skip made non-zero as after training."""
    cfg = UraniumNetConfig.from_dict(json.loads((FIX / "tiny-gtrxl" / "meta.json").read_text())["config"])
    cfg.input = input_dims()
    cfg.variant = VariantConfig(outer=outer, bypass=True, core_skip=True, aux_priv=True, aux_jepa=[1, 8, 32])
    torch.manual_seed(5)
    net = UraniumNet(cfg, action_columns())
    g = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for m in (net.variant.bypass.out, net.variant.core_skip):
            for p in m.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    path.mkdir(parents=True)
    ckpt.save_model(path, net)
    ckpt.save_meta(path, cfg, {}, "f32")
    return path


@pytest.fixture
def shards(tmp_path: Path) -> Path:
    d = tmp_path / "shards"
    _core.write_test_shards(str(d), "varied", 5)
    return d


# ---------------------------------------------------------------- plan-dump


def test_model_rows_match_the_model_functions():
    """jepa_planning S0 / 050-R9: plan-dump's pass gives the `action_seq` log-probs and state, the 032-R24 JEPA
    target and g_k outputs of `forward_seq(aux=True)`, and win_open, on the model 2 path (outer LSTM, bypass,
    core_skip)."""
    from test_variants import SEG, net_of, perturb, tiny

    from uranium.model.auxiliary import jepa_target

    v = VariantConfig(outer=SEG, bypass=True, core_skip=True, aux_priv=True, aux_jepa=[1, 8, 32])
    cfg, b, t, obs, labels, priv = tiny(v)
    net = perturb(net_of(cfg))
    with torch.no_grad():
        for p in net.variant.core_skip.parameters():
            p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(5)) * 0.05)
        f = net.forward_seq(obs, labels, priv, net.init_state(b), t, aux=True)
        _, lp = net.action_seq(obs, net.init_state(b), t)
        o = model_rows(net, obs, net.init_state(b), t, [8, 32, 5])
    assert torch.equal(o["logp"], lp) and torch.equal(o["logp"], f["dists"]["action"])
    assert torch.equal(o["y"], jepa_target(f["jepa_target"]))
    assert set(o["g"]) == {8, 32} and all(torch.equal(o["g"][k], f["aux_jepa"][k]) for k in (8, 32))
    assert torch.equal(o["win_open"], f["win_open"])
    for x, y in zip(torch.utils._pytree.tree_leaves(o["state"]), torch.utils._pytree.tree_leaves(f["state"])):
        assert torch.equal(x, y)


def test_own_columns_and_state():
    names, types, of_action = own_columns()
    assert names[-3:] == ["supply_used", "supply_total", "army"] and "bunker" in names and "siege_tank" in names
    assert names[of_action[tb.action_id("Build_Bunker")]] == "bunker" and names[of_action[tb.action_id("Train_Marine")]] == "marine"
    assert tb.action_id("Train_NuclearMissile") not in of_action
    sc = np.zeros((2, 3000), np.float32)
    d, bld = tb.sc_own_cols(tb.unit_id("Terran_Marine"))
    sc[0, d], sc[0, bld] = tb.encode_count(3), tb.encode_count(2)
    sc[0, tb.sc_own_cols(tb.unit_id("Terran_Siege_Tank_Siege_Mode"))[0]] = tb.encode_count(1)
    sc[0, 4], sc[0, 5] = 18 / 400, 20 / 400
    o = own_state(sc)
    assert o[0, names.index("marine")] == 5 and o[0, names.index("siege_tank")] == 1 and o[0, names.index("army")] == 6
    assert o[0, names.index("supply_used")] == 18 and o[0, names.index("supply_total")] == 20 and not o[1].any()
    ops = set(op_actions())
    assert tb.action_id("Build_Bunker") in ops and tb.action_id("Lift") in ops and tb.action_id("Cancel_Train") in ops
    assert tb.action_id("Move_Pos") not in ops and tb.action_id("SetRally_Pos") not in ops and len(ops) == 66


def dump_args(ck, shards, tiny_config, out, *extra):
    return ["diag", "plan-dump", "--checkpoint", str(ck), "--config", str(tiny_config), "--backend", "cpu", "--synthetic-maps",
            "--shards", str(shards), "--splits", "train,val,test", "--out", str(out), *extra]


def test_plan_dump_schema_and_determinism(tiny_config, shards, tmp_path):
    """jepa_planning S0 6.3: every real step a row with the listed fields; per game a contiguous decision index;
    two runs give the same bytes; the policy log-probs are those of `action_seq` (050-R9 path)."""
    from uranium.__main__ import main

    ck = tiny_model2(tmp_path / "ck")
    assert main(dump_args(ck, shards, tiny_config, tmp_path / "d1", "--row-until", "1e9", "--read-until", "0")) == 0
    assert main(dump_args(ck, shards, tiny_config, tmp_path / "d2", "--row-until", "1e9", "--read-until", "0")) == 0
    R, games, man = load(tmp_path / "d1")
    n = len(R["frame"])
    assert man["games"] == 15 and n == man["stats"]["rows"] == sum(e["rows"] for e in games.values()) and n > 0
    assert man["gk"] == [8, 32] and man["own_columns"] == own_columns()[0] and man["op_actions"] == op_actions()
    want = {"game", "split", "dec", "frame", "h", "y", "priv", "priv_ok", "own", "action", "delay", "weight", "logp", "mask", "win_open", "g8", "g32"}
    assert set(R) == want
    net, _ = ckpt.load_model(ck, torch.device("cpu"))
    assert R["h"].shape == (n, net.cfg.core.hidden) and R["y"].shape == (n, net.cfg.entity.d_model + net.cfg.spatial.pool)
    assert R["priv"].shape == (n, net.cfg.input.privileged) and R["mask"].shape == (n, (net.cfg.input.actions + 7) // 8)
    assert R["h"].dtype == np.float16 and R["own"].dtype == np.float32
    # LN target: per row mean 0, variance 1
    y = R["y"].astype(np.float64)
    assert np.abs(y.mean(1)).max() < 1e-2 and np.abs(y.var(1) - 1).max() < 2e-2
    # decision index contiguous from 0 in every game, frames non-decreasing along it
    for g, e in games.items():
        m = R["game"] == g
        d = np.sort(R["dec"][m])
        assert np.array_equal(d, np.arange(m.sum())) and e["dec_rows"] == m.sum()
        fr = R["frame"][m][np.argsort(R["dec"][m])]
        assert (np.diff(fr) >= 0).all()
        assert e["sha"] and e["occ"] == 0 and e["result"] in (-1, 0, 1) and e["race"] in ("T", "P", "Z", "U")
    # the open types and the masked log-probs agree; every row's policy sums to 1
    mask = np.unpackbits(R["mask"], axis=1, count=net.cfg.input.actions).astype(bool)
    p = np.where(mask, np.exp(R["logp"].astype(np.float64)), 0.0)
    assert np.allclose(p.sum(1), 1.0, atol=5e-3) and mask.any(1).all()
    assert np.isfinite(R["win_open"]).all() == (net.value.open is not None)
    # same bytes on a second run (CPU)
    for f in (tmp_path / "d1" / "rows").iterdir():
        assert f.read_bytes() == (tmp_path / "d2" / "rows" / f.name).read_bytes(), f.name
    assert (tmp_path / "d1" / "games.json").read_text() == (tmp_path / "d2" / "games.json").read_text()
    # a finished folder is not overwritten
    with pytest.raises(SystemExit, match="finished"):
        main(dump_args(ck, shards, tiny_config, tmp_path / "d1"))


def test_plan_dump_rows_equal_action_seq_and_read_until(tiny_config, shards, tmp_path):
    """The dumped h' / log-probs are the `action_seq` values of the same stream walk; `--read-until` keeps the
    rows before `--row-until` (the same per-game rows, another order)."""
    from uranium.__main__ import main

    ck = tiny_model2(tmp_path / "ck")
    assert main(dump_args(ck, shards, tiny_config, tmp_path / "whole", "--row-until", "50", "--read-until", "0")) == 0
    assert main(dump_args(ck, shards, tiny_config, tmp_path / "cut", "--row-until", "50")) == 0
    R0, g0, m0 = load(tmp_path / "whole")
    R1, g1, m1 = load(tmp_path / "cut")
    assert m1["read_until"] == 50 and m0["read_until"] is None and m0["stats"]["rows"] == m1["stats"]["rows"] > 0
    assert (R0["frame"] < 50).all()
    k0 = {(g0[int(g)]["sha"], int(d)): i for i, (g, d) in enumerate(zip(R0["game"], R0["dec"]))}
    k1 = {(g1[int(g)]["sha"], int(d)): i for i, (g, d) in enumerate(zip(R1["game"], R1["dec"]))}
    assert set(k0) == set(k1)
    i0 = np.array([k0[k] for k in sorted(k0)])
    i1 = np.array([k1[k] for k in sorted(k0)])
    for f in ("action", "frame", "own", "priv", "mask", "weight"):
        assert np.array_equal(R0[f][i0], R1[f][i1]), f
    assert np.allclose(R0["h"][i0].astype(np.float32), R1["h"][i1].astype(np.float32), atol=2e-3)
    # rows = action_seq of a fresh walk over the train split (game order of the dump)
    net, _ = ckpt.load_model(ck, torch.device("cpu"))
    net.eval()
    cfg = json.loads(_core.bc_config(str(tiny_config), "s1")[0])
    t = cfg["seq_len"]
    stream = _core.EvalStream(str(tiny_config), [str(shards)], split="train", synthetic_maps=True)
    from uranium.model import to_obs

    st = None
    got = []
    while (b := stream.next()) is not None:
        hg = b.group(0, cfg["lanes"], t, cfg["pad_entities"], net.cfg.heads.max_select)
        with torch.no_grad():
            s0 = net.init_state(hg["b"]) if st is None else net.mask_state(st, torch.as_tensor(np.asarray(hg["keep"]).astype(bool)))
            tr, lp = net.action_seq(to_obs(hg, torch.device("cpu")), s0, t)
        st = tr["state"]
        fr = np.asarray(hg["scalar"])[:, 0] * tb.FRAME_SCALE
        real = np.asarray(hg["real"]).astype(bool) & (fr < 50)
        got.append(lp[torch.as_tensor(np.nonzero(real)[0])].numpy())
    got = np.concatenate(got)
    tr_rows = R0["split"] == 0
    assert len(got) == tr_rows.sum()
    # one lane group holds every lane (tiny config: 2 lanes), so the rows come in the same order
    with np.errstate(over="ignore"):  # masked types: -inf in float16
        assert np.array_equal(R0["logp"][tr_rows], got.astype(np.float16))


# ---------------------------------------------------------------- metric math


def test_boot_mean_r2_auc_match_direct_values():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 20)
    x = rng.normal(size=len(groups))
    bt = ps.Boot(groups, 200, 1)
    pt, reps = bt.mean(x)
    assert np.isclose(pt, x.mean()) and reps.shape == (200,) and reps.min() < pt < reps.max()
    # every draw = the mean over the resampled games' rows
    w = bt.W[3][bt.g]
    assert np.isclose(reps[3], (w * x).sum() / w.sum())
    # NaN rows left out
    xn = x.copy()
    xn[::7] = np.nan
    assert np.isclose(bt.mean(xn)[0], np.nanmean(xn))
    # R² with several dimensions
    y = rng.normal(size=(len(groups), 3))
    pred = y + 0.5 * rng.normal(size=y.shape)
    sse = ((pred - y) ** 2).sum(1)
    want = 1 - sse.sum() / ((y - y.mean(0)) ** 2).sum()
    assert np.isclose(bt.r2(sse, y)[0], want)
    w = bt.W[5][bt.g]
    yb = (w[:, None] * y).sum(0) / w.sum()
    assert np.isclose(bt.r2(sse, y)[1][5], 1 - (w * sse).sum() / (w[:, None] * (y - yb) ** 2).sum())
    # AUC (ties count 1/2) equals the unweighted probe AUC; a draw equals the AUC of the explicit resample
    lab = rng.random(len(groups)) < 0.4
    s = np.round(x + lab, 1)
    assert np.isclose(bt.auc(lab, s)[0], probe.auc(lab, s))
    sel = np.repeat(np.arange(len(groups)), bt.W[7][bt.g].astype(int))
    assert np.isclose(bt.auc(lab, s)[1][7], probe.auc(lab[sel], s[sel]))
    # Spearman: a monotone transform is 1, a reversal -1
    assert np.isclose(ps.spearman(x, np.exp(x)), 1.0) and np.isclose(ps.spearman(x, -x ** 3), -1.0)
    sp, sreps = bt.spearman(x, x + rng.normal(size=len(x)))
    assert 0.3 < sp < 1 and np.isfinite(sreps).all()


def test_paired_diff_ratio_and_bce():
    a = (1.0, np.array([0.8, 1.2, 1.0]))
    b = (0.5, np.array([0.4, 0.5, 0.7]))
    d = ps.diff(a, b)
    assert d["pt"] == 0.5 and np.isclose(d["lo"], np.percentile([0.4, 0.7, 0.3], 2.5))
    assert ps.ratio(a, b)["pt"] == 2.0
    z = np.array([-3.0, 0.0, 2.0])
    y = np.array([0.0, 1.0, 1.0])
    want = -(y * np.log(1 / (1 + np.exp(-z))) + (1 - y) * np.log(1 - 1 / (1 + np.exp(-z))))
    assert np.allclose(ps.bce(z, y), want)


# ---------------------------------------------------------------- synthetic plan-dump folders


def fake_dump(path: Path, n_games=24, rows=60, seed=0, kind_shift=0.0, d=16, sha_prefix="a"):
    """A plan-dump folder with known structure: the action type (0..3, 3 = Build_Bunker's own column) moves the
    future own bunker count and y; h carries the game's result."""
    from uranium.diag.plan_dump import RowWriter

    rng = np.random.default_rng(seed)
    names, _, of_action = own_columns()
    bunker = tb.action_id("Build_Bunker")
    acts = np.array([tb.action_id("Move_Pos"), tb.action_id("Train_SCV"), tb.action_id("Gather_Unit"), bunker])
    n_act = 101
    schema = {"game": ("int32", ()), "split": ("int8", ()), "dec": ("int32", ()), "frame": ("float32", ()), "h": ("float16", (d,)), "y": ("float16", (d,)),
              "priv": ("float16", (5,)), "priv_ok": ("uint8", ()), "own": ("float32", (len(names),)), "action": ("int16", ()), "delay": ("int8", ()),
              "weight": ("float32", ()), "logp": ("float16", (n_act,)), "mask": ("uint8", ((n_act + 7) // 8,)), "win_open": ("float32", ()), "g8": ("float16", (d,))}
    path.mkdir(parents=True)
    w = RowWriter(path, schema)
    games = {}
    for g in range(n_games):
        res = 1 if g % 2 else -1
        a = acts[rng.integers(0, 4, rows)]
        cum = 3.0 * (np.cumsum(a == bunker) - (a == bunker))  # built before t: a_t moves the count from t + 1 on
        own = np.zeros((rows, len(names)), np.float32)
        own[:, of_action[bunker]] = cum
        own[:, names.index("scv")] = 3.0 * (np.cumsum(a == acts[1]) - (a == acts[1]))
        h = rng.normal(size=(rows, d)).astype(np.float32)
        h[:, 0] = res + kind_shift
        h[:, 1] = cum
        y = rng.normal(size=(rows, d))
        y[:, 2] = cum * 0.5
        y = (y - y.mean(1, keepdims=True)) / y.std(1, keepdims=True)
        m = np.zeros((rows, n_act), bool)
        m[:, acts] = True
        lp = np.full((rows, n_act), -np.inf)
        lp[:, acts] = np.log([0.4, 0.3, 0.2, 0.1])
        w.add({"game": np.full(rows, g), "split": np.zeros(rows), "dec": np.arange(rows), "frame": 2880 + 24.0 * np.arange(rows), "h": h, "y": y,
               "priv": rng.normal(size=(rows, 5)) + h[:, :1], "priv_ok": np.ones(rows), "own": own, "action": a, "delay": rng.integers(0, 16, rows),
               "weight": np.ones(rows), "logp": lp, "mask": np.packbits(m, axis=1), "win_open": h[:, 0] * 0.1, "g8": y})
        games[g] = {"split": "val", "race": "Z", "sha": f"{sha_prefix}{g:063x}", "occ": 0, "result": res, "rows": rows, "dec_rows": rows, "scv8": int(g % 3 == 0),
                    "cls": "rush" if g % 4 < 2 else "macro", "cls_given": None}
    w.close()
    (path / "games.json").write_text(json.dumps(games))
    (path / "manifest.json").write_text(json.dumps({"checkpoint": "x", "checkpoint_sha256": "s", "own_columns": names, "op_actions": op_actions()}))
    return path


def test_data_pairs_and_split(tmp_path):
    p = fake_dump(tmp_path / "a")
    data = pp.Data([pp.Source("a", "teacher", "split", str(p))], ks=(8,), eval_frac=0.25, split_seed=1)
    assert data.N == 24 * 60 and data.G == 24 and data.g_role_eval.sum() == 6
    rows = load_rows(p)
    for r in (0, 100, 1000):
        n = data.nxt[8][r]
        if data.dec[r] + 8 < 60:
            assert data.game[n] == data.game[r] and data.dec[n] == data.dec[r] + 8
        else:
            assert n == -1
    assert np.array_equal(data.h, np.asarray(rows["h"]))
    # the oracle input: action shares over [t, t+8)
    r = np.array([5])
    hist = data.inputs(r, 8, "H")["hist"].numpy()[0]
    want = np.bincount(data.action[5:13].astype(int), minlength=101) / 8
    assert np.allclose(hist, want)
    # the control: an action of the same game
    assert (data.game[data.partner] == data.game).all()
    # the same split for a second load; a different seed moves it
    d2 = pp.Data([pp.Source("a", "teacher", "split", str(p))], ks=(8,), eval_frac=0.25, split_seed=1)
    d3 = pp.Data([pp.Source("a", "teacher", "split", str(p))], ks=(8,), eval_frac=0.25, split_seed=2)
    assert np.array_equal(d2.g_role_eval, data.g_role_eval) and not np.array_equal(d3.g_role_eval, data.g_role_eval)
    # dropped streams leave
    d4 = pp.Data([pp.Source("a", "teacher", "split", str(p))], ks=(8,), drop={"a": [[f"a{0:063x}", 0]]})
    assert d4.G == 23 and d4.N == 23 * 60


def test_predictor_training_is_deterministic_and_q_all_matches_forward(tmp_path):
    p = fake_dump(tmp_path / "a")
    data = pp.Data([pp.Source("a", "teacher", "split", str(p))], ks=(8,))
    rows = data.pair_rows(8, "train")
    st = data.stats(rows)
    outs = []
    for _ in range(2):
        net, log = pp.train_one(data, "A", 8, rows, st, steps=20, batch=64, lr=1e-3, seed=3, log_every=10)
        outs.append((net, log))
    for (a, la), (b, lb) in [outs]:
        assert la == lb and all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    other, _ = pp.train_one(data, "A", 8, rows, st, steps=20, batch=64, lr=1e-3, seed=4, log_every=10)
    assert not torch.equal(other.l1.weight, outs[0][0].l1.weight)
    net = outs[0][0]
    ev = data.pair_rows(8, "eval")[:50]
    h = torch.from_numpy(data.h[ev].astype(np.float32))
    cand = torch.arange(data.n_act)
    qa = net.q_all(h, cand)
    with torch.no_grad():
        for c in (0, tb.action_id("Build_Bunker"), 77):
            q = net(h, torch.full((len(ev),), c))["q"]
            assert torch.allclose(qa[:, c], q, atol=1e-5)
    # save / load round trip
    pp.save(net, st, tmp_path / "m.pt", {"x": 1})
    n2, st2, meta = pp.load(tmp_path / "m.pt", data)
    assert meta == {"x": 1} and all(torch.equal(x, y) for x, y in zip(net.state_dict().values(), n2.state_dict().values()))


def test_m2b_direction_and_candidates(tmp_path):
    """M2b counts a Build_X candidate whose own-X prediction beats every other candidate's."""
    from uranium.diag.plan_s0 import candidates

    p = fake_dump(tmp_path / "a")
    data = pp.Data([pp.Source("a", "teacher", "eval", str(p))], ks=(8,))
    rows = np.arange(10)
    c = candidates(data, rows)
    # top open types by probability, then a_t when it is not among them (4 open types here: always among them)
    assert (c[:, :4] == np.array([tb.action_id(x) for x in ("Move_Pos", "Train_SCV", "Gather_Unit", "Build_Bunker")])).all()
    assert (c[:, 4:] == -1).all()


def test_plan_s0_end_to_end_on_known_structure(tmp_path):
    """The chain on synthetic folders: training is repeatable, the report has every table and the verdict; the
    action-conditioned predictor wins on the bunker-driven targets and M2b points the right way."""
    from uranium.__main__ import main

    a = fake_dump(tmp_path / "t", n_games=40, seed=1)
    m = fake_dump(tmp_path / "m", n_games=40, seed=2, sha_prefix="b")
    h = fake_dump(tmp_path / "h", n_games=40, seed=3, sha_prefix="c")
    args = ["diag", "plan-s0", "--src", f"teach:teacher:split={a}", "--src", f"mod:model:split={m}", "--src", f"hum:human:train={h}", "--ks", "8",
            "--steps", "150", "--batch", "128", "--lr", "3e-3", "--boot", "50", "--threads", "1", "--m3b-window", "0,1e9"]
    assert main([*args, "--out", str(tmp_path / "o1")]) == 0
    assert main([*args, "--out", str(tmp_path / "o2"), "--stage", "train"]) == 0
    for f in (tmp_path / "o1" / "models").iterdir():
        a1 = torch.load(f, weights_only=False)
        a2 = torch.load(tmp_path / "o2" / "models" / f.name, weights_only=False)
        assert all(torch.equal(x, y) for x, y in zip(a1["state"].values(), a2["state"].values())), f.name
    assert {f.name for f in (tmp_path / "o1" / "models").iterdir()} == {f"{v}_k8.pt" for v in ("U", "A", "Ad", "H", "S", "A-human", "A-model")}
    rep = json.loads((tmp_path / "o1" / "s0.json").read_text())
    assert {"M0", "M1", "M2", "M3a", "M3b", "M3d", "verdict"} <= set(rep)
    op = rep["M1"]["8"]["pooled"]["op"]
    assert "r" in op["y"] and "gk" in op["y"]["gain"] and set(op) == {"y", "p", "o", "win", "scv8", "_rows"}
    # Build_Bunker moves the future bunker count: its candidate predicts the most bunkers
    assert rep["M2"]["8"]["M2b"]["pairs"] > 0 and rep["M2"]["8"]["M2b"]["build"]["rate"]["pt"] > 0.8, rep["M2"]["8"]["M2b"]
    assert rep["M3b"]["8"]["A"]["model"]["rows"] > 0 and "A-human" in rep["M3b"]["8"]
    assert set(rep["verdict"]) == {"win", "scv8"} and len(rep["verdict"]["win"]["rows"]) == 6
    assert rep["settings"]["thresholds"]["m2b"] == 0.9
    md = (tmp_path / "o1" / "s0.md").read_text(encoding="utf-8")
    assert "판정표" in md and "M2 구별" in md
    # report again from the saved models: the same JSON (deterministic)
    assert main([*args, "--out", str(tmp_path / "o1"), "--stage", "report"]) == 0
    rep2 = json.loads((tmp_path / "o1" / "s0.json").read_text())
    rep.pop("secs"), rep2.pop("secs")
    assert rep == rep2
