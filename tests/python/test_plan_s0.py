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


def test_plan_dump_skip_views(tiny_config, shards, tmp_path):
    """jepa_planning S0 second run: `--skip-views` writes no rows for the "bot" streams of a views JSON (games.json
    `skipped`), skips the forward of a lane group holding only such streams, and the other streams' rows are
    those of a full dump."""
    from uranium.__main__ import main

    ck = tiny_model2(tmp_path / "ck")
    common = ("--row-until", "1e9", "--read-until", "0", "--lanes-per-group", "1")
    assert main(dump_args(ck, shards, tiny_config, tmp_path / "full", *common)) == 0
    R0, g0, _ = load(tmp_path / "full")
    bot = [[e["sha"], e["start"]] for gid, e in sorted(g0.items()) if gid % 2 == 0]
    views = tmp_path / "views.json"
    views.write_text(json.dumps({"bot": bot, "model": [[e["sha"], e["start"]] for gid, e in sorted(g0.items()) if gid % 2 == 1]}))
    assert main(dump_args(ck, shards, tiny_config, tmp_path / "skip", *common, "--skip-views", str(views))) == 0
    R1, g1, m1 = load(tmp_path / "skip")
    skipped = {(e["sha"], e["start"]) for e in g1.values() if e.get("skipped")}
    assert skipped == {tuple(x) for x in bot} and all(e["rows"] == 0 for e in g1.values() if e.get("skipped"))
    assert m1["stats"]["skipped_views"] == len(bot) and m1["stats"]["skipped_groups"] > 0
    key = lambda R, g: {(g[int(x)]["sha"], g[int(x)]["start"], int(d)): i for i, (x, d) in enumerate(zip(R["game"], R["dec"]))}  # noqa: E731
    k0, k1 = key(R0, g0), key(R1, g1)
    assert set(k1) == {k for k in k0 if (k[0], k[1]) not in skipped} and len(k1) > 0
    i0 = np.array([k0[k] for k in sorted(k1)])
    i1 = np.array([k1[k] for k in sorted(k1)])
    for f in ("action", "frame", "own", "priv", "mask", "weight"):
        assert np.array_equal(R0[f][i0], R1[f][i1]), f
    assert np.array_equal(R0["h"][i0], R1["h"][i1]) and np.array_equal(R0["logp"][i0], R1["logp"][i1])
    for e in g1.values():
        if not e.get("skipped"):
            e0 = next(x for x in g0.values() if (x["sha"], x["start"]) == (e["sha"], e["start"]))
            assert e["rows"] == e0["rows"] and e["dec_rows"] == e0["dec_rows"] and e["scv8"] == e0["scv8"]


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


def fake_dump(path: Path, n_games=24, rows=60, seed=0, kind_shift=0.0, d=16, sha_prefix="a", race="Z", void_every=0, chain_every=0):
    """A plan-dump folder with known structure: the action type (0..3, 3 = Build_Bunker's own column) moves the
    future own bunker count and y; h carries the game's result. Game g: start slot g % 4, map family g % 3;
    `void_every`: every n-th row has a void label (weight 0); `chain_every`: row i with i % n == n - 1 is observed
    1 frame after row i - 1 (a 040-R4 chained row; rows are 24 frames apart otherwise)."""
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
        frames = 2880 + 24.0 * np.arange(rows)
        if chain_every:
            ch = np.arange(rows) % chain_every == chain_every - 1
            frames[ch] = frames[np.nonzero(ch)[0] - 1] + 1
        w.add({"game": np.full(rows, g), "split": np.zeros(rows), "dec": np.arange(rows), "frame": frames, "h": h, "y": y,
               "priv": rng.normal(size=(rows, 5)) + h[:, :1], "priv_ok": np.ones(rows), "own": own, "action": a, "delay": rng.integers(0, 16, rows),
               "weight": (np.arange(rows) % void_every != 0) if void_every else np.ones(rows), "logp": lp, "mask": np.packbits(m, axis=1), "win_open": h[:, 0] * 0.1, "g8": y})
        games[g] = {"split": "val", "race": race, "sha": f"{sha_prefix}{g:063x}", "occ": 0, "start": g % 4, "map": g % 3, "result": res, "rows": rows, "dec_rows": rows, "scv8": int(g % 3 == 0),
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
    # the control (second run): an action of another game of the source; the first run: of the same game
    assert (data.game[data.partner] != data.game).all() and data.partner_same_game == 0
    v1 = pp.Data([pp.Source("a", "teacher", "split", str(p))], ks=(8,), eval_frac=0.25, split_seed=1, design=pp.V1)
    assert (v1.game[v1.partner] == v1.game).all()
    # the same split for a second load; a different seed moves it
    d2 = pp.Data([pp.Source("a", "teacher", "split", str(p))], ks=(8,), eval_frac=0.25, split_seed=1)
    d3 = pp.Data([pp.Source("a", "teacher", "split", str(p))], ks=(8,), eval_frac=0.25, split_seed=2)
    assert np.array_equal(d2.g_role_eval, data.g_role_eval) and not np.array_equal(d3.g_role_eval, data.g_role_eval)
    # map families
    d5 = pp.Data([pp.Source("a", "teacher", "split", str(p))], ks=(8,), maps=[0, 1])
    assert d5.G == 16 and d5.dropped["a"] == {"bot_view": 0, "map": 8}


def test_bot_views_and_split_digest(tmp_path):
    """jepa_planning S0: a model source with Terran-enemy streams needs the view table (bot views out by replay
    sha256 + own start slot, every stream listed) or --keep-bot-perspective; the split digest names the games and roles."""
    p = fake_dump(tmp_path / "m", race="T")
    src = [pp.Source("m", "model", "split", str(p))]
    with pytest.raises(SystemExit, match="Terran-enemy"):
        pp.Data(src, ks=(8,))
    keep = pp.Data(src, ks=(8,), keep_bot=True)
    assert keep.G == 24
    views = {"bot": [[f"a{g:063x}", g % 4] for g in range(0, 24, 2)], "model": [[f"a{g:063x}", g % 4] for g in range(1, 24, 2)]}
    d = pp.Data(src, ks=(8,), drop={"m": views})
    assert d.G == 12 and d.dropped["m"]["bot_view"] == 12 and all(e["gid"] % 2 == 1 for e in d.game_meta)
    # a stream in neither list (another start slot) is refused
    bad = {"bot": views["bot"], "model": [[sha, (st + 1) % 4] for sha, st in views["model"]]}
    with pytest.raises(SystemExit, match="neither"):
        pp.Data(src, ks=(8,), drop={"m": bad})
    # the digest moves with the split seed and the games kept
    assert pp.Data(src, ks=(8,), drop={"m": views}).split_digest == d.split_digest
    assert pp.Data(src, ks=(8,), drop={"m": views}, split_seed=2).split_digest != d.split_digest
    assert keep.split_digest != d.split_digest


def test_chained_rows_dropped_for_teacher_and_model(tmp_path):
    """jepa_planning S0 second run: a row whose previous decision is a command 1 frame earlier (040-R4 same-frame
    chain) is not an example in teacher / model sources (human rows stay); it stays a t+k target; V1 keeps all."""
    t = fake_dump(tmp_path / "t", chain_every=5)
    h = fake_dump(tmp_path / "h", chain_every=5, sha_prefix="c", seed=1)
    src = [pp.Source("t", "teacher", "split", str(t)), pp.Source("h", "human", "split", str(h))]
    data = pp.Data(src, ks=(8,))
    want = (data.dec % 5 == 4)  # rows 4, 9, ... of every game: 1 frame after a command
    assert np.array_equal(data.chained, want)
    is_t = data.row_kind == pp.KINDS.index("teacher")
    assert np.array_equal(data.chain_drop, want & is_t) and not (data.chain_drop & ~is_t).any()
    assert data.chain_counts["t"]["dropped"] == data.chain_counts["t"]["chained_valid"] == int((want & is_t).sum())
    assert data.chain_counts["h"]["chained_valid"] > 0 and data.chain_counts["h"]["dropped"] == 0
    rows = np.concatenate([data.pair_rows(8, "train"), data.pair_rows(8, "eval")])
    assert not data.chain_drop[rows].any() and (want[rows] & ~is_t[rows]).any()
    assert data.chain_drop[data.nxt[8][rows]].any()  # still a target
    # the first run keeps every valid row
    v1 = pp.Data(src, ks=(8,), design=pp.V1)
    assert not v1.chain_drop.any() and len(v1.pair_rows(8, "train")) > len(data.pair_rows(8, "train"))
    keep = pp.Data(src, ks=(8,), design=pp.Design(chains="keep"))
    assert np.array_equal(keep.usable, keep.valid)


def test_own_input_delta_targets_and_spaces(tmp_path):
    """jepa_planning S0 second run: o_t standardised as an input, the ô target o_{t+k} - o_t in its own
    standardisation, and `own_spaces` maps a prediction to the level and change spaces consistently."""
    p = fake_dump(tmp_path / "a")
    data = pp.Data([pp.Source("a", "teacher", "split", str(p))], ks=(8,))
    rows = data.pair_rows(8, "train")
    st = data.stats(rows, 8)
    b = data.batch(rows[:64], 8, "A", st)
    t = data.nxt[8][rows[:64]]
    bunker = own_columns()[2][tb.action_id("Build_Bunker")]
    want = (data.own[t, bunker] - data.own[rows[:64], bunker] - st["d_mu"][bunker]) / st["d_sd"][bunker]
    assert np.allclose(b["o"][:, bunker].numpy(), want, atol=1e-5)
    assert np.allclose(b["own"][:, bunker].numpy(), (data.own[rows[:64], bunker] - st["o_mu"][bunker]) / st["o_sd"][bunker], atol=1e-5)
    marine = own_columns()[0].index("marine")  # constant: 0 in both
    assert (b["own"][:, marine] == 0).all() and (b["o"][:, marine] == 0).all()
    # the true change, mapped to both spaces, gives the level and change targets
    lvl, dlt = pp.own_spaces(data, b["o"].numpy(), rows[:64], st)
    on = np.isfinite(st["o_sd"])
    assert np.allclose(dlt, b["o"].numpy(), atol=1e-5)
    assert np.allclose(lvl[:, on], ((data.own[t] - st["o_mu"]) / st["o_sd"])[:, on], atol=1e-4)
    # the level design: the same targets from the other side
    lv = pp.Data([pp.Source("a", "teacher", "split", str(p))], ks=(8,), design=pp.Design(own_target="level", own_input=False))
    bl = lv.batch(rows[:64], 8, "U", st)
    assert "own" not in bl
    l2, d2 = pp.own_spaces(lv, bl["o"].numpy(), rows[:64], st)
    assert np.allclose(l2, lvl, atol=1e-4) and np.allclose(d2, dlt, atol=1e-4)


def test_frame_k_pairs_and_oracle_window(tmp_path):
    """`--k-unit frame`: t+k is the first row of the game k frames or more later; the H oracle counts the
    decisions [t, t+k). On rows 24 frames apart, k = 48 frames is 2 decisions."""
    p = fake_dump(tmp_path / "a", chain_every=5)
    src = [pp.Source("a", "teacher", "split", str(p))]
    fr = pp.Data(src, ks=(48,), design=pp.Design(k_unit="frame"))
    dc = pp.Data(src, ks=(2,), design=pp.Design(k_unit="dec"))
    r = fr.pair_rows(48, "train")
    n = fr.nxt[48][r]
    assert (fr.game[n] == fr.game[r]).all() and (fr.frame[n] - fr.frame[r] >= 48).all()
    prev = fr.order[fr.pos[n] - 1]
    assert (fr.frame[prev] - fr.frame[r] < 48).all()
    same = (fr.dec[n] - fr.dec[r]) == 2
    assert same.mean() > 0.5
    rr = r[same & (dc.nxt[2][r] >= 0)]
    assert np.array_equal(fr.inputs(rr, 48, "H")["hist"].numpy(), dc.inputs(rr, 2, "H")["hist"].numpy())
    assert np.array_equal(fr.nxt_dec1, dc.nxt[1])


def test_partner_rows_have_valid_labels_and_sd_floors(tmp_path):
    p = fake_dump(tmp_path / "a", void_every=3)
    q = fake_dump(tmp_path / "b", void_every=3, sha_prefix="b", seed=1)
    v1 = pp.Data([pp.Source("a", "teacher", "split", str(p))], ks=(8,), design=pp.V1)
    assert (v1.game[v1.partner] == v1.game).all() and (v1.weight[v1.partner] > 0).all()
    # jepa_planning S0 second run: another game of the same source (no game identity), a usable row
    data = pp.Data([pp.Source("a", "teacher", "split", str(p)), pp.Source("b", "model", "split", str(q))], ks=(8,), keep_bot=True)
    assert (data.game[data.partner] != data.game).all() and data.usable[data.partner].all()
    assert (data.row_src[data.partner] == data.row_src).all()
    st = data.stats(data.pair_rows(8, "train"), 8)
    on = np.isfinite(st["o_sd"])
    assert (st["o_sd"][on] >= pp.OWN_SD_FLOOR).all() and not on[own_columns()[0].index("marine")]  # constant 0: left out
    assert (st["p_sd"][np.isfinite(st["p_sd"])] >= pp.PRIV_SD_FLOOR).all()
    d_on = np.isfinite(st["d_sd"])
    assert (st["d_sd"][d_on] >= pp.OWN_SD_FLOOR).all() and d_on[own_columns()[2][tb.action_id("Build_Bunker")]]


@pytest.mark.parametrize("design", [pp.Design(), pp.V1], ids=["second", "v1"])
def test_predictor_training_is_deterministic_and_q_all_matches_forward(tmp_path, design):
    p = fake_dump(tmp_path / "a")
    data = pp.Data([pp.Source("a", "teacher", "split", str(p))], ks=(8,), design=design)
    rows = data.pair_rows(8, "train")
    st = data.stats(rows, 8)
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
    own = data.own_in(ev, st) if design.own_input else None
    assert net.own_in == (data.own.shape[1] if design.own_input else 0)
    cand = torch.arange(data.n_act)
    qa = net.q_all(h, cand, own)
    with torch.no_grad():
        for c in (0, tb.action_id("Build_Bunker"), 77):
            q = net(h, torch.full((len(ev),), c), own=own)["q"]
            assert torch.allclose(qa[:, c], q, atol=1e-5)
    if design.own_input:
        with pytest.raises(ValueError, match="o_t"):
            net(h, torch.full((len(ev),), 0))
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
    action-conditioned predictor wins on the bunker-driven targets and M2b points the right way. jepa_planning S0
    second run: chained teacher / model rows are dropped (human kept), the own-state change metrics and the
    own-state criterion are reported; `--v1` is the first run's design and verdict."""
    from uranium.__main__ import main

    a = fake_dump(tmp_path / "t", n_games=40, seed=1, chain_every=7)
    m = fake_dump(tmp_path / "m", n_games=40, seed=2, sha_prefix="b", chain_every=7)
    h = fake_dump(tmp_path / "h", n_games=40, seed=3, sha_prefix="c", chain_every=7)
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
    assert "r" in op["y"] and "gk" in op["y"]["gain"] and set(op) == {"y", "p", "o", "o_all", "od", "od_all", "win", "scv8", "_rows", "_own_cols"}
    assert op["_own_cols"]["bunker"]["headline"] and "gain_A" in op["_own_cols"]["bunker"] and "cut_A" in op["_own_cols"]["bunker"]["delta"]
    # the action decides the bunker change: A cuts the change error, the other-game control does not
    assert op["od"]["cut"]["A"]["lo"] > 0.05 and op["od"]["cut"]["A"]["pt"] > op["od"]["cut"]["S"]["pt"] + 0.1 and op["od"]["cut"]["H"]["pt"] > 0.5, op["od"]["cut"]
    chains = {s["name"]: s["chains"] for s in rep["data"]["sources"]}
    assert chains["teach"]["dropped"] == chains["teach"]["chained_valid"] > 0 and chains["mod"]["dropped"] > 0
    assert chains["hum"]["chained_valid"] > 0 and chains["hum"]["dropped"] == 0
    assert rep["settings"]["design"] == pp.Design().as_dict() and rep["data"]["partner_same_game_rows"] == 0
    assert rep["verdict"]["win"]["own_k"] == 8 and "m1_own" in rep["verdict"]["win"]["criteria"]["8"]
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
    # models trained on another split, or with another design, are refused
    with pytest.raises(SystemExit, match="other data"):
        main([*args, "--out", str(tmp_path / "o1"), "--stage", "report", "--split-seed", "2"])
    with pytest.raises(SystemExit, match="other data"):
        main([*args, "--out", str(tmp_path / "o1"), "--stage", "report", "--v1"])
    with pytest.raises(SystemExit, match="--v1"):
        main([*args, "--out", str(tmp_path / "o3"), "--v1", "--chains", "keep"])
    with pytest.raises(SystemExit, match="th-own-k"):
        main([*args, "--out", str(tmp_path / "o3"), "--th-own-k", "32"])
    # the first run's design: every valid row, h' only, level targets, same-game partner, no own-state criterion
    assert main([*args, "--out", str(tmp_path / "o3"), "--v1"]) == 0
    r1 = json.loads((tmp_path / "o3" / "s0.json").read_text())
    assert r1["settings"]["design"] == pp.V1.as_dict() and r1["verdict"]["win"]["own_k"] is None
    assert all(s["chains"]["dropped"] == 0 for s in r1["data"]["sources"]) and r1["data"]["partner_same_game_rows"] > 0
    ck = torch.load(tmp_path / "o3" / "models" / "A_k8.pt", weights_only=False)
    assert ck["own_in"] == 0 and "design" not in ck["meta"]["data"]
    assert "od" in r1["M1"]["8"]["pooled"]["op"] and r1["M2"]["8"]["M2b"]["pairs"] > 0


def _m1(a, s, h, r):
    ci = lambda pt, lo, hi: {"pt": pt, "lo": lo, "hi": hi}  # noqa: E731
    return {"gain": {"A": ci(*a), "S": ci(*s), "H": ci(*h)}, "r": ci(r, r, r)}


def _report(*, a=(0.05, 0.02, 0.08), s=(0.01, -0.01, 0.03), h=(0.1, 0.05, 0.15), r=0.5, r_med=0.8, m2b=0.95, m3a=0.02, m3b=(0.01, 0.002, 0.02),
            hum=(-0.02, -0.03, -0.01), mod=(0.01, -0.005, 0.02), cut=((0.2, 0.1, 0.3), (0.0, -0.02, 0.02), (0.4, 0.3, 0.5))):
    ci = lambda t: {"pt": t[0], "lo": t[1], "hi": t[2]}  # noqa: E731
    m1 = {q: _m1(a, s, h, r) for q in ("y", "win", "scv8")}
    m1["od"] = {"cut": {"A": ci(cut[0]), "S": ci(cut[1]), "H": ci(cut[2])}}
    return {
        "M1": {"8": {"pooled": {"op": m1}}, "32": {"pooled": {"op": m1}}},
        "M2": {k: {"M2a_y": {"R_median": r_med}, "M2b": {"rate": {"pt": m2b}}} for k in ("8", "32")},
        "M3a": {k: {q: {"all": {"A_minus_U": ci((m3a, m3a - 0.01, m3a + 0.01))}} for q in ("win", "scv8")} for k in ("8", "32")},
        "M3b": {k: {"A": {"model": {q: {"rush_minus_macro": ci(m3b)} for q in ("win", "scv8")}},
                    "A-human": {"both": {q: {"rush_mean": ci(hum)} for q in ("win", "scv8")}},
                    "A-model": {"both": {q: {"rush_mean": ci(mod)} for q in ("win", "scv8")}}} for k in ("8", "32")},
    }


def test_verdict_rows():
    """6.5 table logic (thresholds 기본안(미확정)): True / False / None per row on crafted reports."""
    from uranium.diag.plan_s0 import TH_DEFAULTS, verdict

    th = dict(TH_DEFAULTS)
    rows = lambda rep: [r["matched"] for r in verdict(rep, th, [8, 32])["win"]["rows"]]  # noqa: E731
    # everything passes: planning possible; M3c human-only CI below 0, model-only reaches 0: delusion row
    assert rows(_report()) == [True, False, False, False, True, False]
    # M3b fails: "M1·M2 pass, M3 fails"
    assert rows(_report(m3b=(0.01, -0.002, 0.02)))[:2] == [False, True]
    # A's gain under twice the S control: M1 fails at both k
    assert rows(_report(s=(0.03, 0.0, 0.05)))[:2] == [False, False]
    # the oracle gains nothing: r undefined, M1 cannot be judged
    v = verdict(_report(h=(-0.01, -0.02, 0.0)), th, [8, 32])["win"]
    assert v["criteria"]["8"]["m1_y"] is None and v["rows"][0]["matched"] is None
    # A in the noise, H significant
    assert rows(_report(a=(0.001, -0.001, 0.003), r=0.01))[2] is True
    # neither A nor H significant
    assert rows(_report(a=(0.001, -0.001, 0.003), h=(0.001, -0.002, 0.004), r=1.0))[3] is True
    # model-only CI below 0 too: no delusion row; M2b < 0.9
    out = rows(_report(mod=(-0.02, -0.03, -0.01), m2b=0.5))
    assert out[4] is False and out[5] is True and out[0] is False
    # missing inputs: None
    rep = _report()
    del rep["M3b"]
    assert rows(rep)[0] is None and rows(rep)[4] is None


def test_verdict_own_state_criterion():
    """jepa_planning S0 second run (기본안(미확정)): M1 also needs A's own-state change-error cut at k = 8 (CI above 0,
    >= 2x the S control's cut); "A and H not significant" counts the cuts; own_k 0 is the first run's table."""
    from uranium.diag.plan_s0 import TH_DEFAULTS, verdict

    th = dict(TH_DEFAULTS, own_k=8)
    rows = lambda rep, t=th: [r["matched"] for r in verdict(rep, t, [8, 32])["win"]["rows"]]  # noqa: E731
    assert rows(_report()) == [True, False, False, False, True, False]
    v = verdict(_report(), th, [8, 32])["win"]
    assert v["own_pass"] is True and v["criteria"]["8"]["m1_own"] is True
    # the cut's CI reaches 0: no "planning possible", nor "M3 fails"
    no = _report(cut=((0.05, -0.01, 0.1), (0.0, -0.02, 0.02), (0.4, 0.3, 0.5)))
    assert rows(no)[:2] == [False, False] and rows(no, dict(TH_DEFAULTS, own_k=0))[0] is True
    # the S control cuts as much: fails too
    assert rows(_report(cut=((0.2, 0.1, 0.3), (0.15, 0.1, 0.2), (0.4, 0.3, 0.5))))[0] is False
    # own-state criterion at a k that was not run: cannot be judged
    assert rows(_report(), dict(TH_DEFAULTS, own_k=16))[0] is None
    # A and H not significant on ŷ, but the own-state cut is: not "the action changes nothing"
    flat = dict(a=(0.001, -0.001, 0.003), h=(0.001, -0.002, 0.004), r=1.0)
    assert rows(_report(**flat))[3] is False and rows(_report(**flat), dict(TH_DEFAULTS, own_k=0))[3] is True
    assert rows(_report(**flat, cut=((0.01, -0.01, 0.02), (0.0, -0.02, 0.02), (0.01, -0.02, 0.03))))[3] is True
