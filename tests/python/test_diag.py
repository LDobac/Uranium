"""`python -m uranium diag` (reports/jepa_survey.md 7, 10) on CPU: the probe math (ridge, game-level
folds, the two gain-over-nuisance methods, AUC, bootstrap, shuffled control), the counterfactual
edits, argument parsing of every subcommand, and the whole chain (dump with counterfactual passes
-> probes / intervene / core-split / spatial -> summary) on synthetic shards with a tiny net."""

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from uranium import _core, ckpt
from uranium.diag import cli, probe
from uranium.diag import tables as tb
from uranium.diag.dump import edit_count, edit_start, pad_one
from uranium.diag.summary import SCHEMA, build, validate
from uranium.model import UraniumNet, UraniumNetConfig, action_columns, input_dims

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "burn_parity"


# ---------------------------------------------------------------- probe math


def toy(n_games=40, rows=30, seed=0):
    rng = np.random.default_rng(seed)
    games = np.repeat(np.arange(n_games), rows)
    nuis = rng.normal(size=(len(games), 3)).astype(np.float32)
    signal = rng.normal(size=len(games))
    y = 0.8 * nuis[:, 0] + signal + 0.1 * rng.normal(size=len(games))
    info = np.concatenate([signal[:, None], rng.normal(size=(len(games), 7))], 1).astype(np.float32)
    noise = rng.normal(size=(len(games), 8)).astype(np.float32)
    return games, nuis, y, info, noise


def test_folds_are_game_level():
    games = np.repeat(np.arange(23), 7)
    fold = probe.folds_of(games)
    for g in np.unique(games):
        assert len(set(fold[games == g])) == 1
    assert set(fold) == set(range(probe.FOLDS))


def test_ridge_cv_recovers_a_linear_signal():
    games, nuis, y, info, noise = toy()
    fold = probe.folds_of(games)
    p, alphas = probe.ridge_cv(np.concatenate([info, nuis], 1), y, games, fold)
    assert probe.r2(y, p) > 0.95 and len(alphas) == probe.FOLDS
    p0, _ = probe.ridge_cv(noise, y, games, fold)
    assert probe.r2(y, p0) < 0.05


@pytest.mark.parametrize("method", probe.GAIN_METHODS)
def test_gain_over_nuisance(method):
    """An informative stage gains over the nuisance; a noise stage gains about 0 (resid: never
    clearly below, since the residual ridge can switch the stage off)."""
    games, nuis, y, info, noise = toy()
    fold = probe.folds_of(games)
    base = probe.r2(y, probe.ridge_cv(nuis, y, games, fold)[0])
    good = probe.r2(y, probe.ridge_cv_gain(info, nuis, y, games, fold, method))
    bad = probe.r2(y, probe.ridge_cv_gain(noise, nuis, y, games, fold, method))
    assert good - base > 0.5
    assert abs(bad - base) < 0.03
    with pytest.raises(ValueError):
        probe.ridge_cv_gain(info, nuis, y, games, fold, "stack")


def test_resid_gain_is_the_default():
    a = argparse.ArgumentParser().add_subparsers()
    cli.add_parser(a)
    ns = a.choices["diag"].parse_args(["probes", "d"])
    assert ns.gain == "resid"


def test_auc_matches_pairwise_count():
    rng = np.random.default_rng(1)
    y = rng.random(200) < 0.3
    s = np.round(rng.normal(size=200) + y, 1)  # ties
    pos, neg = s[y], s[~y]
    want = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))
    assert probe.auc(y, s) == pytest.approx(want)
    assert np.isnan(probe.auc(np.ones(5), np.arange(5)))


def test_bootstrap_and_shuffled_control():
    games, nuis, y, info, noise = toy()
    rng = np.random.default_rng(0)
    res = probe.run_task({"info": info, "noise": noise}, nuis, y, False, games, games, rng, boot_n=50, mlp=True, shuffled_key="info")
    m = res["metrics"]
    for k, v in m.items():
        assert v["lo"] <= v["pt"] <= v["hi"] or np.isnan(v["lo"]), k
    assert m["info+nuis|ridge"]["gain"] > 0.5 and m["info+nuis|ridge"]["gain_lo"] > 0
    assert m["info|mlp"]["pt"] > 0.5
    # per-game labels shuffled among games: nothing to learn
    assert m["shuffled|info"]["pt"] < 0.1
    assert res["metric"] == "r2" and res["games"] == 40 and res["gain_method"] == "resid"


def test_shuffled_labels_move_whole_games():
    games = np.repeat(np.arange(10), 3)
    y = np.repeat(np.arange(10.0), 3)
    ys = probe.shuffled_labels(y, games, np.random.default_rng(0))
    for g in range(10):
        assert len(set(ys[games == g])) == 1
    assert sorted(set(ys)) == sorted(set(y))


# ---------------------------------------------------------------- counterfactual edits


def one_token_group(race="Z", hp=300.0, remembered=False, since=0.0, frame=2400.0):
    s = tb.race(race)
    t, n = 2, 4
    cat = np.zeros((t, n, 4), np.int64)
    cont = np.zeros((t, n, 76), np.float32)
    mask = np.zeros((t, n), bool)
    cat[:, 0, 0], cat[:, 0, 1], cat[:, 0, 2] = s["focus"], tb.OWNER_ENEMY, tb.order_id("IncompleteBuilding")
    cont[:, 0, tb.HP_RATIO], cont[:, 0, tb.HP_SQRT] = hp / s["max_hp"], np.sqrt(hp) / 16
    cont[:, 0, tb.REMEMBERED] = float(remembered)
    cont[:, 0, tb.SINCE_SEEN] = since / tb.SINCE_SCALE
    cont[:, 0, tb.CONSTRUCTING] = 1.0
    mask[:, 0] = True
    sc = np.zeros((t, 567), np.float32)
    sc[:, tb.sc_seen_col(s["focus"])] = tb.encode_count(1)
    g = {"entity_cat": cat, "entity_cont": cont, "entity_mask": mask, "scalar": sc}
    for k in ("entity_cell", "entity_pos", "unit_cap", "is_building"):
        g[k] = np.zeros((t, n) + ((2,) if k == "entity_pos" else ()), np.float32)
    return g, np.full(t, frame)


def test_edit_start_is_consistent_with_the_hp_clock():
    """Zerg pool started 1800 seen at 2400: hp 75 + 0.5625 * 600 (the earlier run's formula);
    started at 1000: complete (Nothing order); seen before the start: masked."""
    g, fr = one_token_group()
    anc = np.zeros(2, int)
    e = edit_start(g, fr, anc, "Z", 1800)
    assert e["entity_cont"][0, 0, tb.HP_RATIO] * 750 == pytest.approx(np.rint(75 + 0.5625 * 600))
    assert e["entity_cont"][0, 0, tb.COMPLETED] == 0 and e["entity_cont"][0, 0, tb.CONSTRUCTING] == 1
    done = edit_start(g, fr, anc, "Z", 1000)
    assert done["entity_cont"][0, 0, tb.HP_RATIO] == 1 and done["entity_cont"][0, 0, tb.COMPLETED] == 1
    assert done["entity_cat"][0, 0, 2] == tb.order_id("Nothing")
    gone = edit_start(g, fr, anc, "Z", 3000)
    assert not gone["entity_mask"][0, 0]
    # a remembered token: edited at its last-seen frame (2400 - 600)
    g2, fr2 = one_token_group(remembered=True, since=600.0)
    e2 = edit_start(g2, fr2, anc, "Z", 1500)
    assert e2["entity_cont"][0, 0, tb.HP_RATIO] * 750 == pytest.approx(np.rint(75 + 0.5625 * 300))
    # Protoss: the shield follows
    gp, frp = one_token_group("P", hp=200.0)
    ep = edit_start(gp, frp, anc, "P", 2000)
    assert 0 < ep["entity_cont"][0, 0, tb.SH_RATIO] < 1
    assert g["entity_cont"][0, 0, tb.HP_RATIO] == pytest.approx(300 / 750)  # input not modified


def test_edit_count_scalar_and_token():
    g, _ = one_token_group("T")
    col = tb.sc_seen_col(tb.race("T")["focus"])
    up = edit_count(pad_one(g), "T", 1, True)
    assert tb.decode_count(up["scalar"][:, col]).tolist() == [2, 2]
    assert up["entity_mask"][:, -1].all() and (up["entity_cat"][:, -1, 0] == tb.race("T")["focus"]).all()
    down = edit_count(g, "T", -1, True)
    assert tb.decode_count(down["scalar"][:, col]).tolist() == [0, 0] and not down["entity_mask"][:, 0].any()
    sc_only = edit_count(g, "T", 1, False)
    assert (sc_only["entity_mask"] == g["entity_mask"]).all()


# ---------------------------------------------------------------- CLI


def parse(argv):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    cli.add_parser(sub)
    return p.parse_args(["diag", *argv])


def test_cli_parses_every_subcommand():
    a = parse(["dump", "--checkpoint", "c", "--shards", "s", "--out", "o", "--cf", "--cf-starts", "T:1000,2000"])
    assert a.diag_cmd == "dump" and a.cf and a.act_until == 5760 and a.row_until == 8640 and a.core_points == "gate" and a.splits == "val,test"
    a = parse(["probes", "d", "--gain", "joint", "--races", "Z", "--no-mlp"])
    assert a.gain == "joint" and a.races == "Z" and a.no_mlp and a.boot == 500 and a.cap == 0
    a = parse(["intervene", "d"])
    assert a.window == "2160,5760"
    a = parse(["spatial", "--checkpoint", "a", "--checkpoint", "b", "--shards", "s", "--out", "o.json"])
    assert a.checkpoint == ["a", "b"] and a.batches == 2 and a.skip_batches == 2
    a = parse(["core-split", "d", "--checkpoint", "c"])
    assert a.checkpoint == "c"
    a = parse(["summary", "--point", "S2e2", "--out", "s.json"])
    assert a.point == "S2e2"
    for argv in (["dump", "--shards", "s", "--out", "o"], ["probes", "d", "--gain", "stack"], ["summary", "--out", "x"]):
        with pytest.raises(SystemExit):
            parse(argv)
    from uranium.__main__ import main

    with pytest.raises(SystemExit) as e:
        main(["diag", "--help"])
    assert e.value.code == 0


# ---------------------------------------------------------------- end to end (synthetic shards)


def tiny_checkpoint(path: Path, kind="Gtrxl") -> Path:
    cfg = UraniumNetConfig.from_dict(json.loads((FIX / "tiny-gtrxl" / "meta.json").read_text())["config"])
    cfg.input = input_dims()
    cfg.core.kind = kind
    torch.manual_seed(5)
    net = UraniumNet(cfg, action_columns())
    path.mkdir(parents=True)
    ckpt.save_model(path, net)
    ckpt.save_meta(path, cfg, {}, "f32")
    return path


@pytest.fixture
def shards(tmp_path: Path) -> Path:
    d = tmp_path / "shards"
    _core.write_test_shards(str(d), "varied", 5)
    return d


class _PoolBatch:
    """A synthetic batch whose own SCVs read as unfinished enemy Spawning Pools (the synthetic
    games have no enemy units), HP rising with the frame, the pool in production (privileged)."""

    def __init__(self, b):
        self.b = b

    def __getattr__(self, k):
        return getattr(self.b, k)

    def group(self, *args):
        g = self.b.group(*args)
        cat, cont = np.array(g["entity_cat"]), np.array(g["entity_cont"])
        pool = tb.race("Z")["focus"]
        scv = (cat[..., 0] == tb.unit_id("Terran_SCV")) & np.asarray(g["entity_mask"]).astype(bool)
        frame = np.asarray(g["scalar"])[:, 0] * tb.FRAME_SCALE
        hp = np.clip(75 + 5.0 * frame + 40.0 * (np.asarray(g["scalar_cat"])[:, 2] % 3), 75, 700)[:, None] * np.ones(scv.shape)
        cat[..., 0][scv], cat[..., 1][scv], cat[..., 2][scv] = pool, tb.OWNER_ENEMY, tb.order_id("IncompleteBuilding")
        cont[..., tb.HP_RATIO][scv], cont[..., tb.HP_SQRT][scv] = hp[scv] / 750, np.sqrt(hp[scv]) / 16
        cont[..., tb.COMPLETED][scv], cont[..., tb.CONSTRUCTING][scv] = 0.0, 1.0
        pv = np.array(g["privileged"])
        pv[:, tb.priv_col(pool, True)] = tb.encode_count(1)
        pv[:, tb.priv_col(tb.unit_id("Zerg_Drone"))] = tb.encode_count(4)
        out = dict(g)
        out.update(entity_cat=cat, entity_cont=cont, privileged=pv)
        return out


_EvalStream = _core.EvalStream


class _PoolStream:
    def __init__(self, *args, **kwargs):
        self.s = _EvalStream(*args, **kwargs)

    def next(self):
        b = self.s.next()
        return None if b is None else _PoolBatch(b)

    def finish(self):
        return self.s.finish()


@pytest.mark.parametrize("kind", ["Gtrxl", "Lstm"])
def test_diag_chain_on_synthetic_shards(tiny_config, shards, tmp_path, kind, monkeypatch):
    from uranium.__main__ import main

    monkeypatch.setattr(_core, "EvalStream", _PoolStream)
    ck = tiny_checkpoint(tmp_path / "ck", kind)
    out = tmp_path / "dump"
    common = ["--config", str(tiny_config), "--backend", "cpu", "--synthetic-maps", "--shards", str(shards)]
    # synthetic games last 120 frames: counterfactual starts before them (else both mask the token)
    assert main(["diag", "dump", "--checkpoint", str(ck), "--out", str(out), "--splits", "train,val,test", "--cf", "--cf-starts", "Z:-1000,-200",
                 "--act-until", "1e9", "--row-until", "1e9", *common]) == 0
    man = json.loads((out / "manifest.json").read_text())
    assert man["games"] == 15 and man["stats"]["rows"] > 0 and man["passes"][0] == "real" and len(man["passes"]) == 7
    from uranium.diag.dump import load

    R, A, games, _ = load(out)
    assert len(R["frame"]) == man["stats"]["rows"] and R["pb"].shape[1:] == (7, 3)
    assert set(np.unique(R["game"])) <= set(games)
    real = R["pb"][:, 0].astype(np.float32)
    assert np.isfinite(R["pb"].astype(np.float32)).all() and ((real >= 0) & (real <= 1)).all()
    assert {games[g]["race"] for g in games} == {"Z"} and R["foc_vis"].any() and (R["foc_comp"][R["foc_vis"]] == 0).all()
    # an earlier counterfactual start reads as more HP on the same token
    early, late = R["pb"][:, 1], R["pb"][:, 2]
    assert not np.array_equal(early, late)
    assert A["h"].shape[0] == len(A["idx"]) and A["token"].shape[1] == 32 and len(A["idx"]) == int(R["foc_vis"].sum())
    if kind == "Gtrxl":
        assert {"x_in", "L1.a", "L1.g1", "L1.f", "L1.out", "L2.out"} <= set(A)
        # the last layer's output is h_t
        assert np.allclose(A["L2.out"].astype(np.float32), A["h"].astype(np.float32), atol=1e-3)
    else:
        assert "x_in" not in A
    # the analyses run (synthetic games are few: tasks may be skipped) and the summary is valid
    assert main(["diag", "probes", str(out), "--boot", "20", "--no-mlp", "--min-rows", "1", "--min-games", "2"]) == 0
    assert main(["diag", "intervene", str(out), "--boot", "20"]) == 0
    spj = tmp_path / "spatial.json"
    assert main(["diag", "spatial", "--checkpoint", str(ck), "--out", str(spj), "--split", "train", "--batches", "1", "--skip-batches", "1", *common]) == 0
    sp = json.loads(spj.read_text())
    pts = next(iter(sp["checkpoints"].values()))["points"]
    assert "spatial.s8" in pts and "entity.L0.ffn" in pts and "heads.type_res1.inner" in pts
    assert ("core.L0.ffn" in pts) == (kind == "Gtrxl")
    pr = json.loads((out / "probes.json").read_text())["races"]["Z"]["tasks"]
    hp = pr["hp|unf_token"]["metrics"]
    assert {"nuis|ridge", "token|ridge", "token+nuis|ridge", "h|ridge", "hc|ridge"} <= set(hp)
    assert hp["token|ridge"]["pt"] > 0.5  # the token after the encoder carries its own HP
    iv = json.loads((out / "intervene.json").read_text())["races"]["Z"]
    assert iv["games"] == 15 and iv["edited"] > 0
    args = ["--point", "test", "--probes", str(out / "probes.json"), "--intervene", str(out / "intervene.json"), "--spatial", str(spj)]
    if kind == "Gtrxl":
        assert main(["diag", "core-split", str(out), "--checkpoint", str(ck), "--boot", "10"]) == 0
        args += ["--core-split", str(out / "core_split.json")]
    assert main(["diag", "summary", *args, "--out", str(tmp_path / "summary.json")]) == 0
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["schema"] == SCHEMA and validate(s) == [] and s["checkpoint_sha256"] == ckpt.policy_version(ck)
    assert {"spatial.s8.const", "spatial.pool.const"} <= set(s["dead_channels"]) and s["dead_channels"]["spatial.s8.const"]["dtype"] == "float32"
    # the order ids the focus tokens carried, per race, in the manifest
    assert man["focus_orders"]["Z"]["unfinished"] == {str(tb.order_id("IncompleteBuilding")): int((R["foc_comp"] < 0.5).sum())}
    assert man["cf_orders"]["Z"]["unfinished"] == tb.order_id("IncompleteBuilding")

    # S7: chunks that start at or after --act-until get no probabilities, and intervene refuses a
    # window past it
    late = tmp_path / "late"
    assert main(["diag", "dump", "--checkpoint", str(ck), "--out", str(late), "--splits", "train", "--cf", "--act-until", "40", "--row-until", "1e9", *common]) == 0
    R2, _, _, _ = load(late)
    chunk_start = np.zeros(len(R2["frame"]))
    for i, st in enumerate(R2["step"]):  # rows are written chunk by chunk, step 0 first
        chunk_start[i] = R2["frame"][i] if st == 0 else chunk_start[i - 1]
    lates = chunk_start >= 40
    assert lates.any() and (~lates).any()
    assert np.isnan(R2["pb"][lates]).all() and np.isfinite(R2["pb"][~lates]).all()
    with pytest.raises(SystemExit, match="act_until"):
        main(["diag", "intervene", str(late)])


@pytest.mark.parametrize("kind", ["Gtrxl", "Lstm"])
def test_action_seq_is_forward_seq_action_head_050_r9(tiny_config, shards, tmp_path, kind):
    """050-R9 `diag dump` runs the trunk and the ActionType head only (`action_seq`): the same log-probs, state and
    head inputs as `forward_seq` (which also runs the location head over a varying row count)."""
    from uranium.diag.capture import Capture
    from uranium.model import to_obs

    net, _ = ckpt.load_model(tiny_checkpoint(tmp_path / "ck", kind), torch.device("cpu"))
    net.eval()
    cfg = json.loads(_core.bc_config(str(tiny_config), "s1")[0])
    t = cfg["seq_len"]
    stream = _core.EvalStream(str(tiny_config), [str(shards)], split="train", synthetic_maps=True)
    cap = Capture(net, "gate")
    st_f = st_a = None
    checked = 0
    while (b := stream.next()) is not None and checked < 4:
        hg = b.group(0, cfg["lanes"], t, cfg["pad_entities"], net.cfg.heads.max_select)
        keep = torch.as_tensor(np.asarray(hg["keep"]).astype(bool))
        obs = to_obs(hg, torch.device("cpu"))
        with torch.no_grad():
            s0 = net.init_state(hg["b"]) if st_f is None else net.mask_state(st_f, keep)
            o = net.forward_seq(obs, net.labels(hg, torch.device("cpu")), torch.as_tensor(hg["privileged"]).float(), s0, t)
            s1 = net.init_state(hg["b"]) if st_a is None else net.mask_state(st_a, keep)
            a, acts = cap.run(obs, s1, t)
        assert torch.equal(o["dists"]["action"], a["action"])
        for x, y in zip(torch.utils._pytree.tree_leaves(o["state"]), torch.utils._pytree.tree_leaves(a["state"])):
            assert torch.equal(x, y)
        st_f, st_a = o["state"], a["state"]
        assert {"h", "c", "E", "sp_pool"} <= set(acts) and acts["h"].shape[0] == hg["b"] * t
        checked += 1
    assert checked == 4
    cap.close()


@pytest.mark.parametrize("variant", ["outer", "model2", "model2_all"])
def test_action_seq_with_outer_and_bypass_050_r9(variant):
    """`action_seq` = `forward_seq(...)["dists"]["action"]` (and the state) with the outer LSTM, the head bypass
    (032-R21, made non-zero) and core_skip (032-R23) on: the model 2 path of the diagnostics."""
    from test_variants import SEG, net_of, perturb, tiny

    from uranium.model.config import VariantConfig

    v = {"outer": VariantConfig(outer=SEG), "model2": VariantConfig(outer=SEG, bypass=True, core_skip=True),
         "model2_all": VariantConfig(terrain=True, ctx_token=True, attn_pool=True, outer=SEG, bypass=True, core_skip=True)}[variant]
    cfg, b, t, obs, labels, priv = tiny(v)
    net = perturb(net_of(cfg))
    if net.variant.core_skip is not None:  # zero-initialised: non-zero as after training
        with torch.no_grad():
            for p in net.variant.core_skip.parameters():
                p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(5)) * 0.05)
    with torch.no_grad():
        o = net.forward_seq(obs, labels, priv, net.init_state(b), t)
        a, lp = net.action_seq(obs, net.init_state(b), t)
    assert torch.equal(o["dists"]["action"], lp)
    for x, y in zip(torch.utils._pytree.tree_leaves(o["state"]), torch.utils._pytree.tree_leaves(a["state"])):
        assert torch.equal(x, y)


def test_dump_read_until_keeps_the_rows_050_r9(tiny_config, shards, tmp_path, monkeypatch):
    """`diag dump --read-until`: each game read only up to that frame; the rows (and their game ids) are the
    dumped rows without it, in another order (the lanes run shorter games); a cut before the row window is refused."""
    from uranium.__main__ import main
    from uranium.diag.dump import load

    monkeypatch.setattr(_core, "EvalStream", _PoolStream)
    ck = tiny_checkpoint(tmp_path / "ck")
    common = ["--checkpoint", str(ck), "--config", str(tiny_config), "--backend", "cpu", "--synthetic-maps", "--shards", str(shards),
              "--splits", "train,val,test", "--row-until", "50", "--act-until", "40"]
    assert main(["diag", "dump", *common, "--out", str(tmp_path / "whole")]) == 0
    assert main(["diag", "dump", *common, "--read-until", "50", "--out", str(tmp_path / "cut")]) == 0
    R0, A0, g0, m0 = load(tmp_path / "whole")
    R1, A1, g1, m1 = load(tmp_path / "cut")
    assert m1["read_until"] == 50 and m0["read_until"] is None
    lane_free = lambda g: {k: {f: v for f, v in e.items() if f != "lane"} for k, e in g.items()}  # noqa: E731
    assert lane_free(g0) == lane_free(g1) and {e["lane"] for e in g0.values()} != {None}  # the same game ids; lanes may differ
    assert m0["stats"]["rows"] == m1["stats"]["rows"] and m0["stats"]["act_rows"] == m1["stats"]["act_rows"]
    assert m1["stats"]["batches"] < m0["stats"]["batches"]  # the steps after the cut were not read
    o0, o1 = np.argsort(R0["game"], kind="stable"), np.argsort(R1["game"], kind="stable")
    for k in R0:
        a, b = R0[k][o0], R1[k][o1]
        if k == "pb":
            assert np.allclose(a, b, atol=1e-6, equal_nan=True)
        else:
            assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), k
    # activation rows: the same (game, step) rows
    key = lambda R, A: sorted(zip(R["game"][A["idx"]].tolist(), R["frame"][A["idx"]].tolist()))  # noqa: E731
    assert key(R0, A0) == key(R1, A1)
    with pytest.raises(SystemExit, match="cuts rows before"):
        main(["diag", "dump", *common, "--read-until", "45", "--out", str(tmp_path / "bad")])


def test_summary_schema_validation():
    s = build("p")
    assert validate(s) == []
    bad = dict(s, schema="x", gain_method="stack", dead_channels={"spatial.s8": {"pos_frac": 0.0, "never_active": 9, "channels": 8},
                                                                    "spatial.s8.const": {"const": 9, "channels": 8, "threshold": 1e-4, "dtype": "float32"}})
    errs = validate(bad)
    assert any("schema" in e for e in errs) and any("gain_method" in e for e in errs)
    assert "dead_channels.spatial.s8" in errs and "dead_channels.spatial.s8.const" in errs


def test_summary_inputs_must_be_one_checkpoint():
    """S6: inputs of different checkpoints are refused; the spatial entry is picked by sha256."""
    entry = lambda sha: {"sha256": sha, "points": {"spatial.s8": {"pos_frac": 0.0, "never_active": 8, "channels": 8, "elements": 64}}}  # noqa: E731
    spatial = {"backend": "cpu", "precision": "f32", "checkpoints": {"a": entry("aa"), "b": entry("bb")}}
    probes = {"dump": {"checkpoint": "b", "checkpoint_sha256": "bb"}, "gain_method": "resid", "races": {}}
    s = build("p", probes=probes, spatial=spatial)
    assert s["checkpoint_sha256"] == "bb" and validate(s) == []
    with pytest.raises(ValueError, match="different checkpoints"):
        build("p", probes=probes, intervene={"dump": {"checkpoint": "a", "checkpoint_sha256": "aa"}, "races": {}})
    with pytest.raises(ValueError, match="no checkpoint with sha256"):
        build("p", probes={"dump": {"checkpoint_sha256": "cc"}, "races": {}}, spatial=spatial)
    with pytest.raises(ValueError, match="2 checkpoints"):
        build("p", spatial=spatial)
