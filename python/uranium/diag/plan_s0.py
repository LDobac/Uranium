"""`python -m uranium diag plan-s0` (jepa_planning S0, reports/jepa_planning.md 6.4-6.5; 050-R9 diagnostics):
the S0 feasibility test on `diag plan-dump` folders, on CPU. Measurement only (no model, spec or
planner change).

Stages (`--stage`): `train` trains the predictors (`uranium.diag.plan_pred`) into `<out>/models/`
(an existing model file is kept unless `--retrain`), `report` computes the tables into `<out>/s0.json`
and `<out>/s0.md` (`--tag`: s0-<tag>.*), `all` both. Per k of `--ks`: U, A, Ad (A+d), H, S on every training row, and for M3c
A on the human-only and the model-only training rows. Same steps, batch and seed for every predictor.

Tables (eval games only; CIs: game bootstrap 95 %, `uranium.diag.plan_stats`; the bootstrap unit is the
replay, so both streams of one replay move together):
- Design (`plan_pred.Design`, reports/jepa_planning.md 6.8): the second run's default drops the chained
  rows of teacher / model sources, gives every predictor o_t, predicts the own-state change and draws
  the S partner from another game; `--v1` is the first run (2026-11-11) bit for bit (same models, same
  numbers; the report adds the change-space own-state metrics, the verdict keeps the first run's rows).
- M0 basics (every row): open action types, policy top-m mass (m = 4, 8, 16), frame gap of the next
  decision and of the k pairs (and their decision gap with `--k-unit frame`), operation-action share,
  rows whose policy top 8 has an operation action, chained rows.
- M1 conditioning gain over U on all rows and on operation-action rows (a_t an operation action), per
  k: ŷ 1 - cos (and the model's own g_k as a reference, decision k only), p̂ R² and ô R² (standardised,
  kept dimensions; ô on the 6.3 columns `OWN_HEADLINE`, `o_all` every column, `_own_cols` per column),
  the own-state change R² `od` / `od_all` (o_{t+k} - o_t standardised by its training sd) with the
  relative error cut 1 - err_X / err_U (`cut`, the reported own-state metric), win NLL and 8:00 SCV
  NLL (decided games); gain = L_U - L_X (R²: X - U), the oracle share r = gain_A / gain_H (undefined,
  and the k not judged, when H gains nothing), the control ratio gain_A / gain_S. Per source kind too:
  k decisions are a different frame span per source (M0).
- Views: a `model` source with Terran-enemy streams needs `--drop NAME=<views json>` (bot views out by
  replay sha256 + own start slot) or `--keep-bot-perspective`. Models record the split digest; a report
  on another split is refused.
- M2 on operation-action rows, candidates = policy top 8 open types ∪ {a_t}, predictor A: M2a
  spread R_t = mean pairwise distance of the candidates' predictions / the RMS error of the actual
  action's prediction (ŷ scaled to norm sqrt(d), ô / p̂ standardised, ô in the predictor's own target
  space); M2b direction: share of (row, Build/BuildAddon/Train_X candidate) where ô_X of that candidate
  (the predicted change with `--own-target delta`) exceeds ô_X of every other candidate.
- M3a outcome AUC by time band: q̂_A(h'_t, a_t), q̂_U(h'_t), win_open(h'_t), oracle q̂_U(h'_{t+k}).
- M3b rush check: rows of teacher / model eval games of `--m3b-races` in `--m3b-window` with
  Build_Bunker open; ΔQ = σq̂(Bunker) - Σ_c π(c) σq̂(c) over the open types; rush minus macro mean
  ΔQ and the row AUC of ΔQ for rush vs macro. M3c: the same with A trained on human-only and on
  model-only rows. Build classes: `cls_given` (plan-dump --classes) else the privileged rules.
- M3d advantage agreement: operation-action rows, Spearman of Â_t = σq̂_A(a_t) - Σ_c π(c) σq̂_A(c)
  and G_t = outcome - σq̂_U(h'_t).
- Verdict: the 6.5 table, per scoring target (win, scv8), thresholds `--th-*` (기본안(미확정): the
  user sets them). Not `--v1`: M1 also needs the own-state criterion at k = `--th-own-k` (A's relative
  change-error cut over U on pooled operation-action rows: CI above 0 and at least `--th-s-ratio` times
  the S control's cut), and "A and H both not significant" counts the own-state cuts too."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from uranium.diag import plan_pred as pp
from uranium.diag import plan_stats as ps
from uranium.diag import tables as tb

M0_TOP = (4, 8, 16)
CAND_TOP = 8
OUTCOMES = pp.Q_NAMES  # q̂ heads: game won, 8:00 SCVs >= 8
# 6.5 thresholds, 기본안(미확정) (reports/jepa_planning.md 6.5: the user decides the numbers)
# 6.3 o_t: the own-state columns of the headline ô metric (the predictors train on every column)
OWN_HEADLINE = ("scv", "supply_used", "supply_total", "barracks", "bunker", "command_center", "supply_depot", "refinery", "factory", "army")
TH_DEFAULTS = {"s_ratio": 2.0, "r_min": 0.2, "r_noise": 0.1, "m2b": 0.90, "m2a": 0.5, "m3a": 0.01}


def add_args(p) -> None:
    p.add_argument("--src", action="append", required=True, help="NAME:KIND:ROLE=PATH, a plan-dump folder (KIND human|teacher|model, ROLE train|eval|split), repeatable")
    p.add_argument("--out", required=True, help="output folder (models/, s0.json, s0.md)")
    p.add_argument("--tag", default="", help="report file suffix (s0-<tag>.json / .md), e.g. a second report with other M3b races")
    p.add_argument("--stage", choices=["all", "train", "report"], default="all")
    p.add_argument("--ks", default="8,32")
    p.add_argument("--variants", default=",".join(pp.VARIANTS))
    p.add_argument("--no-m3c", action="store_true", help="skip the human-only / model-only predictors")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0, help="every predictor's seed (init and batches)")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default); results are bit-identical for one value")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--eval-frac", type=float, default=0.2, help="eval share of a `split` source (game level)")
    p.add_argument("--split-seed", type=int, default=1)
    p.add_argument("--drop", action="append", default=[], help="NAME=JSON: the views of source NAME ({\"bot\": [[sha256, start], ...], \"model\": [...]}, "
                   "config/plan_s0/jepa_diag_tp_views.json): bot views left out, every stream must be listed")
    p.add_argument("--keep-bot-perspective", action="store_true", help="accept a model source's Terran-enemy streams without --drop (both TvT views)")
    p.add_argument("--maps", default="", help="only games of these map families (e.g. 0,1 = the target maps; empty = all)")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--boot-seed", type=int, default=7)
    p.add_argument("--bands", default="2880,5760,8640,14400", help="M3a time bands (frame edges; the first..last span is `all`)")
    p.add_argument("--m3b-races", default="Z", help="enemy races of the M3b / M3c rows (6.4: Zerg)")
    p.add_argument("--m3b-window", default="2880,5040", help="M3b frame window (2:00-3:30)")
    p.add_argument("--m3b-macro", default="macro", help="build classes counted as macro (comma-separated)")
    for k, v in TH_DEFAULTS.items():
        p.add_argument(f"--th-{k.replace('_', '-')}", type=float, default=v, help=f"6.5 threshold, 기본안(미확정) {v}")
    p.add_argument("--th-own-k", type=int, default=8, help="k of the M1 own-state criterion (기본안(미확정) 8; 0 = no own-state criterion; --v1: 0)")
    # design (plan_pred.Design; reports/jepa_planning.md 6.8)
    d = pp.Design()
    p.add_argument("--v1", action="store_true", help="the first S0 run's design (every valid row, h' only, level targets, same-game S partner) "
                   "and verdict: its models and numbers")
    p.add_argument("--chains", choices=["drop", "keep"], help="chained rows of --chain-kinds: drop (default) or keep (--v1)")
    p.add_argument("--chain-kinds", help=f"source kinds whose chained rows are dropped (default {','.join(d.chain_kinds)})")
    p.add_argument("--own-input", choices=["on", "off"], help="o_t as a predictor input (default on; --v1 off)")
    p.add_argument("--own-target", choices=["delta", "level"], help="ô target: o_{t+k} - o_t (default) or o_{t+k} (--v1)")
    p.add_argument("--s-partner", choices=["other-game", "same-game"], help="S control partner row: another game of the source (default) or the same game (--v1)")
    p.add_argument("--k-unit", choices=["dec", "frame"], help="k in decisions (default) or frames (t+k = the first row k frames or more later)")


def design_of(a) -> pp.Design:
    """`plan_pred.Design` from the flags (`--v1` takes no other design flag)."""
    given = {"chains": a.chains, "chain_kinds": a.chain_kinds, "own_input": a.own_input, "own_target": a.own_target, "partner": a.s_partner, "k_unit": a.k_unit}
    given = {k: v for k, v in given.items() if v is not None}
    if a.v1:
        if given:
            raise SystemExit(f"plan-s0: --v1 is the first run's design; drop {sorted(given)}")
        return pp.V1
    if "chain_kinds" in given:
        given["chain_kinds"] = tuple(x for x in given["chain_kinds"].split(",") if x)
    if "own_input" in given:
        given["own_input"] = given["own_input"] == "on"
    return pp.Design(**given)


def model_specs(ks, variants, m3c: bool) -> list[tuple[str, str, int, list | None]]:
    out = []
    for k in ks:
        out += [(f"{v}_k{k}", v, k, None) for v in variants]
        if m3c:
            out += [(f"A-human_k{k}", "A", k, ["human"]), (f"A-model_k{k}", "A", k, ["model"])]
    return out


def fingerprint(data: pp.Data) -> dict:
    """What a trained model depends on: checkpoint, rows, sources and the games / roles kept (the split
    digest covers --eval-frac, --split-seed, --drop and --maps) and the design (absent for `V1`, as the
    first run's models have it)."""
    fp = {**data.checkpoint, "rows": int(data.N), "games": int(data.G), "sources": [[s.name, s.kind, s.role] for s in data.sources],
          "split_digest": data.split_digest}
    if data.design != pp.V1:
        fp["design"] = data.design.as_dict()
    return fp


def train_all(data: pp.Data, a, ks, variants, models: Path) -> dict:
    log = {}
    stats = {}
    for name, v, k, kinds in model_specs(ks, variants, not a.no_m3c):
        path = models / f"{name}.pt"
        rows = data.pair_rows(k, "train", kinds)
        if path.exists() and not a.retrain:
            _, _, meta = pp.load(path, data)
            if meta.get("data") != fingerprint(data):
                raise SystemExit(f"{path}: trained on other data ({meta.get('data')}); --retrain")
            log[name] = {"kept": True, "rows": meta["rows"]}
            continue
        if len(rows) == 0:
            log[name] = {"skipped": "no training rows"}
            print(f"[train] {name}: no rows, skipped", flush=True)
            continue
        key = (k, tuple(kinds or ()))
        if key not in stats:
            stats[key] = data.stats(rows, k)
        t0 = time.time()
        net, lg = pp.train_one(data, v, k, rows, stats[key], steps=a.steps, batch=a.batch, lr=a.lr, seed=a.seed)
        meta = {"variant": v, "k": k, "kinds": kinds, "rows": int(len(rows)), "steps": a.steps, "batch": a.batch, "lr": a.lr, "seed": a.seed,
                "threads": torch.get_num_threads(), "log": lg, "secs": round(time.time() - t0, 1), "data": fingerprint(data)}
        pp.save(net, stats[key], path, meta)
        log[name] = {k_: meta[k_] for k_ in ("rows", "secs")} | {"last": lg[-1] if lg else None}
        print(f"[train] {name}: rows {len(rows)} {meta['secs']}s last {lg[-1] if lg else None}", flush=True)
    return log


# ---------------------------------------------------------------- report pieces


def m0(data: pp.Data, ks) -> dict:
    out = {}
    groups = {"all": np.arange(data.N)} | {kind: np.nonzero(data.row_kind == pp.KINDS.index(kind))[0] for kind in pp.KINDS if (data.row_kind == pp.KINDS.index(kind)).any()}
    for gname, sel in groups.items():
        n_open, tops, top8_op = [], {m: [] for m in M0_TOP}, []
        for s in range(0, len(sel), 100_000):
            r = sel[s:s + 100_000]
            m = data.mask_rows(r)
            n_open.append(m.sum(1))
            p = data.probs(r)
            ps_ = -np.sort(-p, 1)
            cs = np.cumsum(ps_, 1)
            for t in M0_TOP:
                tops[t].append(cs[:, min(t, cs.shape[1]) - 1])
            top8 = np.argsort(-p, 1, kind="stable")[:, :CAND_TOP]
            top8_op.append((data.op[top8] & (np.take_along_axis(p, top8, 1) > 0)).any(1))
        cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)  # noqa: E731
        gaps, dgaps = {}, {}
        nx = data.nxt_dec1[sel]
        ok = nx >= 0
        gaps["1"] = ps.quantiles(data.frame[nx[ok]] - data.frame[sel[ok]])
        for k in sorted(set(ks) - ({1} if data.design.k_unit == "dec" else set())):
            nx = data.nxt[k][sel]
            ok = nx >= 0
            gaps[str(k)] = ps.quantiles(data.frame[nx[ok]] - data.frame[sel[ok]])
            dgaps[str(k)] = ps.quantiles((data.pos[nx[ok]] - data.pos[sel[ok]]).astype(float))
        v = data.valid[sel]
        out[gname] = {
            "rows": int(len(sel)), "valid_rows": int(v.sum()), "open_types": ps.quantiles(cat(n_open)),
            "top_mass": {str(t): ps.quantiles(cat(tops[t])) for t in M0_TOP}, "frame_gap": gaps, "dec_gap": dgaps,
            "op_share": float(data.is_op[sel].sum() / max(v.sum(), 1)), "top8_has_op": float(cat(top8_op).mean()) if len(sel) else None,
            "chained_valid": int((v & data.chained[sel]).sum()), "chain_dropped": int((v & data.chain_drop[sel]).sum()),
        }
    return out


def norm_y(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12) * np.sqrt(x.shape[-1])


def targets(data: pp.Data, ev, k, st, with_y: bool = True) -> dict:
    """The t+k targets of rows `ev` in the predictors' standardised space (float32; left-out
    dimensions 0)."""
    t = data.nxt[k][ev]
    p_on = np.isfinite(st["p_sd"])
    o_on = np.isfinite(st["o_sd"])
    d_on = np.isfinite(st["d_sd"])
    p = np.where(p_on, (data.priv[t].astype(np.float32) - st["p_mu"]) / np.where(p_on, st["p_sd"], 1.0), 0.0).astype(np.float32)
    o = np.where(o_on, (data.own[t] - st["o_mu"]) / np.where(o_on, st["o_sd"], 1.0), 0.0).astype(np.float32)
    od = np.where(d_on, (data.own[t] - data.own[ev] - st["d_mu"]) / np.where(d_on, st["d_sd"], 1.0), 0.0).astype(np.float32)
    head = np.isin(np.array(data.own_names), OWN_HEADLINE)
    out = {"p": p, "p_ok": data.priv_ok[t] > 0, "o": o, "p_on": p_on, "o_on": o_on & head, "o_all_on": o_on, "q": data.row_q[ev],
           "od": od, "od_on": d_on & head, "od_all_on": d_on}
    if with_y:
        out["y"] = data.y[t].astype(np.float32)
    return out


def eval_rows(nets: dict, data: pp.Data, ev, k, st, chunk: int = 8192) -> tuple[dict, dict]:
    """Per-row M1 terms of every predictor (and the model's own g_k, ŷ only, decision k) on rows `ev`,
    chunk by chunk (bounded memory) -> ({name: {metric: [rows]}}, {name: q̂ logits [rows, 2]})."""
    rows = {nm: {m: [] for m in ("y", "p", "o", "o_all", "o_cols", "od", "od_all", "od_cols") + OUTCOMES} for nm in nets}
    rows["gk"] = {"y": []}
    qlog = {nm: [] for nm in nets}
    for s in range(0, len(ev), chunk):
        sl = ev[s:s + chunk]
        tg = targets(data, sl, k, st)
        for nm, net in nets.items():
            pr = pp.predict(net, data, sl, k, st, chunk=chunk)
            pr["o"], pr["od"] = pp.own_spaces(data, pr["o"], sl, st)
            for m, v in per_row(pr, tg).items():
                rows[nm][m].append(v)
            qlog[nm].append(pr["q"])
        gk = data.gk(k, sl) if data.design.k_unit == "dec" else None
        if gk is None:
            rows.pop("gk", None)
        elif "gk" in rows:
            rows["gk"]["y"].append(1.0 - cosine(gk, tg["y"]))
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)  # noqa: E731
    return {nm: {m: cat(v) for m, v in d.items()} for nm, d in rows.items()}, {nm: cat(v).reshape(-1, len(OUTCOMES)) for nm, v in qlog.items()}


def cosine(a, b) -> np.ndarray:
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


def per_row(pred: dict, tg: dict) -> dict:
    out = {"y": 1.0 - cosine(pred["y"], tg["y"])}
    out["p"] = np.where(tg["p_ok"], (((pred["p"] - tg["p"]) ** 2) * tg["p_on"]).sum(1), np.nan)
    for sp in ("o", "od"):
        oc = ((pred[sp] - tg[sp]) ** 2 * tg[f"{sp}_all_on"]).astype(np.float32)
        out[sp] = (oc * tg[f"{sp}_on"]).sum(1)
        out[f"{sp}_all"] = oc.sum(1)
        out[f"{sp}_cols"] = oc
    for j, q in enumerate(OUTCOMES):
        yq = tg["q"][:, j]
        out[q] = np.where(np.isfinite(yq), ps.bce(pred["q"][:, j], np.nan_to_num(yq)), np.nan)
    return out


def m1_block(boot: ps.Boot, rows: dict, tg: dict, sel: np.ndarray, names: list) -> dict:
    """One row set: per metric the predictors' values and the gains over U."""
    res = {}
    r2_tgt = {"p": "p", "o": "o", "o_all": "o", "od": "od", "od_all": "od"}
    for met in ("y", "p", "o", "o_all", "od", "od_all") + OUTCOMES:
        vals, errs = {}, {}
        for nm in names:
            if met not in rows[nm]:
                continue
            v = rows[nm][met][sel]
            if met in r2_tgt:
                tgt = tg[r2_tgt[met]][sel] * tg[f"{met}_on"]
                vals[nm] = boot.r2(v, tgt, tg["p_ok"][sel] if met == "p" else None)
                if met in ("od", "od_all"):
                    errs[nm] = boot.mean(v)
            else:
                vals[nm] = boot.mean(v)
        if "U" not in vals:
            continue
        higher = met in r2_tgt
        gain = {nm: ((vals[nm][0] - vals["U"][0], vals[nm][1] - vals["U"][1]) if higher else (vals["U"][0] - vals[nm][0], vals["U"][1] - vals[nm][1]))
                for nm in vals if nm != "U"}
        r = {"metric": "R2" if higher else ("1-cos" if met == "y" else "NLL"), "values": {nm: ps.ci(v) for nm, v in vals.items()},
             "gain": {nm: ps.ci(g) for nm, g in gain.items()}}
        if errs:
            # the reported own-state metric: relative change-error cut 1 - err_X / err_U (paired draws)
            r["cut"] = {nm: ps.rel_cut(e, errs["U"]) for nm, e in errs.items() if nm != "U"}
        if "A" in gain and "H" in gain:
            r["r"] = ps.ratio(gain["A"], gain["H"])
        if "A" in gain and "S" in gain:
            r["a_over_s"] = float(gain["A"][0] / gain["S"][0]) if gain["S"][0] else None
        res[met] = r
    return res


def own_cols_block(boot: ps.Boot, rows: dict, tg: dict, sel: np.ndarray, names: list, own_names: list) -> dict:
    """Per own-state column (kept on the training rows): each predictor's R² and the gain of A over U, on the
    level (o_{t+k}) and under `delta` on the change (o_{t+k} - o_t) with A's relative error cut."""
    out = {}
    for c in np.nonzero(tg["o_all_on"])[0]:
        y = tg["o"][sel][:, [c]]
        vals = {nm: boot.r2(rows[nm]["o_cols"][sel, c], y) for nm in names if "o_cols" in rows[nm]}
        if "U" not in vals:
            continue
        e = {"values": {nm: ps.ci(v) for nm, v in vals.items()}, "headline": own_names[c] in OWN_HEADLINE}
        if "A" in vals:
            e["gain_A"] = ps.diff(vals["A"], vals["U"])
        if tg["od_all_on"][c]:
            yd = tg["od"][sel][:, [c]]
            dv = {nm: boot.r2(rows[nm]["od_cols"][sel, c], yd) for nm in names if "od_cols" in rows[nm]}
            e["delta"] = {"values": {nm: ps.ci(v) for nm, v in dv.items()}}
            if "A" in dv:
                e["delta"]["gain_A"] = ps.diff(dv["A"], dv["U"])
                e["delta"]["cut_A"] = ps.rel_cut(boot.mean(rows["A"]["od_cols"][sel, c]), boot.mean(rows["U"]["od_cols"][sel, c]))
        out[own_names[c]] = e
    return out


def kind_groups(data: pp.Data, ev) -> dict:
    out = {"pooled": np.arange(len(ev))}
    for kind in pp.KINDS:
        m = data.row_kind[ev] == pp.KINDS.index(kind)
        if m.any():
            out[kind] = np.nonzero(m)[0]
    return out


def candidates(data: pp.Data, rows) -> np.ndarray:
    """[rows, CAND_TOP + 1] action ids (-1 = none): the policy's top open types ∪ {a_t}."""
    p = data.probs(rows)
    top = np.argsort(-p, 1, kind="stable")[:, :CAND_TOP]
    top = np.where(np.take_along_axis(p, top, 1) > 0, top, -1)
    a = data.action[rows].astype(np.int64)
    has = (top == a[:, None]).any(1)
    return np.concatenate([top, np.where(has, -1, a)[:, None]], 1)


def m2(data: pp.Data, net, st, k, ev, boot_n, seed) -> dict:
    from uranium.diag.plan_dump import _group, own_columns

    names, _, of_action = own_columns()
    if names != data.own_names:
        raise SystemExit("plan-s0: own-state columns differ from this code's")
    rows = ev[data.is_op[ev]]
    if not len(rows):
        return {"rows": 0}
    cand = candidates(data, rows)
    tg = targets(data, rows, k, st)
    # ô in the predictor's own target space (the change with --own-target delta: M2b compares predicted changes)
    if data.design.own_target == "delta":
        tg["o"], tg["o_on"] = tg["od"], tg["od_on"]
    C = cand.shape[1]
    D = {"y": np.full(len(rows), np.nan), "o": np.full(len(rows), np.nan), "p": np.full(len(rows), np.nan)}
    err = {"y": [], "o": [], "p": []}
    dir_ok, dir_unit, dir_kind, dir_act = [], [], [], []
    a_t = data.action[rows].astype(np.int64)
    for s in range(0, len(rows), 512):
        r = rows[s:s + 512]
        cs = cand[s:s + 512]
        n = len(r)
        valid = cs >= 0
        flat_r = np.repeat(r, C)
        flat_a = np.where(valid, cs, 0).reshape(-1)
        pr = pp.predict(net, data, flat_r, k, st, action=flat_a)
        out = {"y": norm_y(pr["y"]).reshape(n, C, -1), "o": (pr["o"] * tg["o_on"]).reshape(n, C, -1), "p": (pr["p"] * tg["p_on"]).reshape(n, C, -1)}
        o_dir = pr["o"].reshape(n, C, -1)  # M2b reads the candidate's own column X, every column
        pair = valid[:, :, None] & valid[:, None, :] & ~np.eye(C, dtype=bool)[None]
        for key, x in out.items():
            d = np.linalg.norm(x[:, :, None] - x[:, None, :], axis=-1)
            npair = pair.sum((1, 2))
            D[key][s:s + n] = np.where(npair > 0, (d * pair).sum((1, 2)) / np.maximum(npair, 1), np.nan)
            ai = np.argmax(np.where(valid, cs == a_t[s:s + n, None], False), 1)
            pa = x[np.arange(n), ai]
            tgt = norm_y(tg["y"][s:s + n]) if key == "y" else tg[key][s:s + n] * tg[f"{key}_on"]
            e = ((pa - tgt) ** 2).sum(1)
            if key == "p":
                e = e[tg["p_ok"][s:s + n]]
            err[key].append(e)
        # M2b: Build / BuildAddon / Train_X candidates with an own column X
        for j in range(C):
            col = np.array([of_action.get(int(c), -1) for c in cs[:, j]])
            has = valid[:, j] & (col >= 0)
            if not has.any():
                continue
            ii = np.nonzero(has)[0]
            mine = o_dir[ii, j, col[ii]]
            others = o_dir[ii, :, col[ii]]  # [m, C]
            ov = valid[ii].copy()
            ov[:, j] = False
            some = ov.any(1)
            best = np.where(ov, others, -np.inf).max(1)
            dir_ok.append((mine > best)[some])
            dir_unit.append(data.row_unit[r[ii[some]]])
            dir_act.append(cs[ii[some], j])
            dir_kind.append(np.array(["train" if _group(int(c)) == "train" else "build" for c in cs[ii[some], j]]))
    res = {"rows": int(len(rows)), "candidates": ps.quantiles(((cand >= 0).sum(1)).astype(float))}
    bt = ps.Boot(data.row_unit[rows], boot_n, seed)
    for key in ("y", "o", "p"):
        e = np.concatenate(err[key]) if err[key] else np.zeros(0)
        N = float(np.sqrt(e.mean())) if len(e) else float("nan")
        R = D[key] / N if N > 0 else np.full(len(rows), np.nan)
        res[f"M2a_{key}"] = {"N_rms": N, "R": ps.quantiles(R), "R_median": float(np.nanmedian(R)) if np.isfinite(R).any() else None,
                             "R_mean": ps.ci(bt.mean(R))}
    if dir_ok:
        ok = np.concatenate(dir_ok).astype(np.float64)
        unit = np.concatenate(dir_unit)
        kind = np.concatenate(dir_kind)
        act = np.concatenate(dir_act)
        b2 = ps.Boot(unit, boot_n, seed)
        res["M2b"] = {"pairs": int(len(ok)), "rate": ps.ci(b2.mean(ok))}
        for kk in ("build", "train"):
            m = kind == kk
            if m.any():
                res["M2b"][kk] = {"pairs": int(m.sum()), "rate": ps.ci(b2.mean(np.where(m, ok, np.nan)))}
        names_a = {a["id"]: a["name"] for a in tb._rows("actions")}
        res["M2b"]["by_action"] = {names_a[int(x)]: {"pairs": int((act == x).sum()), "rate": float(ok[act == x].mean())} for x in np.unique(act)}
    else:
        res["M2b"] = {"pairs": 0}
    return res


@torch.no_grad()
def q_all(net, data: pp.Data, rows, st, chunk: int = 256) -> np.ndarray:
    """[rows, A, 2] σq̂ of predictor A (standardisation `st`) for every action type."""
    cand = torch.arange(data.n_act)
    out = []
    for s in range(0, len(rows), chunk):
        r = rows[s:s + chunk]
        own = data.own_in(r, st) if net.own_in else None
        out.append(ps.sigmoid(net.q_all(torch.from_numpy(data.h[r].astype(np.float32)), cand, own).numpy()))
    return np.concatenate(out) if out else np.zeros((0, data.n_act, len(OUTCOMES)))


def delta_q(net, data: pp.Data, rows, action, st) -> np.ndarray:
    """[rows, 2] σq̂(action) - Σ_c π(c) σq̂(c) (`action` [rows] or an int)."""
    Q = q_all(net, data, rows, st)
    pi = data.probs(rows)
    a = np.broadcast_to(np.asarray(action, np.int64), (len(rows),))
    return Q[np.arange(len(rows)), a] - (pi[:, :, None] * Q).sum(1)


def m3b(data: pp.Data, nets: dict, sts: dict, a, boot_n, seed) -> dict:
    bunker = tb.action_id("Build_Bunker")
    lo, hi = (float(x) for x in a.m3b_window.split(","))
    races = set(a.m3b_races.split(","))
    macro = set(a.m3b_macro.split(","))
    sel = (data.row_eval & np.isin(data.row_kind, [pp.KINDS.index("teacher"), pp.KINDS.index("model")]) & (data.frame >= lo) & (data.frame < hi))
    sel &= ~data.chain_drop  # a dropped chained row's observation already shows its chain's commands
    sel &= np.array([data.g_race[g] in races for g in data.game]) if data.N else sel
    rows = np.nonzero(sel)[0]
    if len(rows):
        rows = rows[data.mask_rows(rows)[:, bunker]]
    cls = np.array([data.g_cls[g] for g in data.game[rows]], object)
    rush = cls == "rush"
    mac = np.isin(cls, list(macro))
    rows, rush = rows[rush | mac], rush[rush | mac]
    res = {"window": [lo, hi], "races": sorted(races), "macro_classes": sorted(macro)}
    if not len(rows):
        return res | {"rows": 0}
    groups = {"teacher": data.row_kind[rows] == pp.KINDS.index("teacher"), "model": data.row_kind[rows] == pp.KINDS.index("model")}
    groups["both"] = np.ones(len(rows), bool)
    for pname, net in nets.items():
        dq = delta_q(net, data, rows, bunker, sts[pname])
        res[pname] = {}
        for gname, gm in groups.items():
            if not gm.any():
                res[pname][gname] = {"rows": 0}
                continue
            rr = rows[gm]
            bt = ps.Boot(data.row_unit[rr], boot_n, seed)
            ru = rush[gm]
            ent = {"rows": int(len(rr)), "rush_rows": int(ru.sum()), "macro_rows": int((~ru).sum()),
                   "rush_games": int(len(np.unique(data.game[rr][ru]))), "macro_games": int(len(np.unique(data.game[rr][~ru])))}
            for j, q in enumerate(OUTCOMES):
                d = dq[gm, j]
                mr, mm = bt.mean(np.where(ru, d, np.nan)), bt.mean(np.where(~ru, d, np.nan))
                ent[q] = {"rush_mean": ps.ci(mr), "macro_mean": ps.ci(mm), "rush_minus_macro": ps.diff(mr, mm), "row_auc": ps.ci(bt.auc(ru, d))}
            res[pname][gname] = ent
    res["rows"] = int(len(rows))
    return res


def m3a(data: pp.Data, ev, preds: dict, bands, boot_n, seed) -> dict:
    out = {}
    fr = data.frame[ev]
    edges = list(bands)
    spans = {f"{edges[i]:g}-{edges[i + 1]:g}": (edges[i], edges[i + 1]) for i in range(len(edges) - 1)} | {"all": (edges[0], edges[-1])}
    wo = data.win_open[ev].astype(np.float64)
    for j, q in enumerate(OUTCOMES):
        yq = data.row_q[ev][:, j]
        out[q] = {}
        for sname, (lo, hi) in spans.items():
            m = np.isfinite(yq) & (fr >= lo) & (fr < hi)
            if not m.any():
                out[q][sname] = {"rows": 0}
                continue
            bt = ps.Boot(data.row_unit[ev[m]], boot_n, seed)
            y = yq[m] > 0.5
            sc = {"A": preds["A"]["q"][m, j], "U": preds["U"]["q"][m, j], "oracle_U_t+k": preds["U_next"]["q"][m, j], "win_open": wo[m]}
            au = {nm: bt.auc(y, s) for nm, s in sc.items() if s is not None and np.isfinite(s).any()}
            ent = {"rows": int(m.sum()), "games": int(len(np.unique(data.game[ev[m]]))), "auc": {nm: ps.ci(v) for nm, v in au.items()}}
            if "A" in au and "U" in au:
                ent["A_minus_U"] = ps.diff(au["A"], au["U"])
            out[q][sname] = ent
    return out


def m3d(data: pp.Data, nets: dict, k, ev, boot_n, seed) -> dict:
    rows = ev[data.is_op[ev]]
    if not len(rows) or "A" not in nets or "U" not in nets:
        return {"rows": 0}
    adv = delta_q(nets["A"], data, rows, data.action[rows], nets["A_st"])
    qu = ps.sigmoid(pp.predict(nets["U"], data, rows, k, nets["U_st"])["q"])
    out = {"rows": int(len(rows))}
    bt = ps.Boot(data.row_unit[rows], boot_n, seed)
    for j, q in enumerate(OUTCOMES):
        g = data.row_q[rows][:, j] - qu[:, j]
        out[q] = {"spearman": ps.ci(bt.spearman(adv[:, j], g)), "rows": int(np.isfinite(g).sum())}
    return out


# ---------------------------------------------------------------- verdict


def verdict(rep: dict, th: dict, ks) -> dict:
    """The 6.5 table per scoring target. Each row: matched True / False / None (an input missing)."""
    out = {}

    def g(d, *path):
        for p in path:
            if not isinstance(d, dict) or p not in d:
                return None
            d = d[p]
        return d

    def finite(x):
        return x is not None and np.isfinite(x)

    for q in OUTCOMES:
        crit = {}
        for k in ks:
            ks_ = str(k)
            c = {}
            for met in ("y", q):
                blk = g(rep, "M1", ks_, "pooled", "op", met)
                ga, gs, gh, r = g(blk, "gain", "A"), g(blk, "gain", "S"), g(blk, "gain", "H"), g(blk, "r", "pt")
                name = f"m1_{'y' if met == 'y' else 'outcome'}"
                # r = A / H is undefined when the oracle gains nothing: that k cannot be judged
                if ga is None or gs is None or gh is None or not finite(gh["pt"]) or gh["pt"] <= 0 or not finite(r):
                    c[name] = None
                    continue
                c[name] = bool(ga["pt"] > 0 and ga["lo"] > 0 and ga["pt"] >= th["s_ratio"] * gs["pt"] and r >= th["r_min"])
            r_med = g(rep, "M2", ks_, "M2a_y", "R_median")
            c["m2a"] = None if r_med is None else bool(r_med >= th["m2a"])
            d3a = g(rep, "M3a", ks_, q, "all", "A_minus_U")
            c["m3a"] = None if d3a is None or not finite(d3a["pt"]) else bool(d3a["pt"] >= th["m3a"])
            d3b = g(rep, "M3b", ks_, "A", "model", q, "rush_minus_macro")
            c["m3b"] = None if d3b is None or not finite(d3b["lo"]) else bool(d3b["pt"] > 0 and d3b["lo"] > 0)
            # own-state change: A's relative error cut over U, CI above 0 and >= s_ratio x the S control's cut
            ca, cs = g(rep, "M1", ks_, "pooled", "op", "od", "cut", "A"), g(rep, "M1", ks_, "pooled", "op", "od", "cut", "S")
            c["m1_own"] = None if ca is None or cs is None or not finite(ca["lo"]) or not finite(cs["pt"]) else \
                bool(ca["pt"] > 0 and ca["lo"] > 0 and ca["pt"] >= th["s_ratio"] * cs["pt"])
            crit[ks_] = c
        m2b = g(rep, "M2", str(max(ks)), "M2b", "rate", "pt")
        m2b_pass = None if m2b is None else bool(m2b >= th["m2b"])
        own_k = int(th.get("own_k") or 0)
        own_pass = (crit[str(own_k)]["m1_own"] if str(own_k) in crit else None) if own_k else True

        def all3(vals):
            if any(v is False for v in vals):
                return False
            return None if any(v is None for v in vals) else True

        def any_k(fn):
            vals = [fn(crit[str(k)]) for k in ks]
            if any(v is True for v in vals):
                return True
            return None if any(v is None for v in vals) else False

        m1 = lambda c: all3([c["m1_y"], c["m1_outcome"]])  # noqa: E731
        m3 = lambda c: all3([c["m3a"], c["m3b"]])  # noqa: E731
        row1 = all3([own_pass, any_k(lambda c: all3([m1(c), c["m2a"], m3(c)])), m2b_pass])
        row2 = all3([own_pass, any_k(lambda c: all3([m1(c), c["m2a"], None if m3(c) is None else not m3(c)])), m2b_pass])
        # A gain in the noise (r < r_noise on ŷ, every k) while the oracle's gain is significant (some k)
        # (the k where H is significant; r is defined there, H's gain being > 0)
        rs = [g(rep, "M1", str(k), "pooled", "op", "y", "r", "pt") for k in ks]
        h_lo = [g(rep, "M1", str(k), "pooled", "op", "y", "gain", "H", "lo") for k in ks]
        a_lo = [g(rep, "M1", str(k), "pooled", "op", "y", "gain", "A", "lo") for k in ks]
        sig = [i for i, x in enumerate(h_lo) if finite(x) and x > 0]
        if not all(finite(x) for x in h_lo) or any(not finite(rs[i]) for i in sig):
            row3 = None
        else:
            row3 = bool(sig) and all(rs[i] < th["r_noise"] for i in sig)
        if own_k:  # the own-state change cuts of A and H count too
            a_lo = a_lo + [g(rep, "M1", str(k), "pooled", "op", "od", "cut", "A", "lo") for k in ks]
            h_lo = h_lo + [g(rep, "M1", str(k), "pooled", "op", "od", "cut", "H", "lo") for k in ks]
        row4 = None if not all(finite(x) for x in a_lo + h_lo) else bool(all(x <= 0 for x in a_lo + h_lo))
        # human-only: rush-game ΔQ below 0 (CI), model-only: not (its CI reaches 0 or above)
        r5 = []
        for k in ks:
            hu = g(rep, "M3b", str(k), "A-human", "both", q, "rush_mean", "hi")
            mo = g(rep, "M3b", str(k), "A-model", "both", q, "rush_mean", "hi")
            r5.append(None if not (finite(hu) and finite(mo)) else bool(hu < 0 <= mo))
        row5 = True if any(v is True for v in r5) else (None if any(v is None for v in r5) else False)
        row6 = None if m2b_pass is None else not m2b_pass
        own_txt = f" + own-state change cut (k = {own_k})" if own_k else ""
        out[q] = {
            "criteria": crit, "m2b_k": max(ks), "m2b_rate": m2b, "m2b_pass": m2b_pass, "own_k": own_k or None, "own_pass": own_pass if own_k else None,
            "rows": [
                {"observation": f"M1 (ŷ and outcome) + M2a + M3a + M3b(model games) at one k{own_txt}, and M2b", "meaning": "계획 가능", "next": "S1·S2", "matched": row1},
                {"observation": f"M1·M2 pass{own_txt}, M3 fails", "meaning": "미래는 맞히나 채점이 약함", "next": "채점 대상 바꾸기, 모델 판 늘리기", "matched": row2},
                {"observation": f"M1 A gain in noise (r < {th['r_noise']:g} at every k where the H oracle gain is significant, one at least)", "meaning": "앞으로의 계획 묶음이 중요", "next": "(b) 옵션 검토", "matched": row3},
                {"observation": "M1 A and H gains both not significant (CI includes 0)" + (" on ŷ and the own-state change cut" if own_k else ""),
                 "meaning": "행동이 미래를 거의 안 바꿈", "next": "계획 경로 중단", "matched": row4},
                {"observation": "M3c rush-game ΔQ: human-only CI below 0, model-only CI not below 0", "meaning": "행동이 증거가 되는 착각", "next": "모델 판 위주 학습", "matched": row5},
                {"observation": f"M2b < {th['m2b']:g}", "meaning": "버그나 레이블 어긋남", "next": "파이프라인 확인", "matched": row6},
            ],
        }
    return out


# ---------------------------------------------------------------- markdown


def _f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, dict):
        if "pt" in x:
            return f"{_f(x['pt'], nd)} [{_f(x['lo'], nd)}, {_f(x['hi'], nd)}]"
        return "-"
    if isinstance(x, bool):
        return "예" if x else "아니오"
    if isinstance(x, (int, np.integer)):
        return str(x)
    return "-" if not np.isfinite(x) else f"{x:.{nd}f}"


def markdown(rep: dict) -> str:
    L = ["# S0 가능성 시험 결과 (jepa_planning S0)", "", f"- 체크포인트 sha256: `{rep['checkpoint']['checkpoint_sha256']}`",
         f"- 행 {rep['data']['rows']:,}, 판 {rep['data']['games']:,}, bootstrap {rep['settings']['boot']}회(판 단위, 95%)",
         f"- 판정 문턱(기본안(미확정)): {json.dumps(rep['settings']['thresholds'])}",
         f"- 설계(`plan_pred.Design`): {json.dumps(rep['settings'].get('design'))}" + (" (첫 실행과 같음, `--v1`)" if rep["settings"].get("v1") else ""),
         "", "## 데이터", "", "| 출처 | 종류 | 역할 | 판 | 평가 판 | 행 | 유효 레이블 중 체인 행 | 뺀 체인 행(운영 행동) |", "|---|---|---|---|---|---|---|---|"]
    for s in rep["data"]["sources"]:
        ch = s.get("chains") or {}
        L.append(f"| {s['name']} | {s['kind']} | {s['role']} | {s['games']} | {s['eval_games']} | {s['rows']:,} | {ch.get('chained_valid', 0):,} / {ch.get('valid', 0):,} | "
                 f"{ch.get('dropped', 0):,} ({ch.get('dropped_op', 0):,}) |")
    L += ["", f"- S 대조 상대 행이 같은 판인 행: {rep['data'].get('partner_same_game_rows', '-')}",
          "- 체인 행: 같은 판의 앞 결정이 1프레임 이내의 명령(NoOp 아님)인 행(040-R4가 한 프레임 명령 묶음의 2번째부터를 +1, +2, … 프레임으로 민 것). 관측에 묶음 명령의 효과가 이미 들어 있다."]
    frame_k = (rep["settings"].get("design") or {}).get("k_unit") == "frame"
    L += ["", "## M0 기초", "", "| 묶음 | 행 | 열린 종류 p50 | top4 / top8 / top16 확률 합(평균) | 결정 간격 p50(프레임) | k 쌍 간격 p50 | 운영 행동 비율 | top8에 운영 행동 | 체인 행 / 뺀 행 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for gname, m in rep["M0"].items():
        tm = " / ".join(_f(m["top_mass"][str(t)].get("mean")) for t in M0_TOP)
        dg = m.get("dec_gap") or {}
        gaps = ", ".join(f"k{k} {_f(v.get('p50'), 0)}" + (f" ({_f((dg.get(k) or {}).get('p50'), 0)}결정)" if frame_k else "") for k, v in m["frame_gap"].items() if k != "1")
        L.append(f"| {gname} | {m['rows']:,} | {_f(m['open_types'].get('p50'), 0)} | {tm} | {_f(m['frame_gap']['1'].get('p50'), 0)} | {gaps} | {_f(m['op_share'])} | {_f(m['top8_has_op'])} | "
                 f"{m.get('chained_valid', 0):,} / {m.get('chain_dropped', 0):,} |")
    L += ["", "- k의 단위가 결정이면(`--k-unit dec`) 같은 k라도 출처마다 프레임 간격이 다르다(위 표). 출처 묶음별 M1 표를 따로 보고, 프레임으로 맞추려면 `--k-unit frame`."]
    for k, blk in rep["M1"].items():
        L += ["", f"## M1 조건 이득 (k = {k}, 평가 판)", ""]
        for gname, sets in blk.items():
            for rs, mets in sets.items():
                if not mets:
                    continue
                L += [f"### {gname} · {'운영 행동 행' if rs == 'op' else '전체 행'} (행 {mets.get('_rows', '-')})", "",
                      "| 지표 | U | A 이득 | A+d 이득 | H 이득 | S 이득 | g_k 이득 | r = A/H | A/S |", "|---|---|---|---|---|---|---|---|---|"]
                for met, r in mets.items():
                    if met.startswith("_"):
                        continue
                    gn = r["gain"]
                    L.append(f"| {met} ({r['metric']}) | {_f(r['values'].get('U'))} | {_f(gn.get('A'), 4)} | {_f(gn.get('Ad'), 4)} | {_f(gn.get('H'), 4)} | {_f(gn.get('S'), 4)} | "
                             f"{_f(gn.get('gk'), 4)} | {_f(r.get('r'))} | {_f(r.get('a_over_s'), 2)} |")
                L.append("")
    L += ["## M1 내 상태 변화 오차 감소 (od: o_{t+k} − o_t, 6.3 열, 1 − err_X / err_U)", "",
          "| k | 묶음 | 행 집합 | U R²(변화) | A R²(변화) | A 감소 | A+d 감소 | H 감소 | S 감소 |", "|---|---|---|---|---|---|---|---|---|"]
    for k, blk in rep["M1"].items():
        for gname, sets in blk.items():
            for rs, mets in sets.items():
                od = mets.get("od") if mets else None
                if not od:
                    continue
                cut = od.get("cut", {})
                L.append(f"| {k} | {gname} | {'운영 행동' if rs == 'op' else '전체'} | {_f(od['values'].get('U'))} | {_f(od['values'].get('A'))} | {_f(cut.get('A'))} | "
                         f"{_f(cut.get('Ad'))} | {_f(cut.get('H'))} | {_f(cut.get('S'))} |")
    L += ["", "## M1 내 상태 열별 R² (pooled · 운영 행동 행, 6.3 열)", "", "| k | 열 | U | A − U | 변화 U | 변화 A − U | 변화 A 감소 |", "|---|---|---|---|---|---|---|"]
    for k, blk in rep["M1"].items():
        for col, e in (blk.get("pooled", {}).get("op", {}).get("_own_cols") or {}).items():
            if e["headline"]:
                dl = e.get("delta") or {}
                L.append(f"| {k} | {col} | {_f(e['values'].get('U'))} | {_f(e.get('gain_A'), 4)} | {_f((dl.get('values') or {}).get('U'))} | {_f(dl.get('gain_A'), 4)} | {_f(dl.get('cut_A'))} |")
    L += ["", "## M2 구별 (운영 행동 행, 예측기 A)", "", "| k | 행 | 후보 수 p50 | M2a R 중앙값 ŷ / ô / p̂ | M2b 비율 | M2b 건설 / 생산 |", "|---|---|---|---|---|---|"]
    for k, m in rep["M2"].items():
        if not m.get("rows"):
            L.append(f"| {k} | 0 | - | - | - | - |")
            continue
        r = " / ".join(_f(m[f"M2a_{x}"]["R_median"], 2) for x in ("y", "o", "p"))
        b = m.get("M2b", {})
        L.append(f"| {k} | {m['rows']:,} | {_f(m['candidates'].get('p50'), 0)} | {r} | {_f(b.get('rate'))} | {_f(b.get('build', {}).get('rate'))} / {_f(b.get('train', {}).get('rate'))} |")
    L += ["", "## M3a 결과 순위 AUC (평가 판, 무승부·미정 제외)", "", "| k | 채점 대상 | 시간대 | 행 | q̂_A | q̂_U | win_open | 오라클 q̂_U(t+k) | A − U |", "|---|---|---|---|---|---|---|---|---|"]
    for k, m in rep["M3a"].items():
        for q, bands in m.items():
            for b, e in bands.items():
                au = e.get("auc", {})
                L.append(f"| {k} | {q} | {b} | {e.get('rows', 0):,} | {_f(au.get('A'))} | {_f(au.get('U'))} | {_f(au.get('win_open'))} | {_f(au.get('oracle_U_t+k'))} | {_f(e.get('A_minus_U'))} |")
    L += ["", "## M3b·M3c 러시 정상 확인 (ΔQ = q̂(벙커) − Σπq̂)", "", "| k | 예측기 | 묶음 | 채점 대상 | 행(러시/운영) | 러시 ΔQ | 운영 ΔQ | 러시 − 운영 | 행 AUC |", "|---|---|---|---|---|---|---|---|---|"]
    for k, m in rep["M3b"].items():
        for pname in ("A", "A-human", "A-model"):
            for gname, e in (m.get(pname) or {}).items():
                if not e.get("rows"):
                    L.append(f"| {k} | {pname} | {gname} | - | 0 | - | - | - | - |")
                    continue
                for q in OUTCOMES:
                    x = e[q]
                    L.append(f"| {k} | {pname} | {gname} | {q} | {e['rush_rows']}/{e['macro_rows']} | {_f(x['rush_mean'], 4)} | {_f(x['macro_mean'], 4)} | {_f(x['rush_minus_macro'], 4)} | {_f(x['row_auc'])} |")
    L += ["", "## M3d advantage 일치 (운영 행동 행, Spearman)", "", "| k | 채점 대상 | 행 | Spearman |", "|---|---|---|---|"]
    for k, m in rep["M3d"].items():
        for q in OUTCOMES:
            if q in m:
                L.append(f"| {k} | {q} | {m[q]['rows']:,} | {_f(m[q]['spearman'])} |")
    L += ["", "## 판정표 (6.5, 문턱 기본안(미확정))", ""]
    for q, v in rep["verdict"].items():
        L += [f"### 채점 대상 {q}", "", "| 관찰 | 해석 | 다음 | 해당 |", "|---|---|---|---|"]
        for r in v["rows"]:
            L.append(f"| {r['observation']} | {r['meaning']} | {r['next']} | {'판단 불가(입력 없음)' if r['matched'] is None else _f(r['matched'])} |")
        L += ["", "| k | M1 ŷ | M1 결과 | M1 내 상태 변화 | M2a | M3a | M3b(모델 판) |", "|---|---|---|---|---|---|---|"]
        for k, c in v["criteria"].items():
            L.append(f"| {k} | {_f(c['m1_y'])} | {_f(c['m1_outcome'])} | {_f(c.get('m1_own'))} | {_f(c['m2a'])} | {_f(c['m3a'])} | {_f(c['m3b'])} |")
        L += ["", f"- M2b(k = {v['m2b_k']}) {_f(v['m2b_rate'])}: {_f(v['m2b_pass'])}"]
        if v.get("own_k"):
            L.append(f"- M1 내 상태 변화 기준(k = {v['own_k']}, A 감소 신뢰구간 > 0, S 대조의 {rep['settings']['thresholds']['s_ratio']:g}배 이상): "
                     f"{'판단 불가' if v['own_pass'] is None else _f(v['own_pass'])}")
        L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- report


def report(data: pp.Data, a, ks, variants, models: Path) -> dict:
    th = {k: getattr(a, f"th_{k}") for k in TH_DEFAULTS}
    th["own_k"] = 0 if a.v1 else a.th_own_k
    rep = {"kind": "plan-s0", "checkpoint": data.checkpoint, "data": pp.describe(data),
           "settings": {"ks": ks, "variants": variants, "boot": a.boot, "boot_seed": a.boot_seed, "eval_frac": a.eval_frac, "split_seed": a.split_seed,
                        "design": data.design.as_dict(), "v1": bool(a.v1),
                        "steps": a.steps, "batch": a.batch, "lr": a.lr, "seed": a.seed, "thresholds": th, "bands": a.bands, "m3b_races": a.m3b_races,
                        "m3b_window": a.m3b_window, "m3b_macro": a.m3b_macro, "drop": a.drop, "keep_bot_perspective": a.keep_bot_perspective,
                        "maps": a.maps, "left_out": data.dropped, "own_headline": list(OWN_HEADLINE),
                        "sd_floors": {"own": pp.OWN_SD_FLOOR, "privileged": pp.PRIV_SD_FLOOR}},
           "M0": m0(data, ks), "M1": {}, "M2": {}, "M3a": {}, "M3b": {}, "M3d": {}, "models": {}}
    bands = [float(x) for x in a.bands.split(",")]
    for k in ks:
        ks_ = str(k)
        nets, sts = {}, {}
        for name, v, kk, kinds in model_specs([k], variants, not a.no_m3c):
            path = models / f"{name}.pt"
            if not path.exists():
                continue
            net, st, meta = pp.load(path, data)
            if meta.get("data") != fingerprint(data):
                raise SystemExit(f"{path}: trained on other data; run --stage train (--retrain)")
            label = name.split("_k")[0]
            if "d_mu" not in st:  # first-run models: the change statistics of their training rows, for the change metrics
                st = {**st, **data.delta_stats(data.pair_rows(k, "train", kinds), k)}
            nets[label], sts[label] = net, st
            rep["models"][name] = {x: meta[x] for x in ("rows", "steps", "batch", "lr", "seed", "secs") if x in meta} | {"last": meta["log"][-1] if meta.get("log") else None}
        ev = data.pair_rows(k, "eval")
        if "U" not in nets or not len(ev):
            rep["M1"][ks_] = {}
            continue
        st = sts["U"]  # the main predictors of one k share the standardisation (the same training rows)
        tg = targets(data, ev, k, st, with_y=False)
        rows, qlog = eval_rows({nm: nets[nm] for nm in variants if nm in nets}, data, ev, k, st)
        names = [n for n in ["U", *[v for v in variants if v != "U"], "gk"] if n in rows]
        preds = {nm: {"q": qlog[nm]} for nm in ("A", "U") if nm in qlog}
        is_op = data.is_op[ev]
        rep["M1"][ks_] = {}
        for gname, gsel in kind_groups(data, ev).items():
            rep["M1"][ks_][gname] = {}
            for rs, m in (("all", np.ones(len(gsel), bool)), ("op", is_op[gsel])):
                sel = gsel[m]
                if not len(sel):
                    rep["M1"][ks_][gname][rs] = {}
                    continue
                bt = ps.Boot(data.row_unit[ev[sel]], a.boot, a.boot_seed)
                blk = m1_block(bt, rows, tg, sel, names)
                blk["_rows"] = int(len(sel))
                blk["_own_cols"] = own_cols_block(bt, rows, tg, sel, names, data.own_names)
                rep["M1"][ks_][gname][rs] = blk
        print(f"[report] k={k} M1 done", flush=True)
        if "A" in nets:
            rep["M2"][ks_] = m2(data, nets["A"], sts["A"], k, ev, a.boot, a.boot_seed)
            print(f"[report] k={k} M2 done", flush=True)
            preds["U_next"] = pp.predict(nets["U"], data, ev, k, st, inputs_h=data.nxt[k][ev], keep=("q",))
            for nm in ("A", "U", "U_next"):
                preds[nm]["q"] = ps.sigmoid(preds[nm]["q"])
            rep["M3a"][ks_] = m3a(data, ev, preds, bands, a.boot, a.boot_seed)
            m3nets = {nm: nets[nm] for nm in ("A", "A-human", "A-model") if nm in nets}
            rep["M3b"][ks_] = m3b(data, m3nets, sts, a, a.boot, a.boot_seed)
            rep["M3d"][ks_] = m3d(data, {"A": nets["A"], "U": nets["U"], "U_st": st, "A_st": sts["A"]}, k, ev, a.boot, a.boot_seed)
            print(f"[report] k={k} M3 done", flush=True)
    rep["verdict"] = verdict(rep, th, ks)
    return rep


def run(a) -> int:
    from uranium.diag.dump import redact

    if a.threads:
        torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    ks = [int(x) for x in a.ks.split(",") if x]
    variants = [v for v in a.variants.split(",") if v]
    if "U" not in variants:
        raise SystemExit("plan-s0: U (the baseline) must be among --variants")
    srcs = [pp.parse_src(s) for s in a.src]
    drop = {}
    for spec in a.drop:
        name, sep, path = spec.partition("=")
        if not sep or name not in {s.name for s in srcs}:
            raise SystemExit(f"--drop {spec!r}: expected NAME=JSON with NAME a --src name")
        drop[name] = json.loads(Path(path).read_text(encoding="utf-8"))
    maps = [int(x) for x in a.maps.split(",") if x.strip()] or None
    design = design_of(a)
    if not a.v1 and a.th_own_k and a.th_own_k not in ks:
        raise SystemExit(f"plan-s0: --th-own-k {a.th_own_k} is not among --ks {ks} (0 = no own-state criterion)")
    t0 = time.time()
    data = pp.Data(srcs, ks, a.eval_frac, a.split_seed, drop, keep_bot=a.keep_bot_perspective, maps=maps, design=design)
    print(f"[data] design: {design.as_dict()}", flush=True)
    print(f"[data] left out: {data.dropped}", flush=True)
    print(f"[data] chained rows (valid label / dropped / dropped operation actions): "
          f"{ {n: (c['chained_valid'], c['dropped'], c['dropped_op']) for n, c in data.chain_counts.items()} }", flush=True)
    print(f"[data] rows {data.N} games {data.G} {time.time() - t0:.0f}s", flush=True)
    out = Path(a.out)
    models = out / "models"
    out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("all", "train"):
        log = train_all(data, a, ks, variants, models)
        (out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    if a.stage in ("all", "report"):
        rep = report(data, a, ks, variants, models)
        rep["sources"] = [{"name": s.name, "kind": s.kind, "role": s.role, "path": redact(s.path)} for s in srcs]
        rep["secs"] = round(time.time() - t0, 1)
        stem = "s0" + (f"-{a.tag}" if a.tag else "")
        (out / f"{stem}.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
        (out / f"{stem}.md").write_text(markdown(rep), encoding="utf-8")
        print(f"[out] {out / (stem + '.json')} {out / (stem + '.md')}", flush=True)
    return 0
