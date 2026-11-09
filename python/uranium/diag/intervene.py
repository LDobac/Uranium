"""`python -m uranium diag intervene`: counterfactual effects from a `diag dump --cf` (reports/
jepa_survey.md 7.2 (4), 10.1, 10.2.2), per enemy race.

Per game, window 1:30-4:00 (`--window`) on the steps where the action is allowed (type mask):
- `bunker_any`: P(at least one Build_Bunker) = 1 - prod(1 - p)
- `marine_sum`: expected Train_Marine orders = sum p
- `rax2_any`: P(at least one Build_Barracks) on steps with exactly one own Barracks (the 2nd)
and `label_p`: p(Build_Bunker) at the game's first teacher bunker label in the window.

Effects (game mean and game-bootstrap CI, share of games >= 0.15, share > 0):
- `start`: early - late anchor start; games whose anchor is seen before the window end (`edited`),
  and the subset unfinished at first sight (`edited_unf`)
- `count_sc`, `count_tok`: max-seen count +1 - (-1), scalar only / scalar + token; games where the
  focus type was seen before the window end
- `real`: the unedited pass's values (the base rates)"""

import csv
import json
from pathlib import Path

import numpy as np

from uranium.diag import tables as tb


def add_args(p) -> None:
    p.add_argument("dump", help="a `diag dump --cf` output folder")
    p.add_argument("--races", default="T,P,Z")
    p.add_argument("--window", default="2160,5760", help="frames from,to (default 1:30-4:00)")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", help="output JSON (default <dump>/intervene.json); per-game CSV next to it")


def any_p(p) -> float:
    return float(1 - np.exp(np.log1p(-np.clip(p, 0, 0.999999)).sum())) if len(p) else 0.0


def per_game(r: dict, passes: list, w0: float, w1: float) -> dict:
    fr = r["frame"]
    pb = r["pb"].astype(np.float64)
    win = (fr >= w0) & (fr < w1)
    g = {}
    for name, ai, el, extra in (("bunker_any", 0, "el_b", None), ("marine_sum", 1, "el_m", None), ("rax2_any", 2, "el_r", r["own_rax"] == 1)):
        m = win & r[el].astype(bool)
        if extra is not None:
            m &= extra
        if not m.any() or np.isnan(pb[m][:, :, ai]).any():
            continue
        for pi, p in enumerate(passes):
            v = pb[m, pi, ai]
            g[f"{name}|{p}"] = float(v.sum()) if name.endswith("_sum") else any_p(v)
        g[f"{name}|steps"] = int(m.sum())
    lab = win & r["ok"].astype(bool) & (r["label"] == tb.actions()["bunker"]) & r["el_b"].astype(bool)
    if lab.any():
        i = int(np.argmax(lab))
        if not np.isnan(pb[i, :, 0]).any():
            g["label_frame"] = float(fr[i])
            for pi, p in enumerate(passes):
                g[f"label_p|{p}"] = float(pb[i, pi, 0])
    return g


def ci(v, boot: int, rng) -> dict:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], np.float64)
    if not len(v):
        return {"n": 0}
    reps = rng.choice(v, (boot, len(v))).mean(1)
    return {"n": int(len(v)), "mean": float(v.mean()), "lo": float(np.percentile(reps, 2.5)), "hi": float(np.percentile(reps, 97.5)), "p50": float(np.median(v)),
            "share_ge_0.15": float((v >= 0.15).mean()), "share_pos": float((v > 0).mean())}


def effects(G: list, a_pass: str, b_pass: str, boot: int, rng) -> dict:
    out = {}
    for m in ("bunker_any", "marine_sum", "rax2_any", "label_p"):
        out[m] = ci([g.get(f"{m}|{a_pass}", np.nan) - g.get(f"{m}|{b_pass}", np.nan) for g in G], boot, rng)
    return out


def run(a) -> int:
    from uranium.diag.dump import load
    from uranium.diag.probes import game_table

    R, _, games, man = load(Path(a.dump), stages=[])
    passes = man["passes"]
    if "early" not in passes:
        raise SystemExit(f"{a.dump}: dumped without --cf")
    w0, w1 = (float(x) for x in a.window.split(","))
    gt = game_table(R, games)
    rng = np.random.default_rng(a.seed)
    order = np.lexsort((R["frame"], R["game"]))
    gcol = R["game"][order]
    cuts = np.nonzero(np.diff(gcol))[0] + 1
    G = []
    for seg in np.split(order, cuts):
        gid = int(R["game"][seg[0]])
        if gid not in gt:
            continue
        r = {k: v[seg] for k, v in R.items()}
        e = gt[gid]
        g = {"gid": gid, "race": e["race"], "cls": e["cls"], "anchor_first": e["anchor_first"], "anchor_unf_at_sight": e["anchor_unf_at_sight"],
             "focus_seen": bool((r["sc_focus"][r["frame"] < w1] >= 1).any())}
        g.update(per_game(r, passes, w0, w1))
        G.append(g)
    res = {"dump": {k: man.get(k) for k in ("checkpoint", "checkpoint_sha256", "data", "splits", "cf_starts", "backend", "precision")}, "window": [w0, w1], "races": {}}
    for race in a.races.split(","):
        V = [g for g in G if g["race"] == race]
        E = [g for g in V if g["anchor_first"] is not None and g["anchor_first"] < w1]
        C = [g for g in V if g["focus_seen"]]
        R_ = {"games": len(V), "edited": len(E), "edited_unf": sum(1 for g in E if g["anchor_unf_at_sight"]), "count_edited": len(C), "by_class": {}}
        for cls in ("all", "rush", "macro"):
            sel = lambda xs: [g for g in xs if cls == "all" or g["cls"] == cls]  # noqa: E731
            R_["by_class"][cls] = {
                "start": effects(sel(E), "early", "late", a.boot, rng),
                "start_unf": effects([g for g in sel(E) if g["anchor_unf_at_sight"]], "early", "late", a.boot, rng),
                "count_sc": effects(sel(C), "sc+1", "sc-1", a.boot, rng),
                "count_tok": effects(sel(C), "tok+1", "tok-1", a.boot, rng),
                "real": {m: ci([g.get(f"{m}|real", np.nan) for g in sel(V)], a.boot, rng) for m in ("bunker_any", "marine_sum", "rax2_any", "label_p")},
            }
        res["races"][race] = R_
    path = Path(a.out) if a.out else Path(a.dump) / "intervene.json"
    path.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    keys = sorted({k for g in G for k in g})
    with open(path.with_suffix(".games.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(G)
    print(f"[out] {path}")
    for race, R_ in res["races"].items():
        d = R_["by_class"]["all"]
        s, c = d["start"]["bunker_any"], d["count_tok"]["bunker_any"]
        if s.get("n"):
            print(f"{race}: games {R_['games']} edited {R_['edited']} | start dP(bunker) {s['mean']:+.4f} [{s['lo']:+.4f},{s['hi']:+.4f}] "
                  f"marine {d['start']['marine_sum'].get('mean', float('nan')):+.3f} rax2 {d['start']['rax2_any'].get('mean', float('nan')):+.4f}"
                  + (f" | count(tok) dP(bunker) {c['mean']:+.4f}" if c.get("n") else ""))
    return 0
