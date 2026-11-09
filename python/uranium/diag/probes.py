"""`python -m uranium diag probes`: cue probes per enemy race on a `diag dump` (reports/jepa_survey.md
7.2-7.4; never pooled over races). Offline labels from the privileged counts; the stages are the
model's own activations.

Targets (rows with an enemy focus token before `--act-until` of the dump):
- `hp|unf_token`: the token's HP ratio, unfinished tokens (the cue itself; pooling bottleneck)
- `start|unf_token`, `start|all_token_rows`: the true first start of the focus type (seconds x 1e-3
  frames), Zerg = completion - build time (the HP clock)
- `rush|...`, `macro|...`: the build class (7.3 draft rules), AUC
- `count|focus_rows`: the enemy max-seen count of the focus type (the scalar input the c path
  carries)
- `focus2_by_300|all_token_rows`: two focus buildings started before 3:00 (AUC)

Every target: a ridge and an MLP(64) per stage, game-level 5-fold CV, game bootstrap CIs, the
nuisance-only ridge, each stage's gain over it (`uranium.diag.probe`, `--gain`, default `resid`),
and a shuffled-label control on [h, c] for per-game targets."""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from uranium.diag import tables as tb
from uranium.diag.features import game_labels
from uranium.diag.probe import GAIN_METHODS, onehot, run_task

DEFAULT_STAGES = "token,e_pool,sp_pool,c,x_in,L1.g1,L1.out,h,hc"


def add_args(p) -> None:
    p.add_argument("dump", help="a `diag dump` output folder")
    p.add_argument("--races", default="T,P,Z")
    p.add_argument("--stages", default=DEFAULT_STAGES, help="activation stages (missing ones are skipped); hc = [h, c]")
    p.add_argument("--tasks", default="", help="comma-separated task names to run (default all)")
    p.add_argument("--gain", choices=GAIN_METHODS, default="resid", help="gain over nuisance (uranium.diag.probe)")
    p.add_argument("--boot", type=int, default=500)
    p.add_argument("--cap", type=int, default=0, help="rows per game, evenly spaced (0 = all)")
    p.add_argument("--no-mlp", action="store_true")
    p.add_argument("--device", default="cpu", help="device of the MLP probes")
    p.add_argument("--min-rows", type=int, default=50)
    p.add_argument("--min-games", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", help="output JSON (default <dump>/probes.json)")


def game_table(R: dict, games: dict) -> dict:
    """{gid: labels} for every game of a known race (all its dumped rows, frame order)."""
    order = np.lexsort((R["frame"], R["game"]))
    g = R["game"][order]
    cuts = np.nonzero(np.diff(g))[0] + 1
    out = {}
    for seg in np.split(order, cuts):
        gid = int(R["game"][seg[0]])
        race = games.get(gid, {}).get("race", "U")
        if race not in tb.RACES:
            continue
        e = game_labels({k: v[seg] for k, v in R.items()}, race)
        e["split"] = games[gid]["split"]
        out[gid] = e
    return out


def nuisance(r: dict, first_sight) -> np.ndarray:
    fs = np.nan_to_num(first_sight, nan=float(np.nanmax(r["frame"])))
    cols = [r["frame"] / 1e3, r["vis_worker"], r["sc_worker"], r["sc_town"], r["sc_gas"], r["sc_gas"] > 0, r["sc_army"], r["vis_army"], r["tok_army"], r["tok_bld"],
            r["tok_town"], r["tok_gas"], fs / 1e3, r["foc_rem"]]
    return np.concatenate([np.stack(cols, 1).astype(np.float32), onehot(r["map"]), onehot(r["start"]), onehot(r["split"])], 1)


def raw_input(r: dict, race: str) -> np.ndarray:
    s = tb.race(race)
    hp = np.nan_to_num(r["foc_hp"])
    em = r["est_min"]
    cols = [r["frame"] / 1e3, np.nan_to_num(r["foc_since"]) / 1e3, hp / s["max_hp"], np.sqrt(hp) / 16, np.nan_to_num(r["foc_comp"]), np.nan_to_num(r["foc_cons"]), r["foc_rem"],
            np.nan_to_num(r["foc_seen"]) / 1e3, np.nan_to_num(em, nan=0.0) / 1e3, np.isnan(em), r["n_foc"], r["n_foc_vis"], r["n_foc_unf"], r["n_foc_done"], r["sc_focus"],
            np.nan_to_num(r["foc_shield"]) / max(s["max_shields"], 1)]
    return np.stack(cols, 1).astype(np.float32)


def select_rows(R, A, gt, race, cap):
    """Activation rows of the race's games (index into A), `cap` evenly spaced per game."""
    game = R["game"][A["idx"]]
    ok = np.array([gt.get(int(x), {}).get("race") == race for x in game], bool)
    keep = np.zeros(len(game), bool)
    cand = np.nonzero(ok)[0]
    cand = cand[np.argsort(game[cand], kind="stable")]
    cuts = np.nonzero(np.diff(game[cand]))[0] + 1
    for w in np.split(cand, cuts):
        if cap and len(w) > cap:
            w = w[np.linspace(0, len(w) - 1, cap).round().astype(int)]
        keep[w] = True
    return np.nonzero(keep)[0]


def headroom(gt: dict, race: str) -> dict:
    gs = [e for e in gt.values() if e["race"] == race]
    out = {"games": len(gs), "class": dict(Counter(e["cls"] for e in gs)), "by_class": {}}
    for c in sorted({e["cls"] for e in gs}):
        cg = [e for e in gs if e["cls"] == c]
        seen4 = [e for e in cg if e["first_sight"] is not None and e["first_sight"] < 5760]
        st = [e["start1"] for e in cg if e["start1"] is not None]
        out["by_class"][c] = {"games": len(cg), "start1_p50": float(np.median(st)) if st else None, "seen_lt4": len(seen4),
                              "unfinished_at_first_sight_lt4": sum(1 for e in seen4 if e["sight_comp"] is False),
                              "seen_count2_lt4": sum(1 for e in cg if e["seen2_frame"] is not None and e["seen2_frame"] < 5760)}
    errs = [e["est_err_start1"] for e in gs if e.get("est_err_start1") is not None]
    out["est_err_first_sight_vs_start1"] = {"n": len(errs), "median": float(np.median(errs)) if errs else None, "p90_abs": float(np.percentile(np.abs(errs), 90)) if errs else None}
    return out


def probe_race(R, A, gt, race, a, rng) -> dict:
    ai = select_rows(R, A, gt, race, a.cap)
    res = {"race": race, "rows": int(len(ai)), "headroom": headroom(gt, race)}
    if len(ai) < a.min_rows:
        res["skipped"] = f"{len(ai)} rows"
        return res
    ri = A["idx"][ai]
    r = {k: v[ri] for k, v in R.items()}
    g = r["game"]
    G = lambda key: np.array([np.nan if gt[int(x)].get(key) is None else gt[int(x)][key] for x in g], np.float64)  # noqa: E731
    start1 = G("start1")
    cls = np.array([gt[int(x)]["cls"] for x in g])
    stages = {}
    for st in a.stages.split(","):
        if st == "hc":
            if "h" in A and "c" in A:
                stages["hc"] = np.concatenate([A["h"][ai], A["c"][ai]], 1).astype(np.float32)
        elif st in A:
            stages[st] = A[st][ai].astype(np.float32)
    nuis = nuisance(r, G("first_sight"))
    inp = raw_input(r, race)
    unf = r["foc_comp"] < 0.5
    has_start = ~np.isnan(start1)
    mx = tb.race(race)["max_hp"]
    want = set(filter(None, a.tasks.split(",")))
    tasks = {}
    t0 = time.time()

    def task(name, sel, y, cls_, per_game, with_input=True):
        if want and name not in want:
            return
        if sel.sum() < a.min_rows or len(np.unique(g[sel])) < a.min_games:
            tasks[name] = {"rows": int(sel.sum()), "skipped": "too few rows or games"}
            return
        if cls_:
            pos = len(np.unique(g[sel][y > 0.5]))
            if pos < 5 or pos > len(np.unique(g[sel])) - 5:
                tasks[name] = {"rows": int(sel.sum()), "skipped": f"positive games {pos}"}
                return
        st = {k: v[sel] for k, v in stages.items()}
        if with_input:
            st = {"input": inp[sel], **st}
        tasks[name] = run_task(st, nuis[sel], y, cls_, g[sel], g[sel], rng, boot_n=a.boot, mlp=not a.no_mlp, gain=a.gain,
                               shuffled_key="hc" if per_game else None, mlp_device=a.device)
        print(f"  {race} {name}: rows {int(sel.sum())} {time.time() - t0:.0f}s", flush=True)

    task("hp|unf_token", unf, np.nan_to_num(r["foc_hp"])[unf] / mx, False, False, with_input=False)
    for sub, s in (("unf_token", unf & has_start), ("all_token_rows", has_start)):
        task(f"start|{sub}", s, start1[s] / 1e3, False, True)
    allr = np.ones(len(g), bool)
    for c in ("rush", "macro"):
        for sub, s in (("unf_token", unf), ("all_token_rows", allr)):
            task(f"{c}|{sub}", s, (cls[s] == c).astype(np.float64), True, True)
    task("count|focus_rows", allr, r["sc_focus"].astype(np.float64), False, False)
    f2 = np.array([bool(gt[int(x)]["focus2_by_300"]) for x in g])
    task("focus2_by_300|all_token_rows", allr, f2.astype(np.float64), True, True)
    res["tasks"] = tasks
    res["secs"] = round(time.time() - t0, 1)
    return res


def run(a) -> int:
    from uranium.diag.dump import load

    stages = [s for s in a.stages.split(",") if s != "hc"] + (["h", "c"] if "hc" in a.stages.split(",") else [])
    R, A, games, man = load(Path(a.dump), stages)
    if "idx" not in A:
        raise SystemExit(f"{a.dump}: no activations (dumped without a checkpoint?)")
    gt = game_table(R, games)
    rng = np.random.default_rng(a.seed)
    out = {"dump": {k: man.get(k) for k in ("checkpoint", "checkpoint_sha256", "data", "splits", "act_until", "act_every", "backend", "precision")},
           "gain_method": a.gain, "boot": a.boot, "cap": a.cap, "mlp": not a.no_mlp, "races": {}}
    for race in a.races.split(","):
        print(f"[{race}]", flush=True)
        out["races"][race] = probe_race(R, A, gt, race, a, rng)
    path = Path(a.out) if a.out else Path(a.dump) / "probes.json"
    path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"[out] {path}")
    for race, rr in out["races"].items():
        for tn, t in rr.get("tasks", {}).items():
            if "metrics" not in t:
                continue
            m = t["metrics"]
            line = " ".join(f"{k.split('|')[0]} {v['pt']:.3f}" for k, v in m.items() if k.endswith("|ridge") and "+nuis" not in k)
            print(f"{race} {tn} ({t['metric']}, rows {t['rows']}, games {t['games']}): {line}")
    return 0
