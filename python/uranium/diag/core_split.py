"""`python -m uranium diag core-split`: where inside the GTrXL core a cue is lost (reports/
jepa_survey.md 10.3), on a `diag dump` with the core sub-points (`--core-points gate` or `full`).

Per enemy race, on the rows with an unfinished focus token:
- probes: ridge R² (game-level CV, game bootstrap CI) of the token's HP ratio at every stage from
  the token to h_t, in the order the information flows
- gates: each GRU gate g = (1-z)·x + z·h~ of every layer recomputed from its stored inputs (f32):
  reconstruction error, mean z, variance shares of the kept residual (1-z)·x and the written z·h~,
  and the HP R² of x, (1-z)·x, z·h~, h~ and g
- retention: for a list of row variables, R²(h_t) / R²(core input projection x_in) over every
  activation row of the race (how selectively the core keeps what reaches it)"""

import json
from pathlib import Path

import numpy as np

from uranium.diag import tables as tb
from uranium.diag.probe import boot, folds_of, r2, ridge_cv, ridge_cv_multi

RETENTION_VARS = ("frame", "sc_worker", "sc_focus", "sc_enemy_total", "own_marine", "own_rax", "vis_enemy", "tok_enemy", "n_foc", "foc_rem")


def add_args(p) -> None:
    p.add_argument("dump", help="a `diag dump` folder with core sub-points")
    p.add_argument("--checkpoint", help="the dump's checkpoint (for the gate weights); default: the manifest's, if it is a local path")
    p.add_argument("--races", default="T,P,Z")
    p.add_argument("--boot", type=int, default=300)
    p.add_argument("--max-rows", type=int, default=60000, help="random rows kept for the retention probes")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", help="output JSON (default <dump>/core_split.json)")


def stage_order(A: dict) -> list[str]:
    layers = sorted({int(k[1:].split(".")[0]) for k in A if k.startswith("L") and "." in k})
    out = [k for k in ("token", "e_pool", "core_in", "x_in") if k in A]
    for l in layers:
        out += [f"L{l}.{p}" for p in ("a_pre", "a", "g1", "f_pre", "f", "out") if f"L{l}.{p}" in A]
    return out + [k for k in ("h",) if k in A]


def gate_inputs(l: int, which: int) -> tuple[str, str, str]:
    """(x key, y key, output key) of gate `which` (1 after attention, 2 after the FFN) of layer l."""
    if which == 1:
        return ("x_in" if l == 1 else f"L{l - 1}.out"), f"L{l}.a", f"L{l}.g1"
    return f"L{l}.g1", f"L{l}.f", f"L{l}.out"


def gate_split(gate, x, y):
    """(keep, write, h~, z) of a GTrXL GRU gate (uranium.model.core.GruGate), f32 numpy."""
    import torch

    from uranium.model.core import GATE_BIAS

    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    with torch.no_grad():
        r = torch.sigmoid(gate.wr(yt) + gate.ur(xt))
        z = torch.sigmoid(gate.wz(yt) + gate.uz(xt) - GATE_BIAS)
        hh = torch.tanh(gate.wg(yt) + gate.ug(r * xt))
        keep, write = xt * (1 - z), z * hh
    return keep.numpy(), write.numpy(), hh.numpy(), z.numpy()


def run(a) -> int:
    import torch

    from uranium import _core, ckpt
    from uranium.diag.dump import load
    from uranium.diag.probes import game_table, select_rows

    R, A, games, man = load(Path(a.dump))
    if not any(k.startswith("L1.") for k in A):
        raise SystemExit(f"{a.dump}: no GTrXL sub-points (dump with --core-points gate or full, a GTrXL checkpoint)")
    cdir = a.checkpoint or man.get("checkpoint")
    if not cdir or str(cdir).startswith("<shared>"):
        raise SystemExit("core-split: pass --checkpoint (the manifest's path is redacted)")
    cdir = Path(_core.resolve_checkpoint(str(cdir)))
    net, _ = ckpt.load_model(cdir, torch.device("cpu"))
    if ckpt.policy_version(cdir) != man.get("checkpoint_sha256"):
        raise SystemExit("core-split: --checkpoint is not the dump's checkpoint (sha256 differs)")
    gt = game_table(R, games)
    rng = np.random.default_rng(a.seed)
    order = stage_order(A)
    nl = len(net.core.layers)
    out = {"dump": {k: man.get(k) for k in ("checkpoint", "checkpoint_sha256", "data", "splits", "act_until", "backend", "precision")}, "stages": order, "races": {}}
    for race in a.races.split(","):
        ai = select_rows(R, A, gt, race, 0)
        r = {k: v[A["idx"][ai]] for k, v in R.items()}
        res = {"rows": int(len(ai))}
        unf = r["foc_comp"] < 0.5
        if unf.sum() >= 50 and len(np.unique(r["game"][unf])) >= 10:
            sel = ai[unf]
            games_ = r["game"][unf]
            fold = folds_of(games_)
            y = np.nan_to_num(r["foc_hp"][unf]) / tb.race(race)["max_hp"]
            preds = {st: ridge_cv(A[st][sel].astype(np.float32), y, games_, fold)[0] for st in order}
            res["hp_probes"] = {"rows": int(len(sel)), "games": int(len(np.unique(games_))), "r2": boot(r2, y, preds, games_, a.boot, rng)}
            gates = {}
            for l in range(1, nl + 1):
                layer = net.core.layers[l - 1]
                for which, gate in ((1, layer.gate1), (2, layer.gate2)):
                    xk, yk, gk = gate_inputs(l, which)
                    if not all(k in A for k in (xk, yk, gk)):
                        continue
                    x, yy, gs = (A[k][sel].astype(np.float32) for k in (xk, yk, gk))
                    keep, write, hh, z = gate_split(gate, x, yy)
                    rec = keep + write
                    tot = rec.var(0).sum()
                    d = {"x": xk, "y": yk, "recon_rel_err": float(np.linalg.norm(rec - gs) / max(np.linalg.norm(gs), 1e-12)), "z_mean": float(z.mean()),
                         "var_share_keep": float(keep.var(0).sum() / tot), "var_share_write": float(write.var(0).sum() / tot)}
                    for name, X in (("x", x), ("keep", keep), ("write", write), ("h_tilde", hh), ("g", rec)):
                        d[f"r2_{name}"] = r2(y, ridge_cv(X, y, games_, fold)[0])
                    gates[f"L{l}.gate{which}"] = d
            res["gates"] = gates
        else:
            res["hp_probes"] = {"skipped": f"{int(unf.sum())} unfinished-token rows"}
        if "x_in" in A and "h" in A and len(ai) >= 50:
            sel = ai if len(ai) <= a.max_rows else np.sort(rng.choice(ai, a.max_rows, replace=False))
            rr = {k: v[A["idx"][sel]] for k, v in R.items()}
            games_ = rr["game"]
            fold = folds_of(games_)
            Y = np.stack([np.nan_to_num(rr[v]).astype(np.float64) for v in RETENTION_VARS], 1)
            px = ridge_cv_multi(A["x_in"][sel].astype(np.float32), Y, games_, fold)
            ph = ridge_cv_multi(A["h"][sel].astype(np.float32), Y, games_, fold)
            ret = {}
            for i, v in enumerate(RETENTION_VARS):
                rx, rh = r2(Y[:, i], px[:, i]), r2(Y[:, i], ph[:, i])
                ret[v] = {"constant": True} if not np.std(Y[:, i]) > 0 else {"r2_x_in": rx, "r2_h": rh, "kept": rh / rx if rx > 0.05 else None}
            res["retention"] = {"rows": int(len(sel)), "vars": ret}
        out["races"][race] = res
        print(f"[{race}] rows {res['rows']}", flush=True)
        hp = res.get("hp_probes", {}).get("r2")
        if hp:
            print("  hp R2: " + " ".join(f"{k} {v['pt']:.3f}" for k, v in hp.items()))
        for k, d in res.get("gates", {}).items():
            print(f"  {k}: z {d['z_mean']:.3f} keep {d['var_share_keep']:.2f} | r2 x {d['r2_x']:.3f} keep {d['r2_keep']:.3f} write {d['r2_write']:.3f} g {d['r2_g']:.3f}")
    path = Path(a.out) if a.out else Path(a.dump) / "core_split.json"
    path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"[out] {path}")
    return 0
