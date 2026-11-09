"""Probe math (reports/jepa_survey.md 7.1-7.2): ridge with an inner-CV penalty, a one-hidden-layer
MLP(64), K-fold cross-validation by game (every row of a game in one fold), game bootstrap
confidence intervals, and the gain of a stage over the nuisance baseline.

Gain over nuisance (one definition, `GAIN_METHODS`):
- `resid` (default): a nuisance ridge, then a ridge of the stage on the nuisance ridge's residual,
  each with its own inner-CV penalty (the penalty grid of the second goes high enough to switch the
  stage off). The prediction is their sum, and the gain is metric(sum) - metric(nuisance alone).
  It cannot do worse than the nuisance except by fold noise.
- `joint`: one ridge on [stage, nuisance] with a single penalty (the earlier Zerg run, 10.1). A
  strong penalty for a wide stage also shrinks the nuisance, so the gain can go negative.
The two earlier workers used one each (10.2.3 note); `resid` is the default.
"""

import numpy as np
import torch

ALPHAS = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
ALPHAS_RESID = ALPHAS + [1e6, 1e7, 1e9]  # the top values switch the stage off (gain 0)
GAIN_METHODS = ("resid", "joint")
FOLDS = 5


def standardize(xtr, xte):
    """Columns scaled by the training rows; a (near-)constant training column is dropped (scaled
    to 0), and values are clipped to +-10 so a column constant in training cannot explode on
    the held-out games."""
    mu = xtr.mean(0, keepdims=True)
    sd = xtr.std(0, keepdims=True)
    sd = np.where(sd < 1e-4, np.inf, sd)
    return np.clip((xtr - mu) / sd, -10, 10), np.clip((xte - mu) / sd, -10, 10)


def ridge_fit_pred(xtr, ytr, xte, alphas):
    """Ridge predictions [len(alphas), n_te] (intercept = training mean), float64, one
    eigendecomposition of X^T X for every penalty. `ytr` [n] or [n, k] (then [A, n_te, k])."""
    X = np.asarray(xtr, np.float64)
    y = np.asarray(ytr, np.float64)
    ym = y.mean(0)
    ev, V = np.linalg.eigh(X.T @ X)
    vty = V.T @ (X.T @ (y - ym))
    Xt = np.asarray(xte, np.float64) @ V
    ev = np.clip(ev, 0, None)
    if y.ndim == 2:
        ev = ev[:, None]
    return np.stack([Xt @ (vty / (ev + a)) + ym for a in alphas])


def folds_of(groups, k: int = FOLDS, seed: int = 0) -> np.ndarray:
    """Fold id per row; every row of a group (game) in the same fold."""
    u = np.unique(groups)
    p = np.random.default_rng(seed).permutation(u)
    fid = {g: i % k for i, g in enumerate(p)}
    return np.array([fid[g] for g in groups])


def inner_alpha(xtr, ytr, groups_tr, seed: int, alphas) -> float:
    """The penalty with the least squared error in an inner 3-fold game CV (inputs standardised)."""
    inner = folds_of(groups_tr, 3, seed=seed)
    err = np.zeros(len(alphas))
    for i in range(3):
        a, b = inner != i, inner == i
        if not a.any() or not b.any():
            continue
        xa, xb = standardize(xtr[a], xtr[b])
        err += ((ridge_fit_pred(xa, ytr[a], xb, alphas) - ytr[b][None]) ** 2).mean(1)
    return alphas[int(err.argmin())]


def ridge_cv(x, y, groups, fold):
    """Out-of-fold ridge predictions; the penalty chosen per outer fold by an inner game CV."""
    pred = np.zeros(len(y))
    chosen = []
    for f in range(int(fold.max()) + 1):
        tr, te = fold != f, fold == f
        if not te.any():
            continue
        xtr, xte = standardize(x[tr], x[te])
        al = inner_alpha(xtr, y[tr], groups[tr], f + 11, ALPHAS)
        chosen.append(al)
        pred[te] = ridge_fit_pred(xtr, y[tr], xte, [al])[0]
    return pred, chosen


def ridge_cv_multi(x, Y, groups, fold):
    """Out-of-fold ridge predictions of several targets [n, k], one penalty per target."""
    Y = Y.reshape(len(Y), -1)
    pred = np.zeros(Y.shape)
    for f in range(int(fold.max()) + 1):
        tr, te = fold != f, fold == f
        if not te.any():
            continue
        xtr, xte = standardize(x[tr], x[te])
        inner = folds_of(groups[tr], 3, seed=f + 11)
        err = np.zeros((len(ALPHAS), Y.shape[1]))
        for i in range(3):
            a, b = inner != i, inner == i
            xa, xb = standardize(xtr[a], xtr[b])
            err += ((ridge_fit_pred(xa, Y[tr][a], xb, ALPHAS) - Y[tr][b][None]) ** 2).mean(1)
        best = err.argmin(0)
        p = ridge_fit_pred(xtr, Y[tr], xte, ALPHAS)
        pred[te] = p[best, :, np.arange(Y.shape[1])].T
    return pred


def ridge_cv_resid(x, nz, y, groups, fold):
    """Gain method `resid`: nuisance ridge + a stage ridge on its residual (module docstring)."""
    pred = np.zeros(len(y))
    for f in range(int(fold.max()) + 1):
        tr, te = fold != f, fold == f
        if not te.any():
            continue
        ntr, nte = standardize(nz[tr], nz[te])
        an = inner_alpha(ntr, y[tr], groups[tr], f + 11, ALPHAS)
        pn = ridge_fit_pred(ntr, y[tr], np.concatenate([ntr, nte]), [an])[0]
        ntr_rows = int(tr.sum())
        res = y[tr] - pn[:ntr_rows]
        xtr, xte = standardize(x[tr], x[te])
        ax = inner_alpha(xtr, res, groups[tr], f + 23, ALPHAS_RESID)
        pred[te] = pn[ntr_rows:] + ridge_fit_pred(xtr, res, xte, [ax])[0]
    return pred


def ridge_cv_gain(x, nz, y, groups, fold, method: str = "resid"):
    """Out-of-fold predictions of [stage, nuisance] by the gain method."""
    if method == "resid":
        return ridge_cv_resid(x, nz, y, groups, fold)
    if method == "joint":
        return ridge_cv(np.concatenate([x, nz], 1), y, groups, fold)[0]
    raise ValueError(f"gain method {method!r}: expected one of {GAIN_METHODS}")


def mlp_cv(x, y, fold, cls: bool, device="cpu", epochs: int = 30, hidden: int = 64, seed: int = 0):
    """Out-of-fold predictions of Linear(64) -> ReLU -> Dropout(0.2) -> Linear(1), AdamW 3e-3,
    batch 2048 (logits for a class target, standardised regression otherwise)."""
    dev = torch.device(device)
    pred = np.zeros(len(y))
    for f in range(int(fold.max()) + 1):
        tr, te = fold != f, fold == f
        if not te.any():
            continue
        xtr, xte = standardize(x[tr], x[te])
        ymu, ysd = (0.0, 1.0) if cls else (float(y[tr].mean()), float(y[tr].std()) + 1e-6)
        X = torch.as_tensor(xtr, device=dev, dtype=torch.float32)
        Y = torch.as_tensor((y[tr] - ymu) / ysd, device=dev, dtype=torch.float32)
        torch.manual_seed(seed + f)
        m = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(hidden, 1)).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-2)
        n = len(X)
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, 2048):
                bi = perm[s:s + 2048]
                out = m(X[bi])[:, 0]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(out, Y[bi]) if cls else ((out - Y[bi]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        m.eval()
        with torch.no_grad():
            pred[te] = m(torch.as_tensor(xte, device=dev, dtype=torch.float32))[:, 0].cpu().numpy() * ysd + ymu
    return pred


def auc(y, s) -> float:
    """ROC AUC with average ranks for ties; NaN for a single class."""
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return np.nan
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def r2(y, p) -> float:
    y = np.asarray(y, np.float64)
    den = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / den) if den > 0 else np.nan


def boot(metric, y, preds: dict, groups, n: int, rng, base: str | None = None) -> dict:
    """Game bootstrap: per prediction {"pt", "lo", "hi"} (2.5 / 97.5 percentiles), and against
    `base` the paired difference {"gain", "gain_lo", "gain_hi"}."""
    ug = np.unique(groups)
    gi = {g: np.nonzero(groups == g)[0] for g in ug}
    pts = {k: metric(y, p) for k, p in preds.items()}
    reps = {k: [] for k in preds}
    for _ in range(n):
        sel = np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])
        for k, p in preds.items():
            reps[k].append(metric(y[sel], p[sel]))
    out = {}
    for k in preds:
        v = np.array(reps[k], np.float64)
        out[k] = {"pt": float(pts[k]), "lo": _pct(v, 2.5), "hi": _pct(v, 97.5)}
        if base is not None and k != base and base in reps:
            d = v - np.array(reps[base], np.float64)
            out[k]["gain"] = float(pts[k] - pts[base])
            out[k]["gain_lo"], out[k]["gain_hi"] = _pct(d, 2.5), _pct(d, 97.5)
    return out


def _pct(v, q) -> float:
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if len(v) else float("nan")


def shuffled_labels(y, games, rng) -> np.ndarray:
    """Control target: each game's label (its first row's) moved to another game, rows keep
    their game. Valid for labels constant within a game; for per-row targets the whole sequence
    of the game is not moved, only its first value."""
    ug = np.unique(games)
    first = {g: y[np.argmax(games == g)] for g in ug}
    perm = dict(zip(ug, rng.permutation(ug)))
    return np.array([first[perm[g]] for g in games])


def onehot(x) -> np.ndarray:
    x = np.asarray(x).astype(np.int64)
    u = np.unique(x)
    return (x[:, None] == u[None]).astype(np.float32)


def run_task(stages: dict, nuis, y, cls: bool, groups, game_ids, rng, *, boot_n: int = 500, mlp: bool = True, gain: str = "resid", shuffled_key: str | None = None,
             mlp_device="cpu") -> dict:
    """One probe target on one row set: per stage a ridge (and an MLP), the nuisance-only ridge,
    the gain of each stage over it, and a shuffled-label control on `shuffled_key` (per-game
    labels only). `stages` {name: [rows, d]}, `nuis` [rows, k]; rows already selected."""
    fold = folds_of(groups)
    preds, alphas = {}, {}
    p, al = ridge_cv(nuis, y, groups, fold)
    preds["nuis|ridge"], alphas["nuis"] = p, al
    for st, x in stages.items():
        x = np.asarray(x, np.float32)
        preds[f"{st}|ridge"], alphas[st] = ridge_cv(x, y, groups, fold)
        preds[f"{st}+nuis|ridge"] = ridge_cv_gain(x, nuis, y, groups, fold, gain)
        if mlp:
            preds[f"{st}|mlp"] = mlp_cv(x, y, fold, cls, mlp_device)
    metric = auc if cls else r2
    res = boot(metric, y, preds, groups, boot_n, rng, base="nuis|ridge")
    if shuffled_key is not None and shuffled_key in stages:
        ysh = shuffled_labels(y, game_ids, rng)
        ps, _ = ridge_cv(np.asarray(stages[shuffled_key], np.float32), ysh, groups, fold)
        res[f"shuffled|{shuffled_key}"] = boot(metric, ysh, {"x": ps}, groups, boot_n, rng)["x"]
    ug = np.unique(game_ids)
    pos = None
    if cls:
        pos = int(sum(1 for g in ug if y[np.argmax(game_ids == g)] > 0.5))
    return {"rows": int(len(y)), "games": int(len(ug)), "groups": int(len(np.unique(groups))), "pos_games": pos, "metric": "auc" if cls else "r2", "gain_method": gain,
            "alphas": alphas, "metrics": res}
