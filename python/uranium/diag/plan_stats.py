"""Game-level bootstrap statistics of the S0 feasibility test (jepa_planning S0, reports/jepa_planning.md
6.4; 050-R9 diagnostics). Every confidence interval resamples whole games (with replacement, the same
draws for every predictor, so differences are paired) and gives the 2.5 / 97.5 percentiles.

- `Boot(groups, n, seed)`: the draws, as per-game multiplicities W [n, G] (a row's weight in a draw is
  its game's multiplicity).
- `mean`: mean of a per-row value; `r2`: 1 - SSE / SST over several dimensions (SST about the draw's
  own mean); `auc`: weighted ROC AUC (ties count 1/2); `spearman`: rank correlation (rows resampled
  explicitly); `ci` / `diff`: point, interval and paired difference; `rel_cut`: 1 - a / b of two means
  (relative error reduction), paired."""

import numpy as np


def _pct(v, q) -> float:
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if len(v) else float("nan")


def summary(pt, reps) -> dict:
    return {"pt": float(pt), "lo": _pct(reps, 2.5), "hi": _pct(reps, 97.5)}


class Boot:
    """Game bootstrap draws: `groups` [rows] any hashable game keys (ints here), `n` draws."""

    def __init__(self, groups, n: int, seed: int):
        self.keys, self.g = np.unique(np.asarray(groups), return_inverse=True)
        self.G = len(self.keys)
        self._o = np.argsort(self.g, kind="stable")
        self._starts = np.concatenate([[0], np.nonzero(np.diff(self.g[self._o]))[0] + 1]).astype(np.int64)
        self.n = n
        rng = np.random.default_rng(seed)
        self.W = np.zeros((n, self.G), np.float64)
        for b in range(n):
            self.W[b] = np.bincount(rng.integers(0, self.G, self.G), minlength=self.G)

    def per_game(self, v) -> np.ndarray:
        """Per-game sums [G, ...] (float64) of a per-row array (gathered in its own dtype)."""
        v = np.asarray(v)
        if not len(v):
            return np.zeros((self.G, *v.shape[1:]))
        return np.add.reduceat(v[self._o], self._starts, axis=0, dtype=np.float64)

    def mean(self, v) -> tuple[float, np.ndarray]:
        """(point, draws [n]) of the row mean of `v` (NaN rows left out)."""
        v = np.asarray(v, np.float64)
        ok = np.isfinite(v)
        s = self.per_game(np.where(ok, v, 0.0))
        c = self.per_game(ok.astype(np.float64))
        pt = s.sum() / c.sum() if c.sum() else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(pt), (self.W @ s) / (self.W @ c)

    def r2(self, sse, target, ok=None) -> tuple[float, np.ndarray]:
        """R² = 1 - Σ SSE / Σ_d Σ (y_d - ȳ_d)² over the rows with `ok` (`sse` [rows] the row's squared
        error summed over the dimensions, `target` [rows, D] with 0 in left-out dimensions)."""
        sse = np.asarray(sse, np.float64)
        y = np.asarray(target, np.float32).reshape(len(sse), -1)  # float32 rows (memory), float64 sums
        ok = np.isfinite(sse) if ok is None else (np.asarray(ok, bool) & np.isfinite(sse))
        okf = ok.astype(np.float32)
        e = self.per_game(np.where(ok, sse, 0.0))
        n = self.per_game(ok.astype(np.float64))
        s1 = self.per_game(y * okf[:, None])
        s2 = self.per_game(np.einsum("ij,ij->i", y, y, dtype=np.float64) * ok)

        def r2_of(w):
            nn = w @ n
            sst = w @ s2 - ((w @ s1) ** 2).sum(-1) / np.maximum(nn, 1e-12)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(sst > 0, 1.0 - (w @ e) / sst, np.nan)

        return float(r2_of(np.ones(self.G))), r2_of(self.W)

    def auc(self, y, s, chunk: int = 64) -> tuple[float, np.ndarray]:
        """Weighted ROC AUC of score `s` for binary `y` (rows with a finite score); draws reweight rows
        by their game's multiplicity."""
        y = np.asarray(y).astype(bool)
        s = np.asarray(s, np.float64)
        ok = np.isfinite(s)
        y, s, g = y[ok], s[ok], self.g[ok]
        if not len(y) or y.all() or (~y).all():
            return float("nan"), np.full(self.n, np.nan)
        o = np.argsort(s, kind="stable")
        y, s, g = y[o], s[o], g[o]
        starts = np.concatenate([[0], np.nonzero(np.diff(s))[0] + 1])

        def auc_w(w):  # w [b, rows]
            wp = np.add.reduceat(w * y, starts, axis=1)
            wn = np.add.reduceat(w * ~y, starts, axis=1)
            below = np.cumsum(wn, 1) - wn
            num = (wp * (below + 0.5 * wn)).sum(1)
            den = wp.sum(1) * wn.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(den > 0, num / den, np.nan)

        pt = float(auc_w(np.ones((1, len(y))))[0])
        reps = np.concatenate([auc_w(self.W[i:i + chunk][:, g]) for i in range(0, self.n, chunk)])
        return pt, reps

    def spearman(self, a, b) -> tuple[float, np.ndarray]:
        """Spearman rank correlation of `a` and `b` (finite rows); each draw resamples the rows of
        its games and ranks again."""
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b, g = a[ok], b[ok], self.g[ok]
        if len(a) < 3:
            return float("nan"), np.full(self.n, np.nan)
        reps = np.full(self.n, np.nan)
        idx = np.arange(len(a))
        for k in range(self.n):
            sel = np.repeat(idx, self.W[k][g].astype(np.int64))
            if len(sel) >= 3:
                reps[k] = spearman(a[sel], b[sel])
        return spearman(a, b), reps


def ranks(x) -> np.ndarray:
    """Average ranks (ties share their mean rank), from 1."""
    x = np.asarray(x, np.float64)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]


def spearman(a, b) -> float:
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def ci(stat: tuple) -> dict:
    return summary(stat[0], stat[1])


def diff(a: tuple, b: tuple) -> dict:
    """a - b with the paired interval (the same draws)."""
    return summary(a[0] - b[0], np.asarray(a[1]) - np.asarray(b[1]))


def ratio(a: tuple, b: tuple) -> dict:
    """a / b (the oracle share r of 6.4 M1) with the paired interval; NaN where b is 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        pt = a[0] / b[0] if b[0] else np.nan
        return summary(pt, np.asarray(a[1]) / np.asarray(b[1]))


def rel_cut(a: tuple, b: tuple) -> dict:
    """1 - a / b (relative error reduction of mean error a against b) with the paired interval."""
    with np.errstate(invalid="ignore", divide="ignore"):
        pt = 1.0 - a[0] / b[0] if b[0] else np.nan
        return summary(pt, 1.0 - np.asarray(a[1]) / np.asarray(b[1]))


def bce(logit, y) -> np.ndarray:
    """Per-row binary cross-entropy (nats) of a logit."""
    z = np.asarray(logit, np.float64)
    y = np.asarray(y, np.float64)
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))


def sigmoid(z) -> np.ndarray:
    z = np.asarray(z, np.float64)
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def quantiles(v, qs=(10, 50, 90)) -> dict:
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), **{f"p{q}": float(np.percentile(v, q)) for q in qs}}
