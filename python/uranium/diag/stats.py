import numpy as np


def pct(v, q) -> float:
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if len(v) else float("nan")


def summary(pt, reps) -> dict:
    return {"pt": float(pt), "lo": pct(reps, 2.5), "hi": pct(reps, 97.5)}
