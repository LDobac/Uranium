"""`python -m uranium diag plan-branch` (051-R13): the branch-game analysis on CPU. Measurement only.

Inputs (051-R12): `<dir>/points.jsonl` (one line per decision point) and `<dir>/branches.jsonl` (one
line per branch game). Optional, for M3u / M2c: `--dump`, the `diag plan-dump` folder of the trunk
replays rebuilt without relabelling with the act logs as the memory (041-R28, `uranium-replay build
--memory-log <acts>`), dumped with the fixed checkpoint on CPU; and `--s0`, the `diag plan-s0` folder
whose `models/A_k<k>.pt` are the S0b predictor A (and whose `s0*.json` M3b ΔQ is shown beside the
bunker effects).

Fields read (051 "Interfaces"; extra fields are ignored):
Field names follow `uranium_env::branch` (`PointRecord`, `BranchRecord`, `RuleLog`, `Samples`,
`DilutionEntry`, `BranchEnd`):
- point: `point` (id `<trunk id>#<rule>`), `trunk` {`id`?, `replay` (path, relative to `--root` /
  URANIUM_SHARED_DIR / `<dir>`), `replay_sha256`?, `start`? (own start slot, scalar_cat[2])}, `f_t`,
  `scalar_t`?, `checkpoint_sha256`?, `valid`, `skip_reason`.
- branch: `point`, `alt` (name), `rules` [{`op`, `action` (id or 034 name), `from_s`, `to_s`}],
  `repeat`, `valid`, `invalid` {`reason`, ...}, `intervention` [per rule: `applied`, `force_frame`,
  `missed`, `rejected`], `samples` {`t60`, `t120`, `t240`, `horizon`, `d8`, `d32`: {`frame`, `scalar`}
  or null}, `metrics` {...}, `dilution` [{`action`, `first_frame`, `picks_30s`}], `end`
  {`result`, `end_frame`, `reason`, `before_horizon`}.
- An outcome name (`--outcomes`) is read from `metrics`, then `end`, then the branch line itself, or
  as a dotted path (`end.frame`); `own@<sample>:<column>` is the own-state column (`plan_dump.own_columns`)
  of a sample's scalar vector. true / false read as 1 / 0; a missing value leaves the branch out of
  that outcome. The default outcomes are those of 051-R11 ④ (`OUTCOMES`, and `own@<s>:<c>` for the
  samples `--own-samples` and the columns `--own-cols`, `plan_s0.OWN_HEADLINE`); the sample names
  (`OWN_SAMPLES`, `--sample-k8` / `--sample-k32`) and `end.before_horizon` must match the branch writer.

Tables (bootstrap: whole trunks with replacement, `--boot` draws, fixed `--boot-seed`; one set of draws
for every alternative and outcome, so they are paired):
1. Paired effects: d_p(a) = mean over repeats of Y(a) - mean over repeats of Y(`--baseline`) per point
   with both; D(a) = the point mean (points weighted by their trunk's draw multiplicity) and its 95 %
   interval. Primary: every valid branch of the alternative (refused and missed included); secondary:
   only branches whose intervention ran (every `force` rule fired and was not refused by the executor,
   every `ban` rule closed at least one decision; the baseline keeps all its branches). 0/1 outcomes
   also give counts and rates. Split-half noise ceiling: d_p from the odd and from the even repeat
   numbers, their Pearson correlation over points r and the Spearman-Brown value 2r / (1 + r).
2. M3u (one-decision `force` alternatives, `from_s = to_s = 0`): ΔQ̂(a) = σq̂_A(h'_t, o_t, a) -
   Σ_c π(c|t) σq̂_A(h'_t, o_t, c) on the `scv8` head (`plan_s0.delta_q`) against the measured d_p(a) of
   `--m3u-outcome`: sign agreement (ties, ΔQ̂ = 0 or d = 0, left out and counted), Spearman over (point, alternative) pairs (trunk
   bootstrap), and the ratio to the noise ceiling sqrt(Spearman-Brown) of those pairs. Per alternative
   the mean ΔQ̂ beside D(a) (bunker alternatives also with the S0b M3b ΔQ).
3. M2c: predicted own-state change of predictor A_k (k = `--ks`): ô(a) - Σ_c π(c|t) ô(c) in counts
   (the predictor's standardisation undone) against the measured o(sample k, a) - o(sample k, baseline)
   (repeat means; samples `--sample-k8` / `--sample-k32`). Direction agreement on the column the forced
   action makes (`own_columns` action map; ties, no predicted or measured change, left out and counted;
   the same pairs as M3u: the forced action open at t), and the Spearman
   correlation over every kept column of every pair (trunk bootstrap).
4. h'_t, o_t and π(c|t) are the dump row of the point (051-R13 ④): live frame f_t -> playback frame
   f_t - `--row-lag` (041-R28 ②, the game-start observation at 1) in the game of the trunk replay
   (sha256; several views of one replay: `trunk.start`, else the view whose own state equals
   own(scalar_t)). Exact frames only: the trunk build adds a step at every logged decision's playback
   frame (041-R28 decision-time steps, user decision 2026-11-13), so the row exists by construction
   when the trunk's log was used; points without one (a log not used, another build) are left out of
   M3u / M2c and counted (with the distance to the nearest row of the game). Optional alternatives (`ban`, several rules) get table 1 only.
5. Invalid branches by reason, per alternative the refused / missed rates of its `force` rules, and the
   dilution numbers (051-R9 ③, means of the numeric leaves).
No thresholds (AMB-051-5 기본안(미확정): report only).

Output: `<out>/analysis.json` and `<out>/plan-branch.md` (`--out`, default `<dir>/analysis`)."""

import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from uranium.diag import plan_pred as pp
from uranium.diag import plan_s0
from uranium.diag import plan_stats as ps
from uranium.diag import tables as tb
from uranium.diag.plan_dump import own_columns, own_state

SCV8 = pp.Q_NAMES.index("scv8")  # the q̂ head of 8:00 SCVs >= 8 (= 050-R6 ③, held_8m)
GAME_START_FRAME = 1
# 051-R11 ④ (AMB-051-4): main results, then the auxiliary 050 metrics and the end before the horizon
OUTCOMES = ("held_8m", "held_s", "scv_8m", "scv_min_1m", "bunker_s", "barracks_s", "army_s", "end.before_horizon")
# 051-R9 ① sample names: f_t + 60 / 120 / 240 s, the horizon H, the 8th / 32nd decision after t
OWN_SAMPLES = ("t60", "t120", "t240", "horizon", "d8", "d32")


def add_args(p) -> None:
    p.add_argument("--dir", required=True, help="branch output folder (points.jsonl, branches.jsonl; 051-R12)")
    p.add_argument("--points", help="points file (default <dir>/points.jsonl)")
    p.add_argument("--branches", help="branches file (default <dir>/branches.jsonl)")
    p.add_argument("--out", help="output folder (default <dir>/analysis)")
    p.add_argument("--dump", help="plan-dump folder of the trunk replays (no relabel, act-log memory; M3u / M2c)")
    p.add_argument("--s0", help="plan-s0 folder with models/A_k<k>.pt (the S0b predictor A; M3u / M2c)")
    p.add_argument("--root", help="base of the relative trunk replay paths (default URANIUM_SHARED_DIR, then --dir)")
    p.add_argument("--outcomes", default=",".join(OUTCOMES),
                   help="outcomes Y (051-R11 ④: main held_8m, held_s, then the auxiliary ones; metrics / end fields, dotted paths, own@<sample>:<column>)")
    p.add_argument("--own-samples", default=",".join(OWN_SAMPLES),
                   help="branch samples whose own state is an outcome (051-R11 ④: t + 1 / 2 / 4 min, H = 8:00, t + 8 / 32 decisions; empty = none)")
    p.add_argument("--own-cols", default=",".join(plan_s0.OWN_HEADLINE), help="own-state columns of those outcomes (own@<sample>:<column>)")
    p.add_argument("--m3u-outcome", default="held_8m", help="the measured outcome M3u compares with the scv8 head (050-R6 ③)")
    p.add_argument("--baseline", default="none", help="alternative the effects are taken against (051-R7 no intervention)")
    p.add_argument("--ks", default="8,32", help="predictor k of M3u / M2c (models A_k<k>.pt)")
    p.add_argument("--sample-k8", default="d8", help="branch sample of the 8th decision after t (M2c k = 8)")
    p.add_argument("--sample-k32", default="d32", help="branch sample of the 32nd decision after t (M2c k = 32)")
    p.add_argument("--row-lag", type=int, default=2, help="live frame f_t -> dump row at playback frame f_t - lag (041-R28 ②)")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--boot-seed", type=int, default=7)


# ---------------------------------------------------------------- reading


def read_jsonl(path) -> list[dict]:
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path} line {n}: {e}") from e
    return out


def trunk_of(point: dict) -> str:
    t = point.get("trunk")
    if isinstance(t, dict) and t.get("id"):
        return str(t["id"])
    return str(point["point"]).split("#")[0]


def _num(v) -> float:
    """A number (true / false as 1 / 0), NaN for anything else."""
    return float(v) if isinstance(v, (int, float)) else float("nan")


def _path(d, path: str):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def samples_of(br: dict) -> dict:
    s = br.get("samples") or {}
    if isinstance(s, list):
        s = {x.get("name") or x.get("at"): x for x in s if isinstance(x, dict)}
    return s


def own_of(scalar) -> np.ndarray:
    return own_state(np.asarray(scalar, np.float32)[None])[0]


def sample_own(br: dict, name: str) -> np.ndarray | None:
    """Own state of a branch's sample `name` (None without it), cached on the branch line."""
    cache = br.setdefault("_own", {})
    if name not in cache:
        s = samples_of(br).get(name)
        cache[name] = None if not isinstance(s, dict) or s.get("scalar") is None else own_of(s["scalar"])
    return cache[name]


def outcome(br: dict, name: str) -> float:
    """The value of outcome `name` in a branch line (module docstring), NaN when missing."""
    if name.startswith("own@"):
        sample, _, col = name[4:].partition(":")
        names = own_columns()[0]
        if col not in names:
            raise SystemExit(f"plan-branch: outcome {name}: no own column {col!r} ({names})")
        o = sample_own(br, sample)
        return float("nan") if o is None else float(o[names.index(col)])
    if "." in name:
        return _num(_path(br, name))
    for where in (br.get("metrics") or {}, br.get("end") or {}, br):
        if name in where:
            return _num(where[name])
    return float("nan")


def action_of(x) -> int:
    if isinstance(x, str):
        return tb.action_id(x)
    return int(x)


def one_decision_force(rules) -> int | None:
    """The action of a one-decision `force` alternative (one rule, from_s = to_s = 0), else None."""
    if not rules or len(rules) != 1:
        return None
    r = rules[0]
    if r.get("op") != "force" or _num(r.get("from_s", 0)) != 0 or _num(r.get("to_s", 0)) != 0:
        return None
    return action_of(r["action"])


def _missed(r: dict) -> bool:
    return bool(r.get("missed")) or r.get("frame") == "missed" or r.get("force_frame") == "missed"


def executed(br: dict) -> bool:
    """The intervention ran (module docstring 1, secondary analysis). No rules: True."""
    rules = br.get("rules") or []
    iv = br.get("intervention") or []
    if isinstance(iv, dict):
        iv = iv.get("rules") or [iv]
    for i, rule in enumerate(rules):
        rec = iv[i] if i < len(iv) and isinstance(iv[i], dict) else {}
        if rule.get("op") == "force" and (_missed(rec) or bool(rec.get("rejected")) or not rec):
            return False
        if rule.get("op") == "ban" and not _num(rec.get("applied", 0)) > 0:
            return False
    return True


class Branches:
    """The valid points and branches: cells[(point, alt)] = [(repeat, branch line)]."""

    def __init__(self, points: list[dict], branches: list[dict], baseline: str):
        self.baseline = baseline
        self.all_points = {p["point"]: p for p in points}
        self.points = sorted(k for k, p in self.all_points.items() if p.get("valid", True))
        self.skipped_points = {k: self.all_points[k].get("skip_reason") for k in sorted(self.all_points) if k not in self.points}
        self.trunks = [trunk_of(self.all_points[k]) for k in self.points]
        self.alts, self.rules = [], {}
        self.cells: dict = {}
        self.invalid: dict = {}
        self.outside_points = 0
        self.n_branches = len(branches)
        for b in branches:
            alt = b.get("alt") or ("none" if not b.get("rules") else json.dumps(b["rules"], sort_keys=True))
            if alt not in self.rules:
                self.rules[alt] = b.get("rules") or []
                self.alts.append(alt)
            if not b.get("valid", True):
                inv = b.get("invalid")
                why = inv.get("reason") if isinstance(inv, dict) else inv
                self.invalid.setdefault(alt, {}).setdefault(str(why), 0)
                self.invalid[alt][str(why)] += 1
                continue
            if b.get("point") not in self.all_points or b["point"] not in self.points:
                self.outside_points += 1
                continue
            self.cells.setdefault((b["point"], alt), []).append((int(b.get("repeat", 0)), b))
        self.alts.sort(key=lambda a: (a != baseline, a))
        self.n_valid = sum(len(v) for v in self.cells.values())

    def items(self, p: str, alt: str, only_executed: bool = False, half: int | None = None) -> list[dict]:
        """The valid branches of (point, alternative); `only_executed` drops those whose intervention did not
        run (not the baseline's), `half` keeps the odd (1) or even (0) repeat numbers."""
        return [b for rep, b in self.cells.get((p, alt), [])
                if not (only_executed and alt != self.baseline and not executed(b)) and (half is None or rep % 2 == half)]

    def values(self, p: str, alt: str, fn, only_executed: bool = False, half: int | None = None) -> np.ndarray:
        return np.asarray([fn(b) for b in self.items(p, alt, only_executed, half)], np.float64)


def _mean(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def _own_mean(B: Branches, p: str, alt: str, sample: str) -> np.ndarray | None:
    """Mean own state (`own_columns`) of sample `sample` over the valid branches of (p, alt) that have it."""
    v = [o for b in B.items(p, alt) if (o := sample_own(b, sample)) is not None]
    return np.mean(v, 0) if v else None


def point_effects(B: Branches, alt: str, fn, only_executed=False, half=None) -> np.ndarray:
    """d_p(alt) per point of `B.points` (NaN without both sides)."""
    d = np.full(len(B.points), np.nan)
    for i, p in enumerate(B.points):
        a = _mean(B.values(p, alt, fn, only_executed, half))
        b = _mean(B.values(p, B.baseline, fn, False, half))
        d[i] = a - b
    return d


def split_half(B: Branches, alt_list, fn, sel=None) -> dict:
    """Split-half agreement of d_p over the (point, alternative) pairs of `alt_list` (module docstring 1):
    odd against even repeat numbers; `sel` limits the points (indices into `B.points`)."""
    sel = np.arange(len(B.points)) if sel is None else sel
    h = [np.concatenate([point_effects(B, a, fn, half=x)[sel] for a in alt_list]) if alt_list else np.zeros(0) for x in (1, 0)]
    ok = np.isfinite(h[0]) & np.isfinite(h[1])
    n = int(ok.sum())
    if n < 3 or np.std(h[0][ok]) == 0 or np.std(h[1][ok]) == 0:
        return {"pairs": n, "r": None, "spearman_brown": None}
    r = float(np.corrcoef(h[0][ok], h[1][ok])[0, 1])
    return {"pairs": n, "r": r, "spearman_brown": 2 * r / (1 + r) if r > -1 else None}


def is_binary(B: Branches, fn) -> bool:
    vals = [fn(b) for cell in B.cells.values() for _, b in cell]
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    return bool(len(vals)) and bool(np.isin(vals, (0.0, 1.0)).all())


def paired_table(B: Branches, outcomes: list[str], boot: ps.Boot) -> dict:
    out = {}
    for y in outcomes:
        fn = (lambda b, y=y: outcome(b, y))
        binary = is_binary(B, fn)
        ent = {"binary": binary, "alts": {}}
        if binary:
            ent["counts"] = {}
            for a in B.alts:
                v = np.concatenate([B.values(p, a, fn) for p in B.points]) if B.points else np.zeros(0)
                v = v[np.isfinite(v)]
                ent["counts"][a] = {"n": int(len(v)), "k": int(v.sum()), "rate": float(v.mean()) if len(v) else None}
        for a in B.alts:
            if a == B.baseline:
                continue
            row = {}
            for mode, only in (("all", False), ("executed", True)):
                d = point_effects(B, a, fn, only)
                ok = np.isfinite(d)
                row[mode] = {"points": int(ok.sum()), "trunks": int(len({B.trunks[i] for i in np.nonzero(ok)[0]})),
                             "D": ps.ci(boot.mean(d)) if ok.any() else None}
            row["split_half"] = split_half(B, [a], fn)
            ent["alts"][a] = row
        out[y] = ent
    return out


# ---------------------------------------------------------------- dump rows and predictors


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def replay_sha(point: dict, roots: list[Path], cache: dict | None = None) -> str | None:
    """sha256 of the point's trunk replay (`trunk.replay_sha256`, else the file hashed once per path)."""
    t = point.get("trunk") or {}
    if t.get("replay_sha256"):
        return t["replay_sha256"]
    rp = t.get("replay")
    if not rp:
        return None
    cache = {} if cache is None else cache
    if rp not in cache:
        cands = [Path(rp)] if Path(rp).is_absolute() else [r / rp for r in roots]
        cache[rp] = next((file_sha256(c) for c in cands if c.is_file()), None)
    return cache[rp]


def find_rows(B: Branches, data, roots: list[Path], lag: int) -> tuple[dict, dict]:
    """{point: dump row} and {point: why not} (module docstring 4)."""
    by_sha: dict = {}
    for gi, e in enumerate(data.game_meta):
        by_sha.setdefault(e.get("sha"), []).append(gi)
    frames = np.rint(data.frame.astype(np.float64))
    order = np.argsort(data.game, kind="stable")
    keys, starts = np.unique(data.game[order], return_index=True)
    rows_of = dict(zip(keys.tolist(), np.split(order, starts[1:]))) if len(order) else {}
    sha_cache: dict = {}
    rows, missing = {}, {}
    for p in B.points:
        pt = B.all_points[p]
        sha = replay_sha(pt, roots, sha_cache)
        if sha is None:
            missing[p] = {"reason": "no_replay"}
            continue
        games = by_sha.get(sha, [])
        start = (pt.get("trunk") or {}).get("start")
        if start is not None:
            games = [g for g in games if int(data.game_meta[g].get("start", -1)) == int(start)]
        if not games:
            missing[p] = {"reason": "no_game"}
            continue
        f = max(int(pt["f_t"]) - lag, GAME_START_FRAME)
        hit, near = [], []
        for g in games:
            gr = rows_of.get(g, np.zeros(0, np.int64))
            at = gr[frames[gr] == f]
            if len(at):
                hit.append((g, int(at[np.argmin(data.dec[at])])))
            elif len(gr):
                near.append(float(np.min(np.abs(frames[gr] - f))))
        views = len(hit)
        if views > 1 and pt.get("scalar_t") is not None:
            o = own_of(pt["scalar_t"])
            hit = [h for h in hit if np.array_equal(data.own[h[1]], o)]
        if len(hit) == 1:
            rows[p] = hit[0][1]
        elif views > 1:
            missing[p] = {"reason": "ambiguous_view", "views": views, "own_state_matches": len(hit)}
        else:
            missing[p] = {"reason": "no_row", "nearest_frames": min(near) if near else None}
    return rows, missing


def load_predictor(s0: Path, k: int, data):
    path = s0 / "models" / f"A_k{k}.pt"
    if not path.is_file():
        return None
    net, st, meta = pp.load(path, data)
    if "d_mu" not in st:
        raise SystemExit(f"plan-branch: {path} has no change statistics (a first-run model); use the S0b models")
    return net, st, meta


def predictor_design(s0: Path, ks):
    """The `plan_pred.Design` and checkpoint of the S0b models (from the first model's meta)."""
    for k in ks:
        path = s0 / "models" / f"A_k{k}.pt"
        if path.is_file():
            meta = torch.load(path, map_location="cpu", weights_only=False)["meta"]
            d = (meta.get("data") or {}).get("design")
            design = pp.V1 if d is None else pp.Design(**{**d, "chain_kinds": tuple(d.get("chain_kinds", ()))})
            return design, (meta.get("data") or {}).get("checkpoint_sha256")
    return None, None


def own_change_pred(net, st, data, rows, k: int, chunk_rows: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """([rows, A, O] ô change per candidate action in counts, [rows, O] its policy expectation); columns
    left out of the predictor's target space are NaN."""
    A = data.n_act
    design = data.design
    sd = st["d_sd"] if design.own_target == "delta" else st["o_sd"]
    on = np.isfinite(sd)
    per = []
    for s in range(0, len(rows), chunk_rows):
        r = rows[s:s + chunk_rows]
        flat_r = np.repeat(r, A)
        flat_a = np.tile(np.arange(A), len(r))
        o = pp.predict(net, data, flat_r, k, st, action=flat_a, keep=("o",))["o"].reshape(len(r), A, -1)
        per.append(np.where(on, o * np.where(on, sd, 1.0), np.nan))
    o_all = np.concatenate(per) if per else np.zeros((0, A, len(sd)))
    pi = data.probs(rows)
    return o_all, np.einsum("ra,rao->ro", pi, np.nan_to_num(o_all)) + np.where(on, 0.0, np.nan)


def s0_m3b(s0: Path, k: int) -> dict | None:
    """S0b M3b ΔQ (predictor A, teacher and model rows) of the 8:00 SCV head from `<s0>/s0*.json`."""
    for f in sorted(glob.glob(str(s0 / "s0*.json"))):
        rep = json.loads(Path(f).read_text(encoding="utf-8"))
        blk = (((rep.get("M3b") or {}).get(str(k)) or {}).get("A") or {}).get("both") or {}
        if "scv8" in blk:
            return {"file": Path(f).name, **{x: blk["scv8"].get(x) for x in ("rush_mean", "macro_mean", "rush_minus_macro")}}
    return None


def sign_agreement(pred, meas) -> tuple[np.ndarray, int]:
    """Per pair 1 / 0 = the signs of `pred` and `meas` agree / differ; NaN for a pair left out: not
    finite, or a tie (`pred` == 0 or `meas` == 0, no direction to agree with), counted in the second
    value."""
    pred, meas = np.asarray(pred, np.float64), np.asarray(meas, np.float64)
    fin = np.isfinite(pred) & np.isfinite(meas)
    tie = fin & ((pred == 0) | (meas == 0))
    return np.where(fin & ~tie, (np.sign(pred) == np.sign(meas)).astype(np.float64), np.nan), int(tie.sum())


def _rate(v: np.ndarray) -> float | None:
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else None


def m3u(B: Branches, data, rows: dict, nets: dict, y: str, boot_n: int, seed: int, s0: Path | None) -> dict:
    fn = (lambda b: outcome(b, y))
    forced = {a: one_decision_force(B.rules[a]) for a in B.alts if a != B.baseline}
    forced = {a: x for a, x in forced.items() if x is not None}
    pts = [p for p in B.points if p in rows]
    out = {"outcome": y, "alts": sorted(forced), "points_with_row": len(pts)}
    if not forced or not pts:
        return out
    pi_rows = np.array([rows[p] for p in pts], np.int64)
    pidx = {p: i for i, p in enumerate(B.points)}
    sel = np.array([pidx[p] for p in pts], np.int64)
    eff = {a: point_effects(B, a, fn)[sel] for a in forced}
    # the noise ceiling of the pairs M3u reads (every one-decision force alternative)
    out["split_half"] = split_half(B, list(forced), fn, sel)
    sb = out["split_half"]["spearman_brown"]
    for k, (net, st, _) in nets.items():
        res = {"alts": {}}
        dq_all, d_all, unit = [], [], []
        for a, act in forced.items():
            open_ = data.mask_rows(pi_rows)[:, act]
            dq = plan_s0.delta_q(net, data, pi_rows, act, st)[:, SCV8]
            dq = np.where(open_, dq, np.nan)
            d = eff[a]
            ok = np.isfinite(dq) & np.isfinite(d)
            res["alts"][a] = {"action": act, "pairs": int(ok.sum()), "dq_mean": float(dq[ok].mean()) if ok.any() else None,
                              "d_mean": float(d[ok].mean()) if ok.any() else None,
                              "sign_agree": _rate(sign_agreement(dq[ok], d[ok])[0])}
            dq_all.append(dq[ok])
            d_all.append(d[ok])
            unit.append(np.asarray([B.trunks[i] for i in sel[ok]], object))
        dq_all, d_all, unit = np.concatenate(dq_all), np.concatenate(d_all), np.concatenate(unit)
        res["pairs"] = int(len(d_all))
        if len(d_all):
            bt = ps.Boot(unit.astype(str), boot_n, seed)
            agree, ties = sign_agreement(dq_all, d_all)
            res["sign_agree"] = {"pairs": int(np.isfinite(agree).sum()), "ties": ties, **ps.ci(bt.mean(agree))}
            rho = bt.spearman(dq_all, d_all)
            res["spearman"] = ps.ci(rho)
            ceil = float(np.sqrt(sb)) if sb is not None and sb > 0 else None
            res["noise_ceiling"] = ceil
            res["ratio_to_ceiling"] = (rho[0] / ceil) if ceil else None
        if s0 is not None:
            res["s0b_m3b"] = s0_m3b(s0, k)
        out[f"k{k}"] = res
    return out


def m2c(B: Branches, data, rows: dict, nets: dict, samples: dict, boot_n: int, seed: int) -> dict:
    names, _, of_action = own_columns()
    if list(names) != list(data.own_names):
        raise SystemExit("plan-branch: the dump's own-state columns differ from this code's")
    forced = {a: one_decision_force(B.rules[a]) for a in B.alts if a != B.baseline}
    forced = {a: x for a, x in forced.items() if x is not None}
    pts = [p for p in B.points if p in rows]
    out = {"alts": sorted(forced), "points_with_row": len(pts), "columns": list(names)}
    if not forced or not pts:
        return out
    pi_rows = np.array([rows[p] for p in pts], np.int64)
    trunk = [B.trunks[B.points.index(p)] for p in pts]
    for k, (net, st, _) in nets.items():
        sname = samples.get(k)
        res = {"sample": sname, "alts": {}}
        if sname is None:
            out[f"k{k}"] = res | {"skipped": f"no branch sample for k = {k}"}
            continue
        o_all, o_pi = own_change_pred(net, st, data, pi_rows, k)
        P, M, U, dirs = [], [], [], []
        base = [_own_mean(B, p, B.baseline, sname) for p in pts]
        for a, act in forced.items():
            pred = o_all[:, act] - o_pi  # [points, O]
            meas = np.full_like(pred, np.nan)
            for i, p in enumerate(pts):
                ma = _own_mean(B, p, a, sname)
                if ma is not None and base[i] is not None:
                    meas[i] = ma - base[i]
            col = of_action.get(int(act))
            # the pairs M3u reads: the forced action open at t (051-R10)
            ok = np.isfinite(meas).all(1) & data.mask_rows(pi_rows)[:, act]
            ent = {"action": int(act), "column": None if col is None else names[col], "pairs": int(ok.sum())}
            if col is not None and ok.any():
                pm, mm = pred[ok, col], meas[ok, col]
                # ties (no predicted or no measured change) and columns the predictor leaves out (NaN) are not counted
                agree, ties = sign_agreement(pm, mm)
                ent["direction"] = {"pairs": int(np.isfinite(agree).sum()), "ties": ties, "agree": _rate(agree),
                                    "pred_mean": float(pm.mean()) if np.isfinite(pm).all() else None, "meas_mean": float(mm.mean())}
                dirs.append((agree, [trunk[i] for i in np.nonzero(ok)[0]]))
            res["alts"][a] = ent
            P.append(pred[ok])
            M.append(meas[ok])
            U += [trunk[i] for i in np.nonzero(ok)[0]]
        if dirs:
            v = np.concatenate([d for d, _ in dirs])
            u = np.concatenate([np.asarray(t, object) for _, t in dirs]).astype(str)
            res["direction"] = {"pairs": int(np.isfinite(v).sum()), **ps.ci(ps.Boot(u, boot_n, seed).mean(v))}
        if U:
            P, M = np.concatenate(P), np.concatenate(M)
            kept = np.isfinite(P).all(0)
            n_c = int(kept.sum())
            pv, mv = P[:, kept].reshape(-1), M[:, kept].reshape(-1)
            unit = np.repeat(np.asarray(U, object), n_c).astype(str)
            res["columns_kept"] = [names[i] for i in np.nonzero(kept)[0]]
            res["spearman_all_columns"] = ps.ci(ps.Boot(unit, boot_n, seed).spearman(pv, mv)) if len(pv) else None
        out[f"k{k}"] = res
    return out


def diagnostics(B: Branches) -> dict:
    """Invalid branches, refused / missed rates of the force rules, dilution means (module docstring 5)."""
    inv_total = sum(sum(v.values()) for v in B.invalid.values())
    out = {"branches": B.n_branches, "valid": B.n_valid, "invalid": inv_total,
           "invalid_rate": inv_total / B.n_branches if B.n_branches else None, "invalid_by_alt": B.invalid,
           "outside_points": B.outside_points, "points": len(B.points), "skipped_points": B.skipped_points, "alts": {}}
    for a in B.alts:
        brs = [b for p in B.points for _, b in B.cells.get((p, a), [])]
        ent = {"branches": len(brs), "executed": sum(executed(b) for b in brs)}
        forces = [i for i, r in enumerate(B.rules[a]) if r.get("op") == "force"]
        if forces and brs:
            recs = []
            for b in brs:
                iv = b.get("intervention") or []
                recs += [iv[i] if i < len(iv) and isinstance(iv[i], dict) else {} for i in forces]
            ent["missed_rate"] = float(np.mean([_missed(r) for r in recs]))
            ent["refused_rate"] = float(np.mean([bool(r.get("rejected")) for r in recs]))
        leaves: dict = {}
        for b in brs:
            for k, v in _leaves(b.get("dilution") or {}):
                leaves.setdefault(k, []).append(v)
        ent["dilution"] = {k: {"n": len(v), "mean": float(np.mean(v))} for k, v in sorted(leaves.items())}
        out["alts"][a] = ent
    return out


def _leaves(d, prefix=""):
    """(dotted key, number) of every numeric leaf; a list of `{action, ...}` entries (051-R9 ③
    `DilutionEntry`) is keyed by the action."""
    if isinstance(d, list):
        for i, v in enumerate(d):
            key = v.get("action", i) if isinstance(v, dict) else i
            yield from _leaves({k: x for k, x in v.items() if k != "action"} if isinstance(v, dict) else v, f"{prefix}{key}.")
    elif isinstance(d, dict):
        for k, v in d.items():
            yield from _leaves(v, f"{prefix}{k}.")
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        yield prefix[:-1], float(d)


# ---------------------------------------------------------------- report


def _f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, dict):
        return f"{_f(x['pt'], nd)} [{_f(x['lo'], nd)}, {_f(x['hi'], nd)}]" if "pt" in x else "-"
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return str(x)
    return "-" if not np.isfinite(x) else f"{x:.{nd}f}"


def markdown(rep: dict) -> str:
    s = rep["settings"]
    dg = rep["diagnostics"]
    inv = "" if dg["invalid_rate"] is None else f", {dg['invalid_rate'] * 100:.2f}%"
    L = ["# 분기 게임 분석 (051-R13)", "",
         f"- 시점 {dg['points']}(제외 {len(dg['skipped_points'])}), 분기 판 {dg['branches']}(유효 {dg['valid']}, 무효 {dg['invalid']}{inv})",
         f"- 기준 대안 `{s['baseline']}`, bootstrap {s['boot']}회(줄기 판 단위, seed {s['boot_seed']}, 95%), 판정 문턱 없음(AMB-051-5)",
         f"- 체크포인트 확인: {json.dumps(rep.get('checks', {}), ensure_ascii=False)}", "", "## 1. 짝 효과 D(a)", ""]
    empty = [y for y, ent in rep["paired"].items() if not any(r["all"]["points"] for r in ent["alts"].values())]
    if empty:
        L += [f"- 값이 없는 결과(분기 기록에 없음): {', '.join(empty)}", ""]
    own = [y for y in rep["paired"] if y.startswith("own@") and y not in empty]
    alts = [x for x in rep["alts"] if x != s["baseline"]]
    for y, ent in rep["paired"].items():
        if y in empty or y in own:
            continue
        L += [f"### {y}" + (" (0/1)" if ent["binary"] else ""), "",
              "| 대안 | 시점(줄기) | D 모든 분기 | 시점(줄기) | D 실행된 분기만 | split-half r | Spearman–Brown |", "|---|---|---|---|---|---|---|"]
        for a, r in ent["alts"].items():
            al, ex, sh = r["all"], r["executed"], r["split_half"]
            L.append(f"| {a} | {al['points']}({al['trunks']}) | {_f(al['D'])} | {ex['points']}({ex['trunks']}) | {_f(ex['D'])} | {_f(sh['r'])} | {_f(sh['spearman_brown'])} |")
        if ent["binary"]:
            L += ["", "| 대안 | 판 | 1 | 비율 |", "|---|---|---|---|"]
            for a, c in ent["counts"].items():
                L.append(f"| {a} | {c['n']} | {c['k']} | {_f(c['rate'])} |")
        L.append("")
    if own:
        L += ["### 내 상태(표본:열, D 모든 분기)", "", "| 결과 | " + " | ".join(alts) + " |", "|---|" + "---|" * len(alts)]
        for y in own:
            L.append(f"| {y[4:]} | " + " | ".join(_f(rep["paired"][y]["alts"][x]["all"]["D"], 2) for x in alts) + " |")
        L.append("")
    m = rep.get("M3u") or {}
    L += ["## 2. M3u (scv8 머리 ΔQ̂ 대 분기 효과)", ""]
    if m.get("skipped"):
        L += [f"- 건너뜀: {m['skipped']}", ""]
    else:
        L += [f"- 결과 `{m.get('outcome')}`, 한 결정 강제 대안 {m.get('alts')}, 덤프 행이 있는 시점 {m.get('points_with_row')}, "
              f"잡음 천장 split-half {json.dumps(m.get('split_half'))}", "",
              "| k | 짝 | 부호 일치 | Spearman | 천장 | 천장 대비 |", "|---|---|---|---|---|---|"]
        for key in sorted(x for x in m if x.startswith("k")):
            r = m[key]
            L.append(f"| {key[1:]} | {r.get('pairs')} | {_f(r.get('sign_agree'))} | {_f(r.get('spearman'))} | {_f(r.get('noise_ceiling'))} | {_f(r.get('ratio_to_ceiling'))} |")
        L += ["", "| k | 대안 | 짝 | 평균 ΔQ̂ | 평균 d | 부호 일치 |", "|---|---|---|---|---|---|"]
        for key in sorted(x for x in m if x.startswith("k")):
            for a, r in m[key]["alts"].items():
                L.append(f"| {key[1:]} | {a} | {r['pairs']} | {_f(r['dq_mean'])} | {_f(r['d_mean'])} | {_f(r['sign_agree'])} |")
            if m[key].get("s0b_m3b"):
                L.append(f"| {key[1:]} | (S0b M3b ΔQ 러시 평균, {m[key]['s0b_m3b']['file']}) | - | {_f(m[key]['s0b_m3b'].get('rush_mean'))} | - | - |")
        L.append("")
    b = rep.get("bunker") or {}
    if b:
        L += ["### 벙커 대안", "", "| 대안 | 규칙 | D(`" + s["m3u_outcome"] + "`) 모든 분기 | D 실행된 분기만 | 평균 ΔQ̂(k) | S0b M3b ΔQ 러시 평균(k) |",
              "|---|---|---|---|---|---|"]
        for a, r in b.items():
            dq = ", ".join(f"{k}: {_f(v)}" for k, v in r.get("dq_mean", {}).items()) or "-"
            m3b = ", ".join(f"{k}: {_f(v)}" for k, v in r.get("s0b_m3b_rush", {}).items()) or "-"
            L.append(f"| {a} | {json.dumps(r['rules'], ensure_ascii=False)} | {_f(r['all'])} | {_f(r['executed'])} | {dq} | {m3b} |")
        L.append("")
    m = rep.get("M2c") or {}
    L += ["## 3. M2c (내 상태 변화: 예측 대 분기)", ""]
    if m.get("skipped"):
        L += [f"- 건너뜀: {m['skipped']}", ""]
    else:
        L += ["| k | 표본 | 방향 일치(강제 행동 열) | Spearman(열 전체) |", "|---|---|---|---|"]
        for key in sorted(x for x in m if x.startswith("k")):
            r = m[key]
            L.append(f"| {key[1:]} | {r.get('sample')} | {_f(r.get('direction'))} | {_f(r.get('spearman_all_columns'))} |")
        L += ["", "| k | 대안 | 열 | 짝 | 일치 | 예측 평균 | 측정 평균 |", "|---|---|---|---|---|---|---|"]
        for key in sorted(x for x in m if x.startswith("k")):
            for a, r in m[key].get("alts", {}).items():
                d = r.get("direction") or {}
                L.append(f"| {key[1:]} | {a} | {r['column']} | {d.get('pairs', r['pairs'])} | {_f(d.get('agree'))} | {_f(d.get('pred_mean'))} | {_f(d.get('meas_mean'))} |")
        L.append("")
    L += ["## 4. 시점 행", "", f"- 행 없음·제외: {json.dumps(rep.get('rows_missing', {}), ensure_ascii=False)}", "",
          "## 5. 무효·거절·missed·희석", "", "| 대안 | 분기 | 실행 | missed 비율 | 거절 비율 | 무효(사유) |", "|---|---|---|---|---|---|"]
    for a, r in dg["alts"].items():
        L.append(f"| {a} | {r['branches']} | {r['executed']} | {_f(r.get('missed_rate'))} | {_f(r.get('refused_rate'))} | {json.dumps(dg['invalid_by_alt'].get(a, {}))} |")
    L += ["", "| 대안 | 희석 항목 | n | 평균 |", "|---|---|---|---|"]
    for a, r in dg["alts"].items():
        for k, v in r["dilution"].items():
            L.append(f"| {a} | {k} | {v['n']} | {_f(v['mean'], 1)} |")
    return "\n".join(L) + "\n"


def analyse(a) -> dict:
    """The report dict (module docstring)."""
    d = Path(a.dir)
    points = read_jsonl(a.points or d / "points.jsonl")
    branches = read_jsonl(a.branches or d / "branches.jsonl")
    B = Branches(points, branches, a.baseline)
    if a.baseline not in B.alts:
        raise SystemExit(f"plan-branch: no branch of the baseline alternative {a.baseline!r} (alternatives {B.alts})")
    outcomes = [x for x in a.outcomes.split(",") if x]
    if a.m3u_outcome not in outcomes:
        outcomes.append(a.m3u_outcome)
    names = own_columns()[0]
    cols = [c for c in a.own_cols.split(",") if c]
    if bad := [c for c in cols if c not in names]:
        raise SystemExit(f"plan-branch: --own-cols {bad} are not own-state columns ({names})")
    outcomes += [f"own@{s}:{c}" for s in a.own_samples.split(",") if s for c in cols]
    ks = [int(x) for x in a.ks.split(",") if x]
    trunks = sorted(set(B.trunks))
    tid = {t: i for i, t in enumerate(trunks)}
    boot = ps.Boot(np.array([tid[t] for t in B.trunks], np.int64), a.boot, a.boot_seed) if B.points else None
    rep = {"kind": "plan-branch", "settings": {"baseline": a.baseline, "outcomes": outcomes, "m3u_outcome": a.m3u_outcome, "ks": ks,
                                               "boot": a.boot, "boot_seed": a.boot_seed, "row_lag": a.row_lag, "own_samples": a.own_samples, "own_cols": cols,
                                               "samples": {"8": a.sample_k8, "32": a.sample_k32}},
           "alts": {x: B.rules[x] for x in B.alts}, "trunks": len(trunks)}
    rep["paired"] = paired_table(B, outcomes, boot) if boot is not None else {}

    # 051-R13 ②: the bunker force / ban effects of the M3u outcome (ΔQ̂ and the S0b M3b ΔQ added below)
    bunker = tb.action_id("Build_Bunker")
    rep["bunker"] = {}
    for x in B.alts:
        if x != a.baseline and any(action_of(r["action"]) == bunker for r in B.rules[x] if r.get("action") is not None):
            r = rep["paired"].get(a.m3u_outcome, {}).get("alts", {}).get(x)
            if r:
                rep["bunker"][x] = {"rules": B.rules[x], "all": r["all"]["D"], "executed": r["executed"]["D"], "dq_mean": {}}
    rep["diagnostics"] = diagnostics(B)
    checks = {"points_checkpoint_sha256": sorted({str(p.get("checkpoint_sha256")) for p in points if p.get("checkpoint_sha256")})}
    if not (a.dump and a.s0):
        why = "no --dump / --s0"
        rep["M3u"], rep["M2c"] = {"skipped": why}, {"skipped": why}
    else:
        s0 = Path(a.s0)
        design, pred_sha = predictor_design(s0, ks)
        if design is None:
            raise SystemExit(f"plan-branch: no models/A_k<k>.pt for --ks {ks} in {s0}")
        data = pp.Data([pp.Source("trunk", "model", "eval", a.dump)], ks, keep_bot=True, design=design)
        dump_sha = data.checkpoint["checkpoint_sha256"]
        checks.update({"dump_checkpoint_sha256": dump_sha, "predictor_checkpoint_sha256": pred_sha})
        if pred_sha and pred_sha != dump_sha:
            raise SystemExit(f"plan-branch: the dump's checkpoint {dump_sha} is not the predictors' {pred_sha}")
        checks["points_match_dump"] = all(c == dump_sha for c in checks["points_checkpoint_sha256"]) if checks["points_checkpoint_sha256"] else None
        roots = [Path(x) for x in (a.root, os.environ.get("URANIUM_SHARED_DIR"), a.dir) if x]
        rows, missing = find_rows(B, data, roots, a.row_lag)
        reasons: dict = {}
        for v in missing.values():
            reasons[v["reason"]] = reasons.get(v["reason"], 0) + 1
        rep["rows_missing"] = {"points": len(missing), "by_reason": reasons, "detail": missing}
        nets = {k: m for k in ks if (m := load_predictor(s0, k, data)) is not None}
        rep["M3u"] = m3u(B, data, rows, nets, a.m3u_outcome, a.boot, a.boot_seed, s0)
        rep["M2c"] = m2c(B, data, rows, nets, {8: a.sample_k8, 32: a.sample_k32}, a.boot, a.boot_seed)
        for x, ent in rep["bunker"].items():
            for k in ks:
                r = rep["M3u"].get(f"k{k}", {})
                ent["dq_mean"][str(k)] = (r.get("alts", {}).get(x) or {}).get("dq_mean")
                ent.setdefault("s0b_m3b_rush", {})[str(k)] = (r.get("s0b_m3b") or {}).get("rush_mean")
    rep["checks"] = checks
    return rep


def run(a) -> int:
    torch.use_deterministic_algorithms(True)
    rep = analyse(a)
    out = Path(a.out) if a.out else Path(a.dir) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(rep, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    (out / "plan-branch.md").write_text(markdown(rep), encoding="utf-8")
    print(f"[out] {out / 'analysis.json'} {out / 'plan-branch.md'}", flush=True)
    return 0
