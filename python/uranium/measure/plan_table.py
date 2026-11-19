"""Plan branch table (055-R7): plan effects, plan x opponent table, interaction, split-half, bite, missed and drift"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from uranium.diag.planning import stats as ps
from uranium.diag.planning.dump import own_columns, own_state
from uranium.measure import branch_common as bc
from uranium.measure.branch_collect import score as collect_score

OUTCOMES = ("score", "held_8m", "alive_10m", "inc_10m")
SPLIT_OUTCOMES = ("score", "held_8m", "alive_10m")
DRIFT_OUTCOMES = ("score", "held_8m", "alive_10m")
OWN_SAMPLES = ("t60", "horizon")
OWN_COLS = ("scv", "supply_used", "supply_total", "barracks", "bunker", "command_center", "supply_depot", "refinery", "factory", "army")
CELLS = ("t0-zerg", "t0-zerg-rush", "t0-protoss", "t0-terran")
Z80 = 2.80
FPS = bc.FPS


SCORE_AT: list[int | None] = [None]


def score_of(row: dict) -> float:
    s = row["score"] if "score" in row else collect_score(row.get("labels") or {"_": 0}, row.get("end"))
    if s is not None and SCORE_AT[0] is not None:
        end = row.get("end") or {}
        s = collect_score(row.get("labels") or {"_": 0}, {**end, "before_horizon": int(end.get("end_frame") or 0) < SCORE_AT[0]})
    return float("nan") if s is None else float(s)


def num(v) -> float:
    return float(v) if isinstance(v, (int, float)) else float("nan")


def own_of(row: dict, sample: str) -> np.ndarray | None:
    cache = row.setdefault("_own", {})
    if sample not in cache:
        s = (row.get("samples") or {}).get(sample)
        cache[sample] = None if not isinstance(s, dict) or s.get("scalar") is None else own_state(np.asarray(s["scalar"], np.float32)[None])[0]
    return cache[sample]


def value(row: dict, name: str) -> float:
    if name == "score":
        return score_of(row)
    if name.startswith("own@"):
        sample, _, col = name[4:].partition(":")
        o = own_of(row, sample)
        return float("nan") if o is None else float(o[own_columns()[0].index(col)])
    labels = row.get("labels") or {}
    if name in labels:
        return num(labels[name])
    ops = row.get("ops") or {}
    if name in ops:
        return num(ops[name])
    return float("nan")


def chi2_sf(x: float, k: int) -> float:
    if not math.isfinite(x) or k < 1:
        return float("nan")
    if x <= 0:
        return 1.0
    if k % 2 == 0:
        term = total = 1.0
        for i in range(1, k // 2):
            term *= (x / 2) / i
            total += term
        return math.exp(-x / 2) * total
    total = math.erfc(math.sqrt(x / 2))
    term = math.sqrt(2 * x / math.pi) * math.exp(-x / 2)
    for r in range(1, (k - 1) // 2 + 1):
        if r > 1:
            term *= x / (2 * r - 1)
        total += term
    return total


def mean_finite(v) -> float:
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


class Table:
    def __init__(self, points: list[dict], rows: list[dict], baseline: str = "none", one_rule_list: bool = True):
        self.baseline = baseline
        valid = {p["point"]: p for p in points if p.get("valid", True)}
        self.invalid: dict[str, dict[str, int]] = {}
        self.games: dict[str, int] = {}
        self.cells: dict[tuple[str, str], list[dict]] = {}
        self.rules: dict[str, list[dict]] = {}
        self.outside = 0
        for r in rows:
            a = r["alt"]
            if self.rules.setdefault(a, r.get("rules") or []) != (r.get("rules") or []) and one_rule_list:
                raise bc.Refused(f"alternative {a}: rules differ between branches (a plan has one rule list, 055-R2)")
            self.games[a] = self.games.get(a, 0) + 1
            if not r.get("valid", True):
                why = str((r.get("invalid") or {}).get("reason", "?"))
                self.invalid.setdefault(a, {}).setdefault(why, 0)
                self.invalid[a][why] += 1
                continue
            if r["point"] not in valid:
                self.outside += 1
                continue
            self.cells.setdefault((r["point"], a), []).append(r)
        for v in self.cells.values():
            v.sort(key=lambda r: int(r.get("repeat", 0)))
        self.alts = [self.baseline] + [a for a in self.rules if a != self.baseline]
        self.plans = self.alts[1:]
        self.points = sorted({p for p, _ in self.cells})
        self.f_t = np.array([int(valid[p]["f_t"]) for p in self.points], np.int64)
        self.trunk = np.array([p.split("#")[0] for p in self.points])
        self.cell = np.array([valid[p].get("opponent", "?") for p in self.points])
        self.pi = [{int(k): float(v) for k, v in valid[p].get("policy_top") or []} for p in self.points]
        self.none_repeats = max((len(self.rows(p, self.baseline)) for p in self.points), default=0)
        self.cell_names = [c for c in CELLS if c in set(self.cell)] + sorted(set(self.cell) - set(CELLS))
        self.n_rows = len(rows)
        self.n_valid = sum(len(v) for v in self.cells.values())

    def rows(self, p: str, a: str) -> list[dict]:
        return self.cells.get((p, a), [])

    def means(self, a: str, y: str) -> np.ndarray:
        return np.array([mean_finite([value(r, y) for r in self.rows(p, a)]) for p in self.points])

    def d(self, a: str, y: str) -> np.ndarray:
        return self.means(a, y) - self.means(self.baseline, y)


def boot_of(trunks: np.ndarray, n: int, seed: int) -> ps.Boot | None:
    return ps.Boot(trunks, n, seed) if len(trunks) else None


def effect(d: np.ndarray, trunks: np.ndarray, n: int, seed: int) -> dict:
    ok = np.isfinite(d)
    if not ok.any():
        return {"points": 0, "trunks": 0, "D": None, "se": None, "mde": None}
    b = boot_of(trunks[ok], n, seed)
    pt, reps = b.mean(d[ok])
    reps = np.asarray(reps, np.float64)
    reps = reps[np.isfinite(reps)]
    se = float(reps.std(ddof=1)) if len(reps) > 1 else float("nan")
    return {"points": int(ok.sum()), "trunks": int(len(set(trunks[ok]))), "D": ps.ci((pt, reps)), "se": se, "mde": Z80 * se}


def interaction(cells: dict[str, dict]) -> dict:
    used = [k for k, c in cells.items() if c["D"] is not None and c["se"] and math.isfinite(c["se"]) and c["se"] > 0]
    est = [(cells[k]["D"]["pt"], cells[k]["se"] ** 2) for k in used]
    if len(est) < 2:
        return {"chi2": None, "df": None, "p": None, "cells": used}
    w = np.array([1 / v for _, v in est])
    x = np.array([d for d, _ in est])
    pooled = float((w * x).sum() / w.sum())
    chi2 = float((w * (x - pooled) ** 2).sum())
    return {"chi2": chi2, "df": len(est) - 1, "p": chi2_sf(chi2, len(est) - 1), "pooled": pooled, "cells": used}


def corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def half_corr(T: Table, ds: dict[str, np.ndarray], in_a: np.ndarray) -> tuple[float, float]:
    tabs = [np.array([[mean_finite(ds[a][side & (T.cell == c)]) for c in T.cell_names] for a in T.plans]) for side in (in_a, ~in_a)]
    res = [t - np.nanmean(t, axis=1, keepdims=True) for t in tabs]
    return corr(tabs[0].ravel(), tabs[1].ravel()), corr(res[0].ravel(), res[1].ravel())


def split_half(T: Table, ds: dict[str, np.ndarray], splits: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    by_cell = {c: sorted(set(T.trunk[T.cell == c])) for c in T.cell_names}
    rs = {"table": [], "interaction": []}
    for _ in range(splits):
        half_a: set[str] = set()
        for c in T.cell_names:
            ts = by_cell[c]
            half_a |= {ts[i] for i in rng.permutation(len(ts))[:len(ts) // 2]}
        r_table, r_inter = half_corr(T, ds, np.array([t in half_a for t in T.trunk]))
        rs["table"].append(r_table)
        rs["interaction"].append(r_inter)
    out = {"splits": splits, "seed": seed}
    for k, v in rs.items():
        r = mean_finite(v)
        out[k] = {"r": r, "spearman_brown": 2 * r / (1 + r) if math.isfinite(r) and r > -1 else float("nan")}
    return out


def first_frame(row: dict | None, action: str) -> tuple[bool, int | None]:
    for e in (row or {}).get("dilution") or []:
        if e.get("action") == action:
            return True, e.get("first_frame")
    return False, None


def rule_bite(rule: dict, rec: dict, none_row: dict | None, f_t: int) -> float:
    known, f0 = first_frame(none_row, rule["action"])
    if not known:
        return float("nan")
    if rule["op"] == "ban":
        return float(f0 is not None and f_t + FPS * int(rule["from_s"]) <= f0 < f_t + FPS * int(rule["to_s"]))
    ff = rec.get("force_frame")
    if rec.get("missed") or ff is None:
        return 0.0
    return float(f0 is None or f0 > ff)


def plan_first(rule: dict, rec: dict, row: dict, horizon: int) -> int:
    if rule["op"] == "force" and rec.get("force_frame") is not None:
        return int(rec["force_frame"])
    _, f = first_frame(row, rule["action"])
    return horizon if f is None else int(f)


def bite(T: Table, a: str, horizon: int, n: int, seed: int, outcomes: tuple[str, ...]) -> dict:
    rules = T.rules[a]
    per_rule = [{"bite": [], "diff_s": [], "missed": [], "rejected": [], "applied": [], "at_t": [], "pi_t": []} for _ in rules]
    ids = bc.action_ids()
    plan_bite = np.full(len(T.points), np.nan)
    for i, p in enumerate(T.points):
        nones = T.rows(p, T.baseline)
        none_row = nones[0] if nones else None
        bits = []
        for row in T.rows(p, a):
            iv = row.get("intervention") or []
            rb = []
            for k, rule in enumerate(rules):
                rec = iv[k] if k < len(iv) and isinstance(iv[k], dict) else {}
                b = rule_bite(rule, rec, none_row, int(T.f_t[i]))
                rb.append(b)
                acc = per_rule[k]
                acc["bite"].append(b)
                acc["applied"].append(num(rec.get("applied", 0)))
                if rule["op"] == "force":
                    missed = bool(rec.get("missed")) or rec.get("force_frame") is None
                    acc["missed"].append(float(missed))
                    if not missed:
                        acc["rejected"].append(float(bool(rec.get("rejected"))))
                        at_t = int(rec["force_frame"]) == int(T.f_t[i])
                        acc["at_t"].append(float(at_t))
                        if at_t and T.pi[i]:
                            acc["pi_t"].append(T.pi[i].get(ids[rule["action"]], 0.0))
                known, f0 = first_frame(none_row, rule["action"])
                if known:
                    acc["diff_s"].append((plan_first(rule, rec, row, horizon) - (horizon if f0 is None else int(f0))) / FPS)
            rb = np.array(rb)
            bits.append(1.0 if (rb == 1).any() else (float("nan") if np.isnan(rb).any() else 0.0))
        if bits:
            plan_bite[i] = mean_finite(bits)
    out = {"rate": mean_finite(plan_bite), "points": int(np.isfinite(plan_bite).sum()), "rules": [], "bitten": {}}
    for rule, acc in zip(rules, per_rule):
        out["rules"].append({"rule": rule, "bite": mean_finite(acc["bite"]), "n": int(np.isfinite(np.asarray(acc["bite"], float)).sum()),
                             "first_frame_diff_s": mean_finite(acc["diff_s"]), "applied_mean": mean_finite(acc["applied"]),
                             "missed": mean_finite(acc["missed"]) if rule["op"] == "force" else None,
                             "rejected": mean_finite(acc["rejected"]) if rule["op"] == "force" else None,
                             "forced_at_t": mean_finite(acc["at_t"]) if rule["op"] == "force" else None,
                             "pi_at_t": mean_finite(acc["pi_t"]) if rule["op"] == "force" else None})
    sel = plan_bite > 0.5
    for y in outcomes:
        d = T.d(a, y)
        out["bitten"][y] = effect(np.where(sel, d, np.nan), T.trunk, n, seed)
    return out


def drift(T: Table, other: Table, outcomes: tuple[str, ...], n: int, seed: int) -> dict:
    idx = {p: i for i, p in enumerate(other.points)}
    common = [i for i, p in enumerate(T.points) if p in idx]
    out = {"points": len(common)}
    if not common:
        return out
    trunks = T.trunk[common]
    for y in outcomes:
        mine = T.means(T.baseline, y)[common]
        theirs = other.means(other.baseline, y)[[idx[T.points[i]] for i in common]]
        out[y] = {"mean_b3": mean_finite(mine), "mean_compare": mean_finite(theirs), **effect(mine - theirs, trunks, n, seed)}
    return out


def analyse(job: Path, compare: Path | None, n: int = 2000, seed: int = 7, splits: int = 200, split_seed: int = 7,
            horizon: int = 17280, baseline: str = "none") -> dict:
    T = Table(bc.read_jsonl(job / "points.jsonl"), bc.read_jsonl(job / "branches.jsonl"), baseline)
    if baseline not in T.rules:
        raise bc.Refused(f"{job}: no {baseline} branch")
    own = [f"own@{s}:{c}" for s in OWN_SAMPLES for c in OWN_COLS]
    inv = sum(sum(v.values()) for v in T.invalid.values())
    rep: dict = {"kind": "plan-table", "job": str(job.name), "settings": {"boot": n, "boot_seed": seed, "splits": splits, "split_seed": split_seed,
                                                                         "horizon": horizon, "baseline": baseline},
                 "games": T.n_rows, "valid": T.n_valid, "invalid": inv, "invalid_rate": inv / T.n_rows if T.n_rows else None,
                 "invalid_by_alt": T.invalid, "games_by_alt": T.games, "outside_points": T.outside,
                 "points": len(T.points), "trunks": int(len(set(T.trunk))), "none_repeats_max": T.none_repeats, "cells": {c: int((T.cell == c).sum()) for c in T.cell_names},
                 "plans": {a: T.rules[a] for a in T.plans}, "effects": {}, "strata": {}, "own": {}, "split_half": {}, "bite": {}}
    for y in OUTCOMES:
        base = T.means(baseline, y)
        ent = {"none": mean_finite(base), "none_by_cell": {c: mean_finite(base[T.cell == c]) for c in T.cell_names}, "plans": {}}
        strata = {}
        for a in T.plans:
            d = T.d(a, y)
            ent["plans"][a] = effect(d, T.trunk, n, seed)
            cells = {c: effect(np.where(T.cell == c, d, np.nan), T.trunk, n, seed) for c in T.cell_names}
            strata[a] = {"cells": cells, "interaction": interaction(cells)}
        rep["effects"][y] = ent
        rep["strata"][y] = strata
    for y in own:
        rep["own"][y] = {"none": mean_finite(T.means(baseline, y)), "plans": {a: effect(T.d(a, y), T.trunk, n, seed) for a in T.plans}}
    for y in SPLIT_OUTCOMES:
        rep["split_half"][y] = split_half(T, {a: T.d(a, y) for a in T.plans}, splits, split_seed)
    for a in T.plans:
        rep["bite"][a] = bite(T, a, horizon, n, seed, ("score",))
    if compare is not None:
        other = Table(bc.read_jsonl(compare / "points.jsonl"), bc.read_jsonl(compare / "branches.jsonl"), baseline, one_rule_list=False)
        rep["drift"] = {"compare": str(compare.name), **drift(T, other, DRIFT_OUTCOMES, n, seed)}
    return rep


def f(x, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, dict):
        return f"{f(x['pt'], nd)} [{f(x['lo'], nd)}, {f(x['hi'], nd)}]" if x.get("pt") is not None else "-"
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return str(x)
    return "-" if not math.isfinite(x) else f"{x:.{nd}f}"


def markdown(rep: dict) -> str:
    plans = list(rep["plans"])
    cells = list(rep["cells"])
    inv = "-" if rep["invalid_rate"] is None else f"{rep['invalid_rate'] * 100:.2f}%"
    L = ["# 계획 분기 표 (055-R7)", "",
         f"- job `{rep['job']}`: 분기 {rep['games']}(유효 {rep['valid']}, 무효 {rep['invalid']} = {inv}), 시점 {rep['points']}, 줄기 {rep['trunks']}, 층 {json.dumps(rep['cells'])}",
         *([f"- score: {rep['score_at']} 프레임 시점(그때까지 안 끝난 판은 무, 055-R13)"] if rep.get("score_at") is not None else []),
         f"- 구간: 줄기 bootstrap {rep['settings']['boot']}회(seed {rep['settings']['boot_seed']}, 95%), 최소 검출 효과 = 2.80 × SE, 판정 문턱 없음(055-R12)", "",
         "## 1. 계획 효과 D (계획 − none)", "", "| 결과 | none | " + " | ".join(plans) + " |", "|---|---|" + "---|" * len(plans)]
    for y, ent in rep["effects"].items():
        nd = 1 if y == "inc_10m" else 3
        L.append(f"| {y} | {f(ent['none'], nd)} | " + " | ".join(f(ent["plans"][a]["D"], nd) for a in plans) + " |")
    L += ["", "| 결과 | 값 | " + " | ".join(plans) + " |", "|---|---|" + "---|" * len(plans)]
    for y, ent in rep["effects"].items():
        nd = 1 if y == "inc_10m" else 3
        L.append(f"| {y} | 시점(줄기) | " + " | ".join(f"{ent['plans'][a]['points']}({ent['plans'][a]['trunks']})" for a in plans) + " |")
        L.append(f"| {y} | SE / MDE | " + " | ".join(f"{f(ent['plans'][a]['se'], nd)} / {f(ent['plans'][a]['mde'], nd)}" for a in plans) + " |")
    L += ["", "## 2. 층 표 (계획 × T0 상대)", ""]
    for y, strata in rep["strata"].items():
        nd = 1 if y == "inc_10m" else 3
        base = rep["effects"][y]["none_by_cell"]
        L += [f"### {y}", "", "| 계획 | " + " | ".join(cells) + " | 상호작용 χ²(df) | p |", "|---|" + "---|" * len(cells) + "---|---|",
              "| none 평균 | " + " | ".join(f(base[c], nd) for c in cells) + " | | |"]
        for a in plans:
            it = strata[a]["interaction"]
            chi = "-" if it["chi2"] is None else f"{it['chi2']:.2f}({it['df']}{'' if len(it['cells']) == len(cells) else ', 칸 ' + '·'.join(it['cells'])})"
            L.append(f"| {a} | " + " | ".join(f(strata[a]["cells"][c]["D"], nd) for c in cells) + f" | {chi} | {f(it['p'])} |")
        L += ["", "| 계획 | " + " | ".join(f"{c} 시점 / MDE" for c in cells) + " |", "|---|" + "---|" * len(cells)]
        for a in plans:
            L.append(f"| {a} | " + " | ".join(f"{strata[a]['cells'][c]['points']} / {f(strata[a]['cells'][c]['mde'], nd)}" for c in cells) + " |")
        L.append("")
    L += ["## 3. 잡음 천장 (층 안 줄기 반반 split-half)", "", "| 결과 | 표 전체 r | SB | 상호작용 잔차 r | SB | 나눔 |", "|---|---|---|---|---|---|"]
    for y, s in rep["split_half"].items():
        L.append(f"| {y} | {f(s['table']['r'])} | {f(s['table']['spearman_brown'])} | {f(s['interaction']['r'])} | {f(s['interaction']['spearman_brown'])} | {s['splits']}(seed {s['seed']}) |")
    L += ["", "## 4. 내 상태 (D, 계획 − none)", "", "| 표본:열 | none | " + " | ".join(plans) + " |", "|---|---|" + "---|" * len(plans)]
    for y, ent in rep["own"].items():
        L.append(f"| {y[4:]} | {f(ent['none'], 2)} | " + " | ".join(f(ent["plans"][a]["D"], 2) for a in plans) + " |")
    L += ["", "## 5. 물림·missed·거절", "", "| 계획 | 물림 비율(시점) | 물린 시점만 score D | 시점(줄기) |", "|---|---|---|---|"]
    for a in plans:
        b = rep["bite"][a]
        bb = b["bitten"]["score"]
        L.append(f"| {a} | {f(b['rate'])}({b['points']}) | {f(bb['D'])} | {bb['points']}({bb['trunks']}) |")
    L += ["", "| 계획 | 규칙 | 물림 | 첫 행동 프레임 차(초, 계획 − none) | 적용 결정 수 평균 | missed | 거절 | t 결정 강제 | 그때 π(행동) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a in plans:
        for r in rep["bite"][a]["rules"]:
            rule = r["rule"]
            L.append(f"| {a} | {rule['op']} {rule['action']} [{rule['from_s']}, {rule['to_s']}) | {f(r['bite'])} | {f(r['first_frame_diff_s'], 1)} | "
                     f"{f(r['applied_mean'], 1)} | {f(r['missed'])} | {f(r['rejected'])} | {f(r['forced_at_t'])} | {f(r['pi_at_t'])} |")
    L += ["", "| 대안 | 분기 | 무효(사유) |", "|---|---|---|"]
    for a in [rep["settings"]["baseline"], *plans]:
        L.append(f"| {a} | {rep['games_by_alt'].get(a, 0)} | {json.dumps(rep['invalid_by_alt'].get(a, {}))} |")
    if "drift" in rep:
        dr = rep["drift"]
        L += ["", f"## 6. 환경 드리프트 (`none`: B3 − {dr['compare']}, 같은 시점 {dr['points']})", "", "| 결과 | B3 | 비교 | 차 |", "|---|---|---|---|"]
        for y in DRIFT_OUTCOMES:
            if y in dr:
                L.append(f"| {y} | {f(dr[y]['mean_b3'])} | {f(dr[y]['mean_compare'])} | {f(dr[y]['D'])} |")
    return "\n".join(L) + "\n"


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--job", required=True, help="B3 branch job folder with points.jsonl and branches.jsonl (as given if it exists, else under URANIUM_SHARED_DIR)")
    ap.add_argument("--compare", default=None, help="earlier branch job on the same points whose none branches the drift check compares with (055-R7 6)")
    ap.add_argument("--root", default=None, help="shared drive (default URANIUM_SHARED_DIR)")
    ap.add_argument("--out", default=None, help="output folder (default <job>/analysis)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=7)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--split-seed", type=int, default=7)
    ap.add_argument("--horizon", type=int, default=17280, help="horizon frame H; a rule action never picked counts as H")
    ap.add_argument("--score-at", type=int, default=None, help="score as of this frame from the end record: a game not over by then is a draw (055-R13)")
    a = ap.parse_args(argv)
    try:
        exists = all(x is None or Path(x).exists() for x in (a.job, a.compare))
        root = Path(".") if exists and not a.root else bc.shared_root(a.root)
        job = bc.job_dir(a.job, root)
        compare = None if a.compare is None else bc.job_dir(a.compare, root)
        SCORE_AT[0] = a.score_at
        rep = analyse(job, compare, a.boot, a.boot_seed, a.splits, a.split_seed, a.horizon)
        rep["score_at"] = a.score_at
    except bc.Refused as e:
        raise SystemExit(f"REFUSED: {e}") from e
    out = Path(a.out) if a.out else job / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan-table.json").write_bytes((json.dumps(clean(rep), ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    (out / "plan-table.md").write_bytes(markdown(rep).encode("utf-8"))
    print(f"{out / 'plan-table.md'}: {rep['games']} branches, {rep['valid']} valid, {rep['points']} points")
