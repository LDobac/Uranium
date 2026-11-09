"""Per-lineage-point diagnostic summary (the measurement protocol records one per checkpoint):
`python -m uranium diag summary` merges the outputs of `probes`, `intervene`, `spatial` and
`core-split` into one small JSON of the headline numbers.

Schema `uranium.diag.summary/1` (every section optional except the header; a missing input is
`null`):

    {
      "schema": "uranium.diag.summary/1",
      "point": str,                       # lineage point id, e.g. "S2e2", "r1", "rl-s1"
      "checkpoint": str | null,           # path (shared-drive paths as "<shared>/...")
      "checkpoint_sha256": str | null,    # 020-R5 policy version of the weights
      "git_commit": str | null,           # commit of the measuring code
      "created": str,                     # UTC ISO time
      "data": {"sources": [str], "splits": [str]} | null,
      "device": {"backend": str, "precision": str} | null,
      "gain_method": "resid" | "joint" | null,
      "cue_probes": {                     # per enemy race (T / P / Z), per task, per stage
        race: {task: {"metric": "r2" | "auc", "rows": int, "games": int,
                      "nuisance": {"pt", "lo", "hi"},
                      "stages": {stage: {"pt", "lo", "hi", "gain", "gain_lo", "gain_hi", "mlp"}},
                      "shuffled": {"pt", "lo", "hi"} | null}}},
      "intervention": {                   # per race, all classes
        race: {"games": int, "edited": int,
               effect: {metric: {"n", "mean", "lo", "hi", "share_ge_0.15"}}}},
                                          # effect in start / start_unf / count_sc / count_tok / real,
                                          # metric in bunker_any / marine_sum / rax2_any / label_p
      "dead_channels": {point: {"pos_frac": float, "never_active": int, "channels": int}},
      "core_split": {race: {"hp_r2": {stage: float}, "gates": {gate: {"z_mean", "var_share_keep",
                                                                       "r2_x", "r2_keep", "r2_write", "r2_g"}}}}
    }

`stage` "gain" is the stage's metric with the nuisance minus the nuisance alone (`gain_method`,
uranium.diag.probe); "mlp" the MLP(64) probe's point value (null when not run)."""

import datetime
import json
import subprocess
from pathlib import Path

SCHEMA = "uranium.diag.summary/1"
HEADER = ("schema", "point", "checkpoint", "checkpoint_sha256", "git_commit", "created")
SECTIONS = ("data", "device", "gain_method", "cue_probes", "intervention", "dead_channels", "core_split")
PT = ("pt", "lo", "hi")


def add_args(p) -> None:
    p.add_argument("--point", required=True, help="lineage point id")
    p.add_argument("--probes", help="`diag probes` JSON")
    p.add_argument("--intervene", help="`diag intervene` JSON")
    p.add_argument("--spatial", help="`diag spatial` JSON")
    p.add_argument("--core-split", help="`diag core-split` JSON")
    p.add_argument("--out", required=True)


def _pick(d: dict | None, keys) -> dict | None:
    return None if d is None else {k: d.get(k) for k in keys}


def cue_section(probes: dict) -> dict:
    out = {}
    for race, rr in probes.get("races", {}).items():
        tasks = {}
        for tn, t in rr.get("tasks", {}).items():
            m = t.get("metrics")
            if not m:
                continue
            stages = {}
            for k, v in m.items():
                name, kind = k.split("|", 1)
                if kind != "ridge" or name.endswith("+nuis") or name in ("nuis",) or name.startswith("shuffled"):
                    continue
                g = m.get(f"{name}+nuis|ridge", {})
                stages[name] = {**_pick(v, PT), "gain": g.get("gain"), "gain_lo": g.get("gain_lo"), "gain_hi": g.get("gain_hi"),
                                "mlp": m.get(f"{name}|mlp", {}).get("pt")}
            sh = next((v for k, v in m.items() if k.startswith("shuffled|")), None)
            tasks[tn] = {"metric": t["metric"], "rows": t["rows"], "games": t["games"], "nuisance": _pick(m.get("nuis|ridge"), PT), "stages": stages,
                         "shuffled": _pick(sh, PT)}
        out[race] = tasks
    return out


def intervention_section(iv: dict) -> dict:
    out = {}
    for race, rr in iv.get("races", {}).items():
        allc = rr["by_class"]["all"]
        out[race] = {"games": rr["games"], "edited": rr["edited"],
                     **{eff: {m: _pick(v, ("n", "mean", "lo", "hi", "share_ge_0.15")) for m, v in d.items()} for eff, d in allc.items()}}
    return out


def dead_section(sp: dict) -> dict:
    cks = list(sp.get("checkpoints", {}).values())
    if not cks:
        return {}
    pts = cks[0]["points"]
    return {k: {"pos_frac": v["pos_frac"], "never_active": v["never_active"], "channels": v["channels"]} for k, v in pts.items()}


def core_section(cs: dict) -> dict:
    out = {}
    for race, rr in cs.get("races", {}).items():
        hp = rr.get("hp_probes", {}).get("r2")
        out[race] = {"hp_r2": None if hp is None else {k: v["pt"] for k, v in hp.items()},
                     "gates": {k: _pick(v, ("z_mean", "var_share_keep", "r2_x", "r2_keep", "r2_write", "r2_g")) for k, v in rr.get("gates", {}).items()}}
    return out


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build(point: str, probes=None, intervene=None, spatial=None, core_split=None, commit: str | None = None) -> dict:
    """The summary dict from the loaded command outputs (each may be None)."""
    src = next((d["dump"] for d in (probes, intervene, core_split) if d and d.get("dump")), None)
    ck = src.get("checkpoint") if src else None
    sha = src.get("checkpoint_sha256") if src else None
    if ck is None and spatial and spatial.get("checkpoints"):
        ck, v = next(iter(spatial["checkpoints"].items()))
        sha = v.get("sha256")
    dev = None
    if src and src.get("backend"):
        dev = {"backend": src["backend"], "precision": src.get("precision")}
    elif spatial:
        dev = {"backend": spatial.get("backend"), "precision": spatial.get("precision")}
    return {
        "schema": SCHEMA, "point": point, "checkpoint": ck, "checkpoint_sha256": sha, "git_commit": commit,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "data": None if not src else {"sources": src.get("data"), "splits": src.get("splits")},
        "device": dev,
        "gain_method": probes.get("gain_method") if probes else None,
        "cue_probes": cue_section(probes) if probes else None,
        "intervention": intervention_section(intervene) if intervene else None,
        "dead_channels": dead_section(spatial) if spatial else None,
        "core_split": core_section(core_split) if core_split else None,
    }


def validate(s: dict) -> list[str]:
    """Problems of a summary against the schema (empty = valid)."""
    errs = []
    if s.get("schema") != SCHEMA:
        errs.append(f"schema {s.get('schema')!r} != {SCHEMA!r}")
    for k in HEADER + SECTIONS:
        if k not in s:
            errs.append(f"missing key {k}")
    if not isinstance(s.get("point"), str) or not s.get("point"):
        errs.append("point must be a non-empty string")
    if s.get("gain_method") not in (None, "resid", "joint"):
        errs.append(f"gain_method {s.get('gain_method')!r}")
    for race, tasks in (s.get("cue_probes") or {}).items():
        for tn, t in tasks.items():
            if t.get("metric") not in ("r2", "auc"):
                errs.append(f"cue_probes.{race}.{tn}.metric")
            for st, v in t.get("stages", {}).items():
                if not all(k in v for k in PT + ("gain",)):
                    errs.append(f"cue_probes.{race}.{tn}.stages.{st}")
    for p, v in (s.get("dead_channels") or {}).items():
        if not (0.0 <= float(v["pos_frac"]) <= 1.0 and 0 <= int(v["never_active"]) <= int(v["channels"])):
            errs.append(f"dead_channels.{p}")
    return errs


def run(a) -> int:
    load = lambda p: None if not p else json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
    s = build(a.point, load(a.probes), load(a.intervene), load(a.spatial), load(a.core_split), git_commit())
    errs = validate(s)
    if errs:
        raise SystemExit("summary does not match the schema: " + "; ".join(errs))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(s, indent=1, default=float), encoding="utf-8")
    print(f"[out] {a.out}")
    return 0
