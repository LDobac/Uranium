"""041-R25 dead-channel record summary of a training run log (050-R9 `dead_channels`, the training-log part;
config/protocol/lineage.toml [metric.dead_channels.train_log], docs/measurement_protocol.md).

A BC run logs its `[monitor] spatial_every` records as JSON lines on stderr (`{"step", "loss": {sp_*...}}`,
041-R25 ②) and a WARNING line per alarm (041-R25 ③); an RL run has the same values as train.csv columns
(042-R21 ③, rows of one agent). From the records at updates >= `from_update` (the alarm's start, 600): the
max dead share per ReLU output layer (stem, c0-c3, r0-r5, the alarm layers), the max const S8 / s_pool, the
min positive share per layer, the alarm lines / records, the last record's dead counts; for a BC log also
the training-loss mean of the last 100 updates. Channel counts come from the checkpoint's model config
(`--checkpoint`, meta.json `config.spatial`) or the SpatialConfig defaults.

    python tools/eval/spatial_logs.py <log>... [--checkpoint <ckpt dir>] [--agent main] [--out <json>]

Moved from the 041-R24 lambda-run script (<shared>/eval/m2-lambda-20261110/scripts/spatial_logs.py) with the
same rules; the per-run JSON is the value of one log's entry.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ALARM_LINE = "spatial encoder collapse alarm"
CONST = ("sp_const_s8", "sp_const_pool")
TAIL_KEYS = ("action", "total", "aux_priv", "aux_jepa", "jepa_tgt_std")


def alarm_rules() -> tuple[int, float, tuple[str, ...]]:
    """(from update, share, ReLU output layers) of the 041-R25 alarm (uranium.model.monitor)."""
    from uranium.model.monitor import ALARM_FROM, ALARM_LAYERS, ALARM_SHARE

    return ALARM_FROM, ALARM_SHARE, ALARM_LAYERS


def channels(spatial: dict | None = None) -> dict[str, int]:
    """Channels per 041-R25 alarm layer and const point of a SpatialConfig (dict form; None = defaults):
    stem, c<i> / r<i> = stage i, r4 / r5 (bottleneck) and S8 = the last stage, s_pool = pool."""
    if spatial is None:
        from uranium.model.config import SpatialConfig

        sc = SpatialConfig()
        spatial = {"stem_channels": sc.stem_channels, "stages": list(sc.stages), "pool": sc.pool}
    st = list(spatial["stages"])
    ch = {"stem": int(spatial["stem_channels"])}
    for i, c in enumerate(st):
        ch[f"c{i}"] = ch[f"r{i}"] = int(c)
    ch["r4"] = ch["r5"] = int(st[-1])
    ch["sp_const_s8"], ch["sp_const_pool"] = int(st[-1]), int(spatial["pool"])
    return ch


def checkpoint_channels(ckpt: Path) -> dict[str, int]:
    meta = json.loads((Path(ckpt) / "meta.json").read_text(encoding="utf-8"))
    return channels(meta["config"]["spatial"])


def read_records(path: Path, agent: str = "main") -> tuple[list[dict], list[str]]:
    """({"step", "loss": {...}} records, alarm lines) of a BC stderr log or an RL train.csv."""
    if path.suffix == ".csv":
        recs = []
        with path.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if "agent" in r and r["agent"] != agent:
                    continue
                vals = {k: float(v) for k, v in r.items() if k and k.startswith("sp_") and v not in ("", None)}
                rec = {"step": int(float(r["update"])), "loss": vals}
                if "loss" in r and r["loss"] not in ("", None):
                    rec["loss"]["total"] = float(r["loss"])
                recs.append(rec)
        return recs, []
    recs, warns = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ALARM_LINE in line:
            warns.append(line.strip())
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in r and isinstance(r.get("loss"), dict):
            recs.append(r)
    return recs, warns


def scan(path: Path, ch: dict[str, int], from_update: int | None = None, agent: str = "main") -> dict:
    alarm_from, share, layers = alarm_rules()
    from_update = alarm_from if from_update is None else from_update
    recs, warns = read_records(Path(path), agent)
    sp = [r for r in recs if "sp_dead_stem" in r["loss"]]
    late = [r for r in sp if r["step"] >= from_update]
    out = {"log": Path(path).name, "from_update": from_update, "alarm_share": share, "channels": ch, "last_step": recs[-1]["step"] if recs else None,
           "sp_records": len(sp), "sp_records_from": len(late), "alarm_lines": warns,
           "alarm_records": [r["step"] for r in recs if r.get("spatial_alarm") or r["loss"].get("spatial_alarm")]}
    if late:
        dead = {}
        for k in layers:
            v, s = max((r["loss"][f"sp_dead_{k}"], r["step"]) for r in late)
            dead[k] = {"max": v, "max_share": v / ch[k], "at": s, "last": late[-1]["loss"][f"sp_dead_{k}"]}
        out["dead"] = dead
        out["worst_dead_layer"] = max(dead, key=lambda k: dead[k]["max_share"])
        out["worst_dead_share"] = dead[out["worst_dead_layer"]]["max_share"]
        out["const"] = {k: {"max": max(r["loss"][k] for r in late), "share": max(r["loss"][k] for r in late) / ch[k]} for k in CONST}
        out["min_pos"] = {k: min(r["loss"][f"sp_pos_{k}"] for r in late) for k in layers}
        out["min_pos_any"] = min(out["min_pos"].values())
        out["dead_last"] = {k: late[-1]["loss"][f"sp_dead_{k}"] for k in layers} | {"step": late[-1]["step"]}
        # the alarm rule recomputed from the records (041-R25 ③): records crossing it
        out["crossing_records"] = [r["step"] for r in late if any(r["loss"][f"sp_dead_{k}"] >= share * ch[k] for k in layers)
                                   or any(r["loss"][k] >= share * ch[k] for k in CONST)]
    tail = recs[-100:]
    if tail:
        out["train_tail100_mean"] = {k: sum(r["loss"][k] for r in tail) / len(tail) for k in TAIL_KEYS if all(r["loss"].get(k) is not None for r in tail)}
        out["train_tail100_steps"] = [tail[0]["step"], tail[-1]["step"]]
    return out


def line(name: str, r: dict) -> str:
    if "dead" not in r:
        return f"{name}: no sp records from update {r['from_update']} (last step {r.get('last_step')})"
    c, w = r["const"], r["worst_dead_layer"]
    return (f"{name}: last {r['last_step']} records>={r['from_update']} {r['sp_records_from']} worst dead {w} {r['worst_dead_share']:.1%} "
            f"(max {r['dead'][w]['max']:g} @ {r['dead'][w]['at']}) const s8 {c['sp_const_s8']['max']:g} pool {c['sp_const_pool']['max']:g} "
            f"min pos {r['min_pos_any']:.3f} alarms {len(r['alarm_lines'])} lines / {len(r['alarm_records'])} records / {len(r['crossing_records'])} crossing")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("logs", nargs="+")
    p.add_argument("--checkpoint", help="a checkpoint folder of the run (its meta.json model config gives the channel counts)")
    p.add_argument("--from-update", type=int, default=None, help="default: the alarm's start (lineage.toml train_log from_update)")
    p.add_argument("--agent", default="main", help="RL train.csv rows of this agent")
    p.add_argument("--out")
    a = p.parse_args(argv)
    ch = checkpoint_channels(Path(a.checkpoint)) if a.checkpoint else channels()
    res = {Path(x).stem: scan(Path(x), ch, a.from_update, a.agent) for x in a.logs}
    if a.out:
        Path(a.out).write_text(json.dumps(res if len(res) > 1 else next(iter(res.values())), indent=1) + "\n", encoding="utf-8")
    for n, r in res.items():
        print(line(n, r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
