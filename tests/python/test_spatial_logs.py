import json
from pathlib import Path

from eval_helpers import DEAD

import spatial_logs


LAYERS = ("stem", "c0", "c1", "c2", "c3", "r0", "r1", "r2", "r3", "r4", "r5")


def bc_log(path: Path, records: list[tuple[int, dict]], warn: list[str] | None = None) -> Path:
    base = {f"sp_dead_{n}": 0 for n in LAYERS} | {f"sp_pos_{n}": 0.3 for n in LAYERS} | {"sp_const_s8": 0, "sp_const_pool": 0}
    lines = [json.dumps({"step": step, "loss": {"action": 1.0, "total": 2.0, **base, **sp}}) for step, sp in records]
    path.write_text("[bc] start\n" + "\n".join(lines + (warn or [])) + "\n", encoding="utf-8")
    return path


def test_spatial_logs_summary_041_r25(tmp_path):
    ch = spatial_logs.channels()
    assert (ch["stem"], ch["c2"], ch["r4"], ch["r5"], ch["sp_const_s8"], ch["sp_const_pool"]) == (32, 128, 128, 128, 128, 256)
    assert spatial_logs.default_from_update() == DEAD["from_update"]
    log = bc_log(tmp_path / "s1.log", [(500, {"sp_dead_r4": 100}), (600, {"sp_dead_r4": 51, "sp_pos_c3": 0.1}), (700, {"sp_dead_c3": 64, "sp_const_s8": 3}),
                                       (800, {"sp_dead_r4": 10}), (900, {"sp_dead_stem": 32})],
                 warn=["[bc] WARNING: update 700: spatial encoder collapse alarm (041-R25): sp_dead_c3 = 64 of 128",
                       "[bc] WARNING: update 900: spatial encoder collapse alarm (041-R25): sp_dead_stem = 32 of 32"])
    r = spatial_logs.scan(log, ch, until_update=800)
    assert (r["sp_records"], r["sp_records_from"], r["last_step"], r["until_update"]) == (4, 3, 800, 800)
    assert r["dead"]["r4"]["max"] == 51 and r["dead"]["r4"]["at"] == 600 and r["dead"]["r4"]["last"] == 10
    assert r["worst_dead_layer"] == "c3" and r["worst_dead_share"] == 0.5 and r["crossing_records"] == [700]
    assert r["const"]["sp_const_s8"]["max"] == 3 and r["min_pos_any"] == 0.1 and len(r["alarm_lines"]) == 1
    assert r["train_tail100_mean"]["action"] == 1.0
    assert spatial_logs.scan(log, ch)["crossing_records"] == [700, 900]
    cols = [f"sp_{k}_{n}" for k in ("pos", "dead") for n in LAYERS] + ["sp_const_s8", "sp_const_pool"]
    rows = ["time_s,agent,update,loss," + ",".join(cols)]
    for agent, up, dead in (("main", 600, "2"), ("exploiter", 600, "99"), ("main", 700, None), ("main", 800, "5")):
        vals = ["" if dead is None else "0.3" if c.startswith("sp_pos") else dead for c in cols]
        rows.append(f"1,{agent},{up},0.5," + ",".join(vals))
    csv_log = tmp_path / "train.csv"
    csv_log.write_text("\n".join(rows) + "\n", encoding="utf-8")
    rc = spatial_logs.scan(csv_log, ch)
    assert rc["sp_records_from"] == 2 and rc["dead"]["stem"]["max"] == 5 and rc["dead_last"]["step"] == 800 and rc["alarm_lines"] == []
