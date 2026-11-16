import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS_EVAL = str(ROOT / "tools" / "eval")
if TOOLS_EVAL not in sys.path:
    sys.path.insert(0, TOOLS_EVAL)

import eval_common as ec  # noqa: E402
import protocol  # noqa: E402
import tables  # noqa: E402

LIN = protocol.load_lineage()

CLEAN_GIT = {"commit": "0" * 40, "protocol_files_dirty": False, "dirty_paths": [], "dirty_diff_sha256": None}


def dump(path: Path, cmds: list, frames: int = 20000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rep": "x", "frames": frames, "players": [{"id": 0, "race": 1}, {"id": 1, "race": 0}], "cmds": cmds}), encoding="utf-8")


def build(frame: int, x: int, y: int, ty: int) -> list:
    h = "1e" + "".join(f"{v & 0xFF:02x}{v >> 8:02x}" for v in (x, y, ty))
    return [frame, 0, 12, h]


def le16(*vals: int) -> str:
    return "".join(f"{v & 0xFF:02x}{v >> 8:02x}" for v in vals)


def rclick(frame: int, x: int, y: int, target: int) -> list:
    return [frame, 0, 20, le16(x, y, target, 228) + "00"]


def order(frame: int, x: int, y: int, target: int, order_id: int) -> list:
    return [frame, 0, 21, le16(x, y, target, 228) + f"{order_id:02x}" + "00"]


def make_job(tmp_path: Path, pid: str, rows: list[dict], ledger: list[dict] | None = None, plan_text: str | None = None) -> Path:
    job = tmp_path / pid
    (job / "plans").mkdir(parents=True)
    (job / "results").mkdir()
    (job / "games").mkdir()
    (job / "plans" / "01.toml").write_text(plan_text or protocol.render(LIN, pid, "checkpoints/m2-r1/ckpt-5", pid), encoding="utf-8")
    (job / "job.toml").write_text('stages = ["plans/01.toml"]\nhosts = "hosts.toml"\n', encoding="utf-8")
    (job / "results" / "h.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (job / "games" / "h.jsonl").write_text("".join(json.dumps(g) + "\n" for g in ledger or []), encoding="utf-8")
    return job


def row(n: int, won, **model) -> dict:
    return {"game_id": 100000 + n, "slot": 0, "won": won, "reason": "Elimination", "end_frame": 1440 * n,
            "model": {"checkpoint": "ckpt-5@0", "zero_last_failed": False, "sample": "stochastic", **model}}


def cmds(n: int, frame: int = 100, player: int = 0) -> list:
    return [[frame, player, 9, "0101"] for _ in range(n)]


def ops_value(mined: int, end_s: float = 600.0, sblock_first_s=None, cc_lift_s: float = 0.0, rush: dict | None = None) -> dict:
    keys = dict.fromkeys(tables.OPS_KEYS, 0.0)
    keys.update(end_s=end_s, mined=mined, sblock_first_s=sblock_first_s, cc_lift_s=cc_lift_s, inc_10m=None if end_s < 600 else 900.0)
    return {"ops": keys, "rush_combat": rush}


def rush_value(pull_s=None, scv_lost: int = 0, kills: int = 0, bunker_load_s=None, focus_hhi=None) -> dict:
    return {"in_s": 200.0, "enemy_max": 6, "pull_s": pull_s, "pull_max": 3, "scv_move_max": 0, "scv_lost": scv_lost, "kills": kills,
            "bunker_marines_max": 0, "bunker_load_s": bunker_load_s, "bunker_empty_s": 0.0, "focus_hhi": focus_hhi, "focus_obs": 1}


def write_ops(cache: Path, n: int, value: dict, slot: int = 0, version: int = ec.OPS_CACHE_VERSION) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    line = {"rep": f"g{n}.rep", "perspective": slot, "frames": 20000, **value, "cache_version": version}
    (cache / f"g{n}.ops.json").write_text(json.dumps(line) + "\n", encoding="utf-8")


CUE, DEAD = LIN["metric"]["cue_diag"], LIN["metric"]["dead_channels"]

TEACHER, HUMAN = CUE["teacher"], CUE["human"]

SHA = "f" * 64

TAILS = ["by_name/hold_orig", "by_name/hold_macro2"]

HUMAN_DS = sorted(HUMAN["datasets"])


def probe_metrics(stages) -> dict:
    m = {f"{s}|ridge": {"pt": 0.5, "lo": 0.4, "hi": 0.6} for s in stages} | {f"{s}+nuis|ridge": {"pt": 0.6, "gain": 0.1, "gain_lo": 0.05, "gain_hi": 0.15} for s in stages}
    return m | {"nuis|ridge": {"pt": 0.3, "lo": 0.2, "hi": 0.4}, "h|mlp": {"pt": 0.45}}


def set_outputs(d: Path, name: str, sha: str = SHA, **edit) -> Path:
    s = CUE[name]
    dm, p = s["dump"], s["probes"]
    data = [f"/h/diag_holdout/{t}" for t in TAILS] if name == "teacher" else [f"<shared>/datasets/{x}" for x in HUMAN_DS]
    man = {"checkpoint": "<shared>/checkpoints/m2-s1/ckpt-9", "checkpoint_sha256": sha, "data": data, "splits": ["val", "test"], "games_per_split": s.get("games"),
           "seed": 1, "config": "config/bc.toml", "lanes_per_group": 4, "row_until": 8640.0, "act_until": float(dm["act_until"]), "act_every": dm["act_every"],
           "read_until": float(dm["read_until"]) if dm.get("read_until") else None,
           "passes": ["real", "early", "late", "sc+1", "sc-1", "tok+1", "tok-1"] if dm["cf"] else ["real"],
           "cf_starts": {r: [float(x) for x in v] for r, v in dm["cf_starts"].items()} if dm["cf"] else None,
           "stages": ["token", "e_pool", "sp_pool", "c", "h", "x_in", "L1.a", "L1.g1", "L1.f", "L1.out"], "backend": "rocm", "precision": "f32",
           "games": dm.get("expect_games", 1000), "stats": {"rows": dm.get("expect_rows", 500000), "act_rows": dm.get("expect_act_rows", 90000)}} | edit.get("manifest", {})
    head = {"checkpoint": man["checkpoint"], "checkpoint_sha256": sha, "data": data, "splits": ["val", "test"], "backend": "rocm", "precision": "f32"}
    metric = probe_metrics(p["stages"])
    races = {}
    for race in p["races"]:
        tasks = {t: {"metric": "r2", "rows": 100, "games": 20, "metrics": metric} for t in s["report"]["tasks"]}
        tasks |= {t: {"metric": "r2", "rows": r, "games": g, "metrics": metric} for t, (r, g) in p["expect_tasks"].items()}
        tasks |= {t: {"rows": 36516, "skipped": "constant target 1"} for t in p["expect_skipped"]}
        races[race] = {"tasks": tasks}
    probes_o = {"dump": head, "gain_method": "resid", "boot": p["boot"], "cap": p["cap"], "mlp": True, "races": races} | edit.get("probes", {})
    d.mkdir(parents=True, exist_ok=True)
    (d / "dump").mkdir(exist_ok=True)
    (d / "dump" / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    (d / "dump" / "FAKE_GAME_ROWS").write_text(edit.get("game_rows", dm.get("expect_game_rows_sha256", "")), encoding="utf-8")
    (d / "probes.json").write_text(json.dumps(probes_o), encoding="utf-8")
    if "intervene" in s:
        eff = {"n": 100, "mean": 0.01, "lo": -0.01, "hi": 0.03, "share_ge_0.15": 0.0}
        allc = {e: {m: eff for m in ("bunker_any", "marine_sum", "rax2_any", "label_p")} for e in ("start", "start_unf", "count_sc", "count_tok", "real")}
        inter = {"dump": head, "window": [2160.0, 5760.0], "races": {"Z": {"games": 433, "edited": 300, "edited_unf": 120, "count_edited": 310, "by_class": {"all": allc}}}} | edit.get("intervene", {})
        (d / "intervene.json").write_text(json.dumps(inter), encoding="utf-8")
    return d


def train_log_output(out: Path, **edit) -> Path:
    r = {"log": "s1.log", "from_update": 600, "until_update": 39096, "sp_records_from": 25, "alarm_lines": [], "alarm_records": [], "crossing_records": [],
         "dead": {k: {"max": 3, "max_share": 0.05, "at": 700, "last": 1} for k in DEAD["layers"]} | {"r4": {"max": 51, "max_share": 0.398, "at": 3000, "last": 51}},
         "worst_dead_layer": "r4", "worst_dead_share": 0.398, "const": {"sp_const_s8": {"max": 0, "share": 0.0}, "sp_const_pool": {"max": 0, "share": 0.0}},
         "min_pos_any": 0.2} | edit
    out.mkdir(parents=True, exist_ok=True)
    (out / "spatial_log.json").write_text(json.dumps(r), encoding="utf-8")
    return out


def record(out: Path, metrics=("cue_diag", "dead_channels"), sha: str = SHA, **edit) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    rec = {"metrics": list(metrics), "checkpoint_sha256": sha, "git": dict(CLEAN_GIT), "finished": "2026-11-10T00:00:00+09:00"} | edit
    (out / "protocol.json").write_text(json.dumps(rec), encoding="utf-8")
    return out


def all_outputs(out: Path, sha: str = SHA) -> Path:
    for s in CUE["sets"]:
        set_outputs(out / s, s, sha)
    train_log_output(out)
    return record(out, sha=sha)


def fake_game_order(monkeypatch) -> None:
    real = protocol.game_rows_sha256
    monkeypatch.setattr(protocol, "game_rows_sha256", lambda d: (Path(d) / "FAKE_GAME_ROWS").read_text() if (Path(d) / "FAKE_GAME_ROWS").is_file() else real(d))


def val(top1: dict, nll: dict, bunker: dict | None = None) -> dict:
    d = {"split": "val", "games": 500, "selection_seed": 1,
         "by_enemy_race": {r: {"steps": 100, "action_top1": top1[r], "nll": {"action": nll[r]}} for r in top1}}
    if bunker is not None:
        d["bunker_before"] = {"before_frame": 5760, "by_enemy_race": {r: {"steps": 40, "mean_p": p} for r, p in bunker.items()}}
    return d
