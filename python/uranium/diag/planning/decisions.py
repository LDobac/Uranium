"""Decision rows (053-R5): join model decision / act logs to plan-dump rows, live frame L -> replay row L - 2"""

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from uranium.diag.planning import store as ds
from uranium.provenance import redact

LIVE_LAG = 2
GAME_START_FRAME = 1
NOOP = 0
FRAME_KEY = 1 << 32
SIDECAR = "decisions"
POLICY_SHA = re.compile(r"@([0-9a-f]{6,64})(?::|$)")
GAME_ID = re.compile(r"^g(\d+)")


def add_args(p) -> None:
    p.add_argument("--dump", required=True, help="plan-dump folder of a no-relabel --memory-log build (053-R5)")
    p.add_argument("--job", action="append", default=[], required=True,
                   help="game folder with replays/, the model's logs in memory/ or acts/, optional results.jsonl; repeatable")
    p.add_argument("--records", action="append", default=[], required=True,
                   help="the build's records (JSON lines): memory.status per replay; a stream without a record is left out; repeatable")
    p.add_argument("--out", help="sidecar folder (default <dump>/decisions)")


def playback(live) -> np.ndarray:
    return np.maximum(np.asarray(live, np.int64) - LIVE_LAG, GAME_START_FRAME)


def policy_sha(policy) -> str | None:
    m = POLICY_SHA.search(policy or "")
    return m.group(1) if m else None


def parse_log(text: str) -> dict | None:
    lines = [ln for ln in text.splitlines() if ln.strip()]

    def bad(why: str) -> dict:
        return {"complete": False, "reason": why, "decisions": []}

    if not lines:
        return bad("empty log")
    try:
        rows = [json.loads(ln) for ln in lines]
    except json.JSONDecodeError as e:
        return bad(f"not JSON: {e}")
    act = "i" in rows[0] or isinstance(rows[0].get("end"), dict)
    out, end = [], None
    for j, r in enumerate(rows):
        if end is not None:
            return bad(f"line {j + 1} after the end line")
        if act:
            if isinstance(r.get("end"), dict):
                end = r["end"]
                continue
            if r.get("i") != j:
                return bad(f"line {j + 1}: observation index {r.get('i')}")
            d = r.get("decision")
            if d is not None:
                out.append((int(r["frame"]), int(d["action"]), int(d["delay_bucket"]), int(bool(d.get("rejected"))), int(bool(r.get("forced")))))
        elif "end_frame" in r and "player" in r:
            end = r
        else:
            if out and out[-1][0] > int(r["frame"]):
                return bad(f"line {j + 1}: frame before the previous decision's")
            out.append((int(r["frame"]), int(r["action"]), int(r["delay_bucket"]), -1, 0))
    if end is None:
        return bad("no end line (the game did not end in the session)")
    if act and str(end.get("policy", "")).startswith("t0"):
        return None
    if int(end.get("decisions", len(out))) != len(out):
        return bad(f"end line counts {end.get('decisions')} decisions, the log has {len(out)}")
    return {"complete": True, "decisions": out, "player": end.get("player"), "end_frame": end.get("end_frame"),
            "policy": end.get("policy") if act else None}


def read_logs(logs: Path, stem: str) -> dict | None:
    one = logs / f"{stem}.jsonl"
    paths = [one] if one.is_file() else sorted(logs.glob(f"{stem}-p*.jsonl"))
    found = None
    for p in paths:
        got = parse_log(p.read_text(encoding="utf-8"))
        if got is None:
            continue
        if found is not None:
            return {"complete": False, "reason": f"two model logs for {stem}", "decisions": []}
        found = got
    return found


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_results(path: Path) -> dict:
    out = {}
    if not path.is_file():
        return out
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("model") is not None and r.get("game_id") is not None:
            out[int(r["game_id"])] = r.get("policy") or ""
    return out


def scan_job(job: Path) -> dict:
    logs = next((job / n for n in ("memory", "acts") if (job / n).is_dir()), None)
    if not (job / "replays").is_dir() or logs is None:
        raise SystemExit(f"plan-decisions: {job} needs replays/ and memory/ or acts/")
    results = read_results(job / "results.jsonl")
    out = {}
    for rep in sorted((job / "replays").glob("*.rep")):
        m = GAME_ID.match(rep.stem)
        gid = int(m.group(1)) if m else None
        out.setdefault(file_sha256(rep), []).append({"job": job.name, "stem": rep.stem, "logs": logs,
                                                       "result_policy": results.get(gid) if gid is not None else None})
    return out


def read_records(paths) -> dict:
    out = {}
    for p in paths:
        for ln in Path(p).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            mem = r.get("memory")
            if r.get("outcome") == "written" and mem is not None:
                out.setdefault(r["replay_sha256"], []).append(mem.get("status"))
    return out


def first_rows(game: np.ndarray, frame: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.rint(np.asarray(frame, np.float64)).astype(np.int64)
    key = np.asarray(game, np.int64) * FRAME_KEY + f
    o = np.lexsort((np.asarray(dec), key))
    k = key[o]
    first = np.ones(len(k), bool)
    first[1:] = k[1:] != k[:-1]
    return k[first], o[first]


def match_game(gid: int, decisions: list, keys: np.ndarray, rows: np.ndarray, row_until: float) -> dict:
    d = np.asarray(decisions, np.int64).reshape(-1, 5)
    live = d[:, 0]
    pb = playback(live)
    inr = np.nonzero(pb < row_until)[0]
    pb_in = pb[inr]
    q = gid * FRAME_KEY + pb_in
    at = np.searchsorted(keys, q)
    hit = (at < len(keys)) & (keys[np.minimum(at, len(keys) - 1)] == q)
    first = np.ones(len(pb_in), bool)
    first[1:] = pb_in[1:] != pb_in[:-1]
    row = np.where(hit & first, rows[np.minimum(at, len(keys) - 1)], -1)
    return {"ord": inr.astype(np.int32), "row": row.astype(np.int64), "live": live[inr].astype(np.int32), "action": d[inr, 1].astype(np.int16),
            "delay": d[inr, 2].astype(np.int8), "rejected": d[inr, 3].astype(np.int8), "forced": d[inr, 4].astype(np.int8),
            "collisions": int((~first).sum()), "beyond": int(len(live) - len(inr))}


def build(dump: Path, jobs: list[Path], records: list, out: Path) -> dict:
    rows, games, man = ds.load(dump)
    row_until = float(man.get("row_until") or 14400)
    game = np.asarray(rows["game"]).astype(np.int64)
    keys, first = first_rows(game, np.asarray(rows["frame"]), np.asarray(rows["dec"]))
    by_sha: dict = {}
    for j in jobs:
        for sha, ents in scan_job(Path(j)).items():
            by_sha.setdefault(sha, []).extend(ents)
    rec = read_records(records)
    cols = {k: [] for k in ("row", "game", "ord", "live", "action", "delay", "rejected", "forced")}
    meta: dict = {}
    streams: dict = {}
    for gid, e in games.items():
        streams.setdefault(e.get("sha"), []).append(gid)
    for sha, gids in streams.items():
        ents = by_sha.get(sha, [])
        if not ents:
            for g in gids:
                meta[g] = {"status": "no_replay"}
            continue
        if len(ents) > 1:
            for g in gids:
                meta[g] = {"status": "duplicate_replay", "replays": [f"{x['job']}/{x['stem']}" for x in ents]}
            continue
        ent = ents[0]
        base = {"job": ent["job"], "replay": ent["stem"]}
        log = read_logs(ent["logs"], ent["stem"])
        bad = [s for s in rec.get(sha, []) if s != "ok"] if sha in rec and "ok" not in rec[sha] else []
        if sha not in rec:
            status = "no_record"
        elif log is None:
            status = "no_log"
        elif not log["complete"]:
            status = "incomplete"
        elif bad:
            status = f"record_{bad[0]}"
        else:
            status = "ok"
        if status != "ok":
            for g in gids:
                meta[g] = {**base, "status": status, **({"reason": log["reason"]} if log and log.get("reason") else {})}
            continue
        pol = log.get("policy") or ent["result_policy"]
        tries = {g: match_game(g, log["decisions"], keys, first, row_until) for g in gids}
        best = max(gids, key=lambda g: (int((tries[g]["row"] >= 0).sum()), -g))
        for g in gids:
            if g != best:
                meta[g] = {**base, "status": "other_view"}
        t = tries[best]
        n_in = len(t["row"])
        meta[best] = {**base, "status": "ok", "decisions": len(log["decisions"]), "in_range": n_in, "matched": int((t["row"] >= 0).sum()),
                      "unmatched": int((t["row"] < 0).sum()), "collisions": t["collisions"], "beyond": t["beyond"],
                      "noop": int((t["action"] == NOOP).sum()), "rejected": int((t["rejected"] == 1).sum()), "policy": pol, "policy_sha": policy_sha(pol)}
        for k in cols:
            cols[k].append(np.full(n_in, best, np.int32) if k == "game" else t[k])
    arrs = {k: (np.concatenate(v) if v else np.zeros(0, np.int64)) for k, v in cols.items()}
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "rows.npz", **arrs)
    (out / "games.json").write_text(json.dumps({str(k): meta[k] for k in sorted(meta)}), encoding="utf-8")
    st: dict = {}
    for m in meta.values():
        st[m["status"]] = st.get(m["status"], 0) + 1
    ok = [m for m in meta.values() if m["status"] == "ok"]
    summary = {"kind": "plan-decisions", "dump": redact(dump), "checkpoint_sha256": man.get("checkpoint_sha256"), "row_until": row_until,
               "rule": "live L -> row at replay frame max(L - 2, 1), first row of that frame; later decisions on the same row frame unmatched",
               "jobs": [redact(j) for j in jobs], "records": [redact(r) for r in records], "streams": len(meta), "status": st,
               "decisions": sum(m["decisions"] for m in ok), "in_range": sum(m["in_range"] for m in ok), "matched": sum(m["matched"] for m in ok),
               "unmatched": sum(m["unmatched"] for m in ok), "collisions": sum(m["collisions"] for m in ok), "noop": sum(m["noop"] for m in ok),
               "rejected": sum(m["rejected"] for m in ok),
               "played_by_dump_checkpoint": sum(bool(m["policy_sha"]) and str(man.get("checkpoint_sha256", "")).startswith(m["policy_sha"]) for m in ok)}
    (out / "manifest.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def parse_explore(text: str) -> dict | None:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {"complete": False, "reason": "empty log", "points": [], "decisions": 0}
    try:
        rows = [json.loads(ln) for ln in lines]
    except json.JSONDecodeError as e:
        return {"complete": False, "reason": f"not JSON: {e}", "points": [], "decisions": 0}
    end = rows[-1].get("end")
    if not isinstance(end, dict):
        return {"complete": False, "reason": "no end line", "points": [], "decisions": 0}
    if str(end.get("policy", "")).startswith("t0"):
        return None
    pts, n = [], 0
    for r in rows[:-1]:
        d = r.get("decision")
        if d is None:
            continue
        e = r.get("explore")
        if isinstance(e, dict) and "coin" in e:
            pts.append({"ord": n, "live": int(r["frame"]), "action": int(d["action"]), "rejected": bool(d.get("rejected")), "coin": e["coin"],
                        "propensity": float(e["propensity"]), "a0": int(e["a0"]), "c": None if e.get("c") is None else int(e["c"]),
                        "cands": [int(x[0]) for x in e.get("cands") or []], "point": int(e.get("point", 0)), "epsilon": float(e["epsilon"])})
        n += 1
    return {"complete": int(end.get("decisions", n)) == n, "reason": None, "points": pts, "decisions": n}


def read_explore(job: Path, games: dict) -> tuple[dict, dict]:
    acts = Path(job) / "acts"
    if not acts.is_dir():
        raise SystemExit(f"plan-s0: {job} has no acts/ (054-R4 act logs)")
    out, st = {}, {}
    for gid, m in games.items():
        if m.get("status") != "ok" or m.get("job") != Path(job).name:
            continue
        stem = m["replay"]
        one = acts / f"{stem}.jsonl"
        found, two = None, False
        for p in [one] if one.is_file() else sorted(acts.glob(f"{stem}-p*.jsonl")):
            got = parse_explore(p.read_text(encoding="utf-8"))
            if got is not None:
                two = found is not None
                found = got
        if two:
            why = "two_model_logs"
        elif found is None:
            why = "no_act_log"
        elif not found["complete"]:
            why = "incomplete"
        elif found["decisions"] != int(m.get("decisions", -1)):
            why = "decision_count"
        else:
            why = "ok"
            out[int(gid)] = found["points"]
        st[why] = st.get(why, 0) + 1
    return out, st


def load(dump) -> tuple[dict, dict]:
    d = Path(dump) / SIDECAR
    if not (d / "rows.npz").is_file():
        raise SystemExit(f"{dump}: no decision rows ({SIDECAR}/rows.npz); run diag plan-decisions first (053-R5)")
    with np.load(d / "rows.npz") as z:
        arrs = {k: z[k] for k in z.files}
    games = {int(k): v for k, v in json.loads((d / "games.json").read_text(encoding="utf-8")).items()}
    return arrs, games


def counted(action, rejected) -> np.ndarray:
    return (np.asarray(action) != NOOP) & (np.asarray(rejected) != 1)


def next_decision(game: np.ndarray, ord_: np.ndarray, count: np.ndarray, k: int) -> np.ndarray:
    o = np.lexsort((ord_, game))
    g, nn = np.asarray(game, np.int64)[o], np.asarray(count, bool)[o]
    c = np.cumsum(nn)
    starts = np.ones(len(g), bool)
    starts[1:] = g[1:] != g[:-1]
    base = np.maximum.accumulate(np.where(starts, c - nn, 0))
    cnt = c - base
    key = g * FRAME_KEY + cnt
    q = np.searchsorted(key, key + k, side="left")
    ok = q < len(key)
    qq = np.minimum(q, max(len(key) - 1, 0))
    ok &= (g[qq] == g) & (cnt[qq] == cnt + k) & nn[qq]
    out = np.full(len(g), -1, np.int64)
    out[o[ok]] = o[qq[ok]]
    return out


def run(a) -> int:
    t0 = time.time()
    out = Path(a.out) if a.out else Path(a.dump) / SIDECAR
    s = build(Path(a.dump), [Path(j) for j in a.job], a.records, out)
    s["secs"] = round(time.time() - t0, 1)
    print(json.dumps(s), flush=True)
    return 0
