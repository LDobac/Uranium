"""Outcome labels of plan-dump games (053-R2): held_8m, held_s, alive_10m and win from the session results, joined by replay sha256"""

import json
import re
import tomllib
from pathlib import Path

from uranium.diag.planning.dump import F8M
from uranium.measure import eval_common as ec
from uranium.provenance import redact

F10M = 10 * 60 * 24
HEADS = ("held_8m", "held_s", "alive_10m", "win")
DUMP_HEADS = ("win", "scv8")
ALL_HEADS = HEADS + ("scv8",)
SIDES = ("model", "scripted")
BOT_KINDS = ("teacher", "model")
DEFAULT_SIDE = {"teacher": "scripted", "model": "model"}
BLANK = b"\x00 \t\r\n\f\v"
NONE = {h: None for h in HEADS}


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def held_8m(scv_8m, won, end_frame):
    scv, end = _num(scv_8m), _num(end_frame)
    if scv is not None and scv >= 8:
        return 1
    if end is not None and end < F8M:
        return None if won is None else int(bool(won))
    return None if scv is None else 0


def alive_10m(won, end_frame):
    end = _num(end_frame)
    if won is False and end is not None and end < F10M:
        return 0
    if won is True or (end is not None and end >= F10M):
        return 1
    return None


def bot_heads(rec: dict, side: str) -> dict:
    if rec.get("error") is not None:
        return dict(NONE)
    m = rec.get(side) or {}
    won = rec.get("won")
    won = won if isinstance(won, bool) else None
    smin = _num(m.get("scv_min_1m"))
    return {"held_8m": held_8m(m.get("scv_8m"), won, rec.get("end_frame")), "held_s": None if smin is None else int(smin > 4),
            "alive_10m": alive_10m(won, rec.get("end_frame")), "win": None if won is None else int(won)}


def human_heads(e: dict) -> dict:
    res = int(e.get("result") or 0)
    win = None if res == 0 else int(res > 0)
    last = e.get("last_frame")
    if e.get("scv8") is None:
        held = None
    else:
        held = win if last is not None and last < F8M else e["scv8"]
    return {"held_8m": held, "held_s": None, "alive_10m": None, "win": win}


def read_rows(files) -> tuple[list, int]:
    rows, bad = [], 0
    for f in files:
        for line in Path(f).read_bytes().splitlines():
            for piece in re.split(rb"\x00+", line):
                s = piece.strip(BLANK)
                if not s:
                    continue
                try:
                    r = json.loads(s.decode("utf-8"))
                except ValueError:
                    bad += 1
                    continue
                if isinstance(r, dict):
                    rows.append(r)
                else:
                    bad += 1
    return rows, bad


def _job_of(path: Path) -> Path | None:
    job = path.parent.parent
    return job if path.parent.name == "results" and (job / "job.toml").is_file() else None


def sessions(paths) -> list[tuple[Path | None, list[Path]]]:
    out, seen = [], set()
    for p in paths:
        p = Path(p)
        job = _job_of(p)
        key = (job or p).resolve()
        if key in seen:
            continue
        seen.add(key)
        out.append((job, sorted((job / "results").glob("*.jsonl")) if job else [p]))
    return out


def p0_builds(plan_dir: Path) -> dict:
    plan, games = plan_dir / "plan.toml", plan_dir / "games.jsonl"
    if not plan.is_file() or not games.is_file():
        return {}
    ms = tomllib.loads(plan.read_text(encoding="utf-8")).get("matchup") or []
    out = {}
    for g in read_rows([games])[0]:
        i = g.get("matchup")
        if isinstance(i, int) and 0 <= i < len(ms):
            p0 = ms[i].get("p0") or {}
            if p0.get("kind") == "t0" and p0.get("build"):
                out[int(g["game_id"])] = p0["build"]
    return out


def session_games(job: Path | None, files: list, side: str) -> tuple[dict, dict]:
    stages = {k: (p.get("name") or Path(f).stem) for k, f, p in ec.job_stages(job)} if job else {}
    rows, bad = read_rows(files)
    builds = p0_builds(Path(files[0]).parent) if side == "scripted" and not job and files else {}
    st = {"files": len(files), "rows": len(rows), "bad_lines": bad, "bot_rows": 0, "other_rows": 0, "error_rows": 0, "retries_dropped": 0,
          "ambiguous_slot": 0, "games": 0, "p0_build_known": bool(builds)}
    best: dict = {}
    for i, r in enumerate(rows):
        gid = int(r["game_id"])
        if r.get(side) is None or (builds and r.get("t0_build") != builds.get(gid)):
            st["other_rows"] += 1
            continue
        st["bot_rows"] += 1
        st["error_rows"] += r.get("error") is not None
        if job:
            key, attempt = (ec.stage_of(gid), ec.game_key(gid) % ec.STAGE_BASE), ec.attempt_of(gid)
        else:
            key, attempt = (None, gid), 0
        best.setdefault(key, []).append((attempt, i, r))
    out = {}
    for key, cands in best.items():
        top = max(a for a, _, _ in cands)
        st["retries_dropped"] += sum(a < top for a, _, _ in cands)
        last = [r for a, _, r in cands if a == top]
        if len({r.get("slot") for r in last}) > 1:
            st["ambiguous_slot"] += 1
            continue
        stage, n = key
        if job:
            if stage not in stages:
                raise SystemExit(f"plan-label: {job}: game {n} of stage {stage}, the job has stages {sorted(stages)}")
            replay = job / stages[stage] / "replays" / f"g{n}.rep"
        else:
            replay = Path(files[0]).parent / "replays" / f"g{n}.rep"
        out[key] = {"replay": replay, "row": last[-1]}
    st["games"] = len(out)
    return out, st


def outcomes_by_sha(paths, side: str) -> tuple[dict, dict]:
    if side not in SIDES:
        raise SystemExit(f"plan-label: side {side!r} is not one of {SIDES}")
    by_sha, stats, dup, no_replay = {}, {}, 0, 0
    for job, files in sessions(paths):
        games, st = session_games(job, files, side)
        st["no_replay"] = 0
        for key, g in sorted(games.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1])):
            if not g["replay"].is_file():
                st["no_replay"] += 1
                continue
            sha = ec.file_sha256(g["replay"])
            if sha in by_sha:
                dup += 1
                by_sha[sha] = None
                continue
            by_sha[sha] = {"heads": bot_heads(g["row"], side), "error": g["row"].get("error") is not None, "game": key[1], "stage": key[0]}
        no_replay += st["no_replay"]
        stats[redact(job / "results" if job else files[0])] = st
    ok = {k: v for k, v in by_sha.items() if v is not None}
    return ok, {"files": stats, "replays": len(ok), "same_replay_twice": dup, "no_replay": no_replay}


def join(games: dict, kind: str, by_sha: dict | None, views: dict | None = None) -> tuple[dict, dict]:
    labels = {}
    how = {"results": 0, "dump": 0, "unmatched": 0, "skipped_view": 0, "other_view": 0, "ambiguous_view": 0, "error": 0}
    live = {gid: e for gid, e in games.items() if not e.get("skipped")}
    how["skipped_view"] = len(games) - len(live)
    per_sha: dict = {}
    for gid, e in live.items():
        per_sha.setdefault(e.get("sha") or "", []).append(gid)
    own = None if not views else {(x[0], int(x[1])) for x in views.get("model", [])}
    for gid in sorted(live, key=int):
        e = live[gid]
        base = {"sha": e.get("sha"), "start": e.get("start")}
        if kind == "human":
            labels[int(gid)] = {**human_heads(e), **base, "from": "dump"}
            how["dump"] += 1
            continue
        if own is not None and (e.get("sha"), int(e.get("start", -1))) not in own:
            labels[int(gid)] = {**NONE, **base, "from": "other_view"}
            how["other_view"] += 1
            continue
        if own is None and len(per_sha.get(e.get("sha") or "", [])) > 1:
            labels[int(gid)] = {**NONE, **base, "from": "ambiguous_view"}
            how["ambiguous_view"] += 1
            continue
        hit = (by_sha or {}).get(e.get("sha") or "")
        if hit is None:
            labels[int(gid)] = {**NONE, **base, "from": None}
            how["unmatched"] += 1
            continue
        labels[int(gid)] = {**hit["heads"], **base, "from": "error" if hit["error"] else "results"}
        how["error" if hit["error"] else "results"] += 1
    return labels, how


def coverage(labels: dict) -> dict:
    n = len(labels)
    out = {"games": n, "error": sum(v.get("from") == "error" for v in labels.values())}
    for h in HEADS:
        k = sum(v.get(h) is not None for v in labels.values())
        pos = sum(v.get(h) == 1 for v in labels.values())
        out[h] = {"labeled": k, "rate": k / n if n else None, "positive": pos}
    return out


def build(dump, kind: str, results=(), side: str | None = None, views: dict | None = None) -> dict:
    dump = Path(dump)
    games = json.loads((dump / "games.json").read_text(encoding="utf-8"))
    if kind not in ("human",) + BOT_KINDS:
        raise SystemExit(f"plan-label: kind {kind!r}")
    side = side or DEFAULT_SIDE.get(kind)
    by_sha, rstats = (None, None)
    if kind in BOT_KINDS:
        if not results:
            raise SystemExit(f"plan-label: a {kind} source needs its session results (053-R2 ①)")
        by_sha, rstats = outcomes_by_sha(results, side)
    labels, how = join(games, kind, by_sha, views)
    return {"kind": "plan-label", "dump": redact(dump), "source_kind": kind, "side": side if kind in BOT_KINDS else None,
            "results": [redact(p) for p in results], "heads": list(HEADS), "results_stats": rstats, "join": how,
            "coverage": coverage(labels), "games": {str(k): v for k, v in labels.items()}}


def read(path, dump) -> dict:
    js = json.loads(Path(path).read_text(encoding="utf-8"))
    if js.get("kind") != "plan-label":
        raise SystemExit(f"{path}: not a plan-label file")
    games = json.loads((Path(dump) / "games.json").read_text(encoding="utf-8"))
    live = {int(k) for k, e in games.items() if not e.get("skipped")}
    lab = {int(k): v for k, v in js["games"].items()}
    bad = [k for k, v in lab.items() if str(k) not in games or games[str(k)].get("sha") != v.get("sha") or games[str(k)].get("start") != v.get("start")]
    if bad or set(lab) != live:
        raise SystemExit(f"{path}: labels of another dump than {dump} (games {sorted(bad)[:5]} differ in sha / start, "
                         f"{len(live ^ set(lab))} games only on one side); run diag plan-label again")
    return lab


def add_args(p) -> None:
    p.add_argument("--dump", required=True, help="plan-dump folder (its games.json)")
    p.add_argument("--kind", required=True, choices=("human",) + BOT_KINDS, help="source kind (human: labels from the dump)")
    p.add_argument("--results", default="", help="session results.jsonl files, comma-separated (bot kinds; a job's <job>/results/*.jsonl are read together)")
    p.add_argument("--side", choices=SIDES, help="results field of the bot's session (default teacher: scripted, model: model)")
    p.add_argument("--views", help="views JSON ({\"bot\": [...], \"model\": [[sha256, start], ...]}): only the model views get labels")
    p.add_argument("--out", required=True, help="labels file to write (e.g. <dump>/labels.json)")


def run(a) -> int:
    res = [x for x in a.results.split(",") if x]
    views = json.loads(Path(a.views).read_text(encoding="utf-8")) if a.views else None
    js = build(a.dump, a.kind, res, a.side, views)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(js, indent=1), encoding="utf-8")
    print(json.dumps({"join": js["join"], "coverage": js["coverage"], "results": js["results_stats"] and {k: v for k, v in js["results_stats"].items() if k != "files"}}), flush=True)
    print(f"[out] {out}", flush=True)
    return 0
