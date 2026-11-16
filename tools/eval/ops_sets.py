"""052-R7 replay values of the comparison sets outside the eval jobs (`tables.py --ops` covers those):"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_common as ec  # noqa: E402


def teacher_games(run: Path, build: str) -> dict[int, dict]:
    out = {}
    for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line) if line.strip() else {}
        if r.get("t0_build") == build and r.get("self_race") == "Terran":
            if r["game_id"] in out:
                raise SystemExit(f"{run}: game {r['game_id']} has two {build} rows")
            out[r["game_id"]] = r
    return out


def sha_index(shared: Path) -> dict[str, Path]:
    raw, idx = shared / "datasets/replays/raw", {}
    for f in sorted((shared / "datasets/replays/build/hashes").glob("*.sha256")):
        for line in f.read_text(encoding="utf-8").splitlines():
            sha, _, rel = line.partition(" ")
            if rel:
                idx.setdefault(sha, raw / rel.lstrip("*"))
    return idx


def measure(items: list[tuple[str, Path, int]], cache: Path, jobs: int) -> dict[str, dict]:
    got, todo = {}, []
    for key, rep, slot in items:
        j = cache / f"{key}.ops.json"
        if j.exists():
            d = json.loads(j.read_text(encoding="utf-8"))
            if d.get("perspective") == slot and d.get("ops") is not None and d.get("cache_version") == ec.OPS_CACHE_VERSION:
                got[key] = d
                continue
        todo.append((key, rep, slot))
    binary = ec.replay_bin()
    print(f"opsdump: {len(todo)} of {len(items)} replay(s) to play ({jobs} at a time)", file=sys.stderr)
    cache.mkdir(parents=True, exist_ok=True)

    def one(t):
        key, rep, slot = t
        d = ec._opsdump(binary, rep, slot) if rep.exists() else {"error": f"no replay {rep}"}
        if d.get("ops") is not None:
            (cache / f"{key}.ops.json").write_text(json.dumps({**d, "cache_version": ec.OPS_CACHE_VERSION}) + "\n", encoding="utf-8")
        return key, d

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for k, (key, d) in enumerate(pool.map(one, todo), 1):
            got[key] = d
            if k % 50 == 0:
                print(f"opsdump: {k} / {len(todo)}", file=sys.stderr)
    return got


def write_set(out: Path, name: str, rows: list[dict], got: dict[str, dict]) -> None:
    n_bad = 0
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            d = got.get(r["game"], {})
            n_bad += d.get("ops") is None
            f.write(json.dumps({"set": name, **r, **{k: d.get(k) for k in ("rep", "perspective", "ops", "rush_combat")},
                                **({"error": d.get("error") or "no ops"} if d.get("ops") is None else {})}) + "\n")
    print(f"{name}: {len(rows)} games, {n_bad} failed -> {out / f'{name}.jsonl'}", file=sys.stderr)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", choices=("botgen", "human"))
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=1, help="parallel opsdump processes (OpenBW playbacks: not on a training host)")
    ap.add_argument("--teacher-build", default="terran_react")
    a = ap.parse_args(argv)
    ec.load_env()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.kind == "botgen":
        for run in a.inputs:
            rows, items = [], []
            for n, r in sorted(teacher_games(run, a.teacher_build).items()):
                key = f"{run.name}-g{n}"
                held = {k: (r.get("scripted") or {}).get(k) for k in ("scv_min_1m", "scv_8m")}
                rows.append({"game": key, "matchup": f"Tv{(r.get('enemy_race') or 'Unknown')[:1]}", "held": held})
                items.append((key, run / "replays" / f"g{n}.rep", r["slot"]))
            write_set(a.out, run.name, rows, measure(items, a.out / "cache" / run.name, a.jobs))
    else:
        idx = sha_index(ec.shared_dir())
        for lst in a.inputs:
            rows, items = [], []
            for line in lst.read_text(encoding="utf-8").splitlines():
                g = json.loads(line)
                key = f"{g['replay_sha256'][:16]}-p{g['perspective_slot']}"
                rows.append({"game": key, "matchup": g["matchup"], "src": g["src"], "map_family": g["map_family"], "held": None})
                items.append((key, idx.get(g["replay_sha256"], Path(f"<sha {g['replay_sha256'][:16]} not in the hash lists>")), g["perspective_slot"]))
            write_set(a.out, lst.stem, rows, measure(items, a.out / "cache" / "human", a.jobs))


if __name__ == "__main__":
    main()
