"""One teacher-forced pass over shard games; each real step before --row-until is one plan-s0 row"""

import functools
import json
import time
from pathlib import Path

import numpy as np

from uranium import tables as tb
from uranium.diag.features import fill_race_cols, game_labels, lane_features, race_of

F8M = 8 * 60 * 24
SCV8_MIN = 8
LABEL_KEYS = ("frame", "foc_vis", "foc_comp", "foc_hp", "foc_seen", "a_has", "a_comp", "sc_focus")
OP_GROUPS = ("build", "addon", "train", "research", "upgrade", "cancel")
OP_EXTRA = ("Lift", "Land_Pos", "Land_CommandCenter_Pos")
SIEGE = ("Terran_Siege_Tank_Tank_Mode", "Terran_Siege_Tank_Siege_Mode")


def op_actions() -> list[int]:
    return sorted(a["id"] for a in tb.rows("actions") if a["group"] in OP_GROUPS or a["name"] in OP_EXTRA)


@functools.cache
def own_columns() -> tuple[list[str], dict[int, list[int]], dict[int, int]]:
    terran = set(tb.terran())
    names, types, of_action = [], {}, {}
    for a in sorted(tb.rows("actions"), key=lambda a: a["id"]):
        u = a.get("unit_type")
        if a["group"] not in ("build", "addon", "train") or u not in terran or a["name"] == "Train_NuclearMissile":
            continue
        name = tb.units()[u]["name"]
        ids = [tb.unit_id(n) for n in SIEGE] if name == SIEGE[0] else [u]
        of_action[a["id"]] = len(names)
        types[len(names)] = ids
        names.append("siege_tank" if name == SIEGE[0] else name.removeprefix("Terran_").lower())
    trained = [types[c] for aid, c in of_action.items() if _group(aid) == "train" and names[c] != "scv"]
    names += ["supply_used", "supply_total", "army"]
    types[len(names) - 1] = [i for ids in trained for i in ids]
    return names, types, of_action


@functools.cache
def _group(aid: int) -> str:
    return next(a["group"] for a in tb.rows("actions") if a["id"] == aid)


def own_state(scalar: np.ndarray) -> np.ndarray:
    names, types, _ = own_columns()
    sc = np.asarray(scalar, np.float32)
    out = np.zeros((len(sc), len(names)), np.float32)
    for c, ids in types.items():
        for u in ids:
            d, b = tb.sc_own_cols(u)
            out[:, c] += tb.decode_count(sc[:, d]) + tb.decode_count(sc[:, b])
    out[:, names.index("supply_used")] = np.rint(sc[:, 4] * 400.0)
    out[:, names.index("supply_total")] = np.rint(sc[:, 5] * 400.0)
    return out


def scv_done(scalar: np.ndarray) -> np.ndarray:
    d, _ = tb.sc_own_cols(tb.unit_id("Terran_SCV"))
    return tb.decode_count(np.asarray(scalar)[:, d])


def add_args(p) -> None:
    p.add_argument("--checkpoint", required=True, help="checkpoint folder or a run folder (its LATEST)")
    p.add_argument("--dataset", action="append", default=[], help="dataset under the shared drive, repeatable")
    p.add_argument("--shards", action="append", default=[], help="shard folder, repeatable")
    p.add_argument("--splits", default="val,test", help="comma-separated splits to read")
    p.add_argument("--games", type=int, help="only this many games per split (fixed-seed shuffle; n is a prefix of a larger n)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--config", default="config/bc.toml")
    p.add_argument("--backend", choices=["cuda", "rocm", "cpu"])
    p.add_argument("--out", required=True, help="output folder (must not hold a finished dump)")
    p.add_argument("--row-until", type=float, default=14400, help="rows before this frame (default 10:00)")
    p.add_argument("--read-until", type=float, default=-1,
                   help="read each game only up to this frame (default: --row-until; 0 = whole games); the rows are the same")
    p.add_argument("--gk", default="8,32", help="g_k outputs to keep (the k the model has; empty = none)")
    p.add_argument("--classes", help="JSON {replay_sha256: class or {\"class\": ...}}: a given build class per game (cls_given)")
    p.add_argument("--skip-views", help="views JSON {\"bot\": [[sha256, start], ...], ...}: streams in the \"bot\" list get no rows")
    p.add_argument("--lanes-per-group", type=int, default=8)
    p.add_argument("--limit-batches", type=int, default=0)
    p.add_argument("--maps-dir", help="map folder (else from .env)")
    p.add_argument("--synthetic-maps", action="store_true", help="tests: synthetic maps")


class RowWriter:
    def __init__(self, out: Path, schema: dict):
        self.dir = out / "rows"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.files = {k: open(self.dir / f"{k}.bin", "wb") for k in schema}
        self.n = 0
        self._write_schema(None)

    def _write_schema(self, rows):
        js = {"rows": rows, "fields": {k: {"dtype": d, "shape": list(s)} for k, (d, s) in self.schema.items()}}
        (self.dir / "schema.json").write_text(json.dumps(js, indent=1), encoding="utf-8")

    def add(self, cols: dict) -> None:
        n = None
        for k, (d, s) in self.schema.items():
            v = np.ascontiguousarray(cols[k], dtype=d)
            if v.shape[1:] != tuple(s) or (n is not None and len(v) != n):
                raise ValueError(f"plan-dump field {k}: shape {v.shape}, expected [{n}, *{tuple(s)}]")
            n = len(v)
            self.files[k].write(v.tobytes())
        self.n += n or 0

    def close(self) -> None:
        for f in self.files.values():
            f.close()
        self._write_schema(self.n)


def load_rows(dump, fields=None) -> dict:
    d = Path(dump) / "rows"
    js = json.loads((d / "schema.json").read_text(encoding="utf-8"))
    if js["rows"] is None:
        raise SystemExit(f"{dump}: unfinished plan-dump (no row count)")
    n = js["rows"]
    out = {}
    for k, f in js["fields"].items():
        if fields is not None and k not in fields:
            continue
        shape = (n, *f["shape"])
        out[k] = np.memmap(d / f"{k}.bin", dtype=f["dtype"], mode="r", shape=shape) if n else np.zeros(shape, f["dtype"])
    return out


def load(dump) -> tuple[dict, dict, dict]:
    dump = Path(dump)
    man = json.loads((dump / "manifest.json").read_text(encoding="utf-8"))
    games = {int(k): v for k, v in json.loads((dump / "games.json").read_text(encoding="utf-8")).items()}
    return load_rows(dump), games, man


def model_rows(net, obs: dict, s0, t: int, gk: list[int]) -> dict:
    import torch

    from uranium.model.auxiliary import jepa_target

    h, e, sp, sc, e_pool, state = net.trunk_seq(obs, s0, t, aux=True)
    lp = net.heads.action_logp(h, sc["c"], obs["type_mask"], net._bypass(h, sc, e, obs))
    y = jepa_target(torch.cat([e_pool, sp["pool"].to(e_pool.dtype)], 1))
    wo = net.value.win_open(h, sc["z"], sc["race"])
    v = net.variant
    g = {}
    if v is not None and v.aux_jepa is not None:
        g = {k: v.aux_jepa[str(k)](h) for k in gk if str(k) in v.aux_jepa}
    return {"h": h, "y": y, "logp": lp, "win_open": wo, "g": g, "state": state}


def read_skip_views(path) -> set:
    if not path:
        return set()
    got = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(x[0], int(x[1])) for x in got["bot"]}


def read_classes(path) -> dict:
    if not path:
        return {}
    got = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: (v["class"] if isinstance(v, dict) else v) for k, v in got.items()}


def finish_game(lg: dict, games: dict, read_until: float) -> None:
    e = games[lg["gid"]]
    acc = lg.pop("acc")
    e["dec_rows"] = lg["dec"]
    e["last_frame"] = lg["last"]
    readable = (not read_until or read_until > F8M + 1) and not lg["skip"]
    e["scv8"] = None if not readable else int(lg["last"] >= F8M and lg["scv_at8"] >= SCV8_MIN)
    e["cls"] = e["start1"] = e["first_sight"] = None
    if acc and lg["race"] in tb.RACES:
        r = {k: np.concatenate([a[k] for a in acc]) for k in acc[0]}
        if len(r["frame"]):
            lab = game_labels(r, lg["race"])
            e["cls"], e["start1"], e["first_sight"] = lab["cls"], lab["start1"], lab["first_sight"]


def run(a) -> int:
    import torch

    from uranium import _core, ckpt
    from uranium.device import autocast
    from uranium.diag.dump import lane_slice, redact
    from uranium.model import to_obs
    from uranium.passes import ForcedPass, fixed_device

    read_until = a.row_until if a.read_until < 0 else a.read_until
    if read_until and read_until < a.row_until:
        raise SystemExit(f"diag plan-dump: --read-until {read_until:g} cuts rows before --row-until {a.row_until:g}")
    out = Path(a.out)
    if (out / "manifest.json").exists():
        raise SystemExit(f"diag plan-dump: {out} already holds a finished dump")
    dirs = [str(p) for p in _core.dataset_dirs(a.dataset, a.shards)]
    if not dirs:
        raise SystemExit("diag plan-dump: no data (--dataset / --shards)")
    backend, device, precision = fixed_device(a.backend)
    fp = ForcedPass(a.config, device, precision, lanes_per_group=a.lanes_per_group)
    T, LANES, G = fp.t, fp.lanes, fp.g
    ckpt_dir = Path(_core.resolve_checkpoint(a.checkpoint))
    net = fp.load(ckpt_dir, frozen=True)
    want_gk = [int(x) for x in a.gk.split(",") if x.strip()]
    v = net.variant
    gk = [k for k in want_gk if v is not None and v.aux_jepa is not None and str(k) in v.aux_jepa]
    classes = read_classes(a.classes)
    skip_views = read_skip_views(a.skip_views)
    names, _, _ = own_columns()
    d_h = net.cfg.core.hidden
    d_y = net.cfg.entity.d_model + net.cfg.spatial.pool
    n_act = net.cfg.input.actions
    schema = {
        "game": ("int32", ()), "split": ("int8", ()), "dec": ("int32", ()), "frame": ("float32", ()),
        "h": ("float16", (d_h,)), "y": ("float16", (d_y,)), "priv": ("float16", (net.cfg.input.privileged,)), "priv_ok": ("uint8", ()),
        "own": ("float32", (len(names),)), "action": ("int16", ()), "delay": ("int8", ()), "weight": ("float32", ()),
        "logp": ("float16", (n_act,)), "mask": ("uint8", ((n_act + 7) // 8,)), "win_open": ("float32", ()),
        **{f"g{k}": ("float16", (d_y,)) for k in gk},
    }
    out.mkdir(parents=True, exist_ok=True)
    writer = RowWriter(out, schema)
    games: dict[int, dict] = {}
    seen_sha: dict[str, int] = {}
    t0 = time.time()
    stats = {"batches": 0, "groups": 0, "rows": 0, "frame_resets": 0, "skipped_views": 0, "skipped_groups": 0}
    splits = [s for s in a.splits.split(",") if s]
    for si, split in enumerate(splits):
        try:
            until = {"until_frame": int(np.ceil(read_until)) + 1} if read_until else {}
            stream = fp.stream(dirs, split, games=a.games, seed=a.seed, has_open=False, maps_dir=a.maps_dir, synthetic_maps=a.synthetic_maps, **until)
        except RuntimeError as e:
            if "no game" not in str(e).lower() and "empty" not in str(e).lower():
                raise
            print(f"[{split}] skipped: {e}", flush=True)
            continue
        gstate = fp.states
        lane_game: dict[int, dict] = {}
        nb = 0
        batch_meta = (None, None, None)
        for bi, b, gi, start, _, steps, hg in fp.groups(stream, a.limit_batches):
            if bi != batch_meta[0]:
                shas = b.lane_games() if hasattr(b, "lane_games") else [""] * LANES
                races = b.lane_enemy_races() if hasattr(b, "lane_enemy_races") else [""] * LANES
                batch_meta = (bi, shas, races)
                if bi + 1 > nb:
                    nb = bi + 1
                    stats["batches"] += 1
                    if nb % 50 == 0:
                        print(f"[{split}] batch {nb} games {len(games)} rows {stats['rows']} {time.time() - t0:.0f}s", flush=True)
            _, shas, races = batch_meta
            bsz = hg["b"]
            keep = np.asarray(hg["keep"]).astype(bool)
            scal = np.asarray(hg["scalar"])
            frame_all = scal[:, 0] * tb.FRAME_SCALE
            lanes = []
            for li in range(bsz):
                lane = start + li
                n = steps[lane]
                if n == 0:
                    continue
                fr = frame_all[li * T:li * T + n]
                lg = lane_game.get(lane)
                sha = shas[lane] if lane < len(shas) else ""
                if lg is None or not keep[li] or fr[0] < lg["last"] - 1 or (sha and sha != lg["sha"]):
                    if lg is not None:
                        finish_game(lg, games, read_until)
                    sub = lane_slice(hg, li, T)
                    race = race_of(sub, np.arange(n))
                    gid = len(games)
                    sc_cat = np.asarray(hg["scalar_cat"])[li * T]
                    skip = (sha, int(sc_cat[2])) in skip_views
                    lg = {"gid": gid, "race": race, "last": -1.0, "anchor": None, "sha": sha, "dec": 0, "scv_at8": 0, "acc": [], "skip": skip}
                    lane_game[lane] = lg
                    occ = seen_sha.get(sha, 0)
                    seen_sha[sha] = occ + 1
                    games[gid] = {"split": split, "lane": lane, "race": race, "enemy_race": races[lane] if lane < len(races) else "", "sha": sha, "occ": occ,
                                  "map": int(sc_cat[1]), "start": int(sc_cat[2]), "result": int(np.sign(np.asarray(hg["results"])[li * T])), "rows": 0,
                                  "cls_given": classes.get(sha), **({"skipped": True} if skip else {})}
                    stats["skipped_views"] += skip
                if (np.diff(fr) < -1).any():
                    stats["frame_resets"] += 1
                lg["last"] = float(fr[-1])
                if lg["skip"]:
                    lanes.append((li, np.zeros(0, np.int64), lg["gid"], lg["dec"]))
                    lg["dec"] += n
                    continue
                rows = li * T + np.arange(n)
                at8 = rows[fr <= F8M]
                if len(at8):
                    lg["scv_at8"] = int(scv_done(scal[at8[-1:]])[0])
                sel = np.nonzero(fr < a.row_until)[0]
                f, _, _ = lane_features(lane_slice(hg, li, T), lg["race"], lg, n)
                fill_race_cols(f, T)
                if len(sel):
                    lg["acc"].append({k: np.asarray(f[k])[sel] for k in LABEL_KEYS + tuple(k for k in f if k.startswith("pv"))})
                lanes.append((li, sel, lg["gid"], lg["dec"]))
                games[lg["gid"]]["rows"] += len(sel)
                lg["dec"] += n
            if not lanes:
                continue
            if all(lane_game[start + li]["skip"] for li, _, _, _ in lanes):
                stats["skipped_groups"] += 1
                continue
            stats["groups"] += 1
            obs = to_obs(hg, device)
            s0 = fp.state0(hg, gstate[gi])
            with torch.no_grad(), autocast(device, precision):
                o = model_rows(net, obs, s0, T, gk)
            gstate[gi] = o["state"]
            idx = np.concatenate([li * T + sel for li, sel, _, _ in lanes])
            if not len(idx):
                continue
            it = torch.as_tensor(idx, device=device)
            pick = lambda x, dt: x[it].float().to(dt).cpu().numpy()  # noqa: E731
            n_rows = len(idx)
            cols = {
                "game": np.concatenate([np.full(len(sel), g, np.int32) for _, sel, g, _ in lanes]),
                "split": np.full(n_rows, si, np.int8),
                "dec": np.concatenate([d + sel for _, sel, _, d in lanes]).astype(np.int32),
                "frame": frame_all[idx].astype(np.float32),
                "h": pick(o["h"], torch.float16), "y": pick(o["y"], torch.float16),
                "priv": np.asarray(hg["privileged"])[idx].astype(np.float16), "priv_ok": np.asarray(hg["priv_present"])[idx].astype(np.uint8),
                "own": own_state(scal[idx]), "action": np.asarray(hg["action"])[idx].astype(np.int16), "delay": np.asarray(hg["delay"])[idx].astype(np.int8),
                "weight": np.asarray(hg["weights"])[idx].astype(np.float32), "logp": pick(o["logp"], torch.float16),
                "mask": np.packbits(np.asarray(hg["type_mask"]).astype(bool)[idx], axis=1),
                "win_open": np.full(n_rows, np.nan, np.float32) if o["win_open"] is None else pick(o["win_open"], torch.float32),
                **{f"g{k}": pick(o["g"][k], torch.float16) for k in gk},
            }
            writer.add(cols)
            stats["rows"] += n_rows
        for lg in lane_game.values():
            finish_game(lg, games, read_until)
        stream.finish()
        print(f"[{split}] done: batches {nb} games {len(games)} rows {stats['rows']} {time.time() - t0:.0f}s", flush=True)
    writer.close()
    (out / "games.json").write_text(json.dumps(games), encoding="utf-8")
    man = {
        "kind": "plan-dump", "checkpoint": redact(ckpt_dir), "checkpoint_sha256": ckpt.policy_version(ckpt_dir),
        "data": [redact(d) for d in dirs], "splits": splits, "games_per_split": a.games, "seed": a.seed, "config": Path(a.config).as_posix(),
        "seq_len": T, "lanes": LANES, "lanes_per_group": G, "row_until": a.row_until, "read_until": read_until or None, "gk": gk,
        "own_columns": names, "op_actions": op_actions(), "classes": a.classes and redact(a.classes),
        "skip_views": a.skip_views and redact(a.skip_views),
        "core": net.cfg.core.kind, "outer": net.outer is not None, "backend": backend, "precision": precision,
        "games": len(games), "stats": stats, "secs": round(time.time() - t0, 1),
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k not in ("data", "own_columns", "op_actions")}), flush=True)
    return 0
