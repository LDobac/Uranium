"""`python -m uranium diag plan-dump` (jepa_planning S0, reports/jepa_planning.md 6.3; 050-R9 diagnostics
tooling, 032-R24 aux outputs): one teacher-forced pass of a checkpoint over the games of a shard source;
every real step before `--row-until` becomes one row of the S0 feasibility test (`diag plan-s0`).

Measurement only, the model is not changed: the trunk and the ActionType head as `UraniumNet.action_seq`
(the numbers of `forward_seq`'s `dists["action"]`; the location head is not run), with the trunk's
encoder summaries kept (`trunk_seq(..., aux=True)`, same values), and on h' the training-only JEPA
predictors g_k and win_open. No counterfactual pass. `diag dump` is a separate command and unchanged.

Rows (`rows/<field>.bin`, raw arrays appended lane group by lane group, row order = write order;
`rows/schema.json` has each field's dtype and per-row shape and the row count; `load_rows`):
  game int32 (id in games.json), split int8 (index into manifest `splits`), dec int32 (the step's
  decision index in its game, from 0, every real step counted), frame float32,
  h float16 [d] (h', what the heads read), y float16 [d_y] (LN([e_pool, s_pool]), the JEPA target,
  aux.jepa_target), priv float16 [P] (privileged summary: a training target only, never an input),
  priv_ok uint8 (the step has one), own float32 [O] (own state from the scalar input, `OWN_COLUMNS`),
  action int16 / delay int8 / weight float32 (the shard label: action type, delay bucket, label weight;
  0 = void), logp float16 [A] (ActionType log-probs, masked), mask uint8 [ceil(A/8)] (type_mask,
  np.packbits), win_open float32 (NaN without the head), g<k> float16 [d_y] (g_k(h'), the k of `--gk`
  the model has).
games.json {gid: {split, lane, race (privileged), enemy_race, sha (replay_sha256), occ (n-th stream of
  that replay in dump order; not a player identity), map (family index, 0/1 = target maps), start (own
  start slot, scalar_cat[2]: with sha it names the view, e.g. one player of a TvT replay), result (+1 / -1 / 0 = undecided), rows, dec_rows, last_frame, scv8 (own SCVs >= 8 at 8:00,
  0 if the game ended before, None if not readable), cls (build class from privileged, jepa_survey
  7.3 draft rules, `features.game_labels`), cls_given (`--classes`), start1, first_sight, skipped
  (`--skip-views`: a listed stream, no rows)}}.
manifest.json: checkpoint (sha256), data, selection, windows, backend, precision, stats, own columns.

Memory: bounded (rows streamed to disk per lane group; per lane only the open game's label columns)."""

import functools
import json
import time
from pathlib import Path

import numpy as np

from uranium.diag import tables as tb
from uranium.diag.features import fill_race_cols, game_labels, lane_features, race_of
from uranium.diag.stream import lane_groups

F8M = 8 * 60 * 24  # 8:00 (frames, 24 per game second): the held_8m survival check (docs/measurement_protocol.md)
SCV8_MIN = 8
# game_labels inputs kept per open game (the rest of lane_features is not needed for the classes)
LABEL_KEYS = ("frame", "foc_vis", "foc_comp", "foc_hp", "foc_seen", "a_has", "a_comp", "sc_focus")
# The 2.2 operation actions (build 12, addon 6, train 14, research 11, upgrade 15, lift / land, cancel)
OP_GROUPS = ("build", "addon", "train", "research", "upgrade", "cancel")
OP_EXTRA = ("Lift", "Land_Pos", "Land_CommandCenter_Pos")
SIEGE = ("Terran_Siege_Tank_Tank_Mode", "Terran_Siege_Tank_Siege_Mode")


def op_actions() -> list[int]:
    """Action ids of the operation actions (reports/jepa_planning.md 2.2, 6.4 M0)."""
    return sorted(a["id"] for a in tb._rows("actions") if a["group"] in OP_GROUPS or a["name"] in OP_EXTRA)


@functools.cache
def own_columns() -> tuple[list[str], dict[int, list[int]], dict[int, int]]:
    """(column names, {column: Terran unit ids summed}, {action id: column}) of the own-state vector:
    done + in-production count (scalar input, own vision) of every Terran type a Build / BuildAddon /
    Train action makes (siege tank: both modes), then supply used / total (BWAPI half-supply units:
    a Marine is 2) and the army (every trained type but the SCV), 6.3 o_t. `{action: column}` is the M2b direction map."""
    terran = set(tb.terran())
    names, types, of_action = [], {}, {}
    for a in sorted(tb._rows("actions"), key=lambda a: a["id"]):
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
    return next(a["group"] for a in tb._rows("actions") if a["id"] == aid)


def own_state(scalar: np.ndarray) -> np.ndarray:
    """[rows, O] float32 own-state vector (`own_columns`) from the 030 §4 scalar input."""
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
    p.add_argument("--checkpoint", required=True, help="checkpoint folder or a 061 run folder (its LATEST)")
    p.add_argument("--dataset", action="append", default=[], help="dataset under the shared drive (041-R9), repeatable")
    p.add_argument("--shards", action="append", default=[], help="shard folder, repeatable")
    p.add_argument("--splits", default="val,test", help="comma-separated splits to read")
    p.add_argument("--games", type=int, help="only this many games per split (fixed-seed shuffle, 032-R17 ②; n is a prefix of a larger n)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--config", default="config/bc.toml")
    p.add_argument("--backend", choices=["cuda", "rocm", "cpu"])
    p.add_argument("--out", required=True, help="output folder (must not hold a finished dump)")
    p.add_argument("--row-until", type=float, default=14400, help="rows before this frame (default 10:00, 6.2)")
    p.add_argument("--read-until", type=float, default=-1,
                   help="read each game only up to this frame (default: --row-until; 0 = whole games); the rows are the same")
    p.add_argument("--gk", default="8,32", help="g_k outputs to keep (the k the model has; empty = none)")
    p.add_argument("--classes", help="JSON {replay_sha256: class or {\"class\": ...}}: a given build class per game (cls_given)")
    p.add_argument("--skip-views", help="views JSON ({\"bot\": [[sha256, start], ...], ...}, config/plan_s0/jepa_diag_tp_views.json): the streams "
                   "of the \"bot\" list get no rows (games.json `skipped`), and a lane group whose lanes are all such streams skips its forward")
    p.add_argument("--lanes-per-group", type=int, default=8)
    p.add_argument("--limit-batches", type=int, default=0)
    p.add_argument("--maps-dir", help="map folder (else from .env)")
    p.add_argument("--synthetic-maps", action="store_true", help="tests: synthetic maps")


class RowWriter:
    """Append-only raw columns (module docstring); the schema's row count is written by `close`."""

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
    """{field: read-only np.memmap [rows, *shape]} of a plan-dump folder (`fields` limits them)."""
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
    """(rows memmaps, games {gid: meta}, manifest) of a plan-dump folder."""
    dump = Path(dump)
    man = json.loads((dump / "manifest.json").read_text(encoding="utf-8"))
    games = {int(k): v for k, v in json.loads((dump / "games.json").read_text(encoding="utf-8")).items()}
    return load_rows(dump), games, man


def model_rows(net, obs: dict, s0, t: int, gk: list[int]) -> dict:
    """One teacher-forced pass over b*t rows: the `action_seq` path with the encoder summaries kept
    (`trunk_seq(aux=True)` only also returns e_pool; same values) -> {h, y (f32 LN target),
    logp, win_open (None without the head), g {k: [rows, d_y]}, state}."""
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
    """{(replay sha256, own start slot)} of the "bot" list of a views JSON (`--skip-views`)."""
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
    """Per-game labels of a lane's finished (or cut) game into `games` (module docstring)."""
    e = games[lg["gid"]]
    acc = lg.pop("acc")
    e["dec_rows"] = lg["dec"]
    e["last_frame"] = lg["last"]
    # the 8:00 SCV check needs the steps up to 8:00 read; a game whose last step is before 8:00 ended there
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
    from uranium.device import autocast, configure, resolve
    from uranium.diag.dump import lane_slice, redact
    from uranium.model import to_obs

    read_until = a.row_until if a.read_until < 0 else a.read_until
    if read_until and read_until < a.row_until:
        raise SystemExit(f"diag plan-dump: --read-until {read_until:g} cuts rows before --row-until {a.row_until:g}")
    out = Path(a.out)
    if (out / "manifest.json").exists():
        raise SystemExit(f"diag plan-dump: {out} already holds a finished dump")
    dirs = [str(p) for p in _core.dataset_dirs(a.dataset, a.shards)]
    if not dirs:
        raise SystemExit("diag plan-dump: no data (--dataset / --shards)")
    backend, device, precision = resolve(a.backend)
    configure(backend)
    torch.backends.cudnn.benchmark = False
    cfg = json.loads(_core.bc_config(a.config, "s1")[0])
    T, LANES = cfg["seq_len"], cfg["lanes"]
    G = min(a.lanes_per_group, LANES)
    ckpt_dir = Path(_core.resolve_checkpoint(a.checkpoint))
    net, _ = ckpt.load_model(ckpt_dir, device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
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
            stream = _core.EvalStream(a.config, dirs, split=split, games=a.games, seed=a.seed, has_open=False, maps_dir=a.maps_dir, synthetic_maps=a.synthetic_maps, **until)
        except RuntimeError as e:
            if "no game" not in str(e).lower() and "empty" not in str(e).lower():
                raise
            print(f"[{split}] skipped: {e}", flush=True)
            continue
        gstate: list = [None] * -(-LANES // G)
        lane_game: dict[int, dict] = {}
        nb = 0
        batch_meta = (None, None, None)
        for bi, b, gi, start, steps, hg in lane_groups(stream, LANES, G, T, cfg["pad_entities"], net.cfg.heads.max_select, gstate, a.limit_batches):
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
                    # --skip-views: no rows, no labels; the decision index still counts the steps
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
                # every lane with steps is a skipped view: no forward. The group state stays as it was;
                # only skipped streams continue from it (a new stream on a lane starts from a reset
                # state, `keep` False), so the written rows are those of a full walk.
                stats["skipped_groups"] += 1
                continue
            stats["groups"] += 1
            obs = to_obs(hg, device)
            st = gstate[gi]
            s0 = net.init_state(bsz, device) if st is None else net.mask_state(st, torch.as_tensor(keep, device=device))
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
