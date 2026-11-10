"""`python -m uranium diag dump`: one teacher-forced pass of a checkpoint over the games of a shard
source (reports/jepa_survey.md 7.2), per-row features and offline labels, the activations at the
stages, and with `--cf` the counterfactual passes of the intervention (7.2 (4)), each with its own
recurrent state from the game start.

Rows: every real row before `--row-until` (labels need the first minutes). Model passes: chunks
that start before `--act-until`. Activations: rows before `--act-until` with an enemy focus
token, every `--act-every`-th step. `--read-until`: each game is read only up to that frame (the
same rows; the lanes run shorter games, so the rows' order in the parts differs). A model pass is
the trunk and the ActionType head (`UraniumNet.action_seq`, the values of `forward_seq`).

Counterfactual passes (`--cf`), the anchor = the game's first-seen focus token, edited in every
row that has it (visible or remembered, at the token's last-seen frame):
  early / late: the anchor started at the race's early / late frame (`--cf-starts`): HP (and
      Protoss shield) ratio and sqrt, COMPLETED, CONSTRUCTING, order (Nothing / the race's unfinished
      order, UNFINISHED_ORDER);
      a token seen before the counterfactual start is masked. Spatial channels are left as they are.
  sc+1 / sc-1: the enemy max-seen count of the focus type +-1 (rows where it is >= 1), scalar only
  tok+1 / tok-1: the same, and one focus token duplicated into an extra slot / masked

Output (`--out`): rows_pNN.npz, acts_pNN.npz (float16, `idx` = row index in the part), games.json,
manifest.json. Paths under the shared drive are written as `<shared>/...` (001-R10)."""

import json
import os
import time
from pathlib import Path

import numpy as np

from uranium.diag import tables as tb
from uranium.diag.features import fill_race_cols, lane_features, race_of
from uranium.diag.stream import lane_groups

PASSES = ("real", "early", "late", "sc+1", "sc-1", "tok+1", "tok-1")
# early / late start frames per race: the rush / macro class medians of the earlier runs (10.1, 10.2.2)
CF_STARTS = {"T": (1280, 2080), "P": (1720, 3030), "Z": (1800, 3240)}
# Protoss shield of an unfinished building: (a + b (hp_ratio - 0.1) / 0.9) * max (fit on 978 rows, corr 0.994)
SHIELD_A, SHIELD_B = 0.104, 0.925
# The order an unfinished enemy focus building carries in the observations, per race (as observed:
# Terran / Protoss model games, 10.2.2 counterfactual setup; Zerg teacher holdout, 37,149 unfinished
# pool rows, 2026-11-09: all IncompleteBuilding, not IncompleteWarping / ZergBuildingMorph). The
# dump records the observed orders in its manifest (`focus_orders`) and warns when they differ.
UNFINISHED_ORDER = {"T": "IncompleteBuilding", "P": "IncompleteBuilding", "Z": "IncompleteBuilding"}
ENT_KEYS = ("entity_cat", "entity_cont", "entity_mask", "entity_cell", "entity_pos", "unit_cap", "is_building")


def unfinished_order(race: str) -> int:
    return tb.order_id(UNFINISHED_ORDER[race])
STAGES = ("token", "e_pool", "sp_pool", "c", "h")


def add_args(p) -> None:
    p.add_argument("--checkpoint", required=True, help="checkpoint folder, a 061 run folder (its LATEST), or `none` (features only)")
    p.add_argument("--dataset", action="append", default=[], help="dataset under the shared drive (041-R9), repeatable")
    p.add_argument("--shards", action="append", default=[], help="shard folder, repeatable")
    p.add_argument("--splits", default="val,test", help="comma-separated splits to read (holdouts: val,test)")
    p.add_argument("--games", type=int, help="only this many games per split (fixed-seed shuffle, 032-R17 ②)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--config", default="config/bc.toml")
    p.add_argument("--backend", choices=["cuda", "rocm", "cpu"])
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--row-until", type=float, default=8640, help="rows before this frame (default 6:00)")
    p.add_argument("--act-until", type=float, default=5760, help="model passes and activations before this frame (default 4:00)")
    p.add_argument("--act-every", type=int, default=1, help="activations of every k-th step")
    p.add_argument("--read-until", type=float, default=0,
                   help="read each game only up to this frame (0 = whole games; at least --row-until and --act-until): the same rows, "
                        "fewer steps to extract (the rows' order in the parts differs)")
    p.add_argument("--lanes-per-group", type=int, default=8)
    p.add_argument("--core-points", choices=["none", "gate", "full"], default="gate", help="GTrXL sub-points to keep")
    p.add_argument("--cf", action="store_true", help="also run the counterfactual passes (`diag intervene`)")
    p.add_argument("--cf-starts", default="", help="override early,late per race, e.g. T:1280,2080;Z:1800,3240")
    p.add_argument("--part-batches", type=int, default=200, help="write a part every this many batches")
    p.add_argument("--limit-batches", type=int, default=0)
    p.add_argument("--maps-dir", help="map folder (else from .env)")
    p.add_argument("--synthetic-maps", action="store_true", help="tests: synthetic maps")


def redact(path) -> str:
    """A path under the shared drive as `<shared>/...` (001-R10); other paths as given."""
    p = Path(path).resolve()
    sd = os.environ.get("URANIUM_SHARED_DIR")
    if sd:
        try:
            return "<shared>/" + p.relative_to(Path(sd).resolve()).as_posix()
        except ValueError:
            pass
    return Path(path).as_posix()


def cf_starts(text: str) -> dict:
    out = dict(CF_STARTS)
    for part in filter(None, text.split(";")):
        r, v = part.split(":")
        a, b = (float(x) for x in v.split(","))
        out[r.strip()] = (a, b)
    return out


def edit_start(g: dict, frame, anc, race: str, start: float) -> dict:
    """The anchor token as started at `start` (module docstring), every row that has it."""
    s = tb.race(race)
    mx, msh, bt = s["max_hp"], s["max_shields"], s["build_time"]
    cat = np.array(g["entity_cat"], copy=True)
    cont = np.array(g["entity_cont"], copy=True)
    mask = np.array(g["entity_mask"], copy=True)
    r = np.nonzero(anc >= 0)[0]
    k = anc[r]
    if len(r):
        seen = frame[r] - tb.SINCE_SCALE * cont[r, k, tb.SINCE_SEEN] * (cont[r, k, tb.REMEMBERED] > 0.5)
        el = seen - start
        done = el >= bt
        frac = np.clip(el / bt, 0.0, 1.0)
        hp = np.rint(np.where(done, mx, 0.1 * mx + 0.9 * mx * frac))
        cont[r, k, tb.HP_RATIO] = hp / mx
        cont[r, k, tb.HP_SQRT] = np.sqrt(hp) / 16.0
        if msh:
            sh = np.clip(np.rint(np.where(done, msh, SHIELD_A * msh + SHIELD_B * msh * frac)), 0, msh)
            cont[r, k, tb.SH_RATIO] = sh / msh
            cont[r, k, tb.SH_SQRT] = np.sqrt(sh) / 16.0
        cont[r, k, tb.COMPLETED] = done.astype(np.float32)
        cont[r, k, tb.CONSTRUCTING] = (~done).astype(np.float32)
        cat[r, k, 2] = np.where(done, tb.order_id("Nothing"), unfinished_order(race))
        gone = el < 0
        mask[r[gone], k[gone]] = False
    out = dict(g)
    out["entity_cat"], out["entity_cont"], out["entity_mask"] = cat, cont, mask
    return out


def pad_one(g: dict) -> dict:
    """One more (masked) entity slot in every row."""
    out = dict(g)
    for k in ENT_KEYS:
        a = np.asarray(g[k])
        pad = [(0, 0)] * a.ndim
        pad[1] = (0, 1)
        out[k] = np.pad(a, pad)
    return out


def edit_count(g: dict, race: str, d: int, token: bool) -> dict:
    """Enemy max-seen focus count +-1 where it is >= 1; with `token` also one focus token
    duplicated into the last (padded) slot (+1) or the row's highest-index focus token masked (-1)."""
    fid = tb.race(race)["focus"]
    col = tb.sc_seen_col(fid)
    sc = np.array(g["scalar"], copy=True)
    n = tb.decode_count(sc[:, col])
    on = n >= 1
    sc[on, col] = tb.encode_count(np.maximum(n[on] + d, 0))
    out = dict(g)
    out["scalar"] = sc
    if token:
        cat = np.asarray(g["entity_cat"])
        foc = (cat[:, :, 1] == tb.OWNER_ENEMY) & (cat[:, :, 0] == fid) & np.asarray(g["entity_mask"]).astype(bool)
        rr = np.nonzero(on & foc.any(1))[0]
        if d < 0:
            mask = np.array(g["entity_mask"], copy=True)
            k = foc.shape[1] - 1 - np.argmax(foc[:, ::-1], 1)
            mask[rr, k[rr]] = False
            out["entity_mask"] = mask
        else:
            k = np.argmax(foc, 1)
            last = cat.shape[1] - 1
            for key in ENT_KEYS:
                a = np.array(g[key], copy=True)
                a[rr, last] = a[rr, k[rr]]
                out[key] = a
    return out


def cat_state(states):
    s0 = states[0]
    if isinstance(s0, (tuple, list)):
        return type(s0)(cat_state([s[i] for s in states]) for i in range(len(s0)))
    import torch

    return torch.cat(states, 0)


def lane_slice(hg: dict, li: int, t: int) -> dict:
    b = hg["b"]
    return {k: (v[li * t:(li + 1) * t] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == b * t else v) for k, v in hg.items()}


class Parts:
    """Rows and activations written in parts (bounded host memory)."""

    def __init__(self, out: Path):
        self.out = out
        self.part = 0
        self.rows: dict[str, list] = {}
        self.acts: dict[str, list] = {}
        self.idx: list = []
        self.n = 0

    def add_rows(self, f: dict, sel: np.ndarray) -> int:
        base = self.n
        for k, v in f.items():
            self.rows.setdefault(k, []).append(np.asarray(v)[sel])
        self.n += len(sel)
        return base

    def add_acts(self, rows: np.ndarray, acts: dict):
        self.idx.append(rows)
        for k, v in acts.items():
            self.acts.setdefault(k, []).append(v)

    def flush(self):
        if not self.n:
            return
        np.savez_compressed(self.out / f"rows_p{self.part:03d}.npz", **{k: np.concatenate(v) for k, v in self.rows.items()})
        if self.idx:
            np.savez(self.out / f"acts_p{self.part:03d}.npz", idx=np.concatenate(self.idx), **{k: np.concatenate(v) for k, v in self.acts.items()})
        self.part += 1
        self.rows, self.acts, self.idx, self.n = {}, {}, [], 0


def run(a) -> int:
    import torch

    from uranium import _core, ckpt
    from uranium.device import autocast, configure, resolve
    from uranium.diag.capture import Capture
    from uranium.model import to_obs

    if a.read_until and a.read_until < max(a.row_until, a.act_until):
        raise SystemExit(f"diag dump: --read-until {a.read_until:g} cuts rows before --row-until {a.row_until:g} / --act-until {a.act_until:g}")
    dirs = [str(p) for p in _core.dataset_dirs(a.dataset, a.shards)]
    if not dirs:
        raise SystemExit("diag dump: no data (--dataset / --shards)")
    backend, device, precision = resolve(a.backend)
    configure(backend)
    torch.backends.cudnn.benchmark = False
    cfg = json.loads(_core.bc_config(a.config, "s1")[0])
    T, LANES = cfg["seq_len"], cfg["lanes"]
    G = min(a.lanes_per_group, LANES)
    net = cap = None
    ckpt_dir = None
    if a.checkpoint != "none":
        ckpt_dir = Path(_core.resolve_checkpoint(a.checkpoint))
        net, _ = ckpt.load_model(ckpt_dir, device)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        cap = Capture(net, a.core_points)
    max_sel = 64 if net is None else net.cfg.heads.max_select
    passes = PASSES if (a.cf and net is not None) else ("real",)
    starts = cf_starts(a.cf_starts)
    acts_ids = tb.actions()
    act_cols = [acts_ids["bunker"], acts_ids["marine"], acts_ids["rax"]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = Parts(out)
    games: dict[int, dict] = {}
    gid_next = 0
    t0 = time.time()
    stats = {"batches": 0, "groups_forwarded": 0, "rows": 0, "act_rows": 0, "frame_resets": 0}
    focus_orders: dict = {}
    splits = [s for s in a.splits.split(",") if s]
    for si, split in enumerate(splits):
        try:
            # --read-until: a game's steps from that frame on are dropped as it is read (features
            # are causal, so the kept steps are the same); its lane takes the next game sooner
            # (one frame of margin over the float frame the rows are cut by)
            until = {"until_frame": int(np.ceil(a.read_until)) + 1} if a.read_until else {}
            stream = _core.EvalStream(a.config, dirs, split=split, games=a.games, seed=a.seed, has_open=False, maps_dir=a.maps_dir, synthetic_maps=a.synthetic_maps, **until)
        except RuntimeError as e:
            if "no game" not in str(e).lower() and "empty" not in str(e).lower():
                raise
            print(f"[{split}] skipped: {e}", flush=True)
            continue
        # per lane group: {pass: recurrent state}, None = fresh (a drained group, lane_groups)
        gstate: list = [None] * -(-LANES // G)
        lane_game: dict[int, dict] = {}
        nb = 0

        def batch_done(n_done):
            stats["batches"] += 1
            if n_done % a.part_batches == 0:
                parts.flush()
            if n_done % 50 == 0:
                print(f"[{split}] batch {n_done} games {gid_next} rows {stats['rows']} act_rows {stats['act_rows']} {time.time() - t0:.0f}s", flush=True)

        for bi, b, gi, start, steps, hg in lane_groups(stream, LANES, G, T, cfg["pad_entities"], max_sel, gstate, a.limit_batches):
            if bi + 1 > nb:
                if nb:
                    batch_done(nb)
                nb = bi + 1
            if gstate[gi] is None:
                gstate[gi] = {p: None for p in passes}
            states = gstate[gi]
            bsz = hg["b"]
            keep = np.asarray(hg["keep"]).astype(bool)
            frame_all = np.asarray(hg["scalar"])[:, 0] * tb.FRAME_SCALE
            lane_rows = []  # (li, features, tok_k, a_k, n, race)
            for li in range(bsz):
                lane = start + li
                n = steps[lane]
                if n == 0:
                    continue
                fr = frame_all[li * T:li * T + n]
                lg = lane_game.get(lane)
                if lg is None or not keep[li] or fr[0] < lg["last"] - 1:
                    sub = lane_slice(hg, li, T)
                    race = race_of(sub, np.arange(n))
                    lg = {"gid": gid_next, "race": race, "last": -1.0, "anchor": None}
                    lane_game[lane] = lg
                    games[gid_next] = {"split": split, "race": race, "lane": lane}
                    gid_next += 1
                if (np.diff(fr) < -1).any():
                    stats["frame_resets"] += 1  # a new game inside a chunk: not expected (one game per chunk)
                lg["last"] = float(fr[-1])
                f, tok, anc = lane_features(lane_slice(hg, li, T), lg["race"], lg, n)
                fill_race_cols(f, T)
                lane_rows.append((li, f, tok, anc, n, lg))
            fwd = net is not None and any(frame_all[li * T] < a.act_until for li, *_ in lane_rows)
            pb = np.full((bsz * T, len(passes), 3), np.nan, np.float32)
            acts = None
            if fwd:
                stats["groups_forwarded"] += 1
                keep_t = torch.as_tensor(keep, device=device)
                for pi, p in enumerate(passes):
                    g = hg
                    if p != "real":
                        g = dict(hg)
                        if p == "tok+1":
                            g = pad_one(g)
                        ent = {k: np.array(g[k], copy=True) for k in ENT_KEYS + ("scalar",)}
                        for li, f, tok, anc, n, lg in lane_rows:
                            if lg["race"] not in tb.RACES:
                                continue
                            sub = {k: v[li * T:(li + 1) * T] for k, v in ent.items()}
                            if p in ("early", "late"):
                                e = edit_start(sub, f["frame"], anc, lg["race"], starts[lg["race"]][0 if p == "early" else 1])
                            else:
                                e = edit_count(sub, lg["race"], 1 if p.endswith("+1") else -1, p.startswith("tok"))
                            for k in ent:
                                ent[k][li * T:(li + 1) * T] = e[k]
                        g.update(ent)
                    obs = to_obs(g, device)
                    st = states[p]
                    s0 = net.init_state(bsz, device) if st is None else net.mask_state(st, keep_t)
                    # The trunk and the ActionType head only (`action_seq`: the numbers of
                    # forward_seq's dists["action"]; the other heads are not needed, and the
                    # location head's convolutions over a varying row count made the ROCm
                    # backend search a new kernel for every new count, 050-R9 human hold-out).
                    with torch.no_grad(), autocast(device, precision):
                        if p == "real":
                            o, acts = cap.run(obs, s0, T)
                        else:
                            tr, lp = net.action_seq(obs, s0, T)
                            o = {"state": tr["state"], "action": lp}
                    states[p] = o["state"]
                    pb[:, pi] = o["action"].float().exp()[:, act_cols].cpu().numpy()
                for li, *_ in lane_rows:
                    if frame_all[li * T] >= a.act_until:
                        pb[li * T:(li + 1) * T] = np.nan  # outside the model-pass window, like an unforwarded group
            for li, f, tok, anc, n, lg in lane_rows:
                if lg["race"] in tb.RACES:
                    # the order ids the focus tokens carry (the counterfactual's per-race order, manifest)
                    real = (np.arange(T) < n) & ~np.isnan(f["foc_comp"])
                    for done, key in ((False, "unfinished"), (True, "finished")):
                        c = focus_orders.setdefault(lg["race"], {}).setdefault(key, {})
                        for o_, k_ in zip(*np.unique(f["foc_order"][real & ((f["foc_comp"] > 0.5) == done)], return_counts=True)):
                            c[int(o_)] = c.get(int(o_), 0) + int(k_)
                sel = np.nonzero((np.arange(T) < n) & (f["frame"] < a.row_until))[0]
                if not len(sel):
                    continue
                f["game"] = np.full(T, lg["gid"], np.int64)
                f["split"] = np.full(T, si, np.int8)
                f["step"] = np.arange(T, dtype=np.int16)
                f["pb"] = pb[li * T:(li + 1) * T]  # f32: counterfactual differences are small
                base = parts.add_rows(f, sel)
                stats["rows"] += len(sel)
                if acts is None:
                    continue
                arow = sel[(tok[sel] >= 0) & (f["frame"][sel] < a.act_until)]
                if a.act_every > 1:
                    arow = arow[(arow % a.act_every) == 0]
                if not len(arow):
                    continue
                at = torch.as_tensor(li * T + arow, device=device)
                kk = torch.as_tensor(tok[arow], device=device)
                rec = {"token": acts["E"][at, kk], "e_pool": acts["e_pool"][at], "sp_pool": acts["sp_pool"][at], "c": acts["c"][at], "h": acts["h"][at]}
                for k in cap.keys():
                    if k in acts:
                        rec[k] = acts[k][at]
                parts.add_acts(base + np.searchsorted(sel, arow), {k: v.float().half().cpu().numpy() for k, v in rec.items()})
                stats["act_rows"] += len(arow)
        if nb:
            batch_done(nb)
        stream.finish()
        print(f"[{split}] done: batches {nb} games {gid_next} rows {stats['rows']} {time.time() - t0:.0f}s", flush=True)
    parts.flush()
    (out / "games.json").write_text(json.dumps(games), encoding="utf-8")
    man = {
        "checkpoint": None if ckpt_dir is None else redact(ckpt_dir),
        "checkpoint_sha256": None if ckpt_dir is None else ckpt.policy_version(ckpt_dir),
        "data": [redact(d) for d in dirs], "splits": splits, "games_per_split": a.games, "seed": a.seed, "config": Path(a.config).as_posix(),
        "seq_len": T, "lanes": LANES, "lanes_per_group": G, "row_until": a.row_until, "act_until": a.act_until, "act_every": a.act_every,
        "read_until": a.read_until or None, "passes": list(passes), "cf_starts": {k: list(v) for k, v in starts.items()} if len(passes) > 1 else None,
        "pb_actions": ["Build_Bunker", "Train_Marine", "Build_Barracks"], "stages": list(STAGES) + ([] if cap is None else cap.keys()),
        "core": None if net is None else net.cfg.core.kind, "outer": None if net is None else (net.outer is not None),
        "backend": backend, "precision": precision, "games": gid_next, "parts": parts.part, "stats": stats, "secs": round(time.time() - t0, 1),
        "focus_orders": focus_orders, "cf_orders": {r: {"unfinished": unfinished_order(r), "finished": tb.order_id("Nothing")} for r in tb.RACES},
    }
    for r, d in focus_orders.items():
        seen = d.get("unfinished", {})
        if seen and max(seen, key=seen.get) != unfinished_order(r):
            print(f"warning: unfinished {r} focus tokens mostly carry order {max(seen, key=seen.get)}, the counterfactual writes {unfinished_order(r)} (UNFINISHED_ORDER)", flush=True)
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k not in ("data",)}), flush=True)
    if cap is not None:
        cap.close()
    return 0


def load(dump: Path, stages: list[str] | None = None) -> tuple[dict, dict, dict, dict]:
    """(rows {name: [R]}, acts {"idx": [A], stage: [A, d] float32}, games {gid: meta}, manifest)
    of a dump folder; `stages` limits the activation arrays read."""
    dump = Path(dump)
    man = json.loads((dump / "manifest.json").read_text(encoding="utf-8"))
    games = {int(k): v for k, v in json.loads((dump / "games.json").read_text(encoding="utf-8")).items()}
    rows: dict[str, list] = {}
    acts: dict[str, list] = {}
    off = 0
    for p in sorted(dump.glob("rows_p*.npz")):
        z = np.load(p)
        n = len(z["frame"])
        for k in z.files:
            rows.setdefault(k, []).append(z[k])
        ap = dump / p.name.replace("rows_", "acts_")
        if ap.exists():
            az = np.load(ap)
            acts.setdefault("idx", []).append(az["idx"] + off)
            for k in az.files:
                if k != "idx" and (stages is None or k in stages):
                    acts.setdefault(k, []).append(az[k])
        off += n
    R = {k: np.concatenate(v) for k, v in rows.items()}
    A = {k: np.concatenate(v) for k, v in acts.items()}
    return R, A, games, man
