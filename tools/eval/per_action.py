"""Per-action teacher-forced accuracy of a checkpoint on the teacher hold-out (050-R9 protocol metric
`per_action`, config/protocol/lineage.toml; 041-R21): the val split of the botgen teacher games, scored with the
same loop as python/uranium/bc/eval.py evaluate() plus per-label-action counts, top-1 / top-5, mean
p(label), delay MAE, target tile / unit accuracy, selection F1 and the argmax confusion of key actions.
Read-only; writes one JSON (input of tools/eval/per_action_table.py).

    python tools/eval/per_action.py --checkpoint <dir> --out <json> [--holdout orig|full] [--backend cuda]

--holdout orig (the protocol value): the 2026-11-06 teacher shards only (165 val games, 94,433 steps),
the hold-out of the BC / FT / clone / bc-mix columns of Checkpoint 1. `full` = the parts registered in the botgen
manifest but the 022 TvT / TvP teacher's teacher-tp-* (teacher-20261107-macro2 and teacher-20261109-macro3 too;
041-R27 is about the Zerg teacher, 044-R8 adds teacher-tp-*-v2 to botgen). --shards DIR... scores other source folders.

Moved from runs/mix/per_action.py (a copy of runs/adv8/per_action.py) with the loop unchanged. The
`orig` source folder (four parts of botgen) is rebuilt from links under <URANIUM_CACHE_DIR or runs>/
per_action_holdout/botgen, as runs/mix/holdout_orig/botgen was; `full` likewise under per_action_holdout_full/botgen
(the manifest's shard files; refused when the part folders or file sizes on disk differ from the manifest).

041-R27 (AMB-041-16): the hold-out games are also split by the teacher game's opponent, `rush` (the
four rush opponents of plan teacher_botgen_rush) or `macro` (the five macro opponents of plans
teacher_botgen_macro / macro2 / macro3, 040-R23). Each game's class comes from the generating plan's
games.jsonl (`opponent`) and the SHA-256 of its replay file, which is the shard's `replay_sha256`
(<shared>/datasets/replays/raw/botgen/teacher-*/teacher_botgen_*/, cached in <URANIUM_CACHE_DIR or
runs>/per_action_holdout/teacher_classes.json); teacher-tp-* (the TvT / TvP teacher, 044-R8) has no class and is
in no hold-out. The JSON then also has `by_class` (the same per-action
rows per class, and the mean p(Build_Bunker) over the scored steps before 4:00) and `r27`: the bunker
top-1 of the bc-mix extension rule (041-R21 6, counted on rush-opponent games only) and the macro
games' mean p(Build_Bunker) before 4:00 (recorded only). The `per_action` list over every game is
unchanged. The recorded macro value is over the macro games of the `full` hold-out (main-session interim
decision 2026-11-10: 117 games instead of the 36 of `orig`): with `--holdout orig` the other botgen parts
(macro2, macro3) are scored in a second pass for it only (`r27.macro_holdout`); `by_class` and every other
number stay those of the scored hold-out. `--classes none` turns the split off, `--classes <json>` ({replay_sha256: rush|macro})
gives the classes of other shard folders. Needs `uranium._core` with `Batch.lane_games`.

041-R30 ⑤ (user decision 2026-11-12, AMB-041-19): the hold-out is scored both ways. Every field above is the
current-observation score (each step its own observation, comparable with Checkpoint 1; the Checkpoint 1 lineage's
bc-mix extension rule `r27.rush_bunker_top1` < 70% and side-arm threshold < 50% read these, recorded only for
lineage 2). `chain_start` holds the same report (`games`, `steps`, `per_action`, and with classes `by_class` / `r27`)
scored with the chain-start observation of the `--chain-start` sources (default botgen); lineage 2 (041-R29 3) judges
the extension rule and the side-arm threshold on its `r27.rush_bunker_top1`, same 70% / 50% (user decision 2026-11-12). `--chain-start none` skips it (half the time). The
config's [data] chain_start_sources is not followed: only this option turns it on.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_common as ec  # noqa: E402

ORIG = ["teacher-20261106-a", "teacher-20261106-b", "teacher-20261106-c", "teacher-20261106-d"]
KEY = {"Build_Bunker", "Load_Unit", "Attack_Unit", "Repair_Unit", "Move_Pos", "Build_Barracks", "Build_SupplyDepot", "Build_CommandCenter", "Hold",
       "Attack_Pos", "Gather_Unit", "Smart_Unit", "Train_Marine", "Train_SCV", "NoOp"}

# 041-R27 / 040 teacher botgen opponents (spec/040-replay-dataset.md, teacher botgen row; 040-R23 macro3
# reuses the macro five), as the generating plans' games.jsonl names them (`<name>@<version>`).
RUSH_OPPONENTS = ("t0-zerg-rush", "ualbertabot12poollings", "ualbertabot9pool10lings", "ualbertabot9poolhydra")
MACRO_OPPONENTS = ("ualbertabot2hatchhydra", "ualbertabot3hatchmuta", "steamhammer2hatchmuta", "t0-zerg-muta", "t0-zerg")
CLASSES = ("rush", "macro")
# 044-R8 / 022-R12: the TvT / TvP teacher's botgen folders (raw and shards botgen/teacher-tp-*) are not part of the
# 041-R27 Zerg-teacher hold-out: no rush / macro class, not in the `full` macro pass.
TP_PREFIX = "teacher-tp-"
# 041-R27: "before 4:00" = observation frame < 5,760 (24 frames per game second, 042-R14).
BUNKER_BEFORE = 4 * 60 * ec.FPS


def opponent_class(opponent: str) -> str:
    """rush / macro of a teacher botgen game's `opponent` (`<name>@<version>`); another name is an error
    (a new plan's opponents must be added here, not guessed)."""
    name = opponent.split("@", 1)[0]
    if name in RUSH_OPPONENTS:
        return "rush"
    if name in MACRO_OPPONENTS:
        return "macro"
    raise SystemExit(f"041-R27: teacher opponent {opponent!r} is neither a rush ({', '.join(RUSH_OPPONENTS)}) nor a macro opponent "
                     f"({', '.join(MACRO_OPPONENTS)}); add it to tools/eval/per_action.py")


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def teacher_classes(raw_root: Path, cache: Path | None = None) -> dict[str, dict]:
    """{replay_sha256: {"class", "opponent", "plan"}} of every teacher botgen game under raw_root
    (`teacher-*/teacher_botgen_*/games.jsonl` and `replays/g<game_id>.rep`). A plan folder is hashed again
    only when its games.jsonl changed (cache: {plan folder: {"games_jsonl", "games": {sha: {"opponent"}}}};
    the class is derived from the opponent on every call, so a changed opponent list is never stale). A
    replay hash listed twice with different classes is an error."""
    old = {}
    if cache is not None and cache.exists():
        old = json.loads(cache.read_text(encoding="utf-8"))
    new, out = {}, {}
    for games in sorted(raw_root.glob("teacher-*/teacher_botgen_*/games.jsonl")):
        if games.parent.parent.name.startswith(TP_PREFIX):
            continue  # 044-R8: the 022 teacher's games
        plan = games.parent
        key = plan.relative_to(raw_root).as_posix()
        sig = file_sha256(games)
        if old.get(key, {}).get("games_jsonl") == sig:
            got = old[key]["games"]
        else:
            got = {}
            for line in games.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                g = json.loads(line)
                rep = plan / "replays" / f"g{g['game_id']}.rep"
                if rep.exists():
                    got[file_sha256(rep)] = {"opponent": g["opponent"]}
        new[key] = {"games_jsonl": sig, "games": got}
        for sha, v in got.items():
            e = {"class": opponent_class(v["opponent"]), "opponent": v["opponent"].split("@", 1)[0], "plan": key}
            if sha in out and out[sha]["class"] != e["class"]:
                raise SystemExit(f"041-R27: replay {sha} is {out[sha]['class']} in {out[sha]['plan']} and {e['class']} in {key}")
            out.setdefault(sha, e)
    if not out:
        raise SystemExit(f"041-R27: no teacher botgen plan folder under {raw_root}")
    if cache is not None and new != old:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(new, indent=0), encoding="utf-8")
    return out


def load_classes(spec: str | None, holdout: str | None) -> dict[str, str] | None:
    """{replay_sha256: rush|macro}: `auto` = the teacher botgen plans (only for the --holdout sets),
    `none` = no split, else a JSON file {sha: class} (or {sha: {"class": ...}})."""
    if spec is None:
        spec = "auto" if holdout in ("orig", "full") else "none"
    if spec == "none":
        return None
    if spec == "auto":
        raw = ec.shared_dir() / "datasets" / "replays" / "raw" / "botgen"
        cache = ec.cache_dir() / "per_action_holdout" / "teacher_classes.json"
        got = teacher_classes(raw, cache)
    else:
        got = json.loads(Path(spec).read_text(encoding="utf-8"))
    out = {k: (v["class"] if isinstance(v, dict) else v) for k, v in got.items()}
    bad = sorted({c for c in out.values() if c not in CLASSES})
    if bad:
        raise SystemExit(f"041-R27: classes must be {' / '.join(CLASSES)}, got {bad}")
    return out


SHARD_GLOB = "shard_*.bin.zst"


def is_link(p: Path) -> bool:
    """A symbolic link or (Windows) a directory junction."""
    return p.is_symlink() or p.is_junction()


def remove_entry(p: Path) -> None:
    """Remove a link-tree entry: a link, junction or file (only the entry itself), or a real folder of links (the
    source files stay). An entry already gone (another process building the same tree) is not an error."""
    try:
        if p.is_junction():
            os.rmdir(p)
        elif p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()
    except FileNotFoundError:
        pass


def _under(p: Path, root: Path) -> bool:
    return p.resolve().is_relative_to(root.resolve())


def check_link_root(src_root: Path, parts: list[str], dst_root: Path) -> None:
    """041-R27 / 044-R8: a pruned link tree must be a real folder of its own: dst_root and its part folders are
    neither links / junctions nor inside src_root (nor src_root inside dst_root), so pruning never removes a
    source file."""
    for d in (dst_root, *(dst_root / part for part in parts)):
        if is_link(d):
            raise SystemExit(f"041-R27: link tree folder {d} is a symbolic link or junction; remove it (the tree is rebuilt)")
    if (_under(dst_root, src_root) or _under(src_root, dst_root)
            or any(_under(dst_root / part, src_root) for part in parts if (dst_root / part).exists())):
        raise SystemExit(f"041-R27: link tree {dst_root} lies inside its source {src_root} (or holds it)")


def _link(f: Path, t: Path) -> None:
    """t -> f: hard link, else symbolic link, else copy; a t made meanwhile by another process is kept."""
    try:
        os.link(f, t)
    except FileExistsError:
        pass
    except OSError:
        try:
            os.symlink(f, t)
        except FileExistsError:
            pass
        except OSError:
            shutil.copyfile(f, t)


def link_tree(src_root: Path, parts: list[str], dst_root: Path, prune: bool = False,
              files: dict[str, list[str]] | None = None) -> Path:
    """dst_root/<part>/shard_* linked to src_root/<part>/shard_* (hard link, else symbolic link, else
    copy): a source folder that holds only some parts of a source (the shard reader takes a source
    folder, 040-R13 / 041-R13). `files` {part: [file name]}: the files of each part (default: its shard_* on
    disk). prune (per_action's own trees, 041-R27 / 044-R8): the tree holds exactly these parts and files;
    subfolders of dst_root not in `parts` and unwanted files in a part are removed, after check_link_root.
    Without prune (050-R9 protocol.set_data) nothing is removed: a stray file is left for its fingerprint
    check to refuse. Returns dst_root."""
    if prune:
        check_link_root(src_root, parts, dst_root)
        if dst_root.exists():
            for e in sorted(dst_root.iterdir()):
                if e.name not in parts:
                    remove_entry(e)
    for part in parts:
        d = dst_root / part
        d.mkdir(parents=True, exist_ok=True)
        if files is None:
            want = sorted((src_root / part).glob(SHARD_GLOB))
        else:
            want = [src_root / part / n for n in sorted(files.get(part, []))]
        if prune:
            names = {f.name for f in want}
            for e in sorted(d.iterdir()):
                if e.name not in names:
                    remove_entry(e)
        for f in want:
            t = d / f.name
            try:
                if t.exists():
                    if t.stat().st_size == f.stat().st_size:
                        continue
                    t.unlink()
            except FileNotFoundError:
                pass
            _link(f, t)
    return dst_root


def botgen_src() -> Path:
    return ec.shared_dir() / "datasets" / "replays" / "shards" / "botgen"


def holdout_parts(src: Path) -> tuple[list[str], dict[str, list[str]]]:
    """041-R27 / 044-R8: (parts, {part: shard file names}) of the Zerg-teacher hold-out `full`: the parts registered
    in the botgen manifest (src/manifest.json, 061) except the 022 TvT / TvP teacher's teacher-tp-* (no rush /
    macro class), with the manifest's shard files. Refused when the part folders on disk (teacher-tp-* aside) are
    not the manifest's parts, or a listed shard is missing or its size is not the manifest `bytes`. `orig` is
    ORIG, the rest is added_dirs()."""
    mf = src / "manifest.json"
    if not mf.exists():
        raise SystemExit(f"041-R27: no botgen manifest {mf}")
    listed = json.loads(mf.read_text(encoding="utf-8"))["files"]
    files: dict[str, list[str]] = {}
    for key, entry in listed.items():
        bits = key.split("/")
        if len(bits) != 2 or bits[0].startswith(TP_PREFIX) or not Path(bits[1]).match(SHARD_GLOB):
            continue
        f = src / key
        if not f.is_file():
            raise SystemExit(f"041-R27: botgen manifest file {key} is missing under {src}")
        if f.stat().st_size != entry["bytes"]:
            raise SystemExit(f"041-R27: botgen {key} is {f.stat().st_size} B, the manifest says {entry['bytes']} B")
        files.setdefault(bits[0], []).append(bits[1])
    disk = {p.name for p in src.iterdir() if p.is_dir() and not p.name.startswith(TP_PREFIX)}
    if disk != set(files):
        raise SystemExit(f"041-R27: botgen part folders {sorted(disk)} are not the manifest's parts {sorted(files)} "
                         "(teacher-tp-* aside); push or remove the part first")
    return sorted(files), {k: sorted(v) for k, v in files.items()}


def holdout_dirs(holdout: str) -> tuple[list[str], list[str]]:
    """(datasets, shard folders) of a hold-out name: `orig` = the ORIG parts, `full` = holdout_parts() (the
    parts registered in the botgen manifest but teacher-tp-*), each one source folder of links."""
    if holdout == "full":
        src = botgen_src()
        parts, files = holdout_parts(src)
        work = ec.cache_dir() / "per_action_holdout_full" / "botgen"
        return [], [str(link_tree(src, parts, work, prune=True, files=files))]
    if holdout == "orig":
        src = botgen_src()
        work = ec.cache_dir() / "per_action_holdout" / "botgen"
        return [], [str(link_tree(src, ORIG, work, prune=True))]
    raise SystemExit(f"unknown --holdout {holdout}")


def added_dirs() -> list[str]:
    """041-R27: the `full` parts outside `orig` (teacher-20261107-macro2, teacher-20261109-macro3, ...) as one
    source folder of links (<URANIUM_CACHE_DIR or runs>/per_action_holdout_added/botgen); [] without any."""
    src = botgen_src()
    parts, files = holdout_parts(src)
    parts = [p for p in parts if p not in ORIG]
    if not parts:
        return []
    work = ec.cache_dir() / "per_action_holdout_added" / "botgen"
    return [str(link_tree(src, parts, work, prune=True, files=files))]


def macro_full(r27: dict, added_macro: dict | None) -> dict:
    """041-R27 recorded macro value over the `full` hold-out's macro games (main-session interim decision
    2026-11-10): the scored hold-out's macro tally plus `added_macro` (by_class.macro of the pass over the other
    botgen parts), step-weighted. Returns a new r27."""
    out = {**r27, "macro_holdout": "full" if added_macro is not None else r27.get("macro_holdout")}
    if added_macro is None:
        return out
    a = added_macro["bunker_p_before_4m"]
    n0, n1 = r27["macro_steps_before_4m"], a["steps"]
    s0 = (r27["macro_bunker_p_before_4m"] or 0.0) * n0
    s1 = (a["mean_p"] or 0.0) * n1
    out["macro_steps_before_4m"] = n0 + n1
    out["macro_bunker_p_before_4m"] = (s0 + s1) / (n0 + n1) if n0 + n1 else None
    out["macro_games"] = r27["macro_games"] + added_macro["games"]
    out["macro_added_games"] = added_macro["games"]
    return out


class Tally:
    """Per-label-action counts of a set of scored steps (the per_action report), plus the mean
    p(Build_Bunker) over its steps before 4:00 (041-R27) and the games it saw."""

    def __init__(self):
        self.n = collections.Counter(); self.top1 = collections.Counter(); self.psum = collections.defaultdict(float); self.top5 = collections.Counter()
        self.conf = collections.defaultdict(collections.Counter)
        self.delay_abs = collections.defaultdict(float)
        self.loc_exact = collections.Counter(); self.loc_n = collections.Counter(); self.loc_err = collections.defaultdict(float)
        self.sel_tp = collections.Counter(); self.sel_fp = collections.Counter(); self.sel_fn = collections.Counter()
        self.tu_n = collections.Counter(); self.tu_hit = collections.Counter()
        self.early_n = 0
        self.early_p = 0.0
        self.games: set[str] = set()

    def add_early(self, p_bunker: float) -> None:
        self.early_n += 1
        self.early_p += p_bunker

    def report(self, names: dict, key_ids: set) -> list[dict]:
        total = sum(self.n.values())
        rep = []
        for i, c in sorted(self.n.items(), key=lambda kv: -kv[1]):
            prec = self.sel_tp[i] / max(self.sel_tp[i] + self.sel_fp[i], 1); rec = self.sel_tp[i] / max(self.sel_tp[i] + self.sel_fn[i], 1)
            rep.append({
                "action": names.get(i, str(i)), "n": c, "share": c / total, "top1": self.top1[i] / c, "top5": self.top5[i] / c, "mean_p": self.psum[i] / c,
                "delay_mae": self.delay_abs[i] / c,
                "loc_n": self.loc_n[i], "loc_exact": self.loc_exact[i] / max(self.loc_n[i], 1), "loc_err": self.loc_err[i] / max(self.loc_n[i], 1),
                "tu_n": self.tu_n[i], "tu_top1": self.tu_hit[i] / max(self.tu_n[i], 1),
                "sel_f1": (2 * prec * rec / (prec + rec)) if prec + rec > 0 else None, "sel_n": self.sel_tp[i] + self.sel_fn[i],
                "confusion": {names.get(k, str(k)): v for k, v in self.conf[i].most_common(6)} if i in key_ids else None,
            })
        return rep

    def early_report(self) -> dict:
        """041-R27: mean p(Build_Bunker) over the scored steps observed before 4:00."""
        return {"before_frame": BUNKER_BEFORE, "steps": self.early_n, "mean_p": self.early_p / self.early_n if self.early_n else None}


def action_top1(rep: list[dict], action: str) -> tuple[float | None, int]:
    """(top-1, steps) of one label action in a per_action list."""
    for r in rep:
        if r["action"] == action:
            return r["top1"], r["n"]
    return None, 0


def class_report(tallies: dict, names: dict, key_ids: set) -> tuple[dict, dict]:
    """(by_class, r27) of the per-class tallies (041-R27)."""
    by = {}
    for c in CLASSES:
        tl = tallies[c]
        by[c] = {"games": len(tl.games), "steps": sum(tl.n.values()), "per_action": tl.report(names, key_ids),
                 "bunker_p_before_4m": tl.early_report()}
    top1, n = action_top1(by["rush"]["per_action"], "Build_Bunker")
    r27 = {"rush_bunker_top1": top1, "rush_bunker_steps": n, "rush_games": by["rush"]["games"],
           "macro_bunker_p_before_4m": by["macro"]["bunker_p_before_4m"]["mean_p"],
           "macro_steps_before_4m": by["macro"]["bunker_p_before_4m"]["steps"], "macro_games": by["macro"]["games"]}
    return by, r27


def score(ckpt_dir: str, datasets: list[str], shards: list[str], backend_name: str, config: str,
          classes: dict[str, str] | None = None, maps_dir=None, synthetic_maps: bool = False, split: str = "val",
          chain_start: list[str] | None = None) -> dict:
    """`classes` {replay_sha256: rush|macro}: also tally per class (041-R27); a scored game without a
    class is an error. `chain_start`: 041-R30 ⑤ scoring option, chained steps of these sources observed through
    their chain-start step (None = each step its own observation)."""
    import torch

    from uranium import _core, ckpt
    from uranium.device import autocast, configure, resolve
    from uranium.diag.tables import FRAME_SCALE  # observation scalar 0 = frame / 86,400 (030 §1)
    from uranium.model import to_obs

    names = {a["id"]: a["name"] for a in tomllib.loads((ec.ROOT / "crates/uranium-env/tables/actions.toml").read_text(encoding="utf-8"))["action"]}
    key_ids = {i for i, n in names.items() if n in KEY}
    bunker_id = next(i for i, n in names.items() if n == "Build_Bunker")
    backend, device, precision = resolve(backend_name)
    configure(backend)
    cfg = json.loads(_core.bc_config(config, "s1")[0])
    net, _ = ckpt.load_model(Path(ckpt_dir), device)
    net.eval()
    has_open = net.value.open is not None
    dirs = _core.dataset_dirs(datasets, shards)
    print("dirs", dirs, file=sys.stderr)
    cs = {"chain_start_sources": list(chain_start)} if chain_start else {}
    stream = _core.EvalStream(config, dirs, split=split, games=None, seed=1, has_open=has_open, maps_dir=maps_dir, synthetic_maps=synthetic_maps, **cs)
    t, lanes = cfg["seq_len"], cfg["lanes"]
    g = lanes if cfg["micro_lanes"] == 0 else min(cfg["micro_lanes"], lanes)
    groups = -(-lanes // g)
    states = [None] * groups
    delay_frames = torch.tensor(json.loads(_core.feature_dims())["delay_buckets"], dtype=torch.float64)

    every = Tally()
    tallies = {c: Tally() for c in CLASSES} if classes is not None else None
    with torch.no_grad():
        while (b := stream.next()) is not None:
            steps = b.lane_steps()
            lane_cls = None
            if tallies is not None:
                ids = b.lane_games()
                lane_cls = []
                for l, gid in enumerate(ids):
                    if steps[l] == 0:
                        lane_cls.append(None)
                        continue
                    if gid not in classes:
                        raise SystemExit(f"041-R27: hold-out game {gid} has no rush / macro class (--classes)")
                    lane_cls.append(classes[gid])
                    tallies[classes[gid]].games.add(gid)
            for gi in range(groups):
                start, end = gi * g, min((gi + 1) * g, lanes)
                if all(steps[l] == 0 for l in range(start, end)):
                    states[gi] = None
                    continue
                hg = b.group(start, end, t, cfg["pad_entities"], net.cfg.heads.max_select)
                obs, labels = to_obs(hg, device), net.labels(hg, device)
                priv = torch.as_tensor(hg["privileged"]).to(device, torch.float32)
                keep = torch.as_tensor(hg["keep"]).to(device).bool()
                s = states[gi]
                s0 = net.init_state(hg["b"], device) if s is None else net.mask_state(s, keep)
                with autocast(device, precision):
                    out = net.forward_seq(obs, labels, priv, s0, t)
                states[gi] = out["state"]
                has_loc, sel_len = b.label_extras(start, end, t)
                d, lp = out["dists"], out["logp"]
                ok = torch.as_tensor(hg["weights"], device=device) > 0
                lab_a = torch.as_tensor(hg["action"], device=device).long()
                lab_d = torch.as_tensor(hg["delay"], device=device).long()
                a = d["action"].float()
                pred = a.argmax(1)
                want = a.gather(1, lab_a[:, None])
                better = (a > want).sum(1)
                p_lab = lp["action"].float().exp()
                dp = d["delay"].argmax(1)
                df = delay_frames.to(device)
                dabs = (df[dp] - df[lab_d]).abs()
                hl = torch.as_tensor(has_loc, device=device).bool()
                if bool(hl.any()):
                    m = round(d["target_loc"].shape[1] ** 0.5)
                    pl = d["target_loc"].argmax(1); tl = torch.as_tensor(hg["target_loc"], device=device).long()
                    dy, dx = (pl // m - tl // m).float(), (pl % m - tl % m).float()
                    lerr = (dx * dx + dy * dy).sqrt()
                    lex = (pl == tl)
                sl = torch.as_tensor(sel_len, device=device).long()
                uses_tu = d["uses_target_unit"].bool()
                if bool(uses_tu.any()):
                    tu_pred = d["target_unit"].argmax(1); tu_lab = torch.as_tensor(hg["target_unit"], device=device).long()
                if bool((sl > 0).any()):
                    dist = d["selected"]; s_, endtok = dist.shape[1], dist.shape[2] - 1
                    spred = dist.argmax(2); k = torch.arange(s_, device=device)[None]
                    take = (k < torch.clamp(sl + 1, max=s_)[:, None]) & ((spred == endtok).long().cumsum(1) == 0)
                    slab = torch.as_tensor(hg["selected"], device=device).long()
                    hit = ((spred[:, :, None] == slab[:, None, :]) & (slab[:, None, :] != endtok)).any(2) & take
                rows = ok.nonzero().flatten().tolist()
                la = lab_a.tolist(); pr = pred.tolist(); bt = better.tolist(); pp = p_lab.tolist(); da = dabs.tolist(); hll = hl.tolist(); sll = sl.tolist()
                if tallies is not None:
                    # 041-R27: p(Build_Bunker) and the observation frame of each row (scalar 0 = frame / 86,400).
                    pb = d["action"][:, bunker_id].float().exp().tolist()
                    fr = [round(x * FRAME_SCALE) for x in hg["scalar"][:, 0].tolist()]
                for r in rows:
                    i = la[r]
                    for ta in (every,) if tallies is None else (every, tallies[lane_cls[start + r // t]]):
                        ta.n[i] += 1; ta.psum[i] += pp[r]; ta.delay_abs[i] += da[r]
                        if bt[r] == 0: ta.top1[i] += 1
                        if bt[r] < 5: ta.top5[i] += 1
                        if i in key_ids: ta.conf[i][pr[r]] += 1
                        if hll[r]:
                            ta.loc_n[i] += 1; ta.loc_err[i] += float(lerr[r]); ta.loc_exact[i] += int(lex[r])
                        if bool(uses_tu[r]):
                            ta.tu_n[i] += 1; ta.tu_hit[i] += int(tu_pred[r] == tu_lab[r])
                        if sll[r] > 0:
                            tp = int(hit[r].sum()); ta.sel_tp[i] += tp; ta.sel_fp[i] += int(take[r].sum()) - tp; ta.sel_fn[i] += sll[r] - tp
                        if tallies is not None and fr[r] < BUNKER_BEFORE:
                            ta.add_early(pb[r])
                if out["win_open"] is not None:
                    stream.decide(b, start, end, t, out["win_open"].float().cpu().numpy())
    stream.finish()
    res = {"checkpoint": ckpt_dir, "games": stream.games, "steps": sum(every.n.values()), "backend": backend, "precision": precision,
           "per_action": every.report(names, key_ids)}
    if tallies is not None:
        res["by_class"], res["r27"] = class_report(tallies, names, key_ids)
    return res


def print_report(res: dict) -> None:
    rep = res["per_action"]
    print("steps", res["steps"], "games", res["games"])
    print(f"{'action':22} {'n':>7} {'share':>6} {'top1':>6} {'top5':>6} {'meanp':>6} {'dMAE':>5} {'locN':>5} {'locEx':>6} {'locErr':>6} {'tuN':>5} {'tuTop1':>6} {'selF1':>6}")
    for r in rep[:40]:
        print(f"{r['action']:22} {r['n']:7d} {r['share']:6.3f} {r['top1']:6.3f} {r['top5']:6.3f} {r['mean_p']:6.3f} {r['delay_mae']:5.1f} {r['loc_n']:5d} {r['loc_exact']:6.3f} {r['loc_err']:6.1f} {r['tu_n']:5d} {r['tu_top1']:6.3f} {(r['sel_f1'] if r['sel_f1'] is not None else float('nan')):6.3f}")
    print("confusions:")
    for r in rep:
        if r["confusion"]:
            print(" ", r["action"], r["confusion"])
    r27 = res.get("r27")
    if r27:
        bt = "-" if r27["rush_bunker_top1"] is None else f"{r27['rush_bunker_top1'] * 100:.1f}%"
        mp = "-" if r27["macro_bunker_p_before_4m"] is None else f"{r27['macro_bunker_p_before_4m']:.4f}"
        print(f"041-R27: rush-opponent games {r27['rush_games']}: Build_Bunker top-1 {bt} ({r27['rush_bunker_steps']} steps, the bc-mix extension rule input for this observation's lineage); "
              f"macro-opponent games {r27['macro_games']}: mean p(Build_Bunker) before 4:00 {mp} ({r27['macro_steps_before_4m']} steps, recorded only)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, help="checkpoint folder (relative = under URANIUM_SHARED_DIR)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--holdout", default="orig", choices=["orig", "full"])
    ap.add_argument("--shards", nargs="*", default=None, help="score these shard folders instead of a hold-out")
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--config", default=str(ec.ROOT / "config" / "bc.toml"))
    ap.add_argument("--classes", default=None,
                    help="041-R27 rush / macro split: auto (the teacher botgen plans; default for --holdout), none (default for --shards), "
                         "or a JSON file {replay_sha256: rush|macro}")
    ap.add_argument("--chain-start", default="botgen",
                    help="041-R30 ⑤: also score with the chain-start observation of these sources (comma list of botgen, dagger; "
                         "default botgen) into `chain_start`; none = current observation only")
    a = ap.parse_args(argv)
    ec.load_env()
    ck = Path(a.checkpoint)
    ck = ck if ck.is_absolute() else ec.shared_dir() / ck
    datasets, shards = ([], a.shards) if a.shards else holdout_dirs(a.holdout)
    classes = load_classes(a.classes, None if a.shards else a.holdout)
    chain = chain_sources(a.chain_start)
    holdout = "shards" if a.shards else a.holdout
    # 041-R27 / 044-R8: the parts of the macro pass, once and before any scoring (a botgen manifest mismatch stops
    # here, not after the GPU pass; both observation passes read the same parts).
    extra = added_dirs() if classes is not None and holdout == "orig" else []
    res = score_holdout(str(ck), datasets, shards, a.backend, a.config, classes, holdout, None, extra)
    # The current-observation report (Checkpoint 1 lineage extension rule input) is written before the second pass.
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    if chain:
        # 041-R30 ⑤: the same report with the chain-start observation (lineage 2 extension rule input, 2026-11-12).
        cs = score_holdout(str(ck), datasets, shards, a.backend, a.config, classes, holdout, chain, extra)
        res["chain_start"] = {"sources": chain, **{k: cs[k] for k in ("games", "steps", "per_action", "by_class", "r27") if k in cs}}
        a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print_report(res)
    if chain:
        print(f"\n041-R30 chain-start observation ({', '.join(chain)}; lineage 2 extension rule input; Checkpoint 1 lineage reads the scores above):")
        print_report(res["chain_start"])


def chain_sources(text: str) -> list[str]:
    """041-R30 ⑤ `--chain-start`: [] for none, else the listed sources (botgen, dagger)."""
    if text.strip().lower() == "none":
        return []
    out = [s.strip() for s in text.split(",") if s.strip()]
    bad = [s for s in out if s not in ("botgen", "dagger")]
    if not out or bad:
        raise SystemExit(f"041-R30: --chain-start takes botgen, dagger (comma list) or none, got {text!r}")
    return out


def score_holdout(ck: str, datasets: list[str], shards: list[str], backend: str, config: str, classes: dict | None, holdout: str,
                  chain: list[str] | None, extra: list[str]) -> dict:
    """One scoring of the hold-out (the current observation when `chain` is None, 041-R30 ⑤), with the 041-R27
    recorded macro value over the full hold-out's macro games for `orig`: `extra` = added_dirs(), taken by the
    caller before any scoring ([] = no added part)."""
    res = score(ck, datasets, shards, backend, config, classes, chain_start=chain)
    res["holdout"] = holdout
    if classes is not None:
        res["r27"]["macro_holdout"] = holdout
        if holdout == "orig":
            # 041-R27 interim decision: the recorded macro value over the full hold-out's macro games.
            added = score(ck, [], extra, backend, config, classes, chain_start=chain)["by_class"]["macro"] if extra else None
            res["r27"] = macro_full(res["r27"], added)
    return res


if __name__ == "__main__":
    main()
