"""Per-action teacher-forced accuracy of a checkpoint on the teacher hold-out (050-R9 protocol metric
`per_action`, config/protocol/lineage.toml; 041-R21): the val split of the botgen teacher games, scored with the
same loop as python/uranium/bc/eval.py evaluate() plus per-label-action counts, top-1 / top-5, mean
p(label), delay MAE, target tile / unit accuracy, selection F1 and the argmax confusion of key actions.
Read-only; writes one JSON (input of tools/eval/per_action_table.py).

    python tools/eval/per_action.py --checkpoint <dir> --out <json> [--holdout orig|full] [--backend cuda]

--holdout orig (the protocol value): the 2026-11-06 teacher shards only (165 val games, 94,433 steps),
the hold-out of the BC / FT / clone / bc-mix columns of Checkpoint 1. `full` = the whole botgen source
(teacher-20261107-macro2 too). --shards DIR... scores other source folders.

Moved from runs/mix/per_action.py (a copy of runs/adv8/per_action.py) with the loop unchanged. The
`orig` source folder (four parts of botgen) is rebuilt from links under <URANIUM_CACHE_DIR or runs>/
per_action_holdout/botgen, as runs/mix/holdout_orig/botgen was.
"""

from __future__ import annotations

import argparse
import collections
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


def link_tree(src_root: Path, parts: list[str], dst_root: Path) -> Path:
    """dst_root/<part>/shard_* linked to src_root/<part>/shard_* (hard link, else symbolic link, else
    copy): a source folder that holds only some parts of a source (the shard reader takes a source
    folder, 040-R13 / 041-R13). Returns dst_root."""
    for part in parts:
        d = dst_root / part
        d.mkdir(parents=True, exist_ok=True)
        for f in sorted((src_root / part).glob("shard_*.bin.zst")):
            t = d / f.name
            if t.exists():
                if t.stat().st_size == f.stat().st_size:
                    continue
                t.unlink()
            try:
                os.link(f, t)
            except OSError:
                try:
                    os.symlink(f, t)
                except OSError:
                    shutil.copyfile(f, t)
    return dst_root


def holdout_dirs(holdout: str) -> tuple[list[str], list[str]]:
    """(datasets, shard folders) of a hold-out name."""
    if holdout == "full":
        return ["replays/shards/botgen"], []
    if holdout == "orig":
        src = ec.shared_dir() / "datasets" / "replays" / "shards" / "botgen"
        work = ec.cache_dir() / "per_action_holdout" / "botgen"
        return [], [str(link_tree(src, ORIG, work))]
    raise SystemExit(f"unknown --holdout {holdout}")


def score(ckpt_dir: str, datasets: list[str], shards: list[str], backend_name: str, config: str) -> dict:
    import torch

    from uranium import _core, ckpt
    from uranium.device import autocast, configure, resolve
    from uranium.model import to_obs

    names = {a["id"]: a["name"] for a in tomllib.loads((ec.ROOT / "crates/uranium-env/tables/actions.toml").read_text(encoding="utf-8"))["action"]}
    key_ids = {i for i, n in names.items() if n in KEY}
    backend, device, precision = resolve(backend_name)
    configure(backend)
    cfg = json.loads(_core.bc_config(config, "s1")[0])
    net, _ = ckpt.load_model(Path(ckpt_dir), device)
    net.eval()
    has_open = net.value.open is not None
    dirs = _core.dataset_dirs(datasets, shards)
    print("dirs", dirs, file=sys.stderr)
    stream = _core.EvalStream(config, dirs, split="val", games=None, seed=1, has_open=has_open, maps_dir=None, synthetic_maps=False)
    t, lanes = cfg["seq_len"], cfg["lanes"]
    g = lanes if cfg["micro_lanes"] == 0 else min(cfg["micro_lanes"], lanes)
    groups = -(-lanes // g)
    states = [None] * groups
    delay_frames = torch.tensor(json.loads(_core.feature_dims())["delay_buckets"], dtype=torch.float64)

    n = collections.Counter(); top1 = collections.Counter(); psum = collections.defaultdict(float); top5 = collections.Counter()
    conf = collections.defaultdict(collections.Counter)
    delay_abs = collections.defaultdict(float)
    loc_exact = collections.Counter(); loc_n = collections.Counter(); loc_err = collections.defaultdict(float)
    sel_tp = collections.Counter(); sel_fp = collections.Counter(); sel_fn = collections.Counter()
    tu_n = collections.Counter(); tu_hit = collections.Counter()
    with torch.no_grad():
        while (b := stream.next()) is not None:
            steps = b.lane_steps()
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
                for r in rows:
                    i = la[r]
                    n[i] += 1; psum[i] += pp[r]; delay_abs[i] += da[r]
                    if bt[r] == 0: top1[i] += 1
                    if bt[r] < 5: top5[i] += 1
                    if i in key_ids: conf[i][pr[r]] += 1
                    if hll[r]:
                        loc_n[i] += 1; loc_err[i] += float(lerr[r]); loc_exact[i] += int(lex[r])
                    if bool(uses_tu[r]):
                        tu_n[i] += 1; tu_hit[i] += int(tu_pred[r] == tu_lab[r])
                    if sll[r] > 0:
                        tp = int(hit[r].sum()); sel_tp[i] += tp; sel_fp[i] += int(take[r].sum()) - tp; sel_fn[i] += sll[r] - tp
                if out["win_open"] is not None:
                    stream.decide(b, start, end, t, out["win_open"].float().cpu().numpy())
    stream.finish()
    total = sum(n.values())
    rep = []
    for i, c in sorted(n.items(), key=lambda kv: -kv[1]):
        prec = sel_tp[i] / max(sel_tp[i] + sel_fp[i], 1); rec = sel_tp[i] / max(sel_tp[i] + sel_fn[i], 1)
        rep.append({
            "action": names.get(i, str(i)), "n": c, "share": c / total, "top1": top1[i] / c, "top5": top5[i] / c, "mean_p": psum[i] / c,
            "delay_mae": delay_abs[i] / c,
            "loc_n": loc_n[i], "loc_exact": loc_exact[i] / max(loc_n[i], 1), "loc_err": loc_err[i] / max(loc_n[i], 1),
            "tu_n": tu_n[i], "tu_top1": tu_hit[i] / max(tu_n[i], 1),
            "sel_f1": (2 * prec * rec / (prec + rec)) if prec + rec > 0 else None, "sel_n": sel_tp[i] + sel_fn[i],
            "confusion": {names.get(k, str(k)): v for k, v in conf[i].most_common(6)} if i in key_ids else None,
        })
    return {"checkpoint": ckpt_dir, "games": stream.games, "steps": total, "backend": backend, "precision": precision, "per_action": rep}


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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, help="checkpoint folder (relative = under URANIUM_SHARED_DIR)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--holdout", default="orig", choices=["orig", "full"])
    ap.add_argument("--shards", nargs="*", default=None, help="score these shard folders instead of a hold-out")
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--config", default=str(ec.ROOT / "config" / "bc.toml"))
    a = ap.parse_args(argv)
    ec.load_env()
    ck = Path(a.checkpoint)
    ck = ck if ck.is_absolute() else ec.shared_dir() / ck
    datasets, shards = ([], a.shards) if a.shards else holdout_dirs(a.holdout)
    res = score(str(ck), datasets, shards, a.backend, a.config)
    res["holdout"] = "shards" if a.shards else a.holdout
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print_report(res)


if __name__ == "__main__":
    main()
