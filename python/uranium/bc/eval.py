"""`python -m uranium bc-eval` (041 offline metrics, 041-AC1, AC4): the indicators on one split
(default the target-map test set) with the labels fed to the autoregressive heads (teacher
forcing, as in training): action_type top-1/top-5, delay MAE (frames, E-9), target_loc tile
error, selected F1 (argmax of every selection step against the label set) and the per-head NLL
over the steps that use the head. Games are read without z, symmetry or the race mask; with
`--games N` the N games come from a fixed-seed shuffle (032-R17 ②). With a win_open head
(032-R18) also its MSE and the 031-R12 surrender check replayed per game (041-AC4, in Rust).
Same report as the Burn `uranium-train bc-eval`."""

import argparse
import json
from pathlib import Path

import torch

HEADS = ("action", "delay", "queued", "selected", "target_unit", "target_loc")

# Output fields and their units (E-9), printed with every report.
UNITS = {
    "action_top1 / action_top5": "share of scored steps",
    "delay_mae_frames": "mean |predicted - label| delay in game frames (bucket values, 031-R11)",
    "delay_mae_buckets": "the same in delay bucket indices",
    "target_loc_err_*": "Euclidean distance in target_loc cells (map tiles)",
    "nll.<head>": "mean -log p of the label over the scored steps that use the head (nll_steps.<head>)",
    "steps": "steps with a usable label (041-R11 weight 1); padding and void labels are not scored",
    "win_open_mse": "mean (win_open - result)^2 over the steps of decided games (032-R18, result +1 / -1)",
    "surrender.*": "031-R12 check replayed per game (041-AC4): games, games where it held, the share of those won, first frame it held",
}


def add_parser(sub) -> None:
    p = sub.add_parser("bc-eval", help="041 offline metrics of a checkpoint on a split")
    p.add_argument("--dataset", action="append", default=[])
    p.add_argument("--shards", action="append", default=[])
    p.add_argument("--checkpoint", required=True, help="checkpoint folder, or a 061 run folder (its LATEST)")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--backend", choices=["cuda", "rocm", "cpu"])
    p.add_argument("--core", choices=["lstm", "gtrxl"], help="checked against the checkpoint")
    p.add_argument("--games", type=int, help="score this many games by a fixed-seed shuffle of the split (032-R17 ②)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--config", default="config/bc.toml")
    p.add_argument("--maps-dir", help=argparse.SUPPRESS)
    p.add_argument("--synthetic-maps", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(fn=main)


class Acc:
    def __init__(self, delay_frames: list[int]):
        self.delay_frames = torch.tensor(delay_frames, dtype=torch.float64)
        self.steps = self.top1 = self.top5 = 0
        self.delay_abs = self.delay_abs_frames = 0.0
        self.loc_n = self.loc_exact = 0
        self.loc_err: list[torch.Tensor] = []
        self.sel_n = self.sel_tp = self.sel_fp = self.sel_fn = 0
        self.nll = [0.0] * len(HEADS)
        self.nll_n = [0] * len(HEADS)

    @torch.no_grad()
    def score(self, out: dict, g: dict, has_loc, sel_len) -> None:
        """Adds one lane group (rows with weight 0 are skipped: padding, 041-R11 void labels)."""
        d, lp = out["dists"], out["logp"]
        dev = d["action"].device
        ok = torch.as_tensor(g["weights"], device=dev) > 0
        lab = {k: torch.as_tensor(g[k], device=dev).long() for k in ("action", "delay", "selected", "target_loc")}
        self.steps += int(ok.sum())
        a = d["action"].float()
        want = a.gather(1, lab["action"][:, None])
        better = (a > want).sum(1)
        self.top1 += int(((better == 0) & ok).sum())
        self.top5 += int(((better < 5) & ok).sum())
        dp = d["delay"].argmax(1)
        frames = self.delay_frames.to(dev)
        self.delay_abs += float(((dp - lab["delay"]).abs().double() * ok).sum())
        self.delay_abs_frames += float(((frames[dp] - frames[lab["delay"]]).abs() * ok).sum())

        loc_rows = ok & torch.as_tensor(has_loc, device=dev)
        if bool(loc_rows.any()):
            p = d["target_loc"].argmax(1)[loc_rows]
            t = lab["target_loc"][loc_rows]
            m = round(d["target_loc"].shape[1] ** 0.5)
            dy, dx = (p // m - t // m).float(), (p % m - t % m).float()
            self.loc_err.append((dx * dx + dy * dy).sqrt().cpu())
            self.loc_exact += int((p == t).sum())
            self.loc_n += int(loc_rows.sum())

        sl = torch.as_tensor(sel_len, device=dev).long()
        sel_rows = ok & (sl > 0)
        if bool(sel_rows.any()):
            dist = d["selected"][sel_rows]  # [r, S, n+1]
            s, end = dist.shape[1], dist.shape[2] - 1
            pred = dist.argmax(2)
            n_sel = sl[sel_rows]
            # Teacher-forced steps 0..=len(label): argmax per step until END.
            k = torch.arange(s, device=dev)[None]
            take = (k < torch.clamp(n_sel + 1, max=s)[:, None]) & ((pred == end).long().cumsum(1) == 0)
            labels = lab["selected"][sel_rows]
            hit = ((pred[:, :, None] == labels[:, None, :]) & (labels[:, None, :] != end)).any(2) & take
            tp = int(hit.sum())
            self.sel_n += int(sel_rows.sum())
            self.sel_tp += tp
            self.sel_fp += int(take.sum()) - tp
            self.sel_fn += int(n_sel.sum()) - tp

        uses = {"queued": d["uses_queued"], "selected": d["uses_select"], "target_unit": d["uses_target_unit"], "target_loc": d["uses_target_loc"]}
        for h, k in enumerate(HEADS):
            used = ok if k in ("action", "delay") else ok & uses[k].bool()
            self.nll[h] -= float((lp[k].double() * used).sum())
            self.nll_n[h] += int(used.sum())

    def report(self, ckpt: str, split: str, games: int, seed) -> dict:
        n = max(self.steps, 1)
        errs = torch.sort(torch.cat(self.loc_err)).values if self.loc_err else torch.zeros(0)
        prec = self.sel_tp / max(self.sel_tp + self.sel_fp, 1)
        rec = self.sel_tp / max(self.sel_tp + self.sel_fn, 1)
        top1 = self.top1 / n
        return {
            "checkpoint": ckpt, "split": split, "games": games, "selection_seed": seed, "steps": self.steps,
            "action_top1": top1, "action_top5": self.top5 / n,
            "delay_mae_frames": self.delay_abs_frames / n, "delay_mae_buckets": self.delay_abs / n,
            "target_loc_steps": self.loc_n,
            "target_loc_err_mean": float(errs.double().mean()) if len(errs) else 0.0,
            "target_loc_err_median": float(errs[len(errs) // 2]) if len(errs) else 0.0,
            "target_loc_exact": self.loc_exact / max(self.loc_n, 1),
            "selected_steps": self.sel_n,
            "selected_f1": 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0,
            "nll": {k: self.nll[h] / max(self.nll_n[h], 1) for h, k in enumerate(HEADS)},
            "nll_steps": {k: self.nll_n[h] for h, k in enumerate(HEADS)},
            "ac1_pass": top1 >= 0.5 if split == "test" else None,
            "win_open_mse": None, "surrender": None,
            "units": UNITS,
        }


@torch.no_grad()
def evaluate(dirs: list[str], ckpt_dir: str, split: str, games, seed: int, config: str, device, precision: str, core=None, maps_dir=None, synthetic_maps=False) -> dict:
    from uranium import _core, ckpt
    from uranium.device import autocast
    from uranium.model import to_obs

    cfg = json.loads(_core.bc_config(config, "s1")[0])
    # 032 §2: no dropout at evaluation (eval mode).
    net, _ = ckpt.load_model(Path(ckpt_dir), device)
    net.eval()
    if core is not None and net.cfg.core.kind.lower() != core:
        raise ValueError(f"checkpoint core {net.cfg.core.kind} != requested {core}")
    has_open = net.value.open is not None
    stream = _core.EvalStream(config, dirs, split=split, games=games, seed=seed, has_open=has_open, maps_dir=maps_dir, synthetic_maps=synthetic_maps)
    t, lanes = cfg["seq_len"], cfg["lanes"]
    g = lanes if cfg["micro_lanes"] == 0 else min(cfg["micro_lanes"], lanes)
    groups = -(-lanes // g)
    states = [None] * groups
    acc = Acc(json.loads(_core.feature_dims())["delay_buckets"])
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
            s0 = net.core.init_state(hg["b"], device) if s is None else net.core.mask_state(s, keep)
            with autocast(device, precision):
                out = net.forward_seq(obs, labels, priv, s0, t)
            states[gi] = out["state"]
            has_loc, sel_len = b.label_extras(start, end, t)
            acc.score(out, hg, has_loc, sel_len)
            if out["win_open"] is not None:
                stream.decide(b, start, end, t, out["win_open"].float().cpu().numpy())
    rep = acc.report(ckpt_dir, split, stream.games, seed if games is not None else None)
    tail = json.loads(stream.finish())
    rep["win_open_mse"], rep["surrender"] = tail["win_open_mse"], tail["surrender"]
    return rep


def main(a) -> int:
    from uranium import _core
    from uranium.device import resolve

    dirs = _core.dataset_dirs(a.dataset, a.shards)
    ckpt_dir = str(_core.resolve_checkpoint(a.checkpoint))
    _, device, precision = resolve(a.backend)
    rep = evaluate(dirs, ckpt_dir, a.split, a.games, a.seed, a.config, device, precision, a.core, a.maps_dir, a.synthetic_maps)
    print(json.dumps(rep, indent=2))
    return 0
