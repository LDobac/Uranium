import argparse
import json
from pathlib import Path

import numpy as np
import torch

NOISE_VERSION = "rowhash-1"
FRAMES_PER_MINUTE = 24 * 60


def add_parser(sub) -> None:
    p = sub.add_parser("distill-label", help="policy labels for distill_pre shards (self-game distillation)")
    p.add_argument("--checkpoint", required=True, help="labeller checkpoint folder, or a run folder (its LATEST)")
    p.add_argument("--mode", choices=["sample", "argmax"], default="sample", help="sample: one draw at temperature 1 (training data); argmax: tool option only")
    p.add_argument("--label-seed", type=int, required=True, help="seed of the per-view draws (with the replay hash and perspective)")
    p.add_argument("--in", dest="input", required=True, help="distill_pre shard folder (uranium-replay build --src distill --out)")
    p.add_argument("--out", required=True, help="new folder for the labelled shards, e.g. <shared>/datasets/replays/shards/distill_m2/d<R>")
    p.add_argument("--records", required=True, help="new records file (.jsonl, not named distill*)")
    p.add_argument("--config", default="config/bc.toml", help="BC config: lanes and seq_len of the pass (data settings as bc-eval)")
    p.add_argument("--backend", choices=["cuda", "rocm", "cpu"])
    p.add_argument("--micro-lanes", type=int, help="lanes per forward (default: the config's micro_lanes, as bc-eval)")
    p.add_argument("--threads", type=int, help="shard check threads (default URANIUM_BC_FEED_THREADS)")
    p.add_argument("--maps-dir", help=argparse.SUPPRESS)
    p.add_argument("--synthetic-maps", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(fn=main)


def _rows(x, sl):
    if isinstance(x, torch.Tensor):
        return x[sl]
    if isinstance(x, dict):
        return {k: _rows(v, sl) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_rows(v, sl) for v in x)
    return x


def _host(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().to("cpu", torch.int64).numpy())


@torch.no_grad()
def label_games(input_dir: str, ckpt_dir: str, out: str, records: str, config: str, device, precision: str, mode: str, label_seed: int, backend: str = "cpu",
                micro_lanes: int | None = None, threads: int | None = None, maps_dir=None, synthetic_maps: bool = False, trace=None) -> dict:
    from uranium import _core, ckpt
    from uranium.device import autocast
    from uranium.diag.summary import git_commit
    from uranium.model import to_obs
    from uranium.distill.noise import RowNoise, row_keys
    from uranium.model.net import OBS_KEYS

    todo = _core.distill_check(input_dir, out, records, mode)
    prior = json.loads(_core.distill_records(records))
    sha = ckpt.policy_version(Path(ckpt_dir))
    if prior["policy"] is not None:
        for key, want in (("checkpoint_sha256", sha), ("mode", mode), ("label_seed", label_seed), ("noise", NOISE_VERSION)):
            if prior["policy"].get(key) != want:
                raise ValueError(f"--records {records} continues a labelling with {key} {prior['policy'].get(key)!r}, this run {want!r}: "
                                 "one round folder has one labeller policy (044-R3); give a new --out and --records")
    cfg_json, fingerprint = _core.bc_config(config, "s1")
    cfg = json.loads(cfg_json)
    net, _ = ckpt.load_model(Path(ckpt_dir), device)
    net.eval()
    temp = 1.0 if mode == "sample" else 0.0
    t, lanes = cfg["seq_len"], cfg["lanes"]
    mlanes = micro_lanes if micro_lanes is not None else cfg["micro_lanes"]
    g = lanes if mlanes == 0 else min(mlanes, lanes)
    groups = -(-lanes // g)
    max_select = net.cfg.heads.max_select
    policy = {
        "checkpoint": str(ckpt_dir), "checkpoint_sha256": sha, "mode": mode, "label_seed": label_seed, "noise": NOISE_VERSION,
        "temperature": temp, "device": backend, "precision": precision, "git_commit": git_commit(),
        "config": str(config), "config_fingerprint": fingerprint, "seq_len": t, "lanes": lanes, "micro_lanes": g, "pad_entities": cfg["pad_entities"],
    }
    lab = _core.DistillLabeller(config, input_dir, label_seed, maps_dir=maps_dir, synthetic_maps=synthetic_maps, only=todo)
    lab.open(out, records, json.dumps(policy), threads)
    states = [None] * groups
    names = _action_names()
    bands: dict = {}
    steps_total = 0
    while (b := lab.next()) is not None:
        steps = b.lane_steps()
        seeds = [None if x is None else x[1] for x in lab.lanes(b)]
        frames_of = {lane: b.lane_frames(lane) for lane in range(lanes) if steps[lane]}
        views = list(zip(b.lane_games(), b.lane_perspectives()))
        for gi in range(groups):
            start, end = gi * g, min((gi + 1) * g, lanes)
            live = [lane for lane in range(start, end) if steps[lane]]
            if not live:
                states[gi] = None
                continue
            hg = b.group(start, end, t, cfg["pad_entities"], max_select)
            obs = to_obs(hg, device)
            keep = torch.as_tensor(hg["keep"]).to(device).bool()
            s = states[gi]
            s0 = net.init_state(hg["b"], device) if s is None else net.mask_state(s, keep)
            starts = [(lane - start) * t for lane in live]
            idx = torch.tensor([r0 + k for r0, lane in zip(starts, live) for k in range(steps[lane])], dtype=torch.int64, device=device)
            with autocast(device, precision):
                h, e, sp, sc, _, state = net.trunk_seq(obs, s0, t)
                u = net._bypass(h, sc, e, obs)
                states[gi] = state
                n_ent = e.shape[1]
                noise = None
                if temp > 0:
                    keys = row_keys([seeds[lane] for lane in live for _ in range(steps[lane])], [f for lane in live for f in frames_of[lane]], device)
                    noise = RowNoise(keys, n_ent, max_select, net.cfg.input.actions, net.cfg.heads.delay_buckets, net.cfg.input.map_size ** 2)
                o = net.heads(None, temp, h[idx], sc["c"][idx], e[idx], _rows(sp, idx), _rows(obs, idx), net.table, noise, u=None if u is None else u[idx])
                if noise is not None:
                    noise.finished()
            chosen = {key: _host(v) for key, v in o["chosen"].items()}
            at = 0
            for lane, r0 in zip(live, starts):
                k = steps[lane]
                ch = {key: v[at:at + k] for key, v in chosen.items()}
                at += k
                lab.put(b, lane, ch["action"], ch["delay"], ch["queued"], ch["selected"], n_ent, ch["target_unit"], ch["target_loc"])
                steps_total += k
                for f, a in zip(frames_of[lane], ch["action"].tolist()):
                    band = bands.setdefault(str(f // FRAMES_PER_MINUTE), {})
                    name = names.get(a, str(a))
                    band[name] = band.get(name, 0) + 1
                if trace is not None:
                    trace({"game": views[lane][0], "slot": views[lane][1], "frames": frames_of[lane], "chosen": ch, "end": n_ent,
                           "obs": {key: np.asarray(hg[key])[r0:r0 + k] for key in OBS_KEYS}})
        lab.flush()
    written = json.loads(lab.finish())
    summary = {"input": str(input_dir), "out": str(out), "records": str(records), "input_files": len(todo), "views_before": prior["views"],
               "games": lab.games, "steps_labelled": steps_total, **written,
               "label_policy": policy, "actions_by_minute": dict(sorted(bands.items(), key=lambda kv: int(kv[0])))}
    return summary


def _action_names() -> dict:
    from uranium.diag import tables as tb

    return {a["id"]: a["name"] for a in tb._rows("actions")}


def main(a) -> int:
    from uranium import _core
    from uranium.device import configure, resolve

    ckpt_dir = str(_core.resolve_checkpoint(a.checkpoint))
    backend, device, precision = resolve(a.backend)
    configure(backend)
    summary = label_games(a.input, ckpt_dir, a.out, a.records, a.config, device, precision, a.mode, a.label_seed, backend, a.micro_lanes, a.threads, a.maps_dir, a.synthetic_maps)
    text = json.dumps(summary, indent=2)
    print(text)
    rec = Path(a.records)
    side = rec.with_name(rec.stem + ".summary.json")
    tmp = side.with_name(side.name + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(side)
    return 0
