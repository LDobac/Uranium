"""`python -m uranium distill-label` (044-R3, 040-R24): the policy labeller of the self-game
distillation. It reads the `distill_pre` shards of `uranium-replay build --src distill` (044-R2:
one step per playback frame at the game start and the model's decisions, every label an `unknown`
slot, the model's own decisions in `memory_action`) and labels every step with the labeller
checkpoint (`m2-s2e2/ckpt-11387`, AMB-044-2).

① Input: exactly what the BC loader gives the trainer, through `uranium._core` (030 features, 034
masks, 041-R28 ③ memory written from `memory_action` before each observation, never from a label,
map precomputes, `last_failed` 0), read as `bc-eval` reads a split: no z (`z_present` false), no
symmetry, no enemy race mask, no chain-start observation; the core state runs from each view's
first step to its last.
② Answer: `--mode sample` draws one action per step from the labeller's distribution (temperature
1, the heads' autoregressive order and masks as in a live session, Gumbel-max, 032-R9), with a
torch RNG per view seeded by a hash of (`replay_sha256`, perspective, `--label-seed`)
(`_core.distill_game_seed`); the draws of a view are those of its own steps in order, `seq_len`
rows per head call. `--mode argmax` (temperature 0) is a tool option and never training data.
③ The head values become 040 `action_label`s in Rust (`label_of_choice`): action id, the delay
bucket's lower bound, queued, the selection's unit ids in pick order, the target unit id, the
location cell's tile; the BC loader reads them back as the same head indices (044-AC2).
④ The 040-R19 invalid-label filter (20%) runs on the labelled views as the build runs it, and the
shards are written with the 040-R17 verified copy, one output file per input file (same relative
path under `--out`). ⑤ One records line per view with `label_policy` (checkpoint path and sha256,
mode, label seed, the view's seed, device, precision, git commit, invalid labels, config).

An existing `--records` file continues its labelling (044-R5 ③: the round's top-up games are built
into the same `distill_pre/d<R>` and labelled into the same `--out`): the input files it names are
skipped, the others are labelled with the same checkpoint, mode and label seed (else an error),
and their lines are appended. A run stopped between files continues the same way.

Records are not named `distill*.jsonl` (the `--src distill` build's resume scan reads those). A
summary (views, steps, invalid labels, action types per minute of game time) is printed and
written next to the records (`<records stem>.summary.json`)."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Game minutes per band of the action-type summary (044 risk: labels in states the labeller never saw).
FRAMES_PER_MINUTE = 24 * 60


def add_parser(sub) -> None:
    p = sub.add_parser("distill-label", help="044-R3 policy labels for distill_pre shards (self-game distillation)")
    p.add_argument("--checkpoint", required=True, help="labeller checkpoint folder, or a 061 run folder (its LATEST)")
    p.add_argument("--mode", choices=["sample", "argmax"], default="sample", help="sample: one draw at temperature 1 (training data); argmax: tool option only")
    p.add_argument("--label-seed", type=int, required=True, help="seed of the per-view draws (with the replay hash and perspective)")
    p.add_argument("--in", dest="input", required=True, help="distill_pre shard folder (uranium-replay build --src distill --out)")
    p.add_argument("--out", required=True, help="new folder for the labelled shards, e.g. <shared>/datasets/replays/shards/distill_m2/d<R>")
    p.add_argument("--records", required=True, help="new records file (.jsonl, not named distill*)")
    p.add_argument("--config", default="config/bc.toml", help="BC config: lanes and seq_len of the pass (data settings as bc-eval)")
    p.add_argument("--backend", choices=["cuda", "rocm", "cpu"])
    p.add_argument("--micro-lanes", type=int, help="lanes per forward (default: the config's micro_lanes, as bc-eval)")
    p.add_argument("--threads", type=int, help="040-R19 check threads (default URANIUM_BC_FEED_THREADS)")
    p.add_argument("--maps-dir", help=argparse.SUPPRESS)
    p.add_argument("--synthetic-maps", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(fn=main)


def _rows(x, sl):
    """Rows `sl` of every tensor in a (nested) dict / list of encoder outputs or observations."""
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
    """Labels every step of the `distill_pre` games under `input_dir` and writes the shards and
    records (044-R3). `trace(view)`, tests only: called per lane chunk with `game` / `slot` /
    `frames`, the chosen head values `chosen` (numpy, one row per step; the selection's END is
    `end`) and the chunk's observation rows `obs` (numpy). Returns the summary."""
    from uranium import _core, ckpt
    from uranium.device import autocast
    from uranium.diag.summary import git_commit
    from uranium.model import to_obs
    from uranium.model.net import OBS_KEYS

    # The input files left to label: an existing records file is continued (044-R5 ③ top-up into
    # the same round folder, or a run stopped between files), with the same labeller policy.
    todo = _core.distill_check(input_dir, out, records, mode)
    prior = json.loads(_core.distill_records(records))
    sha = ckpt.policy_version(Path(ckpt_dir))
    if prior["policy"] is not None:
        for key, want in (("checkpoint_sha256", sha), ("mode", mode), ("label_seed", label_seed)):
            if prior["policy"].get(key) != want:
                raise ValueError(f"--records {records} continues a labelling with {key} {prior['policy'].get(key)!r}, this run {want!r}: "
                                 "one round folder has one labeller policy (044-R3); give a new --out and --records")
    cfg_json, fingerprint = _core.bc_config(config, "s1")
    cfg = json.loads(cfg_json)
    net, _ = ckpt.load_model(Path(ckpt_dir), device)
    net.eval()  # 032 §2: no dropout
    temp = 1.0 if mode == "sample" else 0.0
    t, lanes = cfg["seq_len"], cfg["lanes"]
    mlanes = micro_lanes if micro_lanes is not None else cfg["micro_lanes"]
    g = lanes if mlanes == 0 else min(mlanes, lanes)
    groups = -(-lanes // g)
    lab = _core.DistillLabeller(config, input_dir, label_seed, maps_dir=maps_dir, synthetic_maps=synthetic_maps, only=todo)
    states = [None] * groups
    gens: list = [None] * lanes  # per lane: (game key, torch.Generator) of the game in it
    names = _action_names()
    bands: dict = {}
    steps_total = 0
    while (b := lab.next()) is not None:
        steps = b.lane_steps()
        for lane, info in enumerate(lab.lanes(b)):
            if info is None:
                gens[lane] = None
            elif gens[lane] is None or gens[lane][0] != info[0]:
                gen = torch.Generator(device=device)
                gen.manual_seed(info[1])
                gens[lane] = (info[0], gen)
        frames_of = {lane: b.lane_frames(lane) for lane in range(lanes) if steps[lane]}
        views = list(zip(b.lane_games(), b.lane_perspectives()))
        for gi in range(groups):
            start, end = gi * g, min((gi + 1) * g, lanes)
            if all(steps[lane] == 0 for lane in range(start, end)):
                states[gi] = None
                continue
            hg = b.group(start, end, t, cfg["pad_entities"], net.cfg.heads.max_select)
            obs = to_obs(hg, device)
            keep = torch.as_tensor(hg["keep"]).to(device).bool()
            s = states[gi]
            s0 = net.init_state(hg["b"], device) if s is None else net.mask_state(s, keep)
            with autocast(device, precision):
                h, e, sp, sc, _, state = net.trunk_seq(obs, s0, t)
                u = net._bypass(h, sc, e, obs)
                states[gi] = state
                n_ent = e.shape[1]  # the selection's END index (032 §2.5 #4)
                for lane in range(start, end):
                    k = steps[lane]
                    if k == 0:
                        continue
                    r0 = (lane - start) * t
                    rows = slice(r0, r0 + k)
                    # The heads of one view's rows with that view's RNG: its draws never depend on
                    # the other lanes (044-R3 ②, per-game seed).
                    gen = gens[lane][1] if temp > 0 else None
                    o = net.heads(None, temp, h[rows], sc["c"][rows], e[rows], _rows(sp, rows), _rows(obs, rows), net.table, gen, u=None if u is None else u[rows])
                    ch = {key: _host(v) for key, v in o["chosen"].items()}
                    lab.put(b, lane, ch["action"], ch["delay"], ch["queued"], ch["selected"], n_ent, ch["target_unit"], ch["target_loc"])
                    steps_total += k
                    for f, a in zip(frames_of[lane], ch["action"].tolist()):
                        band = bands.setdefault(str(f // FRAMES_PER_MINUTE), {})
                        name = names.get(a, str(a))
                        band[name] = band.get(name, 0) + 1
                    if trace is not None:
                        trace({"game": views[lane][0], "slot": views[lane][1], "frames": frames_of[lane], "chosen": ch, "end": n_ent,
                               "obs": {key: np.asarray(hg[key])[r0:r0 + k] for key in OBS_KEYS}})
    policy = {
        "checkpoint": str(ckpt_dir), "checkpoint_sha256": sha, "mode": mode, "label_seed": label_seed,
        "temperature": temp, "device": backend, "precision": precision, "git_commit": git_commit(),
        "config": str(config), "config_fingerprint": fingerprint, "seq_len": t, "lanes": lanes, "micro_lanes": g,
    }
    written = json.loads(lab.write(out, records, json.dumps(policy), threads))
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
