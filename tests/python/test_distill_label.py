"""044-R3 policy labeller `python -m uranium distill-label` with a tiny model on synthetic
`distill_pre` games (044-AC2): the labelled shards read back through the BC loader give the
labeller's head choices and the same model inputs; the same seed gives the same labels, another
seed other draws, argmax ignores the seed; a view's draws do not depend on the other games; the
records carry `label_policy`. CPU."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from uranium import _core, ckpt
from uranium.distill.label import label_games
from uranium.model import action_columns
from uranium.model.net import OBS_KEYS

from test_aux import CPU, make_net

POLICY_KEYS = {"checkpoint", "checkpoint_sha256", "mode", "label_seed", "game_seed", "temperature", "device", "precision", "git_commit", "invalid_labels",
               "config", "config_fingerprint", "seq_len", "lanes", "micro_lanes", "pad_entities", "noise"}


@pytest.fixture
def labeller(tmp_path) -> Path:
    """A tiny untrained checkpoint (its sampled actions are spread over the masks)."""
    d = tmp_path / "ck"
    d.mkdir()
    net = make_net()
    ckpt.save_model(d, net)
    ckpt.save_meta(d, net.cfg, {}, "f32")
    return d


@pytest.fixture
def pre(tmp_path) -> Path:
    d = tmp_path / "distill_pre"
    _core.write_test_shards(str(d), "distill", 4)
    return d


def run(pre: Path, ck: Path, config: Path, root: Path, name: str, mode: str = "sample", seed: int = 5, trace=None) -> dict:
    return label_games(str(pre), str(ck), str(root / name / "out"), str(root / name / "label.jsonl"), str(config), CPU, "f32", mode, seed,
                       synthetic_maps=True, trace=trace)


def read_back(out: Path, config: Path, max_select: int) -> dict:
    """{(game, slot, frame): (label row dict, obs row dict, end)} of every step the BC loader reads
    from `out` (bc-eval's stream: no z, symmetry or race mask), over all splits."""
    cfg = json.loads(_core.bc_config(str(config), "s1")[0])
    t = cfg["seq_len"]
    rows = {}
    for split in ("train", "val", "test"):
        try:
            stream = _core.EvalStream(str(config), [str(out)], split=split, synthetic_maps=True)
        except RuntimeError:
            continue  # no game in this split
        while (b := stream.next()) is not None:
            steps, games, slots = b.lane_steps(), b.lane_games(), b.lane_perspectives()
            for lane in range(b.lanes):
                if not steps[lane]:
                    continue
                g = b.group(lane, lane + 1, t, cfg["pad_entities"], max_select)
                for k, f in enumerate(b.lane_frames(lane)):
                    lab = {key: np.asarray(g[key])[k] for key in ("action", "delay", "queued", "selected", "target_unit", "target_loc", "weights")}
                    obs = {key: np.asarray(g[key])[k] for key in OBS_KEYS}
                    rows[(games[lane], slots[lane], f)] = (lab, obs, g["n"])
    return rows


def collect():
    seen = {}

    def trace(v):
        for k, f in enumerate(v["frames"]):
            key = (v["game"], v["slot"], f)
            assert key not in seen, "one label per step"
            seen[key] = ({h: a[k] for h, a in v["chosen"].items()}, {o: a[k] for o, a in v["obs"].items()}, v["end"])
    return seen, trace


def test_labels_read_back_as_the_labellers_choices_044_ac2(tiny_config, tmp_path, labeller, pre):
    """044-AC2: every decision step read back through the BC loader has the head indices the
    labeller drew (action, delay bucket, queued, selection, target unit, location cell; no
    augmentation) and the inputs the labeller saw; the invalid labels (weight 0) are the records'
    count; every view is labelled and every records line has the `label_policy` fields."""
    chosen, trace = collect()
    rep = run(pre, labeller, tiny_config, tmp_path, "a", trace=trace)
    cols = action_columns()
    net = make_net()
    back = read_back(tmp_path / "a" / "out", tiny_config, net.cfg.heads.max_select)
    lines = [json.loads(x) for x in (tmp_path / "a" / "label.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rep["views"] == len(lines) == 15 and rep["filtered"] == 0 and rep["written"] == 15
    assert rep["steps_written"] == rep["steps_labelled"] == len(chosen) == len(back) == sum(x["steps"] for x in lines)
    assert set(back) == set(chosen)
    zero = 0
    for key, (c, obs, end) in chosen.items():
        lab, obs_back, n_back = back[key]
        a = int(c["action"])
        assert int(lab["action"]) == a and int(lab["delay"]) == int(c["delay"]), key
        if cols["uses_queued"][a]:
            assert int(lab["queued"]) == int(c["queued"]), key
        if cols["uses_select"][a]:
            picks = sorted(int(x) for x in c["selected"] if x < end)
            assert sorted(int(x) for x in lab["selected"] if x < n_back) == picks, key
        if cols["target_kind"][a] == 1:
            assert int(lab["target_unit"]) == int(c["target_unit"]), key
        if cols["target_kind"][a] == 2:
            assert int(lab["target_loc"]) == int(c["target_loc"]), key
        zero += lab["weights"] == 0
        for k in OBS_KEYS:
            assert np.array_equal(obs[k], obs_back[k]), (key, k)
    assert zero == rep["invalid_labels"] == sum(x["invalid"] for x in lines)
    sha = ckpt.policy_version(labeller)
    for x in lines:
        lp = x["label_policy"]
        assert set(lp) == POLICY_KEYS
        assert (lp["checkpoint_sha256"], lp["mode"], lp["label_seed"], lp["device"], lp["precision"], lp["temperature"]) == (sha, "sample", 5, "cpu", "f32", 1.0)
        assert lp["game_seed"] == _core.distill_game_seed(x["replay_sha256"], x["perspective_slot"], 5)
        assert (x["src"], x["status"], x["shard"]) == ("distill", "written", x["input"])
    # The drawn actions are not one action (the sampling is live).
    assert len({int(c["action"]) for c, _, _ in chosen.values()}) > 3


def shard_bytes(out: Path) -> dict:
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("shard_*.bin.zst"))}


def test_same_seed_same_labels_other_seed_other_draws_argmax_ignores_it_044_ac2(tiny_config, tmp_path, labeller, pre):
    """044-R3 ②: the same input, checkpoint, seed and device give the same shards byte for byte;
    another label seed draws other labels; argmax labels do not depend on the seed."""
    runs = {}
    for name, mode, seed in (("s5", "sample", 5), ("s5b", "sample", 5), ("s6", "sample", 6), ("m5", "argmax", 5), ("m6", "argmax", 6)):
        chosen, trace = collect()
        run(pre, labeller, tiny_config, tmp_path, name, mode, seed, trace)
        runs[name] = ({k: int(v[0]["action"]) for k, v in chosen.items()}, shard_bytes(tmp_path / name / "out"))
    assert runs["s5"][1] == runs["s5b"][1] and runs["s5"][1]
    assert runs["s5"][1] != runs["s6"][1]
    differ = sum(runs["s5"][0][k] != runs["s6"][0][k] for k in runs["s5"][0])
    assert differ > len(runs["s5"][0]) // 4, differ
    assert runs["m5"][1] == runs["m6"][1]
    assert runs["m5"][0] != runs["s5"][0]


@pytest.mark.parametrize("variant", [
    {"lanes = 2": "lanes = 3"},
    {"micro_lanes = 1": "micro_lanes = 0"},
    {"lanes = 2": "lanes = 3", "micro_lanes = 1": "micro_lanes = 2"},
    {"pad_entities = 0": "pad_entities = 64"},
    {"seq_len = 4": "seq_len = 8", "micro_lanes = 1": "micro_lanes = 0"},
], ids=["lanes3", "micro0", "micro2", "pad64", "seq8"])
def test_labels_do_not_depend_on_the_batch_shape_044_r3(tiny_config, tmp_path, labeller, pre, variant):
    """044-R3 ②: a step's draw is a hash of (view seed, frame, head draw, column), so the labelled
    shards are byte-identical whatever the lane count, the lanes per forward (games sharing a
    forward: the group's entity count N sets the selection / target widths), the entity padding or
    the seq_len chunks (CPU, 032-R1)."""
    base = run(pre, labeller, tiny_config, tmp_path, "base")
    text = tiny_config.read_text(encoding="utf-8")
    for a, b in variant.items():
        assert a in text
        text = text.replace(a, b)
    other_cfg = tmp_path / "bc_other.toml"
    other_cfg.write_text(text, encoding="utf-8")
    other = run(pre, labeller, other_cfg, tmp_path, "other")
    shape = ("lanes", "micro_lanes", "pad_entities", "seq_len")
    assert [other["label_policy"][k] for k in shape] != [base["label_policy"][k] for k in shape]
    assert shard_bytes(tmp_path / "other" / "out") == shard_bytes(tmp_path / "base" / "out")


def test_row_noise_depends_only_on_the_row_key_and_column_044_r3():
    """The noise of a column does not depend on the width drawn, the other rows or the row order;
    the selection's END has its own column (not N); different keys, draws and columns differ."""
    from uranium.distill.noise import RowNoise, row_keys

    keys = row_keys([5, 5, 2 ** 62 + 7], [1, 9, 1], "cpu")
    assert len(set(keys.tolist())) == 3

    def draws(k, n):
        nz = RowNoise(k, n, 2)
        return [nz.uniform((k.shape[0], w), "cpu") for w in (10, 16, 2, n + 1, n + 1, n, 30)]

    a, b = draws(keys, 4), draws(keys[[2, 0]], 9)
    for i, (x, y) in enumerate(zip(a, b)):
        w = min(x.shape[1], y.shape[1]) - (1 if i in (3, 4) else 0)
        assert torch.equal(x[[2, 0], :w], y[:, :w]), i
        if i in (3, 4):
            assert torch.equal(x[[2, 0], -1], y[:, -1]), "END column"
    assert not torch.equal(a[0][0], a[0][1]) and not torch.equal(a[3][:, :4], a[4][:, :4])
    assert float(a[6].min()) > 0.0 and float(a[6].max()) < 1.0
    with pytest.raises(ValueError, match="selection draw"):
        nz = RowNoise(keys, 4, 2)
        [nz.uniform((3, w), "cpu") for w in (10, 16, 2, 4)]


def test_inputs_and_outputs_are_checked_044_r3(tiny_config, tmp_path, labeller, pre, monkeypatch):
    """Refusals before any labelling: records named like build records, an output folder with
    shards, argmax output inside the BC shards; and a folder whose games are not distill_pre."""
    with pytest.raises(RuntimeError, match="resume scan"):
        label_games(str(pre), str(labeller), str(tmp_path / "o"), str(tmp_path / "distill_d1.jsonl"), str(tiny_config), CPU, "f32", "sample", 1, synthetic_maps=True)
    monkeypatch.setenv("URANIUM_SHARED_DIR", str(tmp_path / "shared"))
    bc_out = tmp_path / "shared" / "datasets" / "replays" / "shards" / "distill_m2" / "x"
    with pytest.raises(RuntimeError, match="never training data"):
        label_games(str(pre), str(labeller), str(bc_out), str(tmp_path / "l.jsonl"), str(tiny_config), CPU, "f32", "argmax", 1, synthetic_maps=True)
    other = tmp_path / "varied"
    _core.write_test_shards(str(other), "varied", 4)
    with pytest.raises(RuntimeError, match="044-R3"):
        label_games(str(other), str(labeller), str(tmp_path / "o2"), str(tmp_path / "l2.jsonl"), str(tiny_config), CPU, "f32", "sample", 1, synthetic_maps=True)
    with pytest.raises(RuntimeError, match="already holds"):
        label_games(str(pre), str(labeller), str(other), str(tmp_path / "l3.jsonl"), str(tiny_config), CPU, "f32", "sample", 1, synthetic_maps=True)


def test_records_continue_with_the_same_policy_044_r5(tiny_config, tmp_path, labeller, pre):
    """044-R5 ③ top-up: with the round's records, a later pass labels only the new input file into
    the same output folder and appends its lines; another label seed (or checkpoint, mode) is
    refused; the files labelled before are not touched."""
    import shutil

    root = tmp_path / "round"
    first = run(pre, labeller, tiny_config, tmp_path, "round")
    before = shard_bytes(root / "out")
    with pytest.raises(RuntimeError, match="nothing to label"):
        run(pre, labeller, tiny_config, tmp_path, "round")
    more = tmp_path / "more"
    _core.write_test_shards(str(more), "distill", 15)
    shutil.copytree(more / "p0", pre / "p9")
    with pytest.raises(ValueError, match="label_seed"):
        run(pre, labeller, tiny_config, tmp_path, "round", seed=6)
    second = run(pre, labeller, tiny_config, tmp_path, "round")
    assert (second["input_files"], second["views_before"], second["views"]) == (1, first["views"], 15)
    after = shard_bytes(root / "out")
    assert {k: v for k, v in after.items() if k != "p9/shard_00000.bin.zst"} == before and "p9/shard_00000.bin.zst" in after
    lines = [json.loads(x) for x in (root / "label.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 30 and [x["input"] for x in lines[15:]] == ["p9/shard_00000.bin.zst"] * 15


def test_cli_writes_the_summary_044_r3(tiny_config, tmp_path, labeller, pre, python_cmd):
    """`python -m uranium distill-label` end to end: shards, records and the summary file. In a
    child process: the command loads `.env` into its environment (001-R9)."""
    import os
    import subprocess
    import sys

    out, rec = tmp_path / "cli" / "out", tmp_path / "cli" / "label_d1.jsonl"
    env = {**os.environ, "URANIUM_SHARD_STAGING_DIR": str(tmp_path / "staging"), "PYTHONPATH": os.pathsep.join(sys.path)}
    subprocess.run([*python_cmd, "distill-label", "--checkpoint", str(labeller), "--label-seed", "3", "--in", str(pre), "--out", str(out), "--records", str(rec),
                    "--config", str(tiny_config), "--backend", "cpu", "--synthetic-maps"], check=True, capture_output=True, env=env)
    s = json.loads((tmp_path / "cli" / "label_d1.summary.json").read_text(encoding="utf-8"))
    assert s["views"] == 15 and s["label_policy"]["label_seed"] == 3 and s["actions_by_minute"]["0"]
    assert sum(sum(b.values()) for b in s["actions_by_minute"].values()) == s["steps_labelled"]
    assert shard_bytes(out) and len(rec.read_text(encoding="utf-8").splitlines()) == 15
