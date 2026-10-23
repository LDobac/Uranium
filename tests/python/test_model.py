"""032 model properties (the Burn `uranium-model/tests/model.rs` checks, AMB-000-4): sizes, masks,
privileged isolation, teacher forcing vs sampling, step vs sequence, location rows, win_open.
Inputs are the tiny Burn-parity batch (tests/fixtures/burn_parity)."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from uranium.model import UraniumNet, UraniumNetConfig, action_columns, input_dims, to_labels, to_obs
from uranium.model.config import CoreConfig, ValueConfig
from uranium.model.heads import gather_cap, gather_loc

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "burn_parity"
CORES = ["tiny-lstm", "tiny-gtrxl"]


def tiny(name: str):
    meta = json.loads((FIX / name / "meta.json").read_text())
    inp = load_file(FIX / name / "inputs.safetensors")
    cfg = UraniumNetConfig.from_dict(meta["config"])
    return cfg, meta, to_obs(inp, "cpu"), to_labels(inp, "cpu"), inp["privileged"]


def net_of(cfg, seed=3):
    torch.manual_seed(seed)
    return UraniumNet(cfg, action_columns()).eval()


@pytest.mark.parametrize("kind,want", [("Lstm", 11_343_764), ("Gtrxl", 21_715_364)])
def test_default_size_032_r2(kind, want):
    """The default config (Burn `UraniumNetConfig::new()`, as the benches) has the Burn sizes;
    the training config takes its input widths from the feature schema (`input_dims`)."""
    cfg = UraniumNetConfig(core=CoreConfig(kind=kind))
    assert sum(p.numel() for p in UraniumNet(cfg, action_columns()).parameters()) == want
    real = UraniumNetConfig(input=input_dims(), core=CoreConfig(kind=kind))
    assert 10_000_000 <= sum(p.numel() for p in UraniumNet(real, action_columns()).parameters()) <= 40_000_000


@pytest.mark.parametrize("name", CORES)
def test_masks_hold_032_ac4(name):
    """Sampled actions always satisfy the masks (032-R4, 034-R7, 034-R9)."""
    cfg, _, obs, _, _ = tiny(name)
    net = net_of(cfg)
    cols = action_columns()
    b, n = obs["entity_mask"].shape
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for _ in range(30):
            out = net.act(obs, net.init_state(b), 1.0, g)
            a = out["action"]
            act = a["action"]
            assert obs["type_mask"][torch.arange(b), act].all(), "masked action_type"
            info = net.table.info(act)
            sel_ok = gather_cap(obs["unit_cap"], info["select_cap"]) & (obs["entity_cat"][..., 1] == 0) & obs["entity_mask"]
            for r in range(b):
                picked = [i for i in a["selected"][r].tolist() if i != n]
                if cols["uses_select"][act[r]]:
                    assert picked and len(picked) <= cols["max_select"][act[r]] and len(set(picked)) == len(picked)
                    assert all(sel_ok[r, i] for i in picked), "ineligible selection"
                    buildings = sum(bool(obs["is_building"][r, i]) for i in picked)
                    assert buildings == 0 or len(picked) == 1, "034-R7 building with others"
                else:
                    assert not picked
                if cols["target_kind"][act[r]] == 1:
                    tgt_ok = gather_cap(obs["unit_cap"], info["target_cap"]) & obs["entity_mask"]
                    assert tgt_ok[r, a["target_unit"][r]]
                if cols["target_kind"][act[r]] == 2:
                    assert gather_loc(obs["loc_class_mask"], info["loc_class"])[r, a["target_loc"][r]]


@pytest.mark.parametrize("name", CORES)
def test_privileged_only_reaches_value_032_r11(name):
    cfg, meta, obs, labels, priv = tiny(name)
    net = net_of(cfg)
    priv = priv.clone().requires_grad_(True)
    out = net.forward_seq(obs, labels, priv, net.init_state(meta["b"]), meta["t"])
    policy = sum(v.sum() for v in out["logp"].values())
    (g,) = torch.autograd.grad(policy, priv, allow_unused=True, retain_graph=True)
    assert g is None or not g.any(), "privileged input reached a policy head"
    (gv,) = torch.autograd.grad(out["value"].sum(), priv)
    assert gv.abs().sum() > 0, "privileged input reaches the value heads"
    if out["win_open"] is not None:
        (go,) = torch.autograd.grad(out["win_open"].sum(), priv, allow_unused=True)
        assert go is None or not go.any(), "032-R18: win_open takes no privileged input"


@pytest.mark.parametrize("name", CORES)
def test_teacher_matches_sampling_032_r3(name):
    """Teacher forcing with the sampled (argmax) action gives the sampling pass's log-probs."""
    cfg, _, obs, _, _ = tiny(name)
    net = net_of(cfg)
    b = obs["entity_mask"].shape[0]
    with torch.no_grad():
        s = net.act(obs, net.init_state(b), 0.0)
        out = net.forward_seq(obs, s["action"], None, net.init_state(b), 1)
    teacher = torch.stack([out["logp"][k] for k in ("action", "delay", "queued", "selected", "target_unit", "target_loc")], 1)
    assert torch.allclose(teacher, s["head_log_prob"], atol=1e-5)


@pytest.mark.parametrize("name", CORES)
def test_step_matches_sequence_032_r5(name):
    cfg, meta, obs, _, _ = tiny(name)
    net = net_of(cfg)
    b, t = meta["b"], meta["t"]
    with torch.no_grad():
        core_in, *_ = net.encode(obs)
        x = core_in.view(b, t, -1)
        seq, _ = net.core.forward_seq(x, net.init_state(b))
        s = net.init_state(b)
        steps = []
        for k in range(t):
            h, s = net.core.step(x[:, k], s)
            steps.append(h)
    assert torch.allclose(seq, torch.stack(steps, 1), atol=1e-5)


@pytest.mark.parametrize("name", CORES)
def test_location_on_used_rows_matches_all_rows_032_r10(name):
    cfg, meta, obs, labels, priv = tiny(name)
    net = net_of(cfg)
    with torch.no_grad():
        a = net.forward_seq(obs, labels, priv, net.init_state(meta["b"]), meta["t"])["logp"]["target_loc"]
        net.heads.location_all_rows = True
        b = net.forward_seq(obs, labels, priv, net.init_state(meta["b"]), meta["t"])["logp"]["target_loc"]
    assert torch.allclose(a, b, atol=1e-6)


def test_win_open_head_leaves_the_rest_unchanged_032_ac9():
    """The head does not move the other modules' initial weights (032-R18 ③)."""
    cfg, *_ = tiny("tiny-gtrxl")
    off = UraniumNetConfig.from_dict(cfg.to_dict())
    off.value = ValueConfig(hidden=cfg.value.hidden, privileged=cfg.value.privileged, win_open=False)
    a, b = net_of(cfg, 9).state_dict(), net_of(off, 9).state_dict()
    assert any(k.startswith("value.open.") for k in a) and not any(k.startswith("value.open.") for k in b)
    for k, v in b.items():
        assert torch.equal(a[k], v), k
