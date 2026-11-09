"""032 §2.8 model 2 plumbing (032-R21..R24, AC13, AC14; 041-R23 ④; 042-R21 ①): the `[model]`
switches and their config handling, the `[aux]` key / module start check, the hook points the
model 2 modules plug into (`Variants.bypass`, `core_skip`, `aux_priv`, `aux_jepa`) and the
`forward_seq(aux=...)` outputs. The modules themselves come later; here the hooks are exercised
with stand-in modules, and every switch off must leave the model as it was (Checkpoint 1)."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open

from uranium import ckpt
from uranium.model import UraniumNet, UraniumNetConfig, action_columns
from uranium.model.config import AUX_JEPA_SLOTS, MODEL2_KEYS, VariantConfig, check_aux
from uranium.model.net import join_parts

from test_variants import HEADS, SEG, VARIANTS, net_of, rows_at, tiny

ROOT = Path(__file__).resolve().parents[2]
RELEASES = ROOT / "releases"
ALL_ARCH = VARIANTS["all"]


def test_switches_default_off_032_ac13():
    v = VariantConfig()
    assert not v.any() and not v.model2() and not v.aux() and v.tokens() == 0
    assert VariantConfig.from_table(None) == v == VariantConfig.from_table({})
    # The model 2 switches add no entity token.
    m2 = VariantConfig(bypass=True, spatial_preact=True, core_skip=True, aux_priv=True, aux_jepa=[1, 8, 32])
    assert m2.any() and m2.model2() and not m2.arch() and m2.aux() and m2.tokens() == 0
    assert replace(ALL_ARCH, aux_priv=True).tokens() == ALL_ARCH.tokens() == 2
    for k in MODEL2_KEYS:
        assert getattr(v, k) in (False, []), k


def test_model_table_032_r21_r24():
    """`[model]` accepts the 032 §2.8 keys (bool, and the `aux_jepa` integer list) and refuses
    wrong types, bad k lists and keys outside the spec (032-AC13: unknown key / bad value = start
    error), F2 `recency_init` included (not adopted, 032-R23)."""
    v = VariantConfig.from_table({"terrain": True, "outer": 32, "bypass": True, "spatial_preact": True, "core_skip": True, "aux_priv": True, "aux_jepa": [1, 8, 32]})
    assert v == VariantConfig(terrain=True, outer=32, bypass=True, spatial_preact=True, core_skip=True, aux_priv=True, aux_jepa=[1, 8, 32])
    assert VariantConfig.from_table({"aux_jepa": []}) == VariantConfig()
    bad = [
        {"bypass": 1}, {"core_skip": "yes"}, {"aux_priv": 1}, {"aux_jepa": 8}, {"aux_jepa": (1, 8)}, {"aux_jepa": [1.0]}, {"aux_jepa": [True]},
        {"aux_jepa": [0]}, {"aux_jepa": [-1]}, {"aux_jepa": [8, 8]}, {"aux_jepa": [1, 2, 4, 8]}, {"bypas": True},
    ]
    for t in bad:
        with pytest.raises(ValueError):
            VariantConfig.from_table(t)
    assert len(VariantConfig.from_table({"aux_jepa": list(range(1, AUX_JEPA_SLOTS + 1))}).aux_jepa) == AUX_JEPA_SLOTS
    for k in ({"recency_init": True}, {"recency_init": False}):
        with pytest.raises(ValueError, match="unknown"):
            VariantConfig.from_table(k)


def test_config_dict_keeps_old_forms_032_ac13():
    """`to_dict` (the checkpoint meta) leaves the model 2 keys out while off: the default model
    has no `variant`, a §2.7 model the four keys of before; on, they round-trip."""
    assert "variant" not in UraniumNetConfig().to_dict()
    assert UraniumNetConfig(variant=ALL_ARCH).to_dict()["variant"] == {"terrain": True, "ctx_token": True, "attn_pool": True, "outer": SEG}
    on = UraniumNetConfig(variant=VariantConfig(bypass=True, aux_jepa=[1, 8]))
    d = on.to_dict()["variant"]
    assert d == {"terrain": False, "ctx_token": False, "attn_pool": False, "outer": 0, "bypass": True, "aux_jepa": [1, 8]}
    back = UraniumNetConfig.from_dict(json.loads(json.dumps(on.to_dict())))
    assert back.variant == on.variant and back.to_dict() == on.to_dict()


@pytest.mark.parametrize("release", ["checkpoint-1", "checkpoint-1-rl"])
def test_checkpoint1_meta_round_trip_032_ac13(release):
    """Checkpoint 1's model config reads and writes back unchanged (no model 2 key appears)."""
    meta = json.loads((RELEASES / release / "meta.json").read_text(encoding="utf-8"))
    cfg = UraniumNetConfig.from_dict(meta["config"])
    assert not cfg.variant.model2() and cfg.variant.arch()
    assert cfg.to_dict() == meta["config"]


def test_checkpoint1_loads_with_the_same_parameters_032_ac13():
    """Checkpoint 1 loads strictly into the new code: the model's parameter list is exactly the
    weights file's (no model 2 module is built with the switches off)."""
    d = RELEASES / "checkpoint-1"
    net, meta = ckpt.load_model(d)
    with safe_open(str(d / ckpt.WEIGHTS_FILE), "pt") as f:
        keys = set(f.keys())
    assert set(net.state_dict()) == keys
    v = net.variant
    assert v is not None and v.bypass is None and v.core_skip is None and v.aux_priv is None and v.aux_jepa is None


def test_check_aux_041_r23_042_r21():
    """041-R23 ④ / 042-R21 ①: an [aux] key without its module, or a module without its key, is a
    start error; weights must be finite and >= 0; 0 is a valid key (the loss is not added)."""
    off = VariantConfig()
    assert check_aux(off, None) == {} and check_aux(off, {}) == {}
    priv, jepa = VariantConfig(aux_priv=True), VariantConfig(aux_jepa=[1, 8, 32])
    both = VariantConfig(aux_priv=True, aux_jepa=[1])
    assert check_aux(priv, {"priv": 0.2}) == {"priv": 0.2}
    assert check_aux(jepa, {"jepa": 0}) == {"jepa": 0.0}
    assert check_aux(both, {"priv": 0.2, "jepa": 0.1}) == {"priv": 0.2, "jepa": 0.1}
    for v, aux in [(off, {"priv": 0.2}), (off, {"jepa": 0.0}), (priv, {}), (priv, {"jepa": 0.1}), (jepa, {"priv": 0.1}), (both, {"priv": 0.1})]:
        with pytest.raises(ValueError, match="041-R23"):
            check_aux(v, aux)
    for w in (-0.1, float("nan"), float("inf"), True, "0.2"):
        with pytest.raises(ValueError):
            check_aux(priv, {"priv": w})
    with pytest.raises(ValueError, match="unknown"):
        check_aux(priv, {"priv": 0.2, "privileged": 0.2})


def test_trainers_refuse_aux_mismatch_041_r23():
    """The BC trainer and the RL learner check [aux] against the model at start."""
    from uranium.bc.trainer import BcTrainer
    from uranium.rl.learner import Learner

    cfg, *_ = tiny()
    net = net_of(cfg)
    bc = {"lr": 1e-3, "weight_decay": 0.0, "seq_len": 4, "lanes": 2, "micro_lanes": 0, "value": {}}
    with pytest.raises(ValueError, match="041-R23"):
        BcTrainer(dict(bc, aux={"priv": 0.2}), net, torch.device("cpu"), "f32")
    assert BcTrainer(bc, net, torch.device("cpu"), "f32").aux == {}
    with pytest.raises(ValueError, match="041-R23"):
        Learner({"aux": {"jepa": 0.1}}, net, net_of(cfg), torch.device("cpu"), "f32", "vtrace")


def test_bc_config_passes_aux_to_python_041_r23(tiny_config):
    """`[aux]` / `[monitor]` reach the Python trainer's config JSON through the Rust BcConfig;
    absent, the JSON and the fingerprint are those of before the keys."""
    from uranium import _core

    js, fp = _core.bc_config(str(tiny_config), "s1")
    assert "aux" not in json.loads(js) and "monitor" not in json.loads(js)
    p = tiny_config.with_name("bc_aux.toml")
    p.write_text(tiny_config.read_text(encoding="utf-8") + "\n[aux]\npriv = 0.25\njepa = 0.5\n\n[monitor]\nspatial_every = 100\n", encoding="utf-8")
    js2, fp2 = _core.bc_config(str(p), "s1")
    d = json.loads(js2)
    assert d["aux"] == {"priv": 0.25, "jepa": 0.5} and d["monitor"] == {"spatial_every": 100}
    assert fp2 != fp
    m = tiny_config.with_name("bc_mon.toml")
    m.write_text(tiny_config.read_text(encoding="utf-8") + "\n[monitor]\nspatial_every = 100\n", encoding="utf-8")
    assert _core.bc_config(str(m), "s1")[1] == fp, "041-R25: [monitor] is not in the fingerprint"
    bad = tiny_config.with_name("bc_bad.toml")
    bad.write_text(tiny_config.read_text(encoding="utf-8") + "\n[aux]\nprv = 0.25\n", encoding="utf-8")
    with pytest.raises(Exception):
        _core.bc_config(str(bad), "s1")


def test_bc_group_has_the_aux_masks_041_r23(tiny_config, tmp_path):
    """The collated lane group carries `real` and `priv_present` [rows] bool (032 §2.8)."""
    import numpy as np

    from uranium import _core

    shards = tmp_path / "shards"
    _core.write_test_shards(str(shards), "mixed")
    f = _core.Feeder(str(tiny_config), [str(shards)], prefetch=0, synthetic_maps=True)
    b = f.next()
    g = b.group(0, 2, 4, 0, 12)
    for k in ("real", "priv_present"):
        assert g[k].shape == (8,) and g[k].dtype == np.bool_, k
    steps = b.lane_steps()
    want = np.array([k < steps[lane] for lane in range(2) for k in range(4)])
    assert (g["real"] == want).all()
    # Rows of a real step whose label is void still count as real (aux targets need no label).
    assert ((g["weights"] > 0) <= g["real"]).all()


def test_rl_group_has_the_aux_masks_042_r21():
    """RL lane groups carry `real` (non-fill steps, the bootstrap step included) and
    `priv_present` (`Step::has_privileged`, before the first summary false)."""
    import numpy as np

    from uranium import _core

    t = 4
    batch = _core.test_segments(2, t, priv_from=1)
    g = batch.group(0, 2, t, 12)
    assert g["real"].dtype == np.bool_ and g["priv_present"].dtype == np.bool_
    assert g["real"].all(), "full segments: every row is a real step"
    assert (g["priv_present"].reshape(2, t) == np.array([False, True, True, True])).all()
    # The non-terminal lane's last step only bootstraps (mask 0) but is a real step.
    assert g["mask"].reshape(2, t)[1, -1] == 0 and g["real"].reshape(2, t)[1, -1]


@pytest.mark.parametrize("switch", ["bypass", "core_skip", "aux_priv", "aux_jepa"])
def test_unimplemented_switch_is_refused(switch):
    """Until a 032 §2.8 module exists, turning its switch on builds no silently-default model."""
    cfg, *_ = tiny(VariantConfig(**{switch: [1] if switch == "aux_jepa" else True}))
    with pytest.raises(NotImplementedError, match=switch):
        net_of(cfg)


# --- Hook points, with stand-in modules for the model 2 modules (032 §2.8, `Variants` docstring).


class Bypass(nn.Module):
    """Stand-in for 032-R21: u = Linear(h) (the hook's contract: (h, c, e, obs) -> [rows, ar])."""

    def __init__(self, d_core: int, d_ar: int, zero: bool):
        super().__init__()
        self.lin = nn.Linear(d_core, d_ar)
        if zero:
            nn.init.zeros_(self.lin.weight)
            nn.init.zeros_(self.lin.bias)

    def forward(self, h, c, e, obs):
        return self.lin(h)


def with_hooks(cfg, zero: bool, seed: int = 3):
    """A net with the §2.7 variant `cfg` and stand-ins in every 032 §2.8 hook."""
    net = net_of(cfg, seed)
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        d_core, d_in = cfg.core.hidden, cfg.entity.d_model + cfg.spatial.pool + cfg.scalar.hidden
        net.variant.bypass = Bypass(d_core, cfg.heads.ar, zero)
        net.variant.core_skip = nn.Linear(d_in, d_core)
        for p in net.variant.core_skip.parameters():
            p.copy_(torch.zeros_like(p) if zero else torch.randn(p.shape, generator=g) * 0.05)
        d_y = cfg.entity.d_model + cfg.spatial.pool
        net.variant.aux_priv = nn.Linear(d_core, cfg.input.privileged)
        net.variant.aux_jepa = nn.ModuleDict({str(k): nn.Linear(d_core, d_y) for k in (1, 2)})
    return net.eval()


@pytest.mark.parametrize("name", ["pool", "all"])
def test_zero_hooks_keep_outputs_032_ac13(name):
    """Zero-initialised bypass and core skip (032-R21 ②, 032-R23) give the outputs of the model
    without them, through every path (BC sequence, step, RL phased)."""
    cfg, b, t, obs, labels, _ = tiny(VARIANTS[name])
    ref = net_of(cfg).eval()
    net = with_hooks(cfg, zero=True)
    with torch.no_grad():
        for kw in ({}, {"phased": True}) if name == "all" else ({},):
            x = ref.forward_seq(obs, labels, None, ref.init_state(b), t, **kw)
            y = net.forward_seq(obs, labels, None, net.init_state(b), t, **kw)
            for k in HEADS:
                assert torch.equal(x["logp"][k], y["logp"][k]), (kw, k)
            assert torch.equal(x["value"], y["value"])
        a = ref.act(rows_at(obs, b, t, 0), ref.init_state(b), 0.0)
        c = net.act(rows_at(obs, b, t, 0), net.init_state(b), 0.0)
    assert torch.equal(a["log_prob"], c["log_prob"]) and torch.equal(a["value"], c["value"])


@pytest.mark.parametrize("name", ["pool", "outer", "all"])
def test_hooks_step_matches_sequence_032_ac13(name):
    """With non-zero stand-ins the hooks act where 032-R23 puts them (heads, value and aux read
    h' = h + skip; the outer h_sum takes the pre-skip h): one decision at a time = the training
    pass (032-R5, outer segments), and the phased RL pass gives the same (032 §2.7)."""
    from uranium.agent.infer import leaves

    cfg, b, t, obs, labels, _ = tiny(VARIANTS[name])
    net = with_hooks(cfg, zero=False)
    ref = net_of(cfg).eval()
    with torch.no_grad():
        seq = net.forward_seq(obs, labels, None, net.init_state(b), t)
        assert not torch.equal(seq["value"], ref.forward_seq(obs, labels, None, ref.init_state(b), t)["value"]), "the hooks are used"
        s = net.init_state(b)
        values, logps = [], []
        for k in range(t):
            o = rows_at(obs, b, t, k)
            out = net.act(o, s, 0.0)
            s = out["state"]
            values.append(out["value"])
        if net.outer is not None:
            ph = net.forward_seq(obs, labels, None, net.init_state(b), t, phased=True)
            assert torch.allclose(ph["value"], seq["value"], atol=1e-5)
            for k in HEADS:
                assert torch.allclose(ph["logp"][k], seq["logp"][k], atol=1e-4), k
    assert torch.allclose(seq["value"].view(b, t, -1), torch.stack(values, 1), atol=1e-5)
    for x, y in zip(leaves(seq["state"]), leaves(s)):
        assert torch.allclose(x.float(), y.float(), atol=1e-5)


@pytest.mark.parametrize("phased", [False, True])
def test_core_skip_leaves_recurrent_and_outer_state_032_r23(phased):
    """F1 (user decision 2026-11-09): h' = h + skip feeds the heads and value only; the core state
    and the outer LSTM state (its h_sum takes the pre-skip h) are those of the model without it."""
    from uranium.agent.infer import leaves

    cfg, b, t, obs, labels, _ = tiny(ALL_ARCH)
    ref = net_of(cfg).eval()
    net = net_of(cfg).eval()
    with torch.no_grad():
        net.variant.core_skip = nn.Linear(cfg.entity.d_model + cfg.spatial.pool + cfg.scalar.hidden, cfg.core.hidden)
        x = ref.forward_seq(obs, labels, None, ref.init_state(b), t, phased=phased)
        y = net.forward_seq(obs, labels, None, net.init_state(b), t, phased=phased)
        sa = ref.act(rows_at(obs, b, t, 0), ref.init_state(b), 0.0)
        sb = net.act(rows_at(obs, b, t, 0), net.init_state(b), 0.0)
    assert not torch.equal(x["value"], y["value"]), "the skip reaches the value head"
    for p, q in zip(leaves(x["state"]), leaves(y["state"])):
        assert torch.equal(p, q)
    for p, q in zip(leaves(sa["state"]), leaves(sb["state"])):
        assert torch.equal(p, q)


def test_bypass_reaches_action_type_and_ar_032_r21():
    """`u` changes the ActionType distribution and, through `ar`, the later heads."""
    cfg, b, t, obs, labels, _ = tiny(VARIANTS["pool"])
    net = net_of(cfg).eval()
    with torch.no_grad():
        x = net.forward_seq(obs, labels, None, net.init_state(b), t)
        net.variant.bypass = Bypass(cfg.core.hidden, cfg.heads.ar, zero=False)
        y = net.forward_seq(obs, labels, None, net.init_state(b), t)
    assert not torch.equal(x["dists"]["action"], y["dists"]["action"])
    assert not torch.equal(x["dists"]["delay"], y["dists"]["delay"])
    assert torch.equal(x["value"], y["value"]), "the bypass is a head-side addition only"


@pytest.mark.parametrize("name", ["pool", "outer", "all"])
def test_aux_outputs_032_ac14(name):
    """`forward_seq(aux=True)` adds `aux_priv`, `aux_jepa` {k} and `jepa_target` (= [e_pool,
    s_pool] before `core_add`, no gradient); `aux=False` (and `act`) runs no aux module and
    returns the same policy outputs."""
    cfg, b, t, obs, labels, priv = tiny(VARIANTS[name])
    net = with_hooks(cfg, zero=False).train()
    calls = []
    for m in (net.variant.aux_priv, *net.variant.aux_jepa.values()):
        m.register_forward_hook(lambda *_: calls.append(1))
    torch.manual_seed(1)
    off = net.forward_seq(obs, labels, priv, net.init_state(b), t)
    assert not calls and not {"aux_priv", "aux_jepa", "jepa_target"} & off.keys()
    torch.manual_seed(1)
    on = net.forward_seq(obs, labels, priv, net.init_state(b), t, aux=True)
    assert len(calls) == 3
    for k in HEADS:
        assert torch.equal(off["logp"][k], on["logp"][k]), k
    d_y = cfg.entity.d_model + cfg.spatial.pool
    assert on["aux_priv"].shape == (b * t, cfg.input.privileged)
    assert set(on["aux_jepa"]) == {1, 2} and all(v.shape == (b * t, d_y) for v in on["aux_jepa"].values())
    y = on["jepa_target"]
    assert y.shape == (b * t, d_y) and not y.requires_grad
    # The target is the encoders' summary before the outer LSTM's core input addition.
    net.eval()
    with torch.no_grad():
        o = rows_at(obs, b, t, 0)
        h_o = torch.randn(b, 256) if net.outer is not None else None
        core_in, _, sp, sc, e_pool = net.encode(o, h_o=h_o)
        joined = torch.cat([e_pool, sp["pool"], sc["c"]], 1)
        if net.outer is None:
            assert torch.equal(core_in, joined)
        else:
            assert torch.allclose(core_in, joined + net.outer.core_add(h_o), atol=1e-6)
            assert not torch.allclose(core_in[:, :e_pool.shape[1]], e_pool)
        calls.clear()
        net.act(o, net.init_state(b), 0.0)
    assert not calls, "032-AC14: act runs no aux module"


@pytest.mark.parametrize("name", ["outer", "all"])
def test_jepa_target_matches_per_decision_032_ac14(name):
    """In the outer paths (BC segments, RL phased) the target rows are each decision's own
    encoder summary, in b-major order."""
    cfg, b, t, obs, labels, _ = tiny(VARIANTS[name])
    net = with_hooks(cfg, zero=False)
    with torch.no_grad():
        seq = net.forward_seq(obs, labels, None, net.init_state(b), t, aux=True)["jepa_target"]
        ph = net.forward_seq(obs, labels, None, net.init_state(b), t, phased=True, aux=True)["jepa_target"]
        s = net.init_state(b)
        ys = []
        for k in range(t):
            o = rows_at(obs, b, t, k)
            _, _, sp, _, e_pool = net.encode(o, h_o=s[1][0])
            ys.append(torch.cat([e_pool, sp["pool"]], 1))
            s = net.act(o, s, 0.0)["state"]
    want = torch.stack(ys, 1).reshape(b * t, -1)
    assert torch.allclose(seq, want, atol=1e-5) and torch.allclose(ph, want, atol=1e-5)


def test_aux_without_modules_adds_nothing_032_ac14():
    cfg, b, t, obs, labels, _ = tiny()
    net = net_of(cfg).eval()
    with torch.no_grad():
        x = net.forward_seq(obs, labels, None, net.init_state(b), t)
        y = net.forward_seq(obs, labels, None, net.init_state(b), t, aux=True)
    assert x.keys() == y.keys()
    for k in HEADS:
        assert torch.equal(x["logp"][k], y["logp"][k])


def test_join_parts_keeps_every_key():
    """The outer paths rebuild the encoder dicts generically, so a key added later survives."""
    a = {"s8": torch.ones(2, 1), "skips": [torch.zeros(2, 1)], "pool": torch.full((2, 1), 2.0), "extra": torch.full((2, 1), 3.0)}
    b = {k: [x + 10 for x in v] if isinstance(v, list) else v + 10 for k, v in a.items()}
    out = join_parts([a, b], lambda xs: torch.cat(xs, 0))
    assert list(out) == list(a)
    assert torch.equal(out["extra"], torch.tensor([[3.0], [3.0], [13.0], [13.0]]))
    assert torch.equal(out["skips"][0], torch.tensor([[0.0], [0.0], [10.0], [10.0]]))
