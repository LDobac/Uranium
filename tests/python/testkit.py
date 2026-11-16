import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from uranium import _core, ckpt
from uranium.bc.trainer import BcTrainer
from uranium.model import UraniumNet, UraniumNetConfig, action_columns, input_dims, to_labels, to_obs
from uranium.model.config import VariantConfig
from uranium.model.state import leaves, rebuild
from uranium.rl.learner import state0_of

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "burn_parity"
TINY_GTRXL = FIX / "tiny-gtrxl"
CPU = torch.device("cpu")
HEADS = ("action", "delay", "queued", "selected", "target_unit", "target_loc")
WEIGHTS = [1.0, 0.3, 0.3, 0.05]
SEG = 2
SEEN = 6
VARIANTS = {
    "terrain": VariantConfig(terrain=True),
    "ctx": VariantConfig(ctx_token=True),
    "pool": VariantConfig(attn_pool=True),
    "outer": VariantConfig(outer=SEG),
    "all": VariantConfig(terrain=True, ctx_token=True, attn_pool=True, outer=SEG),
}
KS = [1, 2, 8]
AUX = VariantConfig(aux_priv=True, aux_jepa=KS)


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT).stdout


def fixture(name: str) -> tuple:
    meta = json.loads((FIX / name / "meta.json").read_text())
    return UraniumNetConfig.from_dict(meta["config"]), meta, load_file(FIX / name / "inputs.safetensors")


def net_config(name: str = "tiny-lstm", kind: str | None = None, win_open: bool | None = None, dropout: float | None = None, variant: VariantConfig | None = None) -> UraniumNetConfig:
    cfg = UraniumNetConfig.from_dict(json.loads((FIX / name / "meta.json").read_text())["config"])
    cfg.input = input_dims()
    if kind is not None:
        cfg.core.kind = kind
    if win_open is not None:
        cfg.value.win_open = win_open
    if dropout is not None:
        cfg.entity.dropout = dropout
    if variant is not None:
        cfg.variant = variant
    return cfg


def tiny_net(seed: int = 5, name: str = "tiny-lstm", **over) -> UraniumNet:
    cfg = net_config(name, **over)
    torch.manual_seed(seed)
    return UraniumNet(cfg, action_columns())


def net_cfg(variant: VariantConfig | None = None, kind: str = "Lstm", dropout: float | None = None) -> UraniumNetConfig:
    return net_config(kind=kind, dropout=dropout, variant=variant)


def make_net(variant: VariantConfig | None = None, seed: int = 7, **kw) -> UraniumNet:
    torch.manual_seed(seed)
    return UraniumNet(net_cfg(variant, **kw), action_columns())


def params_of(net) -> dict:
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def bc_cfg(config: Path, **over) -> dict:
    cfg = json.loads(_core.bc_config(str(config), "s1")[0])
    cfg.update(over)
    return cfg


def feeder(config: Path, shards: Path, restore: str | None = None):
    return _core.Feeder(str(config), [str(shards)], prefetch=0, synthetic_maps=True, check_maps=False, restore=restore)


def rl_cfg(lanes: int | None = None, **over) -> dict:
    cfg = json.loads(_core.rl_config(str(ROOT / "config" / "rl.toml")))
    cfg.update(seq_len=4, micro_lanes=1)
    cfg["optimizer"]["lr"] = 1e-3
    if lanes is not None:
        cfg["lanes"] = lanes
    cfg.update(over)
    return cfg


def tiny_make(variant, aux, cfg_json, start, dir, entity_chunk):
    cfg = json.loads(cfg_json)
    if aux:
        cfg["aux"] = aux
    if start == "new":
        net = make_net(variant)
    else:
        net, _ = ckpt.load_model(Path(dir))
    t = BcTrainer(cfg, net, CPU, "f32", entity_chunk)
    if start == "resume":
        ckpt.load_optimizer(Path(dir), ckpt.OPTIM_MAIN, t.net, t.opt)
    return t


def widen_scalar(cfg, inp: dict) -> dict:
    from uranium.model.variants import SCALAR_BLOCKS, SCALAR_COLUMNS

    fixed = sum(w for _, w in SCALAR_BLOCKS) + len(SCALAR_COLUMNS) + 2 * cfg.input.z_units
    width = fixed + SEEN + cfg.input.actions + cfg.heads.delay_buckets + 1
    sc = torch.as_tensor(inp["scalar"])
    extra = torch.rand(sc.shape[0], width - sc.shape[1], generator=torch.Generator().manual_seed(21)).to(sc.dtype)
    wide = torch.cat([sc, extra], 1)
    cfg.input.scalar = width
    return dict(inp, scalar=wide if isinstance(inp["scalar"], torch.Tensor) else wide.numpy())


def perturb(net, scale: float = 0.05):
    if net.variant is None or net.variant.bypass is None:
        return net
    g = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for p in net.variant.bypass.out.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale)
    return net


def tiny(variant: VariantConfig | None = None, fixture: Path = TINY_GTRXL, wide: bool = False):
    meta = json.loads((fixture / "meta.json").read_text())
    inp = load_file(fixture / "inputs.safetensors")
    cfg = UraniumNetConfig.from_dict(meta["config"])
    if variant is not None:
        cfg.variant = variant
    if wide or cfg.variant.bypass:
        inp = widen_scalar(cfg, inp)
    return cfg, meta["b"], meta["t"], to_obs(inp, "cpu"), to_labels(inp, "cpu"), inp["privileged"]


def net_of(cfg, seed=3):
    torch.manual_seed(seed)
    return UraniumNet(cfg, action_columns()).eval()


def rows_at(obs: dict, b: int, t: int, k: int) -> dict:
    return {key: x.view(b, t, *x.shape[1:])[:, k] for key, x in obs.items()}


def state_after(net, obs, b: int, t: int, counts) -> tuple:
    s = net.init_state(b)
    snaps = [s]
    with torch.no_grad():
        for k in range(max(counts)):
            s = net.act(rows_at(obs, b, t, k % t), s, 0.0)["state"]
            snaps.append(s)
    parts = [leaves(snaps[c]) for c in counts]
    return rebuild(snaps[0], iter([torch.stack([parts[r][i][r] for r in range(b)]) for i in range(len(parts[0]))]))


class Segments:
    def __init__(self, lanes: int, t: int, net, outer: int, r_value_off: tuple = (), pseudo_rewards: bool = False, pseudo_scale: tuple = (), sil_returns: tuple = (), priv_from: int = 0):
        self.batch, self.net = _core.test_segments(lanes, t, list(r_value_off), pseudo_rewards, list(pseudo_scale), list(sil_returns), priv_from), net
        g = torch.Generator().manual_seed(7)
        s = net.init_state(lanes, CPU)
        cs = [torch.randn(x.shape, generator=g) * 0.1 for x in leaves(s if not outer else s[0])]
        self.state = cs
        if outer:
            h_o, c_o, h_sum, _ = s[1]
            counter = torch.tensor([0.0] + [(5.0, 4.0, 3.0, 7.0)[(i - 1) % 4] for i in range(1, lanes)])
            self.state = (cs, (torch.randn(h_o.shape, generator=g), torch.randn(c_o.shape, generator=g), torch.randn(h_sum.shape, generator=g), counter))

    lanes = property(lambda self: self.batch.lanes)
    trained = property(lambda self: self.batch.trained)
    versions = property(lambda self: self.batch.versions)
    sil_returns = property(lambda self: self.batch.sil_returns)
    real_steps = property(lambda self: self.batch.real_steps)
    priv_trained = property(lambda self: self.batch.priv_trained)

    def group(self, start, end, t, max_select):
        g = self.batch.group(start, end, t, max_select)
        g["state0"] = [None if lane == 0 else [(tuple(x[lane:lane + 1].shape), x[lane:lane + 1].numpy().astype(np.float32).ravel()) for x in leaves(self.state)] for lane in range(start, end)]
        with torch.no_grad():
            out = self.net.forward_seq(to_obs(g, CPU), to_labels(g, CPU), torch.as_tensor(g["privileged"]).float(), state0_of(self.net, g, CPU), t, phased=True)
        g["behaviour"] = sum(out["logp"][k] for k in HEADS).numpy().astype(np.float32)
        return g


def kernel_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA / ROCm device")
    from uranium.kernels.attention import available

    if not available("cuda"):
        pytest.skip("attn_bias kernel not built on this host")


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-12)).item()


def fake_dump(path: Path, n_games=24, rows=60, seed=0, kind_shift=0.0, d=16, sha_prefix="a", race="Z", void_every=0, chain_every=0, f32_frames=False):
    from uranium import tables as tb
    from uranium.diag.planning.dump import RowWriter, op_actions, own_columns

    rng = np.random.default_rng(seed)
    names, _, of_action = own_columns()
    bunker = tb.action_id("Build_Bunker")
    acts = np.array([tb.action_id("Move_Pos"), tb.action_id("Train_SCV"), tb.action_id("Gather_Unit"), bunker])
    n_act = 101
    schema = {"game": ("int32", ()), "split": ("int8", ()), "dec": ("int32", ()), "frame": ("float32", ()), "h": ("float16", (d,)), "y": ("float16", (d,)),
              "priv": ("float16", (5,)), "priv_ok": ("uint8", ()), "own": ("float32", (len(names),)), "action": ("int16", ()), "delay": ("int8", ()),
              "weight": ("float32", ()), "logp": ("float16", (n_act,)), "mask": ("uint8", ((n_act + 7) // 8,)), "win_open": ("float32", ()), "g8": ("float16", (d,))}
    path.mkdir(parents=True)
    w = RowWriter(path, schema)
    games = {}
    for g in range(n_games):
        res = 1 if g % 2 else -1
        a = acts[rng.integers(0, 4, rows)]
        cum = 3.0 * (np.cumsum(a == bunker) - (a == bunker))
        own = np.zeros((rows, len(names)), np.float32)
        own[:, of_action[bunker]] = cum
        own[:, names.index("scv")] = 3.0 * (np.cumsum(a == acts[1]) - (a == acts[1]))
        h = rng.normal(size=(rows, d)).astype(np.float32)
        h[:, 0] = res + kind_shift
        h[:, 1] = cum
        y = rng.normal(size=(rows, d))
        y[:, 2] = cum * 0.5
        y = (y - y.mean(1, keepdims=True)) / y.std(1, keepdims=True)
        m = np.zeros((rows, n_act), bool)
        m[:, acts] = True
        lp = np.full((rows, n_act), -np.inf)
        lp[:, acts] = np.log([0.4, 0.3, 0.2, 0.1])
        frames = 2880 + 24.0 * np.arange(rows)
        if chain_every:
            ch = np.arange(rows) % chain_every == chain_every - 1
            frames[ch] = frames[np.nonzero(ch)[0] - 1] + 1
        if f32_frames:
            frames = (frames.astype(np.float32) / np.float32(86400)).astype(np.float32) * np.float32(tb.FRAME_SCALE)
        w.add({"game": np.full(rows, g), "split": np.zeros(rows), "dec": np.arange(rows), "frame": frames, "h": h, "y": y,
               "priv": rng.normal(size=(rows, 5)) + h[:, :1], "priv_ok": np.ones(rows), "own": own, "action": a, "delay": rng.integers(0, 16, rows),
               "weight": (np.arange(rows) % void_every != 0) if void_every else np.ones(rows), "logp": lp, "mask": np.packbits(m, axis=1), "win_open": h[:, 0] * 0.1, "g8": y})
        games[g] = {"split": "val", "race": race, "sha": f"{sha_prefix}{g:063x}", "occ": 0, "start": g % 4, "map": g % 3, "result": res, "rows": rows, "dec_rows": rows, "scv8": int(g % 3 == 0),
                    "cls": "rush" if g % 4 < 2 else "macro", "cls_given": None}
    w.close()
    (path / "games.json").write_text(json.dumps(games))
    (path / "manifest.json").write_text(json.dumps({"checkpoint": "x", "checkpoint_sha256": "s", "own_columns": names, "op_actions": op_actions()}))
    return path
