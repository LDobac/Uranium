"""Architecture variants for the comparison runs (032 §2.7, 032-R20) and the model 2 switches
(032 §2.8, 032-R21..R24): the BC config's `[model]`
switches add or replace modules; with every switch off nothing here is built. All modules are
created after the default model, under a saved global RNG and from their own fixed-seed
generators (as 032-R18 ③), so the shared modules start from the default model's weights."""

import math
from typing import NoReturn

import torch
import torch.nn as nn

from uranium.model.config import UraniumNetConfig
from uranium.model.layers import MASKED
from uranium.model.value import fixed_linear

# 030 §3: the static spatial channels (height x3, walkable, buildable, start_loc, base_loc, choke,
# wall_candidate, valid_map) come first.
TERRAIN_CHANNELS = 10
# 032 §2.7: attention-pool queries per side (own, enemy), 2026-10-25 user decision.
POOL_QUERIES = 4
OUTER_HIDDEN = 256
SEED = 0x3220_0000
# Fixed-seed base of each module's own generators (032-R18 ③ / §2.7 rule, also for the 032 §2.8
# modules, built under `fork_rng` after every other module). A module numbers its parameters
# from its base (+0, +1, ...); bases are 0x10 apart, so up to 16 seeds each. §2.7: terrain 0x10,
# ctx_token 0x20 / 0x21, attn_pool 0x30.., outer 0x40... §2.8 (model 2, reserved here):
SEED_BYPASS = SEED + 0x50  # 032-R21: query, K, V, output projection (the output starts at 0)
SEED_CORE_SKIP = SEED + 0x60  # 032-R23 F1: Linear(core input -> h), starts at 0
SEED_AUX_PRIV = SEED + 0x70  # 032-R24: AuxPriv MLP layers


def seed_aux_jepa(k: int) -> int:
    """032-R24: base of the JEPA predictor g_k, keyed by k (not by list position), so a k starts
    from the same weights whatever the other k in `aux_jepa` are. 0x1000 + 0x10 k never meets the
    bases above."""
    return SEED + 0x1000 + 0x10 * k


def pending(switch: str, unit: str) -> NoReturn:
    """A 032 §2.8 switch whose module is not implemented yet: refuse to build a model that would
    silently be the default one."""
    raise NotImplementedError(f"[model] {switch}: not implemented yet ({unit})")


def core_skip_linear(d_core_in: int, d_core: int) -> nn.Linear:
    """032-R23 F1 `core_skip`: Linear(core input -> core.hidden) with weight and bias at 0, so the
    model with it starts with the outputs of the model without it (032-AC13 ②). Built from its
    reserved seed like the other §2.8 modules (`fixed_linear`, under `fork_rng` in `Variants`),
    then zeroed."""
    # §2.8 construction rule: every new module is drawn from its own reserved seed (never the
    # global RNG); the zero init then overwrites the draw.
    lin = fixed_linear(d_core_in, d_core, SEED_CORE_SKIP)
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.zero_()
    return lin


def fixed_param(shape, seed: int, scale: float, normal: bool = False) -> nn.Parameter:
    """A parameter drawn from its own generator: U(-scale, scale), or N(0, scale) for bias tables
    (the default model's tables use N(0, 0.02))."""
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(shape, generator=g) * scale if normal else torch.empty(shape).uniform_(-scale, scale, generator=g)
    return nn.Parameter(t)


def terrain_channels(cfg: UraniumNetConfig) -> int:
    """TERRAIN_CHANNELS (fewer only in the tiny test models, which have fewer channels)."""
    return min(TERRAIN_CHANNELS, cfg.input.spatial_channels)


def terrain_cells(obs: dict, channels: int):
    """The first `channels` (static) channels at each entity's cell -> [B,N,channels].
    `entity_cell` is y*W + x, the flat index of the spatial plane (as the entity scatter)."""
    sp = obs["spatial"][:, :channels].flatten(2)
    cell = obs["entity_cell"].long()
    b, n = cell.shape
    return sp.gather(2, cell[:, None, :].expand(b, channels, n)).transpose(1, 2)


class AttnPool(nn.Module):
    """032 §2.7 `attn_pool`: learned queries (parameters, the same for every game and step),
    POOL_QUERIES over the own entities and POOL_QUERIES over the enemy ones, masked multi-head
    attention over E, the outputs joined and projected back to d. An empty set gives zeros."""

    def __init__(self, d: int, heads: int, seed: int):
        super().__init__()
        q = 2 * POOL_QUERIES
        self.query = fixed_param((q, d), seed, 1.0 / math.sqrt(d))
        self.k = fixed_linear(d, d, seed + 1)
        self.v = fixed_linear(d, d, seed + 2)
        self.out = fixed_linear(q * d, d, seed + 3)
        self.heads = heads

    def forward(self, e, own, enemy):
        b, n, d = e.shape
        h, dh = self.heads, d // self.heads
        q = self.query.shape[0]
        k = self.k(e).view(b, n, h, dh).transpose(1, 2)
        v = self.v(e).view(b, n, h, dh).transpose(1, 2)
        query = self.query.to(k.dtype).view(q, h, dh).transpose(0, 1)  # [H,Q,dh]
        scores = (query[None] @ k.transpose(2, 3)).float() / math.sqrt(dh)  # [B,H,Q,N]
        sets = torch.cat([own[:, None].expand(b, POOL_QUERIES, n), enemy[:, None].expand(b, POOL_QUERIES, n)], 1)
        w = torch.softmax(scores.masked_fill(~sets[:, None], MASKED), 3)
        w = w * sets.any(2).to(w.dtype)[:, None, :, None]
        out = (w.to(v.dtype) @ v).transpose(1, 2).reshape(b, q * d)  # [B, Q*H*dh]
        return self.out(out)


class OuterLstm(nn.Module):
    """032 §2.7 `outer`: an LSTM cell over the mean core output of each `segment` decisions. State
    (h_o, c_o, h_sum, n): [B,256], [B,256], [B,d_core], [B] (decision counter, f32). h_o goes back
    as an entity token, a FiLM on the spatial stem output and an addition to the core input."""

    def __init__(self, cfg: UraniumNetConfig, d_core_in: int, segment: int, seed: int):
        super().__init__()
        d_core, hid = cfg.core.hidden, OUTER_HIDDEN
        k = 1.0 / math.sqrt(hid)
        # The LSTM cell's parameters (PyTorch LSTMCell layout and init, gates i, f, g, o).
        self.weight_ih = fixed_param((4 * hid, d_core), seed, k)
        self.weight_hh = fixed_param((4 * hid, hid), seed + 1, k)
        self.bias_ih = fixed_param((4 * hid,), seed + 2, k)
        self.bias_hh = fixed_param((4 * hid,), seed + 3, k)
        self.tok = fixed_linear(hid, cfg.entity.d_model, seed + 4)
        self.tok_bias = fixed_param((1, cfg.entity.layers * cfg.entity.heads), seed + 5, 0.02, normal=True)
        self.film = fixed_linear(hid, 2 * cfg.spatial.stem_channels, seed + 6)
        self.core_add = fixed_linear(hid, d_core_in, seed + 7)
        self.segment, self.hidden, self.d_core = segment, hid, d_core

    def zero_state(self, b: int, device):
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        return (z(b, self.hidden), z(b, self.hidden), z(b, self.d_core), z(b))

    def cell(self, x, h, c):
        gates = x @ self.weight_ih.t().to(x.dtype) + self.bias_ih.to(x.dtype) + h.to(x.dtype) @ self.weight_hh.t().to(x.dtype) + self.bias_hh.to(x.dtype)
        i, f, g, o = gates.float().chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c

    def film_of(self, h_o):
        g, b = self.film(h_o).chunk(2, 1)
        return g, b

    def _update(self, s, total, n, active=None):
        """The cell where the counter `n` reached a multiple of the segment (no host branch, so
        it is graph-capture safe, 032-R19). `active` [B]: rows that took decisions (others keep
        their state even on a multiple)."""
        h_o, c_o, _, _ = s
        hn, cn = self.cell((total / self.segment).to(h_o.dtype), h_o, c_o)
        upd = torch.fmod(n, float(self.segment)) == 0
        if active is not None:
            upd = upd & active
        upd = upd[:, None]
        return (torch.where(upd, hn, h_o), torch.where(upd, cn, c_o), torch.where(upd, torch.zeros_like(total), total), n)

    def step(self, h, s):
        """After one decision with core output h [B,d_core]."""
        return self._update(s, s[2] + h.float(), s[3] + 1.0)

    def after_segment(self, h, s):
        """After a whole segment h [B,segment,d_core] that started on a segment boundary (BC
        chunks start at multiples of 64 decisions from the game start, 032 §2.7)."""
        return self._update(s, s[2] + h.float().sum(1), s[3] + float(h.shape[1]))

    def after_rows(self, total, count, s):
        """After `count` [B] decisions per row whose core outputs sum to `total` [B,d_core], each
        row inside one segment from its own phase (RL segments, 032 §2.7); a row with no decision
        keeps its state."""
        return self._update(s, s[2] + total, s[3] + count, count > 0)

    @staticmethod
    def mask(s, keep):
        k = keep.to(s[0].dtype)
        return (s[0] * k[:, None], s[1] * k[:, None], s[2] * k[:, None], s[3] * k)

    @staticmethod
    def detach(s):
        return tuple(t.detach().float() for t in s)


class Variants(nn.Module):
    """The modules of the switches that are on (032 §2.7, §2.8). Built last by `UraniumNet`.

    032 §2.8 hook points (each None when its switch is off; `UraniumNet` calls them):
    - `bypass` (032-R21): `bypass(h, c, e, obs) -> u [rows, heads.ar]`, added to the ActionType
      GLU output and to `ar` (`Heads.forward(..., u=u)`). Reads E as the heads get it (no extra
      tokens) and `obs["entity_mask"]`; `s_raw` it takes from `obs["scalar"]`.
    - `core_skip` (032-R23 F1): `core_skip(core_in) -> [rows, core.hidden]`, h' = h + it
      (`UraniumNet._skip`). Heads, value, win_open and aux read h'; the outer LSTM's h_sum keeps
      the pre-skip h and the core's recurrent state is untouched (user decision 2026-11-09).
    - `aux_priv` (032-R24): `aux_priv(h) -> [rows, input.privileged]`, training only.
    - `aux_jepa` (032-R24): `nn.ModuleDict` keyed by `str(k)`, `aux_jepa[str(k)](h) -> [rows, d]`
      with d = entity.d_model + spatial.pool, training only.
    `spatial_preact` (032-R22) has no module here: it is a flag of the spatial encoder's ResBlocks
    (`SpatialEncoder(preact=...)`, no parameters)."""

    def __init__(self, cfg: UraniumNetConfig, d_core_in: int):
        super().__init__()
        vc, d = cfg.variant, cfg.entity.d_model
        lh = cfg.entity.layers * cfg.entity.heads
        vc.check()
        with torch.random.fork_rng(devices=[]):  # the global RNG is left where it was
            self.terrain_ch = terrain_channels(cfg)
            self.terrain = fixed_linear(self.terrain_ch, d, SEED + 0x10) if vc.terrain else None
            self.ctx = fixed_linear(cfg.scalar.hidden, d, SEED + 0x20) if vc.ctx_token else None
            self.ctx_bias = fixed_param((1, lh), SEED + 0x21, 0.02, normal=True) if vc.ctx_token else None
            self.pool = AttnPool(d, cfg.entity.heads, SEED + 0x30) if vc.attn_pool else None
            self.outer = OuterLstm(cfg, d_core_in, vc.outer, SEED + 0x40) if vc.outer > 0 else None
            # 032 §2.8 (model 2): after the §2.7 modules, from their reserved seeds (above).
            self.bypass = pending("bypass", "032-R21") if vc.bypass else None
            self.core_skip = core_skip_linear(d_core_in, cfg.core.hidden) if vc.core_skip else None
            self.aux_priv = pending("aux_priv", "032-R24") if vc.aux_priv else None
            self.aux_jepa = pending("aux_jepa", "032-R24") if vc.aux_jepa else None

    def entity_inputs(self, obs: dict, c, h_o):
        """(added to the input projection [B,N,d] or None, extra tokens [B,K,d] or None, their bias
        rows [K, layers*heads] or None). Token order: ctx_token, outer (032 §2.7)."""
        add = self.terrain(terrain_cells(obs, self.terrain_ch)) if self.terrain is not None else None
        toks, rows = [], []
        if self.ctx is not None:
            toks.append(self.ctx(c)[:, None])
            rows.append(self.ctx_bias)
        if self.outer is not None:
            toks.append(self.outer.tok(h_o)[:, None])
            rows.append(self.outer.tok_bias)
        if not toks:
            return add, None, None
        return add, torch.cat(toks, 1), torch.cat(rows, 0)
