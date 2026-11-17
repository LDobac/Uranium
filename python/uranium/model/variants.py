import math

import torch
import torch.nn as nn

from uranium.kernels import MASKED
from uranium.model.auxiliary import aux_jepa_modules, aux_priv_module
from uranium.model.config import UraniumNetConfig
from uranium.model.encoders import SpatialNorms
from uranium.model.value import fixed_linear

TERRAIN_CHANNELS = 10
POOL_QUERIES = 4
OUTER_HIDDEN = 256
SEED = 0x3220_0000
SEED_BYPASS = SEED + 0x50
SEED_CORE_SKIP = SEED + 0x60
SEED_AUX_PRIV = SEED + 0x70


def seed_aux_jepa(k: int) -> int:
    return SEED + 0x1000 + 0x10 * k


def core_skip_linear(d_core_in: int, d_core: int) -> nn.Linear:
    lin = fixed_linear(d_core_in, d_core, SEED_CORE_SKIP)
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.zero_()
    return lin


def fixed_param(shape, seed: int, scale: float, normal: bool = False) -> nn.Parameter:
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(shape, generator=g) * scale if normal else torch.empty(shape).uniform_(-scale, scale, generator=g)
    return nn.Parameter(t)


def terrain_channels(cfg: UraniumNetConfig) -> int:
    return min(TERRAIN_CHANNELS, cfg.input.spatial_channels)


def terrain_cells(obs: dict, channels: int):
    sp = obs["spatial"][:, :channels].flatten(2)
    cell = obs["entity_cell"].long()
    b, n = cell.shape
    return sp.gather(2, cell[:, None, :].expand(b, channels, n)).transpose(1, 2)


class AttnPool(nn.Module):
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
        query = self.query.to(k.dtype).view(q, h, dh).transpose(0, 1)
        scores = (query[None] @ k.transpose(2, 3)).float() / math.sqrt(dh)
        sets = torch.cat([own[:, None].expand(b, POOL_QUERIES, n), enemy[:, None].expand(b, POOL_QUERIES, n)], 1)
        w = torch.softmax(scores.masked_fill(~sets[:, None], MASKED), 3)
        w = w * sets.any(2).to(w.dtype)[:, None, :, None]
        out = (w.to(v.dtype) @ v).transpose(1, 2).reshape(b, q * d)
        return self.out(out)


class OuterLstm(nn.Module):
    def __init__(self, cfg: UraniumNetConfig, d_core_in: int, segment: int, seed: int):
        super().__init__()
        d_core, hid = cfg.core.hidden, OUTER_HIDDEN
        k = 1.0 / math.sqrt(hid)
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
        h_o, c_o, _, _ = s
        hn, cn = self.cell((total / self.segment).to(h_o.dtype), h_o, c_o)
        upd = torch.fmod(n, float(self.segment)) == 0
        if active is not None:
            upd = upd & active
        upd = upd[:, None]
        return (torch.where(upd, hn, h_o), torch.where(upd, cn, c_o), torch.where(upd, torch.zeros_like(total), total), n)

    def step(self, h, s):
        return self._update(s, s[2] + h.float(), s[3] + 1.0)

    def after_segment(self, h, s):
        return self._update(s, s[2] + h.float().sum(1), s[3] + float(h.shape[1]))

    def after_rows(self, total, count, s):
        return self._update(s, s[2] + total, s[3] + count, count > 0)

    @staticmethod
    def mask(s, keep):
        k = keep.to(s[0].dtype)
        return (s[0] * k[:, None], s[1] * k[:, None], s[2] * k[:, None], s[3] * k)

    @staticmethod
    def detach(s):
        return tuple(t.detach().float() for t in s)


SCALAR_COLUMNS = ("frame", "log_frame", "minerals", "gas", "supply_used", "supply_total", "supply_max")
SCALAR_BLOCKS = (("upgrades", 63), ("tech", 47), ("researching", 47), ("upgrading", 63))
PRIVILEGED_FIXED = 5 + 63 + 47
S_RAW = ("frame", "log_frame", "minerals", "gas", "supply_used", "enemy_seen")
BYPASS_DIM = 256
BYPASS_HEADS = 8


def scalar_layout(cfg: UraniumNetConfig) -> dict[str, slice]:
    inp = cfg.input
    out, at = {}, 0
    blocks = [*((n, 1) for n in SCALAR_COLUMNS), *SCALAR_BLOCKS, ("own", 2 * inp.z_units)]
    last = inp.actions + cfg.heads.delay_buckets + 1
    seen = inp.scalar - sum(w for _, w in blocks) - last
    if seen < 1:
        raise ValueError(f"scalar width {inp.scalar} does not hold the 030 §4 layout (z_units {inp.z_units}, actions {inp.actions})")
    priv_seen, odd = divmod(inp.privileged - PRIVILEGED_FIXED, 2)
    if priv_seen >= 1 and not odd and priv_seen != seen:
        raise ValueError(f"scalar width {inp.scalar} gives {seen} enemy max-seen columns, privileged width {inp.privileged} gives {priv_seen} observed types (030 §4, 030-R9)")
    for name, w in [*blocks, ("enemy_seen", seen), ("last_action", last)]:
        out[name] = slice(at, at + w)
        at += w
    return out


def s_raw_slices(cfg: UraniumNetConfig) -> list[slice]:
    layout = scalar_layout(cfg)
    out = []
    for name in S_RAW:
        s = layout[name]
        if out and out[-1].stop == s.start:
            out[-1] = slice(out[-1].start, s.stop)
        else:
            out.append(s)
    return out


def zero_linear(d_in: int, d_out: int, seed: int) -> nn.Linear:
    lin = fixed_linear(d_in, d_out, seed)
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.zero_()
    return lin


class HeadBypass(nn.Module):
    def __init__(self, cfg: UraniumNetConfig, seed: int):
        super().__init__()
        self.cols = s_raw_slices(cfg)
        n_s = sum(s.stop - s.start for s in self.cols)
        d_q = cfg.core.hidden + cfg.scalar.hidden + n_s
        self.query = fixed_linear(d_q, BYPASS_DIM, seed)
        self.k = fixed_linear(cfg.entity.d_model, BYPASS_DIM, seed + 1)
        self.v = fixed_linear(cfg.entity.d_model, BYPASS_DIM, seed + 2)
        self.out = zero_linear(BYPASS_DIM + n_s, cfg.heads.ar, seed + 3)
        self.heads = BYPASS_HEADS

    def s_raw(self, scalar):
        return torch.cat([scalar[:, s] for s in self.cols], 1)

    def attend(self, q, e, mask):
        b, n, _ = e.shape
        h, dh = self.heads, BYPASS_DIM // self.heads
        k = self.k(e).view(b, n, h, dh).transpose(1, 2)
        v = self.v(e).view(b, n, h, dh).transpose(1, 2)
        q = q.to(k.dtype).view(b, h, 1, dh)
        scores = (q @ k.transpose(2, 3)).float() / math.sqrt(dh)
        w = torch.softmax(scores.masked_fill(~mask[:, None, None], MASKED), 3)
        w = w * mask.any(1).to(w.dtype)[:, None, None, None]
        return (w.to(v.dtype) @ v).reshape(b, BYPASS_DIM)

    def forward(self, h, c, e, obs):
        s = self.s_raw(obs["scalar"])
        q = self.query(torch.cat([h, c.to(h.dtype), s.to(h.dtype)], 1))
        att = self.attend(q, e, obs["entity_mask"])
        return self.out(torch.cat([att, s.to(att.dtype)], 1))


class Variants(nn.Module):
    def __init__(self, cfg: UraniumNetConfig, d_core_in: int):
        super().__init__()
        vc, d = cfg.variant, cfg.entity.d_model
        lh = cfg.entity.layers * cfg.entity.heads
        vc.check()
        with torch.random.fork_rng(devices=[]):
            self.terrain_ch = terrain_channels(cfg)
            self.terrain = fixed_linear(self.terrain_ch, d, SEED + 0x10) if vc.terrain else None
            self.ctx = fixed_linear(cfg.scalar.hidden, d, SEED + 0x20) if vc.ctx_token else None
            self.ctx_bias = fixed_param((1, lh), SEED + 0x21, 0.02, normal=True) if vc.ctx_token else None
            self.pool = AttnPool(d, cfg.entity.heads, SEED + 0x30) if vc.attn_pool else None
            self.outer = OuterLstm(cfg, d_core_in, vc.outer, SEED + 0x40) if vc.outer > 0 else None
            self.bypass = HeadBypass(cfg, SEED_BYPASS) if vc.bypass else None
            self.core_skip = core_skip_linear(d_core_in, cfg.core.hidden) if vc.core_skip else None
            d_y = d + cfg.spatial.pool
            self.aux_priv = aux_priv_module(cfg.core.hidden, cfg.input.privileged, SEED_AUX_PRIV) if vc.aux_priv else None
            self.aux_jepa = aux_jepa_modules(cfg.core.hidden, d_y, vc.aux_jepa, seed_aux_jepa) if vc.aux_jepa else None
            self.spatial_norm = SpatialNorms(cfg.spatial, vc.spatial_norm) if vc.spatial_norm else None

    def entity_inputs(self, obs: dict, c, h_o):
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
