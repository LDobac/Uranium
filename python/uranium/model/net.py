import functools
import json

import numpy as np
import torch
import torch.nn as nn

from uranium.env import env_choice
from uranium.model.config import UraniumNetConfig
from uranium.model.core import build_core
from uranium.model.encoders import EntityEncoder, ScalarEncoder, SpatialEncoder
from uranium.model.heads import HEAD_NAMES, LOC_ROW_MODES, ActionTable, Heads, location_rows
from uranium.model.layers import CHANNELS_LAST_MODES
from uranium.model.value import ValueHeads
from uranium.model.variants import Variants

OBS_KEYS = ("entity_cat", "entity_cont", "entity_mask", "entity_cell", "entity_pos", "spatial", "scalar", "scalar_cat", "z_build", "z_units", "z_present", "type_mask", "unit_cap", "loc_class_mask", "is_building")
LABEL_KEYS = ("action", "delay", "queued", "selected", "target_unit", "target_loc")
BOOL_KEYS = ("entity_mask", "z_present", "type_mask", "unit_cap", "loc_class_mask", "is_building")
LONG_KEYS = ("entity_cat", "entity_cell", "scalar_cat", "z_build")


def action_columns() -> dict:
    from uranium import _core

    return json.loads(_core.action_columns())


H2D_MODES = ("sync", "pinned")


def to_device(x, device, pinned: bool | None = None) -> torch.Tensor:
    t = torch.as_tensor(x)
    dev = torch.device(device)
    if pinned is None:
        pinned = env_choice("URANIUM_H2D", H2D_MODES, "sync") == "pinned"
    if dev.type == "cuda" and pinned and not t.is_pinned():
        t = t.pin_memory()
    return t.to(dev, non_blocking=True)


def to_obs(group: dict, device, dtype=torch.float32, pinned: bool | None = None) -> dict:
    out = {}
    for k in OBS_KEYS:
        t = to_device(group[k], device, pinned)
        if k in BOOL_KEYS:
            t = t.bool()
        elif k in LONG_KEYS:
            t = t.long()
        elif not (k == "spatial" and t.dtype == torch.bfloat16):
            t = t.to(dtype)
        out[k] = t
    return out


SPATIAL_BF16_MODES = ("off", "on")


def stage(group: dict, device, precision: str, pinned: bool | None = None, spatial_bf16: bool | None = None, keys=OBS_KEYS) -> dict:
    bf16 = precision == "bf16" and (env_choice("URANIUM_SPATIAL_BF16", SPATIAL_BF16_MODES, "off") == "on" if spatial_bf16 is None else spatial_bf16)
    if pinned is None:
        pinned = env_choice("URANIUM_H2D", H2D_MODES, "sync") == "pinned"
    pinned = pinned and torch.device(device).type == "cuda"
    if not (bf16 or pinned):
        return group
    out = dict(group)
    for k in keys:
        v = group[k]
        if not isinstance(v, np.ndarray) or (not pinned and k != "spatial"):
            continue
        dtype = torch.bfloat16 if bf16 and k == "spatial" else None
        src = torch.from_numpy(np.ascontiguousarray(v))
        dst = torch.empty(src.shape, dtype=dtype or src.dtype, pin_memory=pinned)
        out[k] = dst.copy_(src)
    return out


ENTITY_KEYS = ("entity_cat", "entity_cont", "entity_mask", "entity_cell", "entity_pos", "unit_cap", "is_building")


def trim_entity_pad(group: dict, k: int) -> dict:
    m = group["entity_mask"]
    n = m.shape[1]
    if k == 0 or n <= k or m[:, n - k:].any():
        return group
    out = dict(group)
    for key in ENTITY_KEYS:
        out[key] = np.ascontiguousarray(group[key][:, :n - k])
    if "n" in out:
        out["n"] = n - k
    return out


def phase_split(counter: torch.Tensor, t: int, seg: int) -> list[torch.Tensor]:
    dev = counter.device
    pos = torch.arange(t, device=dev)[None]
    first = (seg - torch.fmod(counter, float(seg))).long()[:, None]
    part = torch.where(pos < first, torch.zeros_like(pos), 1 + torch.div(pos - first, seg, rounding_mode="floor")).reshape(-1)
    out = []
    for k in range((t + seg - 1) // seg + 1):
        idx = (part == k).nonzero().squeeze(1)
        if idx.numel():
            out.append(idx)
    return out


def join_parts(parts: list[dict], join) -> dict:
    return {k: [join([p[k][i] for p in parts]) for i in range(len(v))] if isinstance(v, list) else join([p[k] for p in parts]) for k, v in parts[0].items()}


def to_labels(group: dict, device, pinned: bool | None = None) -> dict:
    return {k: to_device(group[k], device, pinned).long() for k in LABEL_KEYS}


class UraniumNet(nn.Module):
    def __init__(self, cfg: UraniumNetConfig, cols: dict | None = None):
        super().__init__()
        assert cfg.input.map_size % 16 == 0, "map_size must be a multiple of 16 (four stride-2 stages)"
        self.cfg = cfg
        d_e = cfg.entity.d_model
        d_in = d_e + cfg.spatial.pool + cfg.scalar.hidden
        self.entity = EntityEncoder(cfg.input, cfg.entity)
        self.spatial = SpatialEncoder(cfg.input, cfg.spatial, d_e, preact=cfg.variant.spatial_preact)
        self.scalar = ScalarEncoder(cfg.input, cfg.scalar)
        self.core = build_core(d_in, cfg.core)
        self.heads = Heads(cfg.input, cfg.heads, cfg.spatial, cfg.core.hidden, cfg.scalar.hidden, d_e)
        self.value = ValueHeads(cfg.input, cfg.value, cfg.core.hidden, cfg.scalar.z_out, cfg.scalar.race_emb)
        self.variant = Variants(cfg, d_in) if cfg.variant.any() else None
        if self.variant is not None and self.variant.spatial_norm is not None:
            self.spatial.set_norms(self.variant.spatial_norm, cfg.variant.spatial_end_relu)
        cols = cols if cols is not None else action_columns()
        assert len(cols["uses_select"]) == cfg.input.actions, "action table size != config.input.actions"
        self._cols = cols
        self.table = ActionTable(cols, "cpu")
        if env_choice("URANIUM_CHANNELS_LAST", CHANNELS_LAST_MODES, "off") == "on":
            self.to(memory_format=torch.channels_last)

    def conv_layout(self, channels_last: bool, stem_pad: bool) -> "UraniumNet":
        self.spatial.channels_last = self.heads.location.channels_last = channels_last
        self.spatial.stem_pad = stem_pad
        return self.to(memory_format=torch.channels_last if channels_last else torch.contiguous_format)

    @property
    def outer(self):
        return None if self.variant is None else self.variant.outer

    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        dev = next(self.parameters()).device
        self.table = self.table.to(dev)
        return out

    def entity_chunked(self, obs: dict, rows: int, add=None, tokens=None, tok_bias=None, pool=None):
        from torch.utils.checkpoint import checkpoint

        n = obs["entity_cat"].shape[0]
        fn = self.entity if pool is None and tok_bias is None else functools.partial(self.entity, tok_bias=tok_bias, pool=pool)
        cut = lambda t, s: None if t is None else t[s:s + rows]  # noqa: E731
        es, pools = [], []
        for s in range(0, n, rows):
            sub = {k: v[s:s + rows] for k, v in obs.items()}
            if add is None and tokens is None:
                e, p = checkpoint(fn, sub, use_reentrant=False, preserve_rng_state=True)
            else:
                e, p = checkpoint(fn, sub, cut(add, s), cut(tokens, s), use_reentrant=False, preserve_rng_state=True)
            es.append(e)
            pools.append(p)
        return torch.cat(es, 0), torch.cat(pools, 0)

    def encode(self, obs: dict, entity_chunk: int = 0, h_o=None):
        v = self.variant
        sc = film = None
        extra = {}
        if v is not None:
            if v.ctx is not None:
                sc = self.scalar(obs)
            add, tokens, tok_bias = v.entity_inputs(obs, None if sc is None else sc["c"], h_o)
            extra = {"add": add, "tokens": tokens, "tok_bias": tok_bias, "pool": v.pool}
            if v.outer is not None:
                film = v.outer.film_of(h_o)
        if entity_chunk and torch.is_grad_enabled() and obs["entity_cat"].shape[0] > entity_chunk:
            e, e_pool = self.entity_chunked(obs, entity_chunk, **extra)
        else:
            e, e_pool = self.entity(obs, **extra)
        sp = self.spatial(obs, e) if film is None else self.spatial(obs, e, film)
        if sc is None:
            sc = self.scalar(obs)
        core_in = torch.cat([e_pool, sp["pool"].to(e_pool.dtype), sc["c"].to(e_pool.dtype)], 1)
        if self.outer is not None:
            core_in = core_in + self.outer.core_add(h_o).to(core_in.dtype)
        return core_in, e, sp, sc, e_pool

    def _skip(self, h, core_in):
        m = None if self.variant is None else self.variant.core_skip
        return h if m is None else h + m(core_in).to(h.dtype)

    def _bypass(self, h, sc: dict, e, obs: dict):
        m = None if self.variant is None else self.variant.bypass
        return None if m is None else m(h, sc["c"], e, obs)

    def _aux(self, h, e_pool, sp_pool) -> dict:
        v = self.variant
        out = {}
        if v is None:
            return out
        if v.aux_priv is not None:
            out["aux_priv"] = v.aux_priv(h)
        if v.aux_jepa is not None:
            out["aux_jepa"] = {int(k): g(h) for k, g in v.aux_jepa.items()}
            out["jepa_target"] = torch.cat([e_pool, sp_pool.to(e_pool.dtype)], 1).detach()
        return out

    def labels(self, group: dict, device, pinned: bool | None = None) -> dict:
        out = to_labels(group, device, pinned)
        mode = env_choice("URANIUM_LOC_ROWS", LOC_ROW_MODES, "off")
        if mode != "off":
            rows, k = location_rows(group["action"], self._cols["target_kind"], mode)
            out["loc_rows"], out["loc_w"] = to_device(rows, device, pinned), to_device((np.arange(len(rows)) < k).astype(np.float32), device, pinned)
        return out

    def init_state(self, b: int, device=None):
        device = device or next(self.parameters()).device
        s = self.core.init_state(b, device)
        return s if self.outer is None else (s, self.outer.zero_state(b, device))

    def mask_state(self, s, keep):
        if self.outer is None:
            return self.core.mask_state(s, keep)
        return (self.core.mask_state(s[0], keep), self.outer.mask(s[1], keep))

    def detach_state(self, s):
        if self.outer is None:
            return self.core.detach_state(s)
        return (self.core.detach_state(s[0]), self.outer.detach(s[1]))

    def act(self, obs: dict, state, temp: float, generator: torch.Generator | None = None) -> dict:
        if self.outer is None:
            core_in, e, sp, sc, _ = self.encode(obs)
            h, state = self.core.step(core_in, state)
            h = self._skip(h, core_in)
        else:
            cs, os_ = state
            core_in, e, sp, sc, _ = self.encode(obs, h_o=os_[0])
            h, cs = self.core.step(core_in, cs)
            state = (cs, self.outer.step(h, os_))
            h = self._skip(h, core_in)
        out = self.heads(None, temp, h, sc["c"], e, sp, obs, self.table, generator, u=self._bypass(h, sc, e, obs))
        value = self.value(h, sc["z"], sc["race"], None)
        win_open = self.value.win_open(h, sc["z"], sc["race"])
        head_logp = torch.stack([out["logp"][k] for k in HEAD_NAMES], 1)
        return {
            "action": out["chosen"], "log_prob": head_logp.sum(1), "head_log_prob": head_logp,
            "entropy": torch.stack([out["entropy"][k] for k in HEAD_NAMES], 1).sum(1),
            "value": value, "win_open": win_open, "state": state, "type_logp": out["dists"]["action"],
        }

    def forward_seq(self, obs: dict, labels: dict, priv, s0, t: int, entity_chunk: int = 0, phased: bool = False, phases: list | None = None, aux: bool = False) -> dict:
        h, e, sp, sc, e_pool, state = self.trunk_seq(obs, s0, t, entity_chunk, phased, phases, aux)
        out = self.heads(labels, 1.0, h, sc["c"], e, sp, obs, self.table, u=self._bypass(h, sc, e, obs))
        value = self.value(h, sc["z"], sc["race"], priv)
        win_open = self.value.win_open(h, sc["z"], sc["race"])
        res = {"logp": out["logp"], "entropy": out["entropy"], "dists": out["dists"], "value": value, "win_open": win_open, "state": state}
        if aux:
            res.update(self._aux(h, e_pool, sp["pool"]))
        return res

    def trunk_seq(self, obs: dict, s0, t: int, entity_chunk: int = 0, phased: bool = False, phases: list | None = None, aux: bool = False):
        if self.outer is None:
            core_in, e, sp, sc, e_pool = self.encode(obs, entity_chunk)
            bt, d = core_in.shape
            assert t > 0 and bt % t == 0, "batch rows must be B*T"
            h, state = self.core.forward_seq(core_in.view(bt // t, t, d), s0)
            h = self._skip(h.reshape(bt, -1), core_in)
            return h, e, sp, sc, e_pool, state
        return self._outer_seq_phased(obs, s0, t, entity_chunk, phases, aux) if phased else self._outer_seq(obs, s0, t, entity_chunk, aux)

    def action_seq(self, obs: dict, s0, t: int) -> tuple[dict, torch.Tensor]:
        h, e, sp, sc, _, state = self.trunk_seq(obs, s0, t)
        lp = self.heads.action_logp(h, sc["c"], obs["type_mask"], self._bypass(h, sc, e, obs))
        return {"h": h, "e": e, "sp": sp, "sc": sc, "state": state}, lp

    def _outer_seq(self, obs: dict, s0, t: int, entity_chunk: int, aux: bool = False):
        seg = self.outer.segment
        bt = obs["entity_cat"].shape[0]
        assert t > 0 and bt % t == 0 and t % seg == 0, "batch rows must be B*T, T a multiple of the outer segment"
        b = bt // t
        part = lambda x, k: x.view(b, t, *x.shape[1:])[:, k * seg:(k + 1) * seg].reshape(b * seg, *x.shape[1:])  # noqa: E731
        cs, os_ = s0
        hs, es, sps, scs, eps = [], [], [], [], []
        for k in range(t // seg):
            sub = {key: part(x, k) for key, x in obs.items()}
            core_in, e, sp, sc, e_pool = self.encode(sub, entity_chunk, os_[0].repeat_interleave(seg, 0))
            h, cs = self.core.forward_seq(core_in.view(b, seg, -1), cs)
            os_ = self.outer.after_segment(h, os_)
            hs.append(self._skip(h, core_in.view(b, seg, -1)).reshape(b * seg, -1))
            es.append(e)
            sps.append(sp)
            scs.append(sc)
            if aux:
                eps.append(e_pool)
        join = lambda xs: torch.cat([x.view(b, seg, *x.shape[1:]) for x in xs], 1).reshape(bt, *xs[0].shape[1:])  # noqa: E731
        return join(hs), join(es), join_parts(sps, join), join_parts(scs, join), join(eps) if aux else None, (cs, os_)

    def _outer_seq_phased(self, obs: dict, s0, t: int, entity_chunk: int, phases: list | None = None, aux: bool = False):
        seg = self.outer.segment
        bt = obs["entity_cat"].shape[0]
        assert t > 0 and bt % t == 0, "batch rows must be B*T"
        b = bt // t
        cs0, os_ = s0
        dev = obs["entity_cat"].device
        core_in = cs = None
        idxs, hs, es, sps, scs, eps = [], [], [], [], [], []
        for idx in phase_split(os_[3], t, seg) if phases is None else phases:
            row = torch.div(idx, t, rounding_mode="floor")
            sub = {key: x.index_select(0, idx) for key, x in obs.items()}
            ci, e, sp, sc, e_pool = self.encode(sub, entity_chunk, os_[0].index_select(0, row))
            core_in = (ci.new_zeros(bt, ci.shape[1]) if core_in is None else core_in).index_copy(0, idx, ci)
            h_all, cs = self.core.forward_seq(core_in.view(b, t, -1), cs0)
            h = h_all.reshape(bt, -1).index_select(0, idx)
            total = torch.zeros(b, h.shape[1], device=dev).index_add(0, row, h.float())
            count = torch.zeros(b, device=dev).index_add(0, row, torch.ones(idx.numel(), device=dev))
            os_ = self.outer.after_rows(total, count, os_)
            idxs.append(idx)
            hs.append(self._skip(h, ci))
            es.append(e)
            sps.append(sp)
            scs.append(sc)
            if aux:
                eps.append(e_pool)
        perm = torch.cat(idxs)
        inv = perm.new_empty(bt).index_copy(0, perm, torch.arange(perm.numel(), device=dev))
        join = lambda xs: torch.cat(xs, 0).index_select(0, inv)  # noqa: E731
        return join(hs), join(es), join_parts(sps, join), join_parts(scs, join), join(eps) if aux else None, (cs, os_)

    def win_open_params(self) -> list[nn.Parameter]:
        return [] if self.value.open is None else list(self.value.open.parameters())

    def aux_params(self) -> list[nn.Parameter]:
        v = self.variant
        return [] if v is None else [p for m in (v.aux_priv, v.aux_jepa) if m is not None for p in m.parameters()]
