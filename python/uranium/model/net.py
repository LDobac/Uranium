"""`UraniumNet` (032 §1, §6). Encoders and heads run over all B (or B·T) rows in parallel; only
the core walks the time axis. Inputs are dicts of tensors in the 030 §1 layout (the keys of a
`uranium._core` lane group)."""

import functools
import json

import numpy as np
import torch
import torch.nn as nn

from uranium.model.config import UraniumNetConfig
from uranium.model.core import build_core
from uranium.model.encoders import EntityEncoder, ScalarEncoder, SpatialEncoder
from uranium.model.heads import HEAD_NAMES, LOC_ROW_MODES, ActionTable, Heads, location_rows
from uranium.model.layers import env_choice
from uranium.model.value import ValueHeads
from uranium.model.variants import Variants

OBS_KEYS = ("entity_cat", "entity_cont", "entity_mask", "entity_cell", "entity_pos", "spatial", "scalar", "scalar_cat", "z_build", "z_units", "z_present", "type_mask", "unit_cap", "loc_class_mask", "is_building")
LABEL_KEYS = ("action", "delay", "queued", "selected", "target_unit", "target_loc")
BOOL_KEYS = ("entity_mask", "z_present", "type_mask", "unit_cap", "loc_class_mask", "is_building")
LONG_KEYS = ("entity_cat", "entity_cell", "scalar_cat", "z_build")


def action_columns() -> dict:
    """034 table columns (032 §5) from the Rust action table."""
    from uranium import _core

    return json.loads(_core.action_columns())


H2D_MODES = ("sync", "pinned")


def to_device(x, device) -> torch.Tensor:
    """A host array on `device`. `URANIUM_H2D=pinned` stages it in page-locked memory so the copy
    is asynchronous: a copy from pageable memory waits for the queued GPU work first, so the host
    cannot launch ahead (AMB-000-4 step 2). Same values either way."""
    t = torch.as_tensor(x)
    dev = torch.device(device)
    if dev.type == "cuda" and env_choice("URANIUM_H2D", H2D_MODES, "sync") == "pinned":
        t = t.pin_memory()
    return t.to(dev, non_blocking=True)


def to_obs(group: dict, device, dtype=torch.float32) -> dict:
    """Observation tensors from a lane group / numpy dict: masks to bool, indices to int64,
    features to `dtype`."""
    out = {}
    for k in OBS_KEYS:
        t = to_device(group[k], device)
        if k in BOOL_KEYS:
            t = t.bool()
        elif k in LONG_KEYS:
            t = t.long()
        else:
            t = t.to(dtype)
        out[k] = t
    return out


# The observations with an entity axis (dim 1, padded to `pad_entities`): the `[r, n, ...]`
# arrays of the Rust group (`put_obs`, crates/uranium-py).
ENTITY_KEYS = ("entity_cat", "entity_cont", "entity_mask", "entity_cell", "entity_pos", "unit_cap", "is_building")


def trim_entity_pad(group: dict, k: int) -> dict:
    """032 §2.7 extra tokens, same math (032-R19): the K tokens join after the N entity slots,
    and N is a multiple of `pad_entities` (64), so N + K starts one more 64-wide tile of the fused
    attention kernel in every group (CUDA host, B1 N512: +6 ms of 115 per pass). When the last K
    slots are padding in every row they are dropped and N - K + K stays on the tile. Padding is
    masked everywhere; the selection's END is index N, and its labels clamp to N. With dropout the
    draws differ from the untrimmed group's (the dropout tensors are smaller), not their law."""
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


def to_labels(group: dict, device) -> dict:
    return {k: to_device(group[k], device).long() for k in LABEL_KEYS}


class UraniumNet(nn.Module):
    def __init__(self, cfg: UraniumNetConfig, cols: dict | None = None):
        super().__init__()
        assert cfg.input.map_size % 16 == 0, "map_size must be a multiple of 16 (four stride-2 stages)"
        self.cfg = cfg
        d_e = cfg.entity.d_model
        d_in = d_e + cfg.spatial.pool + cfg.scalar.hidden
        self.entity = EntityEncoder(cfg.input, cfg.entity)
        self.spatial = SpatialEncoder(cfg.input, cfg.spatial, d_e)
        self.scalar = ScalarEncoder(cfg.input, cfg.scalar)
        self.core = build_core(d_in, cfg.core)
        self.heads = Heads(cfg.input, cfg.heads, cfg.spatial, cfg.core.hidden, cfg.scalar.hidden, d_e)
        self.value = ValueHeads(cfg.input, cfg.value, cfg.core.hidden, cfg.scalar.z_out, cfg.scalar.race_emb)
        # 032 §2.7: the variant modules are built last (after win_open, 032-R18 ③), and only when a
        # switch is on, so the default model has the same parameters and initial weights.
        self.variant = Variants(cfg, d_in) if cfg.variant.any() else None
        cols = cols if cols is not None else action_columns()
        assert len(cols["uses_select"]) == cfg.input.actions, "action table size != config.input.actions"
        self._cols = cols
        self.table = ActionTable(cols, "cpu")

    @property
    def outer(self):
        """The outer LSTM (032 §2.7), or None."""
        return None if self.variant is None else self.variant.outer

    def _apply(self, fn, *args, **kwargs):
        # Keep the action table constants on the module's device.
        out = super()._apply(fn, *args, **kwargs)
        dev = next(self.parameters()).device
        self.table = self.table.to(dev)
        return out

    def entity_chunked(self, obs: dict, rows: int, add=None, tokens=None, tok_bias=None, pool=None):
        """041-R17: the entity encoder over `rows`-row chunks, each recomputed in the backward
        pass (its dropout draws replayed), so no chunk's activations stay alive. Same values as
        one pass; device memory only. The 032 §2.7 per-row inputs are cut with the rows."""
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
        """Encoders over all rows -> (core input [rows,768], E, spatial outputs, scalar outputs).
        `h_o` [rows,256]: the outer LSTM output each row uses (032 §2.7 `outer`)."""
        v = self.variant
        sc = film = None
        extra = {}
        if v is not None:
            if v.ctx is not None:
                sc = self.scalar(obs)  # the context token needs c first
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
        return core_in, e, sp, sc

    def labels(self, group: dict, device) -> dict:
        """`to_labels` plus the location head's row list from the host labels when
        `URANIUM_LOC_ROWS` is not `off` (no device sync, bucketed shapes, AMB-000-4 step 2)."""
        out = to_labels(group, device)
        mode = env_choice("URANIUM_LOC_ROWS", LOC_ROW_MODES, "off")
        if mode != "off":
            # Weight 1 for the real rows, 0 for the padding: no Python row count reaches the
            # forward (torch.compile would specialise on it), and always the same keys and types.
            rows, k = location_rows(group["action"], self._cols["target_kind"], mode)
            out["loc_rows"], out["loc_w"] = to_device(rows, device), to_device((np.arange(len(rows)) < k).astype(np.float32), device)
        return out

    # The recurrent state of B games: the core's, and with `outer` (032 §2.7) the pair
    # (core state, outer LSTM state). Reset per episode by `mask_state` / `init_state` (032-R5).
    def init_state(self, b: int, device=None):
        device = device or next(self.parameters()).device
        s = self.core.init_state(b, device)
        return s if self.outer is None else (s, self.outer.zero_state(b, device))

    def mask_state(self, s, keep):
        """Zero the rows where `keep` [B] is false (a new episode in that row)."""
        if self.outer is None:
            return self.core.mask_state(s, keep)
        return (self.core.mask_state(s[0], keep), self.outer.mask(s[1], keep))

    def detach_state(self, s):
        if self.outer is None:
            return self.core.detach_state(s)
        return (self.core.detach_state(s[0]), self.outer.detach(s[1]))

    def act(self, obs: dict, state, temp: float, generator: torch.Generator | None = None) -> dict:
        """One inference step for B games; sampling inside this forward (032-R9). No privileged
        input (032-R11). `temp` 0 picks the argmax of every head (032-R15, 050-R1)."""
        if self.outer is None:
            core_in, e, sp, sc = self.encode(obs)
            h, state = self.core.step(core_in, state)
        else:
            cs, os_ = state
            core_in, e, sp, sc = self.encode(obs, h_o=os_[0])
            h, cs = self.core.step(core_in, cs)
            state = (cs, self.outer.step(h, os_))
        out = self.heads(None, temp, h, sc["c"], e, sp, obs, self.table, generator)
        value = self.value(h, sc["z"], sc["race"], None)
        win_open = self.value.win_open(h, sc["z"], sc["race"])  # 032-R18 ③: last
        head_logp = torch.stack([out["logp"][k] for k in HEAD_NAMES], 1)
        return {
            "action": out["chosen"], "log_prob": head_logp.sum(1), "head_log_prob": head_logp,
            "entropy": torch.stack([out["entropy"][k] for k in HEAD_NAMES], 1).sum(1),
            "value": value, "win_open": win_open, "state": state,
        }

    def forward_seq(self, obs: dict, labels: dict, priv, s0, t: int, entity_chunk: int = 0) -> dict:
        """Teacher-forced training pass over `t` steps per sequence; rows b-major (032 §6).
        `entity_chunk` > 0: the entity encoder recomputed per row chunk (041-R17)."""
        if self.outer is None:
            core_in, e, sp, sc = self.encode(obs, entity_chunk)
            bt, d = core_in.shape
            assert t > 0 and bt % t == 0, "batch rows must be B*T"
            h, state = self.core.forward_seq(core_in.view(bt // t, t, d), s0)
            h = h.reshape(bt, -1)
        else:
            h, e, sp, sc, state = self._outer_seq(obs, s0, t, entity_chunk)
        out = self.heads(labels, 1.0, h, sc["c"], e, sp, obs, self.table)
        value = self.value(h, sc["z"], sc["race"], priv)
        win_open = self.value.win_open(h, sc["z"], sc["race"])
        return {"logp": out["logp"], "entropy": out["entropy"], "dists": out["dists"], "value": value, "win_open": win_open, "state": state}

    def _outer_seq(self, obs: dict, s0, t: int, entity_chunk: int):
        """032 §2.7 `outer`: the `t` steps in segments of `outer` decisions, one after another
        (each segment's rows in parallel). Segment k uses the outer state updated at the end of
        segment k-1, through which its loss reaches segment k-1. The sequences start on a segment
        boundary (BC chunks start at multiples of 64 decisions from the game start)."""
        seg = self.outer.segment
        bt = obs["entity_cat"].shape[0]
        assert t > 0 and bt % t == 0 and t % seg == 0, "batch rows must be B*T, T a multiple of the outer segment"
        b = bt // t
        part = lambda x, k: x.view(b, t, *x.shape[1:])[:, k * seg:(k + 1) * seg].reshape(b * seg, *x.shape[1:])  # noqa: E731
        cs, os_ = s0
        hs, es, sps, scs = [], [], [], []
        for k in range(t // seg):
            sub = {key: part(x, k) for key, x in obs.items()}
            core_in, e, sp, sc = self.encode(sub, entity_chunk, os_[0].repeat_interleave(seg, 0))
            h, cs = self.core.forward_seq(core_in.view(b, seg, -1), cs)
            os_ = self.outer.after_segment(h, os_)
            hs.append(h.reshape(b * seg, -1))
            es.append(e)
            sps.append(sp)
            scs.append(sc)
        # Back to b-major rows over the whole sequence, as the heads and labels expect.
        join = lambda xs: torch.cat([x.view(b, seg, *x.shape[1:]) for x in xs], 1).reshape(bt, *xs[0].shape[1:])  # noqa: E731
        skips = [join([x["skips"][i] for x in sps]) for i in range(len(sps[0]["skips"]))]
        sp = {"s8": join([x["s8"] for x in sps]), "skips": skips, "pool": join([x["pool"] for x in sps])}
        sc = {key: join([x[key] for x in scs]) for key in scs[0]}
        return join(hs), join(es), sp, sc, (cs, os_)

    def win_open_params(self) -> list[nn.Parameter]:
        """032-R18: the win_open head's parameters (041-R19 clips them apart)."""
        return [] if self.value.open is None else list(self.value.open.parameters())
