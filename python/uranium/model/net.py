"""`UraniumNet` (032 §1, §6). Encoders and heads run over all B (or B·T) rows in parallel; only
the core walks the time axis. Inputs are dicts of tensors in the 030 §1 layout (the keys of a
`uranium._core` lane group)."""

import json

import torch
import torch.nn as nn

from uranium.model.config import UraniumNetConfig
from uranium.model.core import build_core
from uranium.model.encoders import EntityEncoder, ScalarEncoder, SpatialEncoder
from uranium.model.heads import HEAD_NAMES, ActionTable, Heads
from uranium.model.value import ValueHeads

OBS_KEYS = ("entity_cat", "entity_cont", "entity_mask", "entity_cell", "entity_pos", "spatial", "scalar", "scalar_cat", "z_build", "z_units", "z_present", "type_mask", "unit_cap", "loc_class_mask", "is_building")
LABEL_KEYS = ("action", "delay", "queued", "selected", "target_unit", "target_loc")
BOOL_KEYS = ("entity_mask", "z_present", "type_mask", "unit_cap", "loc_class_mask", "is_building")
LONG_KEYS = ("entity_cat", "entity_cell", "scalar_cat", "z_build")


def action_columns() -> dict:
    """034 table columns (032 §5) from the Rust action table."""
    from uranium import _core

    return json.loads(_core.action_columns())


def to_obs(group: dict, device, dtype=torch.float32) -> dict:
    """Observation tensors from a lane group / numpy dict: masks to bool, indices to int64,
    features to `dtype`."""
    out = {}
    for k in OBS_KEYS:
        t = torch.as_tensor(group[k]).to(device, non_blocking=True)
        if k in BOOL_KEYS:
            t = t.bool()
        elif k in LONG_KEYS:
            t = t.long()
        else:
            t = t.to(dtype)
        out[k] = t
    return out


def to_labels(group: dict, device) -> dict:
    return {k: torch.as_tensor(group[k]).to(device, non_blocking=True).long() for k in LABEL_KEYS}


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
        cols = cols if cols is not None else action_columns()
        assert len(cols["uses_select"]) == cfg.input.actions, "action table size != config.input.actions"
        self._cols = cols
        self.table = ActionTable(cols, "cpu")

    def _apply(self, fn, *args, **kwargs):
        # Keep the action table constants on the module's device.
        out = super()._apply(fn, *args, **kwargs)
        dev = next(self.parameters()).device
        self.table = self.table.to(dev)
        return out

    def encode(self, obs: dict):
        e, e_pool = self.entity(obs)
        sp = self.spatial(obs, e)
        sc = self.scalar(obs)
        core_in = torch.cat([e_pool, sp["pool"].to(e_pool.dtype), sc["c"].to(e_pool.dtype)], 1)
        return core_in, e, sp, sc

    def init_state(self, b: int, device=None):
        return self.core.init_state(b, device or next(self.parameters()).device)

    def act(self, obs: dict, state, temp: float, generator: torch.Generator | None = None) -> dict:
        """One inference step for B games; sampling inside this forward (032-R9). No privileged
        input (032-R11). `temp` 0 picks the argmax of every head (032-R15, 050-R1)."""
        core_in, e, sp, sc = self.encode(obs)
        h, state = self.core.step(core_in, state)
        out = self.heads(None, temp, h, sc["c"], e, sp, obs, self.table, generator)
        value = self.value(h, sc["z"], sc["race"], None)
        win_open = self.value.win_open(h, sc["z"], sc["race"])  # 032-R18 ③: last
        head_logp = torch.stack([out["logp"][k] for k in HEAD_NAMES], 1)
        return {
            "action": out["chosen"], "log_prob": head_logp.sum(1), "head_log_prob": head_logp,
            "entropy": torch.stack([out["entropy"][k] for k in HEAD_NAMES], 1).sum(1),
            "value": value, "win_open": win_open, "state": state,
        }

    def forward_seq(self, obs: dict, labels: dict, priv, s0, t: int) -> dict:
        """Teacher-forced training pass over `t` steps per sequence; rows b-major (032 §6)."""
        core_in, e, sp, sc = self.encode(obs)
        bt, d = core_in.shape
        assert t > 0 and bt % t == 0, "batch rows must be B*T"
        h, state = self.core.forward_seq(core_in.view(bt // t, t, d), s0)
        h = h.reshape(bt, -1)
        out = self.heads(labels, 1.0, h, sc["c"], e, sp, obs, self.table)
        value = self.value(h, sc["z"], sc["race"], priv)
        win_open = self.value.win_open(h, sc["z"], sc["race"])
        return {"logp": out["logp"], "entropy": out["entropy"], "dists": out["dists"], "value": value, "win_open": win_open, "state": state}

    def win_open_params(self) -> list[nn.Parameter]:
        """032-R18: the win_open head's parameters (041-R19 clips them apart)."""
        return [] if self.value.open is None else list(self.value.open.parameters())
