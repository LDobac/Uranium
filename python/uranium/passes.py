import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch


class LaneGroup(NamedTuple):
    batch_index: int
    batch: object
    index: int
    start: int
    end: int
    steps: list
    hg: dict


def bc_shape(config: str) -> dict:
    from uranium import _core

    return json.loads(_core.bc_config(config, "s1")[0])


def group_size(lanes: int, micro_lanes: int) -> int:
    return lanes if micro_lanes == 0 else min(micro_lanes, lanes)


def fixed_device(backend_name: str | None) -> tuple[str, torch.device, str]:
    from uranium.device import configure, resolve

    backend, device, precision = resolve(backend_name)
    configure(backend)
    torch.backends.cudnn.benchmark = False
    return backend, device, precision


def lane_groups(source, lanes: int, g: int, t: int, pad: int, max_select: int, states: list, limit_batches: int = 0, after_batch=None):
    groups = -(-lanes // g)
    nb = 0
    while (b := source.next()) is not None:
        if limit_batches and nb >= limit_batches:
            break
        steps = b.lane_steps()
        for gi in range(groups):
            s, e = gi * g, min((gi + 1) * g, lanes)
            if all(steps[l] == 0 for l in range(s, e)):
                states[gi] = None
                continue
            yield LaneGroup(nb, b, gi, s, e, steps, b.group(s, e, t, pad, max_select))
        if after_batch is not None:
            after_batch()
        nb += 1


def real_rows(hg: dict, steps, start: int, t: int) -> np.ndarray:
    m = np.zeros(hg["b"] * t, bool)
    for li in range(hg["b"]):
        m[li * t:li * t + steps[start + li]] = True
    return m


class ForcedPass:
    def __init__(self, config: str, device: torch.device, precision: str, micro_lanes: int | None = None, lanes_per_group: int | None = None, cfg: dict | None = None):
        self.config = config
        self.cfg = bc_shape(config) if cfg is None else cfg
        self.device, self.precision = device, precision
        self.t, self.lanes = self.cfg["seq_len"], self.cfg["lanes"]
        if lanes_per_group is not None:
            self.g = min(lanes_per_group, self.lanes)
        else:
            self.g = group_size(self.lanes, self.cfg["micro_lanes"] if micro_lanes is None else micro_lanes)
        self.net = None
        self.states: list = []
        self.reset()

    def reset(self) -> None:
        self.states = [None] * -(-self.lanes // self.g)

    def load(self, ckpt_dir, frozen: bool = False):
        from uranium import ckpt

        net, _ = ckpt.load_model(Path(ckpt_dir), self.device)
        net.eval()
        if frozen:
            for p in net.parameters():
                p.requires_grad_(False)
        self.net = net
        return net

    def stream(self, dirs: list, split: str, **kw):
        from uranium import _core

        self.reset()
        return _core.EvalStream(self.config, dirs, split=split, **kw)

    def groups(self, source, limit_batches: int = 0, max_select: int | None = None, after_batch=None):
        if max_select is None:
            max_select = self.net.cfg.heads.max_select
        return lane_groups(source, self.lanes, self.g, self.t, self.cfg["pad_entities"], max_select, self.states, limit_batches, after_batch)

    def state0(self, hg: dict, state):
        if state is None:
            return self.net.init_state(hg["b"], self.device)
        return self.net.mask_state(state, torch.as_tensor(np.asarray(hg["keep"]).astype(bool), device=self.device))

    def unload(self) -> None:
        self.net = None
        self.reset()


def lane_slice(hg: dict, li: int, t: int) -> dict:
    b = hg["b"]
    return {k: (v[li * t:(li + 1) * t] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == b * t else v) for k, v in hg.items()}
