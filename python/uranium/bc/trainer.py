"""BC trainer (041): one update per batch of B lanes x T steps. Lanes run forward/backward in
groups of `micro_lanes` and their gradients are summed before one optimizer step (041-R3), so
peak memory follows the group, not B. Each group keeps its core state across batches (masked at
a new game, detached when stored, 041-R2 / 032-R5). Losses are divided by the usable steps of
the whole batch, so the summed group gradients equal one pass over all lanes. The run loop
(schedule, data, checkpoints, locks) is Rust (`uranium._core.run_bc`)."""

import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from uranium import ckpt, kernels
from uranium.bc.clip import clip_global_norm
from uranium.device import adam_impl, autocast, compile_graphs, compiled, impl_settings, mark_dynamic
from uranium.model import UraniumNet, UraniumNetConfig, input_dims, to_obs
from uranium.model.config import VariantConfig
from uranium.model.net import stage, to_device, trim_entity_pad

HEADS = ("action", "delay", "queued", "selected", "target_unit", "target_loc")
VALUE_WIN = 0
# Burn AdamW defaults kept (032-AC10 parity): betas (0.9, 0.999), eps 1e-5, decoupled decay.
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-5
CORES = {"lstm": "Lstm", "gtrxl": "Gtrxl"}


def dropout_seed(seed: int, step: int, group: int) -> int:
    """Dropout draws of one lane group's pass: a function of (data seed, update, group), so a
    resumed run draws what the run that kept going drew (061-R10)."""
    x = (seed * 0x9E3779B97F4A7C15 + (step << 20) + group * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    x ^= x >> 31
    return x & ((1 << 63) - 1)


def new_net_config(core: str, win_open: bool, variant: VariantConfig | None = None) -> UraniumNetConfig:
    """A new run's net (032 §6 defaults, input dims from the feature schema, 032-R18 win_open
    head when the BC config weights it, 041-R19, the BC config's `[model]` variant, 032 §2.7)."""
    cfg = UraniumNetConfig()
    cfg.input = input_dims()
    cfg.core.kind = CORES[core]
    cfg.value.win_open = win_open
    if variant is not None:
        cfg.variant = variant
    return cfg


class BcTrainer:
    def __init__(self, cfg: dict, net: UraniumNet, device: torch.device, precision: str, entity_chunk: int = 0):
        self.cfg = cfg
        self.net = net.to(device).train()
        self.device = device
        self.precision = precision
        self.entity_chunk = entity_chunk
        self.params = list(self.net.parameters())
        self.apart = self.net.win_open_params()
        self.opt = torch.optim.AdamW(self.params, lr=cfg["lr"], betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=cfg["weight_decay"], **adam_impl())
        # Core state per lane group (a group always holds the same lanes).
        self.states: list = []
        # URANIUM_BC_PROFILE=1: per-update phase times (device synchronised at each mark, so it
        # slows the update a little) in the step log, to measure the Rust bridge's share.
        self.phases = os.environ.get("URANIUM_BC_PROFILE", "") not in ("", "0")
        self._ph: dict = defaultdict(float)
        self._mark = 0.0
        self._collate = ThreadPoolExecutor(max_workers=1, thread_name_prefix="collate")
        kernels.warm(device)
        self.forward_seq = compiled(self.net.forward_seq)
        self.graphs = self.forward_seq is not self.net.forward_seq and compile_graphs()
        has_open, weighted = self.net.value.open is not None, cfg["value"].get("win_open") is not None
        if weighted and not has_open:
            print("[bc] WARNING: [value] win_open is set but the model has no win_open head: no win_open loss", flush=True)
        if has_open and not weighted:
            print("[bc] WARNING: the model has a win_open head but [value] win_open is not set: the head is not trained (041-R19)", flush=True)

    @classmethod
    def make(cls, cfg_json: str, start: str, dir: str | None, entity_chunk: int, core: str, device: torch.device, precision: str, variant: VariantConfig | None = None) -> "BcTrainer":
        """The `make` callback of `_core.run_bc`: a new net, or a checkpoint's weights (`init`)
        and optimizer state (`resume`). `variant`: the BC config's `[model]` (032 §2.7)."""
        cfg = json.loads(cfg_json)
        variant = variant or VariantConfig()
        if variant.outer and cfg["seq_len"] % variant.outer:
            raise ValueError(f"[model] outer = {variant.outer} must divide seq_len {cfg['seq_len']} (032 §2.7)")
        if start == "new":
            net = UraniumNet(new_net_config(core, cfg["value"].get("win_open") is not None, variant))
        else:
            net, meta = ckpt.load_model(Path(dir))
            if net.cfg.core.kind != CORES[core]:
                raise ValueError(f"checkpoint core {net.cfg.core.kind} != requested {CORES[core]}")
            # 032-AC11: the run identity (Rust) does not hold [model], so a different structure
            # switch is refused here.
            if net.cfg.variant != variant:
                raise ValueError(f"checkpoint [model] {net.cfg.variant} != requested {variant}")
        t = cls(cfg, net, device, precision, entity_chunk)
        if start == "resume":
            ckpt.load_optimizer(Path(dir), ckpt.OPTIM_MAIN, t.net, t.opt)
        return t

    def group_size(self) -> int:
        m, lanes = self.cfg["micro_lanes"], self.cfg["lanes"]
        return lanes if m == 0 else min(m, lanes)

    def train_batch(self, batch, lr: float, step: int) -> str:
        """One update over `batch` (a `_core.Batch`); returns the loss parts as JSON."""
        cfg, dev = self.cfg, self.device
        if self.phases:
            self._ph.clear()
            started = self._mark = time.perf_counter()
        valid, value_valid = batch.valid, batch.value_valid
        lanes, g = batch.lanes, self.group_size()
        groups = -(-lanes // g)
        if len(self.states) < groups:
            self.states += [None] * (groups - len(self.states))
        self.opt.zero_grad(set_to_none=True)
        parts = torch.zeros(len(HEADS) + 3, device=dev)  # total, heads..., value, win_open
        # The next group is collated on a worker thread (the Rust collate releases the GIL) while
        # this one trains, so the GPU does not wait for it; at most two collated groups are alive.
        # 032 §2.7: the entity padding cut for the extra tokens there too (a host copy).
        k = self.net.cfg.variant.tokens()
        collate = lambda i: stage(trim_entity_pad(batch.group(i * g, min((i + 1) * g, lanes), cfg["seq_len"], cfg["pad_entities"], self.net.cfg.heads.max_select), k), dev, self.precision)  # noqa: E731
        nxt = self._collate.submit(collate, 0)
        for i in range(groups):
            self.tick("other")
            hg = nxt.result()
            self.tick("collate_wait")
            if i + 1 < groups:
                nxt = self._collate.submit(collate, i + 1)
            parts += self.group_loss(hg, i, valid, value_valid, step)
        # 041-R3: one global L2 clip over the summed gradients; 041-R19: win_open apart.
        clip_global_norm(self.params, cfg["grad_clip"], self.apart)
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        self.opt.step()
        self.tick("optimizer")
        v = parts.tolist()
        out = {"total": v[0], **{k: v[1 + j] for j, k in enumerate(HEADS)}, "value": v[-2], "win_open": v[-1], "valid": valid}
        if self.phases:
            self._ph["python_total"] = time.perf_counter() - started
            out["phases"] = {k: round(x, 4) for k, x in self._ph.items()}
        return json.dumps(out)

    def tick(self, phase: str) -> None:
        """Adds the time since the last mark to `phase` (URANIUM_BC_PROFILE only)."""
        if not self.phases:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        self._ph[phase] += now - self._mark
        self._mark = now

    def group_loss(self, hg: dict, group: int, valid: float, value_valid: float, step: int) -> torch.Tensor:
        """Forward, loss and backward of one collated lane group (`Batch.group`); returns the
        detached loss parts."""
        cfg, dev, net = self.cfg, self.device, self.net
        t = cfg["seq_len"]
        b = hg["b"]
        obs, labels = to_obs(hg, dev), net.labels(hg, dev)
        priv = to_device(hg["privileged"], dev).float()
        w = to_device(hg["weights"], dev).float()
        target = to_device(hg["results"], dev).float()
        vw = to_device(hg["value_w"], dev).float()
        keep = to_device(hg["keep"], dev).bool()
        s = self.states[group]
        s0 = net.init_state(b, dev) if s is None else net.mask_state(s, keep)
        self.tick("to_device")
        torch.manual_seed(dropout_seed(cfg["data"]["seed"], step, group))
        if self.forward_seq is not self.net.forward_seq:
            mark_dynamic(obs, labels)
        if self.graphs:
            # 032-R19 CUDA graph replay: this group's forward and backward are one new step; the
            # previous group's outputs are no longer read.
            torch.compiler.cudagraph_mark_step_begin()
        with autocast(dev, self.precision):
            out = self.forward_seq(obs, labels, priv, s0, t, self.entity_chunk)
        self.tick("forward")
        # Detached when stored: a waiting group's state does not keep its graph alive (041-R2).
        # Copied under graph replay: the state outlives the next replay, which overwrites the
        # graph's output buffers (same values).
        state = net.detach_state(out["state"])
        self.states[group] = torch.utils._pytree.tree_map(torch.clone, state) if self.graphs else state

        hw = cfg["heads"]
        ce = [(-out["logp"][k].float() * w).sum() / valid for k in HEADS]
        # 041-R7: the win value regresses the replay result (privileged input), MSE over
        # decided games (A-3: undecided games have value weight 0).
        value_loss = ((out["value"][:, VALUE_WIN].float() - target) ** 2 * vw).sum() / value_valid
        loss = value_loss * cfg["value"]["win"]
        for k, l in zip(HEADS, ce):
            loss = loss + l * hw[k]
        # 041-R19: win_open regresses the same target (detached input, 032-R18), added last.
        open_loss = torch.zeros((), device=dev)
        k_open = cfg["value"].get("win_open")
        if out["win_open"] is not None and k_open is not None:
            open_loss = ((out["win_open"].float() - target) ** 2 * vw).sum() / value_valid
            loss = loss + open_loss * k_open
        self.tick("loss")
        fresh = [p for p in self.params if p.grad is None] if self.graphs else ()
        loss.backward()
        # Under graph replay the compiled backward's gradient outputs live in the graph's memory,
        # which the next group's replay overwrites: a gradient first set by this group gets its
        # own buffer, and the later groups add into it (the same sum, 041-R3).
        for p in fresh:
            if p.grad is not None:
                p.grad = p.grad.clone()
        self.tick("backward")
        return torch.stack([loss, *ce, value_loss, open_loss]).detach()

    def save(self, dir: str, train_json: str) -> None:
        """Weights, optimizer state and metadata (032-R6, 061-R10); the Rust loop adds the
        training state, run identity and configs."""
        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        ckpt.save_model(d, self.net)
        ckpt.save_optimizer(d, ckpt.OPTIM_MAIN, self.net, self.opt)
        ckpt.save_meta(d, self.net.cfg, json.loads(train_json), self.precision, impl_settings())
