import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from uranium import ckpt, kernels
from uranium.clip import clip_global_norm
from uranium.device import adam_impl, autocast, compile_graphs, compiled, impl_settings, mark_dynamic
from uranium.model import UraniumNet, to_obs
from uranium.model.auxiliary import jepa_mean, jepa_moments, jepa_std, jepa_sums, priv_sum
from uranium.model.config import AUX_JEPA_SLOTS, CORES, VariantConfig, check_aux, new_net_config
from uranium.model.monitor import SPEC_COLUMNS, ReluMonitor, channel_counts_of, packed, relu_stats, spatial_alarm, spatial_due, spatial_every
from uranium.model.net import stage, to_device, trim_entity_pad

HEADS = ("action", "delay", "queued", "selected", "target_unit", "target_loc")
PARTS = ("total", *HEADS, "value", "win_open")
AUX_PARTS = ("aux_priv", "aux_jepa", *(f"aux_jepa_slot{i + 1}" for i in range(AUX_JEPA_SLOTS)))
VALUE_WIN = 0
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-5


def dropout_seed(seed: int, step: int, group: int) -> int:
    x = (seed * 0x9E3779B97F4A7C15 + (step << 20) + group * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    x ^= x >> 31
    return x & ((1 << 63) - 1)


def bc_aux_counts(batch, t: int, ks: list) -> dict:
    steps = [min(int(x), t) for x in batch.lane_steps()]
    return {"real": sum(steps), "priv": sum(int(x) for x in batch.lane_priv_steps(t)), "pairs": [sum(max(s - k, 0) for s in steps) for k in ks]}


class BcTrainer:
    def __init__(self, cfg: dict, net: UraniumNet, device: torch.device, precision: str, entity_chunk: int = 0):
        self.cfg = cfg
        self.aux = check_aux(net.cfg.variant, cfg.get("aux"))
        self.ks = list(net.cfg.variant.aux_jepa)
        self.monitor_every = spatial_every(cfg)
        self.net = net.to(device).train()
        self.device = device
        self.precision = precision
        self.entity_chunk = entity_chunk
        self.params = list(self.net.parameters())
        self.apart = self.net.win_open_params()
        self.opt = torch.optim.AdamW(self.params, lr=cfg["lr"], betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=cfg["weight_decay"], **adam_impl())
        self.states: list = []
        self.skipped = 0
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
        cfg = json.loads(cfg_json)
        variant = variant or VariantConfig()
        if variant.outer and cfg["seq_len"] % variant.outer:
            raise ValueError(f"[model] outer = {variant.outer} must divide seq_len {cfg['seq_len']} (032 §2.7)")
        if start == "new":
            seed = (cfg.get("train") or {}).get("model_seed")
            if seed is not None:
                torch.manual_seed(int(seed))
            net = UraniumNet(new_net_config(core, cfg["value"].get("win_open") is not None, variant))
        else:
            net, meta = ckpt.load_model(Path(dir))
            if net.cfg.core.kind != CORES[core]:
                raise ValueError(f"checkpoint core {net.cfg.core.kind} != requested {CORES[core]}")
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
        aux = bc_aux_counts(batch, cfg["seq_len"], self.ks) if self.aux else None
        parts = torch.zeros(len(PARTS) + (len(AUX_PARTS) if aux else 0), device=dev)
        moments = torch.zeros(4, self.net.cfg.entity.d_model + self.net.cfg.spatial.pool, dtype=torch.float64, device=dev) if "jepa" in self.aux else None
        mon = ReluMonitor(self.net, points="spatial") if spatial_due(self.monitor_every, step + 1) else None
        k = self.net.cfg.variant.tokens()
        collate = lambda i: stage(trim_entity_pad(batch.group(i * g, min((i + 1) * g, lanes), cfg["seq_len"], cfg["pad_entities"], self.net.cfg.heads.max_select), k), dev, self.precision)  # noqa: E731
        nxt = self._collate.submit(collate, 0)
        for i in range(groups):
            self.tick("other")
            hg = nxt.result()
            self.tick("collate_wait")
            if i + 1 < groups:
                nxt = self._collate.submit(collate, i + 1)
            parts += self.group_loss(hg, i, valid, value_valid, step, aux, moments, mon)
        extra = []
        if moments is not None:
            extra.append(torch.stack([jepa_std(moments[:2], aux["real"]), jepa_std(moments[2:], aux["real"])]))
        channels = None
        if mon is not None:
            st = mon.stats()
            channels = channel_counts_of(st)
            extra.append(packed(st))
        v = parts.tolist() if not extra else torch.cat([parts.double(), *extra]).tolist()
        tail = v[len(parts):]
        v = v[:len(parts)]
        skip = not all(math.isfinite(x) for x in v[:len(PARTS)] + [x for x, w in zip(v[len(PARTS):], self.aux_weights_of_parts()) if w > 0])
        if skip:
            self.skip_update(step, v)
        else:
            clip_global_norm(self.params, cfg["grad_clip"], self.apart)
            for pg in self.opt.param_groups:
                pg["lr"] = lr
            self.opt.step()
        self.tick("optimizer")
        out = {k: x if math.isfinite(x) else None for k, x in zip(PARTS, v)} | {"valid": valid}
        if aux:
            out.update(self.aux_log(v[len(PARTS):], aux, tail[:2] if moments is not None else (None, None)))
        if mon is not None:
            out.update(self.monitor_log(step + 1, tail[-len(SPEC_COLUMNS):], channels))
        if skip:
            out["skipped"] = True
        if self.skipped:
            out["skipped_steps"] = self.skipped
        if self.phases:
            self._ph["python_total"] = time.perf_counter() - started
            out["phases"] = {k: round(x, 4) for k, x in self._ph.items()}
        return json.dumps(out)

    def aux_weights_of_parts(self) -> list:
        lp, lj = self.aux.get("priv", 0.0), self.aux.get("jepa", 0.0)
        return [lp, lj, *[lj] * AUX_JEPA_SLOTS]

    def aux_log(self, v: list, aux: dict, tgt_std) -> dict:
        fin = lambda x: x if x is not None and math.isfinite(x) else None  # noqa: E731
        jepa = "jepa" in self.aux
        out = {"aux_priv": fin(v[0]) if "priv" in self.aux and aux["priv"] > 0 else None, "aux_jepa": fin(v[1]) if jepa and any(c > 0 for c in aux["pairs"]) else None,
               "jepa_tgt_std": fin(tgt_std[0]), "jepa_tgt_std_raw": fin(tgt_std[1])}
        for i in range(AUX_JEPA_SLOTS):
            live = jepa and i < len(self.ks) and aux["pairs"][i] > 0
            out[f"aux_jepa_slot{i + 1}"] = fin(v[2 + i]) if live else None
        return out

    def monitor_log(self, update: int, vals: list, channels: dict) -> dict:
        rec = dict(zip(SPEC_COLUMNS, vals))
        alarm = spatial_alarm(update, rec, channels)
        if alarm:
            print(f"[bc] WARNING: update {update}: spatial encoder collapse alarm (041-R25): {'; '.join(alarm)}", file=sys.stderr, flush=True)
            rec["spatial_alarm"] = alarm
        return rec

    def skip_update(self, step: int, v: list) -> None:
        self.skipped += 1
        self.opt.zero_grad(set_to_none=True)
        reset = 0
        for i, s in enumerate(self.states):
            leaves = [x for x in torch.utils._pytree.tree_leaves(s) if isinstance(x, torch.Tensor)]
            if not all(bool(torch.isfinite(x).all()) for x in leaves):
                self.states[i] = None
                reset += 1
        bad = [k for k, x in zip(PARTS + AUX_PARTS, v) if not math.isfinite(x)]
        print(f"[bc] update {step + 1}: non-finite loss {bad}: optimizer step skipped ({self.skipped} so far), {reset} lane group core state(s) reset (041-R3)", file=sys.stderr, flush=True)

    def tick(self, phase: str) -> None:
        if not self.phases:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        self._ph[phase] += now - self._mark
        self._mark = now

    def group_loss(self, hg: dict, group: int, valid: float, value_valid: float, step: int, aux: dict | None = None, moments: torch.Tensor | None = None,
                   mon: ReluMonitor | None = None) -> torch.Tensor:
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
            torch.compiler.cudagraph_mark_step_begin()
        with autocast(dev, self.precision):
            out = self.forward_seq(obs, labels, priv, s0, t, self.entity_chunk, **({"aux": True} if aux else {}))
        self.tick("forward")
        state = net.detach_state(out["state"])
        self.states[group] = torch.utils._pytree.tree_map(torch.clone, state) if self.graphs else state

        hw = cfg["heads"]
        ce = [(-out["logp"][k].float() * w).sum() / valid for k in HEADS]
        value_loss = ((out["value"][:, VALUE_WIN].float() - target) ** 2 * vw).sum() / value_valid
        loss = value_loss * cfg["value"]["win"]
        for k, l in zip(HEADS, ce):
            loss = loss + l * hw[k]
        open_loss = torch.zeros((), device=dev)
        k_open = cfg["value"].get("win_open")
        if out["win_open"] is not None and k_open is not None:
            open_loss = ((out["win_open"].float() - target) ** 2 * vw).sum() / value_valid
            loss = loss + open_loss * k_open
        aux_parts = []
        if aux:
            loss, aux_parts = self.aux_loss(out, hg, priv, b, t, aux, moments, loss)
        self.tick("loss")
        fresh = [p for p in self.params if p.grad is None] if self.graphs else ()
        loss.backward()
        for p in fresh:
            if p.grad is not None:
                p.grad = p.grad.clone()
        self.tick("backward")
        if mon is not None:
            rows = to_device(hg["real"], dev).bool()
            with mon:
                relu_stats(net, obs, labels, priv, s0, t, rows=rows, points="spatial", ctx=autocast(dev, self.precision), monitor=mon)
            self.tick("monitor")
        return torch.stack([loss, *ce, value_loss, open_loss, *aux_parts]).detach()

    def aux_loss(self, out: dict, hg: dict, priv, b: int, t: int, aux: dict, moments, loss):
        dev = self.device
        real = to_device(hg["real"], dev).bool()
        zero = torch.zeros((), device=dev)
        l_priv, l_jepa, per = zero, zero, [zero] * AUX_JEPA_SLOTS
        if "priv" in self.aux:
            rows = (real & to_device(hg["priv_present"], dev).bool()).float()
            l_priv = priv_sum(out["aux_priv"], priv, rows) / max(aux["priv"], 1)
            if self.aux["priv"] > 0:
                loss = loss + l_priv * self.aux["priv"]
        if "jepa" in self.aux:
            r = real.view(b, t)
            sums = jepa_sums(out["aux_jepa"], out["jepa_target"], r, r, self.ks)
            l_jepa, per_k = jepa_mean(sums, aux["pairs"])
            per = per_k + [zero] * (AUX_JEPA_SLOTS - len(per_k))
            moments += jepa_moments(out["jepa_target"], real.float())
            if self.aux["jepa"] > 0:
                loss = loss + l_jepa * self.aux["jepa"]
        return loss, [l_priv, l_jepa, *per]

    def save(self, dir: str, train_json: str) -> None:
        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        ckpt.save_model(d, self.net)
        ckpt.save_optimizer(d, ckpt.OPTIM_MAIN, self.net, self.opt)
        ckpt.save_meta(d, self.net.cfg, json.loads(train_json), self.precision, impl_settings())
