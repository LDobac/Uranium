import copy
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch

from uranium.clip import clip_global_norm
from uranium.device import adam_impl, autocast
from uranium.model import to_obs
from uranium.model.auxiliary import jepa_mean, jepa_moments, jepa_std, jepa_sums, priv_sum
from uranium.model.config import AUX_JEPA_SLOTS, check_aux
from uranium.model.monitor import SPEC_COLUMNS, ReluMonitor, channel_counts_of, packed, relu_stats, spatial_alarm, spatial_due, spatial_every
from uranium.model.net import LABEL_KEYS, OBS_KEYS, phase_split, stage, to_device, trim_entity_pad
from uranium.model.state import leaves, rebuild
from uranium.rl.returns import gae_valid, upgo_ratio, vtrace_ratio

HEADS = ("action", "delay", "queued", "selected", "target_unit", "target_loc")
MICRO_LANES_ENV = "URANIUM_RL_MICRO_LANES"
STAGE_KEYS = OBS_KEYS + LABEL_KEYS + ("privileged", "mask", "behaviour")
BASE_PARTS = 20
AUX_PARTS = 2 + AUX_JEPA_SLOTS
PARTS = BASE_PARTS + AUX_PARTS
SIL_PARTS = 3
REF_RACES = ("zerg", "terran", "protoss", "random")


def host_micro_lanes(cfg: dict) -> int:
    v = os.environ.get(MICRO_LANES_ENV, "").strip()
    if not v:
        return int(cfg["micro_lanes"])
    if not v.isdecimal():
        raise ValueError(f"{MICRO_LANES_ENV}={v!r} (host .env, 042 learner lane groups): expected a lane count, 0 = all lanes")
    return int(v)


def kl_coef(cfg: dict, update: int) -> float:
    k = cfg["kl"]
    return max(k["floor"], k["init"] * 0.5 ** (update / k["half_life_updates"]))


def kl_head_scale(cfg: dict) -> list | None:
    s = cfg["kl"].get("head_scale") or {}
    scale = [float(s.get(h, 1.0)) for h in HEADS]
    return None if all(x == 1.0 for x in scale) else scale


def scaled_kl(kl_terms: list, kl_total, scale: list | None):
    return kl_total if scale is None else sum(s * x for s, x in zip(scale, kl_terms))


def sil_settings(cfg: dict) -> dict:
    return {"enabled": False, "mode": "sil", "w_pg": 0.05, "w_v": 0.01, **(cfg.get("sil") or {})}


def sil_lane_count(batch) -> int:
    rets = getattr(batch, "sil_returns", None)
    if rets is None:
        return 0
    rets = list(rets)
    k = sum(r is not None for r in rets)
    if any(r is not None for r in rets[:len(rets) - k]):
        raise ValueError("042-R16: self-imitation buffer lanes must come after the queue lanes")
    return k


def finite_or_none(x: float):
    return x if math.isfinite(x) else None


def kl_tail(kl_steps: list, device) -> torch.Tensor:
    if not kl_steps:
        return torch.full((2,), float("nan"), dtype=torch.float64, device=device)
    x = torch.cat(kl_steps).float()
    hi = torch.nan_to_num(x, nan=float("-inf")).max()
    return torch.stack([torch.nanquantile(x, 0.99), hi]).double()


def kl_per_head(p: dict, q: dict) -> dict:
    full = lambda a, b: (a.exp() * (a - b)).sum(1)  # noqa: E731
    sel = (p["selected"].exp() * (p["selected"] - q["selected"])).sum(2) * p["selected_active"].float()
    return {
        "action": full(p["action"], q["action"]),
        "delay": full(p["delay"], q["delay"]),
        "queued": full(p["queued"], q["queued"]) * p["uses_queued"].float(),
        "selected": sel.sum(1) * p["uses_select"].float(),
        "target_unit": full(p["target_unit"], q["target_unit"]) * p["uses_target_unit"].float(),
        "target_loc": full(p["target_loc"], q["target_loc"]) * p["uses_target_loc"].float(),
    }


def component_weights(g: dict, weights) -> list:
    w = list(weights)
    on = g.get("r_value_policy")
    if on is not None and not bool(torch.as_tensor(on, dtype=torch.bool).all()):
        w[3] = torch.where(torch.as_tensor(on, dtype=torch.bool), w[3], 0.0)[:, None]
    scale = g.get("pseudo_scale")
    if scale is not None:
        scale = torch.as_tensor(scale, dtype=torch.float32).reshape(-1, 2)
        for c in (1, 2):
            col = scale[:, c - 1]
            if not bool((col == 1.0).all()):
                w[c] = (w[c] * col)[:, None]
    return w


def advantage_squares(pg: torch.Tensor, chan: list, valid: torch.Tensor) -> dict:
    return {"adv_sq": float((pg * pg * valid).sum()), "adv_sq_c": [float((x * x * valid).sum()) for x in chan]}


def sil_positive(g: dict, v: torch.Tensor, valid: torch.Tensor) -> dict:
    ret = g.get("sil_return")
    if ret is None:
        return {"sil_pos": 0.0, "sil_steps": 0.0}
    r = torch.as_tensor(ret, dtype=torch.float32).reshape(-1)
    sel = valid & torch.isfinite(r)[:, None]
    pos = sel & ((r[:, None] - v[:, :, 0]) > 0)
    return {"sil_pos": float(pos.sum()), "sil_steps": float(sel.sum())}


def diagnostics(adv_sq: float, adv_sq_c: list, value_ch: list, kl_q: list, n: int) -> dict:
    return {
        "adv_rms": math.sqrt(adv_sq / n), "adv_rms_channels": [math.sqrt(x / n) for x in adv_sq_c],
        "value_loss_channels": list(value_ch), "kl_p99": finite_or_none(kl_q[0]), "kl_max": finite_or_none(kl_q[1]),
    }


def sil_stats(sil: dict, parts: list, pos: float, pos_steps: float) -> dict:
    if not sil["enabled"]:
        return {"sil_loss": None, "sil_pos_frac": None}
    if sil["mode"] == "sil":
        loss, pos, pos_steps = parts
        return {"sil_loss": loss, "sil_pos_frac": pos / pos_steps if pos_steps > 0 else None}
    return {"sil_loss": None, "sil_pos_frac": pos / pos_steps if pos_steps > 0 else None}


def aux_stats(weights: dict, ks: list, vals: list | None = None, counts: dict | None = None, tgt_std: float | None = None) -> dict:
    out = {"aux_priv": None, "aux_jepa": None, "jepa_tgt_std": None, "jepa_tgt_std_raw": None, "aux_jepa_slots": [None] * AUX_JEPA_SLOTS, "spatial": None}
    if not weights or vals is None:
        return out
    fin = lambda x: finite_or_none(x) if x is not None else None  # noqa: E731
    jepa = "jepa" in weights
    if "priv" in weights and counts["priv"] > 0:
        out["aux_priv"] = fin(vals[0])
    if jepa and any(c > 0 for c in counts["pairs"]):
        out["aux_jepa"] = fin(vals[1])
    if jepa:
        out["jepa_tgt_std"], out["jepa_tgt_std_raw"] = fin(tgt_std[0]), fin(tgt_std[1])
    out["aux_jepa_slots"] = [fin(vals[2 + i]) if jepa and i < len(ks) and counts["pairs"][i] > 0 else None for i in range(AUX_JEPA_SLOTS)]
    return out


def rl_aux_counts(batch, t: int, queue: int, ks: list) -> dict:
    trained = [int(x) for x in batch.trained][:queue]
    real = [min(int(x), t) for x in batch.real_steps][:queue]
    priv = [int(x) for x in batch.priv_trained][:queue]
    return {"priv": sum(priv), "real": sum(real), "pairs": [sum(max(0, min(tr, r - k)) for tr, r in zip(trained, real)) for k in ks]}


def kl_ref_stats(sums: list | None, steps: list | None) -> list:
    if sums is None:
        return [None] * len(REF_RACES)
    return [finite_or_none(s / c) if c > 0 else None for s, c in zip(sums, steps)]


def lanes_by_ref(races: list, race_ref: list, lanes: int, size: int) -> list:
    ref = [race_ref[int(r)] for r in list(races)[:lanes]]
    out = []
    for r in sorted(set(ref)):
        mine = [i for i in range(lanes) if ref[i] == r]
        out += [(mine[s:s + size], r) for s in range(0, len(mine), size)]
    return out


def value_channels(values: torch.Tensor, vs: torch.Tensor, mask: torch.Tensor, n: int) -> torch.Tensor:
    return ((values - vs) ** 2 * mask[:, None]).sum(0) * (0.5 / n)


def state0_host(net, g: dict):
    s = net.init_state(g["b"], torch.device("cpu"))
    parts = leaves(s)
    for lane, st in enumerate(g["state0"]):
        if st is None:
            continue
        for leaf, (shape, v) in zip(parts, st):
            leaf[lane] = torch.as_tensor(v).reshape(shape)[0].to(leaf.dtype)
    return s


def state0_of(net, g: dict, device, pinned: bool | None = None):
    s = g["state0_host"] if "state0_host" in g else state0_host(net, g)
    return rebuild(s, iter([to_device(p, device, pinned) for p in leaves(s)]))


class Learner:
    def __init__(self, cfg: dict, net, bc, device: torch.device, precision: str, algo: str, update: int = 0, micro_lanes: int | None = None, refs: list | None = None, race_ref: list | None = None):
        self.cfg = cfg
        self.aux = check_aux(net.cfg.variant, cfg.get("aux"))
        self.ks = list(net.cfg.variant.aux_jepa)
        self.monitor_every = spatial_every(cfg)
        self.device = device
        self.precision = precision
        self.algo = algo
        self.net = net.to(device).train()
        self.bc = bc.to(device).eval()
        for p in self.bc.parameters():
            p.requires_grad_(False)
        self.refs = [self.bc]
        for r in refs or []:
            r = r.to(device).eval()
            for p in r.parameters():
                p.requires_grad_(False)
            self.refs.append(r)
        if race_ref is not None and (len(race_ref) != len(REF_RACES) or any(not 0 <= int(i) < len(self.refs) for i in race_ref)):
            raise ValueError(f"044-R9: race_ref {race_ref} needs one index into the {len(self.refs)} references per race {REF_RACES}")
        self.race_ref = None if race_ref is None else [int(i) for i in race_ref]
        self.update_count = update
        self.kl_scale = kl_head_scale(cfg)
        self.micro_lanes = host_micro_lanes(cfg) if micro_lanes is None else micro_lanes
        self.pinned = device.type == "cuda" and torch.version.hip is None
        self.spatial_bf16 = precision == "bf16"
        self.trim_pad = False
        self.opt = self.new_optimizer()
        self._collate = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rl-collate")
        self.phases = False
        self._ph: dict = defaultdict(float)
        self._mark = 0.0

    def new_optimizer(self) -> torch.optim.Optimizer:
        o = self.cfg["optimizer"]
        return torch.optim.Adam(self.net.parameters(), lr=o["lr"], betas=(o["beta1"], o["beta2"]), eps=o["eps"], weight_decay=o["weight_decay"], **adam_impl())

    def reset_to_bc(self) -> None:
        with torch.no_grad():
            self.net.load_state_dict(self.bc.state_dict())
        self.opt = self.new_optimizer()
        self.update_count = 0

    def group_size(self, lanes: int) -> int:
        m = self.micro_lanes
        return max(1, lanes if m == 0 else min(m, lanes))

    def tick(self, phase: str) -> None:
        if not self.phases:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        self._ph[phase] += now - self._mark
        self._mark = now

    def prepare(self, batch, lanes, t: int) -> dict:
        net = self.net
        ms = net.cfg.heads.max_select
        g = batch.group(lanes[0], lanes[1], t, ms) if isinstance(lanes, tuple) else batch.group_lanes(list(lanes), t, ms)
        g = trim_entity_pad(g, net.cfg.variant.tokens() if self.trim_pad else 0)
        g = stage(g, self.device, self.precision, pinned=self.pinned, spatial_bf16=self.spatial_bf16, keys=STAGE_KEYS)
        s = state0_host(net, g)
        if net.outer is not None:
            g["phases"] = [self.pin(i) for i in phase_split(s[1][3], t, net.outer.segment)]
        g["state0_host"] = rebuild(s, iter([self.pin(p) for p in leaves(s)]))
        return g

    def pin(self, x: torch.Tensor) -> torch.Tensor:
        return x.pin_memory() if self.pinned else x

    def readback(self, logp: torch.Tensor, values: torch.Tensor):
        lp, v = logp.detach(), values.detach()
        if not self.pinned:
            host = (lp.cpu(), v.cpu())
            return lambda: host
        host = [torch.empty(x.shape, dtype=x.dtype, pin_memory=True).copy_(x, non_blocking=True) for x in (lp, v)]
        done = torch.cuda.Event()
        done.record()

        def wait():
            done.synchronize()
            return host[0], host[1]

        return wait

    def targets(self, g: dict, logp: torch.Tensor, values: torch.Tensor, weights) -> dict:
        b, t = g["b"], g["t"]
        weights = component_weights(g, weights)
        l = self.cfg["loss"]
        length = torch.as_tensor(g["trained"], dtype=torch.long)
        terminal = torch.as_tensor(g["terminal"], dtype=torch.bool)
        behaviour = torch.as_tensor(g["behaviour"]).float().view(b, t)
        rewards = torch.as_tensor(g["rewards"]).float().view(b, t, 4)
        lp = logp.float().cpu().view(b, t)
        v = values.float().cpu().view(b, t, 4)
        log_rho = lp - behaviour
        valid = torch.arange(t)[None, :] < length[:, None]
        ratio = log_rho.exp()
        rows = lambda x: x.permute(2, 0, 1).reshape(4 * b, t)  # noqa: E731
        comp = lambda x: [x[c * b:(c + 1) * b] for c in range(4)]  # noqa: E731
        len4, valid4 = length.repeat(4), valid.repeat(4, 1)
        ext = torch.cat([rows(v), torch.zeros(4 * b, 1)], 1)
        boot = torch.where(terminal.repeat(4), torch.zeros(4 * b), ext.gather(1, len4[:, None])[:, 0])
        ext = ext.scatter(1, len4[:, None], boot[:, None])
        r4 = rows(rewards)
        pg = torch.zeros(b, t)
        up = torch.zeros(b, t)
        if self.algo == "vtrace":
            ratio4 = ratio.repeat(4, 1)
            vs4, pg4, rho4 = vtrace_ratio(r4, ext, ratio4, valid4, l["gamma"], l["td_lambda"], l["vtrace_rho_bar"], l["vtrace_c_bar"])
            up4 = upgo_ratio(r4, ext, ratio4, len4, valid4, l["gamma"], l["vtrace_rho_bar"])
            for c, (x, y) in enumerate(zip(comp(pg4), comp(up4))):
                pg += weights[c] * x
                up += weights[c] * y
            rho_sum = float(rho4[:b].sum())
            chan = comp(pg4)
        else:
            adv4, vs4 = gae_valid(r4, ext, valid4, l["gamma"], self.cfg["ppo"]["gae_lambda"])
            for c, x in enumerate(comp(adv4)):
                pg += weights[c] * x
            rho_sum = float((ratio * valid).sum())
            chan = comp(adv4)
        vs = vs4.view(4, b, t).permute(1, 2, 0)
        coef = l["pg"] * pg + l["upgo"] * up if self.algo == "vtrace" else pg
        dev = self.device
        out = {"coef": to_device(coef.reshape(-1), dev, self.pinned), "vs": to_device(vs.reshape(-1, 4), dev, self.pinned), "rho_sum": rho_sum, "adv_sum": float((pg * valid).sum())}
        out.update(advantage_squares(pg, chan, valid))
        out.update(sil_positive(g, v, valid))
        return out

    def update(self, batch, weights) -> dict:
        cfg, dev = self.cfg, self.device
        if self.phases:
            self._ph.clear()
            started = self._mark = time.perf_counter()
        t = cfg["seq_len"]
        lanes = batch.lanes
        g_size = self.group_size(lanes)
        sil = sil_settings(cfg)
        k = sil_lane_count(batch)
        apart_lanes = k if sil["mode"] == "sil" else 0
        normal = lanes - apart_lanes
        multi = self.race_ref is not None and len(set(self.race_ref)) > 1
        if multi:
            groups = [(ls, False, r) for ls, r in lanes_by_ref(batch.ref_races, self.race_ref, normal, g_size)]
        else:
            groups = [((s, min(s + g_size, normal)), False, 0) for s in range(0, normal, g_size)]
        groups += [((s, min(s + g_size, lanes)), True, 0) for s in range(normal, lanes, g_size)]
        coef_kl = kl_coef(cfg, self.update_count)
        epochs = 1 if self.algo == "vtrace" else max(1, cfg["ppo"]["epochs"])
        collate = lambda gi: self.prepare(batch, groups[gi][0], t)  # noqa: E731
        trained = [int(x) for x in batch.trained]
        n = max(sum(trained[:normal]), 1)
        steps = sum(trained[:lanes - k])
        n_sil = max(sum(trained[normal:]), 1)
        versions = [int(v) for v in batch.versions][:lanes - k]
        parts = torch.zeros(PARTS, dtype=torch.float64, device=dev)
        sil_parts = torch.zeros(SIL_PARTS, dtype=torch.float64, device=dev)
        host = defaultdict(float)
        adv_sq_c = [0.0] * 4
        kl_steps: list = []
        norm = None
        ppo_targets: list = [None] * len(groups)
        apart = self.net.win_open_params()
        params = list(self.net.parameters())
        aux = rl_aux_counts(batch, t, lanes - k, self.ks) if self.aux else None
        net_cfg = self.net.cfg
        moments = torch.zeros(4, net_cfg.entity.d_model + net_cfg.spatial.pool, dtype=torch.float64, device=dev) if "jepa" in self.aux else None
        mon = ReluMonitor(self.net, points="spatial") if spatial_due(self.monitor_every, self.update_count + 1) else None
        kl_race = None
        if self.race_ref is not None:
            kl_race = torch.zeros(len(REF_RACES), dtype=torch.float64, device=dev)
            race_steps = [0.0] * len(REF_RACES)
            for r, x in zip(list(batch.ref_races)[:normal], trained[:normal]):
                race_steps[int(r)] += x
        for epoch in range(epochs):
            self.opt.zero_grad(set_to_none=True)
            nxt = self._collate.submit(collate, 0)
            try:
                for gi in range(len(groups)):
                    self.tick("other")
                    g = nxt.result()
                    self.tick("collate_wait")
                    if gi + 1 < len(groups):
                        nxt = self._collate.submit(collate, gi + 1)
                    if groups[gi][1]:
                        part = self.sil_group_loss(g, n_sil, sil)
                        if epoch == 0:
                            sil_parts += part
                        del g
                        continue
                    first = epoch == 0
                    part, tg, kl_step = self.group_loss(g, n, weights, coef_kl, ppo_targets, gi, aux, moments if first else None, mon if first else None, groups[gi][2])
                    if first and kl_race is not None:
                        lane_kl = torch.nan_to_num(kl_step.view(g["b"], -1).double(), nan=0.0).sum(1)
                        kl_race.index_add_(0, to_device(g["ref_race"], dev, self.pinned).long(), lane_kl)
                    del g
                    if epoch == 0:
                        parts += part
                        for key in ("rho_sum", "adv_sum", "adv_sq", "sil_pos", "sil_steps"):
                            host[key] += tg[key]
                        adv_sq_c = [a + x for a, x in zip(adv_sq_c, tg["adv_sq_c"])]
                        kl_steps.append(kl_step)
            finally:
                nxt.exception()
            clipped = clip_global_norm(params, cfg["optimizer"]["grad_clip"], apart)
            if epoch == 0:
                norm = clipped
            for pg in self.opt.param_groups:
                pg["lr"] = cfg["optimizer"]["lr"]
            self.opt.step()
            self.tick("optimizer")
        nan = torch.full((1,), float("nan"), dtype=torch.float64, device=dev)
        extra, channels = [], None
        if moments is not None:
            extra.append(torch.stack([jepa_std(moments[:2], aux["real"]), jepa_std(moments[2:], aux["real"])]))
        if mon is not None:
            st = mon.stats()
            channels = channel_counts_of(st)
            extra.append(packed(st))
        races = [] if kl_race is None else [kl_race]
        vals = torch.cat([parts, nan if norm is None else norm.detach().double().reshape(1), kl_tail(kl_steps, dev), sil_parts, *races, *extra]).tolist()
        at = PARTS + 3 + SIL_PARTS
        kl_race_sums = None if kl_race is None else vals[at:at + len(REF_RACES)]
        tail = vals[at + (0 if kl_race is None else len(REF_RACES)):]
        self.tick("stats")
        stats = {
            "update": self.update_count + 1, "steps": steps, "kl_coef": coef_kl, "policy": vals[0], "value": vals[1], "win_open": vals[3], "kl": vals[2],
            "kl_heads": vals[4:10], "entropy_heads": vals[10:16], "rho_mean": host["rho_sum"] / n, "adv_mean": host["adv_sum"] / n,
            "lag_mean": sum(max(self.update_count - v, 0) for v in versions) / max(len(versions), 1),
            "grad_norm": finite_or_none(vals[PARTS]),
        }
        stats.update(diagnostics(host["adv_sq"], adv_sq_c, vals[16:BASE_PARTS], vals[PARTS + 1:PARTS + 3], n))
        stats.update(sil_stats(sil, vals[PARTS + 3:PARTS + 3 + SIL_PARTS], host["sil_pos"], host["sil_steps"]))
        stats.update(aux_stats(self.aux, self.ks, vals[BASE_PARTS:PARTS], aux, tail[:2] if moments is not None else (None, None)))
        stats["kl_ref"] = kl_ref_stats(kl_race_sums, None if kl_race is None else race_steps)
        if mon is not None:
            rec = dict(zip(SPEC_COLUMNS, tail[-len(SPEC_COLUMNS):]))
            stats["spatial"] = list(rec.values())
            alarm = spatial_alarm(stats["update"], rec, channels)
            if alarm:
                print(f"[rl] WARNING: update {stats['update']}: spatial encoder collapse alarm (041-R25, 042-R21): {'; '.join(alarm)}", file=sys.stderr, flush=True)
        e = cfg["entropy"]
        bonus = sum(e[h] * stats["entropy_heads"][i] for i, h in enumerate(HEADS))
        kl_loss = scaled_kl(stats["kl_heads"], stats["kl"], self.kl_scale)
        stats["loss"] = stats["policy"] + stats["value"] + stats["win_open"] + kl_loss * coef_kl - bonus
        if stats["sil_loss"] is not None:
            stats["loss"] += stats["sil_loss"]
        for key, name in (("priv", "aux_priv"), ("jepa", "aux_jepa")):
            if self.aux.get(key, 0.0) > 0 and stats[name] is not None:
                stats["loss"] += self.aux[key] * stats[name]
        if self.phases:
            self._ph["python_total"] = time.perf_counter() - started
            stats["phases"] = {k: round(x, 4) for k, x in self._ph.items()}
        self.update_count += 1
        return stats

    def group_loss(self, g: dict, n: int, weights, coef_kl: float, ppo_targets: list, gi: int, aux: dict | None = None, moments: torch.Tensor | None = None,
                   mon: ReluMonitor | None = None, ref: int = 0) -> tuple[torch.Tensor, dict, torch.Tensor]:
        cfg, dev, net, pin = self.cfg, self.device, self.net, self.pinned
        t = g["t"]
        obs, labels = to_obs(g, dev, pinned=pin), net.labels(g, dev, pinned=pin)
        priv = to_device(g["privileged"], dev, pin).float()
        mask = to_device(g["mask"], dev, pin).float()
        s0 = state0_of(net, g, dev, pin)
        phases = [to_device(i, dev, pin) for i in g["phases"]] if "phases" in g else None
        self.tick("h2d")
        with autocast(dev, self.precision):
            out = net.forward_seq(obs, labels, priv, s0, t, phased=True, phases=phases, **({"aux": True} if aux else {}))
        logp = sum(out["logp"][k].float() for k in HEADS)
        values = out["value"].float()
        tg = ppo_targets[gi]
        read = self.readback(logp, values) if tg is None else None
        self.tick("forward")
        e = cfg["entropy"]
        ent_terms = [(out["entropy"][h].float() * mask).sum() / n for h in HEADS]
        entropy = sum(e[h] * term for h, term in zip(HEADS, ent_terms))
        with torch.no_grad(), autocast(dev, self.precision):
            bc = self.refs[ref].forward_seq(obs, labels, priv, s0, t, phased=True, phases=phases)
        self.tick("bc_forward")
        kl = kl_per_head({k: (v.float() if v.is_floating_point() else v) for k, v in out["dists"].items()}, {k: (v.float() if v.is_floating_point() else v) for k, v in bc["dists"].items()})
        kl_terms = [(kl[h] * mask).sum() / n for h in HEADS]
        kl_total = sum(kl_terms)
        kl_step = torch.where(mask > 0, sum(kl[h] for h in HEADS).detach(), float("nan"))
        self.tick("kl")
        if tg is None:
            tg = self.targets(g, *read(), weights)
            if self.algo == "ppo":
                ppo_targets[gi] = tg
        self.tick("targets")
        coef = tg["coef"]
        if self.algo == "vtrace":
            policy = -(logp * coef)
        else:
            clip = cfg["ppo"]["clip"]
            old = to_device(g["behaviour"], dev, pin).float()
            ratio = (logp - old).exp()
            policy = -torch.minimum(ratio * coef, ratio.clamp(1 - clip, 1 + clip) * coef)
        policy = (policy * mask).sum() / n
        vs = tg["vs"]
        value = ((values - vs) ** 2).sum(1).mul(mask).sum() * (0.5 * cfg["loss"]["value"] / n)
        win_open = None
        if out["win_open"] is not None:
            win_open = ((out["win_open"].float() - vs[:, 0]) ** 2 * mask).sum() * (0.5 * cfg["loss"]["win_open"] / n)
        loss = policy + value - entropy + scaled_kl(kl_terms, kl_total, self.kl_scale) * coef_kl
        if win_open is not None:
            loss = loss + win_open
        lane_rows = None
        if aux is not None or mon is not None:
            queue = torch.as_tensor(~torch.isfinite(torch.as_tensor(g["sil_return"], dtype=torch.float32)))
            lane_rows = to_device(queue.repeat_interleave(t), dev, pin).bool()
            real = to_device(g["real"], dev, pin).bool()
        aux_part = torch.zeros(AUX_PARTS, device=dev)
        if aux is not None:
            loss, aux_part = self.aux_loss(out, g, priv, mask, real, lane_rows, t, aux, moments, loss)
        self.tick("loss")
        loss.backward()
        self.tick("backward")
        if mon is not None:
            with mon:
                relu_stats(net, obs, labels, priv, s0, t, rows=real & lane_rows, points="spatial", phased=True, phases=phases, ctx=autocast(dev, self.precision), monitor=mon)
            self.tick("monitor")
        zero = torch.zeros((), device=dev)
        value_ch = value_channels(values.detach(), vs, mask, n)
        part = torch.cat([torch.stack([policy.detach(), value.detach(), kl_total.detach(), zero if win_open is None else win_open.detach(), *[x.detach() for x in kl_terms], *[x.detach() for x in ent_terms]]), value_ch, aux_part.detach()]).double()
        return part, tg, kl_step

    def aux_loss(self, out: dict, g: dict, priv, mask, real, lane_rows, t: int, aux: dict, moments, loss):
        dev = self.device
        b = g["b"]
        trained = (mask > 0) & lane_rows
        zero = torch.zeros((), device=dev)
        l_priv, l_jepa, per = zero, zero, [zero] * AUX_JEPA_SLOTS
        if "priv" in self.aux:
            rows = (trained & to_device(g["priv_present"], dev, self.pinned).bool()).float()
            l_priv = priv_sum(out["aux_priv"], priv, rows) / max(aux["priv"], 1)
            if self.aux["priv"] > 0:
                loss = loss + l_priv * self.aux["priv"]
        if "jepa" in self.aux:
            sums = jepa_sums(out["aux_jepa"], out["jepa_target"], trained.view(b, t), real.view(b, t), self.ks)
            l_jepa, per_k = jepa_mean(sums, aux["pairs"])
            per = per_k + [zero] * (AUX_JEPA_SLOTS - len(per_k))
            if moments is not None:
                moments += jepa_moments(out["jepa_target"], (real & lane_rows).float())
            if self.aux["jepa"] > 0:
                loss = loss + l_jepa * self.aux["jepa"]
        return loss, torch.stack([l_priv, l_jepa, *per])

    def sil_group_loss(self, g: dict, n_sil: int, sil: dict) -> torch.Tensor:
        dev, net, pin = self.device, self.net, self.pinned
        t = g["t"]
        obs, labels = to_obs(g, dev, pinned=pin), net.labels(g, dev, pinned=pin)
        priv = to_device(g["privileged"], dev, pin).float()
        mask = to_device(g["mask"], dev, pin).float()
        s0 = state0_of(net, g, dev, pin)
        phases = [to_device(i, dev, pin) for i in g["phases"]] if "phases" in g else None
        with autocast(dev, self.precision):
            out = net.forward_seq(obs, labels, priv, s0, t, phased=True, phases=phases)
        logp = sum(out["logp"][k].float() for k in HEADS)
        v_win = out["value"].float()[:, 0]
        ret = torch.as_tensor(g["sil_return"], dtype=torch.float32).reshape(-1).repeat_interleave(t).to(dev)
        adv = (ret - v_win).clamp(min=0.0)
        a = adv.detach()
        pg = (-(logp * a) * mask).sum() / n_sil
        vl = (0.5 * adv * adv * mask).sum() / n_sil
        loss = sil["w_pg"] * pg + sil["w_v"] * vl
        loss.backward()
        self.tick("sil")
        return torch.stack([loss.detach(), ((a > 0).float() * mask).sum(), mask.sum()]).double()


def frozen_copy(net):
    return copy.deepcopy(net)


def stats_json(stats: dict) -> str:
    return json.dumps(stats)
