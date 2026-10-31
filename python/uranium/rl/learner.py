"""RL learner of one agent (042): one gradient update over a batch of segments. Targets
(V-trace + UPGO, or GAE for the PPO baseline) come from the detached forward; the loss is

  policy (V-trace PG + UPGO, or PPO clip) + value (per reward component, 033-R3)
  - Σ_h entropy_h (used heads, 032-R10) + kl_coef · KL(π ‖ π_BC) (exact, per head, AMB-042-9)
  + win_open (042-R13, regresses the win v-trace target, not in the advantage)

KL re-runs the frozen BC network on the same inputs and taken actions, so both distributions are
the conditionals along the taken action (042, AMB-042-4). Lanes run in groups of `micro_lanes`
(this host's `URANIUM_RL_MICRO_LANES`, else the config's) with gradients summed before one
optimizer step, like BC (041). The same math as the Burn learner it replaces.

032-R19 ⑥ (042 learner optimisation, same formulas): the collate thread also stages a group for
the copy (page-locked on the CUDA host, the spatial planes rounded to bf16 under bf16 autocast,
the entity padding cut for the 032 §2.7 extra tokens, core_state0 and the outer segments' row
indices built on the host), so the training thread's copies do not wait for the GPU there; the
host computes the targets from the detached log-probs and values while the device runs the BC
forward (queued before the targets on every backend), and the targets of the 4 reward
components are one recursion; the statistics stay on the device until one read per update. The
learner as it was before is `uranium.rl.reference` (equivalence tests, `bench rl`)."""

import copy
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch

from uranium.agent.infer import leaves, rebuild
from uranium.bc.clip import clip_global_norm
from uranium.device import adam_impl, autocast
from uranium.model import to_obs
from uranium.model.net import LABEL_KEYS, OBS_KEYS, phase_split, stage, to_device, trim_entity_pad
from uranium.rl.returns import gae_valid, upgo_ratio, vtrace_ratio

HEADS = ("action", "delay", "queued", "selected", "target_unit", "target_loc")
# 042 / 060-R3: this host's lane group size (001-R9 `.env`), over the config's `micro_lanes`.
MICRO_LANES_ENV = "URANIUM_RL_MICRO_LANES"
# The arrays of a lane group the learner copies to the device (staged on the collate thread).
STAGE_KEYS = OBS_KEYS + LABEL_KEYS + ("privileged", "mask", "behaviour")
# Per lane group, the loss parts summed on the device (042-R2): policy, value, KL, win_open,
# KL per head, entropy per head.
PARTS = 16


def host_micro_lanes(cfg: dict) -> int:
    """Lanes per lane group (0 = all lanes): `URANIUM_RL_MICRO_LANES` when set and not empty, else
    `micro_lanes` of `config/rl.toml` (042 learner lane groups, 032-R19 ⑥). Execution only: the
    summed gradients are the same up to float rounding, so hosts may differ within a run."""
    v = os.environ.get(MICRO_LANES_ENV, "").strip()
    if not v:
        return int(cfg["micro_lanes"])
    if not v.isdecimal():
        raise ValueError(f"{MICRO_LANES_ENV}={v!r} (host .env, 042 learner lane groups): expected a lane count, 0 = all lanes")
    return int(v)


def kl_coef(cfg: dict, update: int) -> float:
    """`max(floor, init * 0.5^(update / half_life))` (042 KL schedule)."""
    k = cfg["kl"]
    return max(k["floor"], k["init"] * 0.5 ** (update / k["half_life_updates"]))


def kl_per_head(p: dict, q: dict) -> dict:
    """Per-row KL(p ‖ q) per head [rows], times the head's usage (032-R10)."""
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


def state0_host(net, g: dict):
    """Core state per lane (042 core_state0) on the host; a lane without one starts from zero
    (032-R5)."""
    s = net.init_state(g["b"], torch.device("cpu"))
    parts = leaves(s)
    for lane, st in enumerate(g["state0"]):
        if st is None:
            continue
        for leaf, (shape, v) in zip(parts, st):
            leaf[lane] = torch.as_tensor(v).reshape(shape)[0].to(leaf.dtype)
    return s


def state0_of(net, g: dict, device, pinned: bool | None = None):
    """`state0_host` on `device` (the collate thread's copy when it made one): one copy per state
    tensor, not per lane and tensor."""
    s = g["state0_host"] if "state0_host" in g else state0_host(net, g)
    return rebuild(s, iter([to_device(p, device, pinned) for p in leaves(s)]))


class Learner:
    """One learning agent: the trained net, the frozen BC reference, Adam (042 optimizer table)."""

    def __init__(self, cfg: dict, net, bc, device: torch.device, precision: str, algo: str, update: int = 0, micro_lanes: int | None = None):
        self.cfg = cfg
        self.device = device
        self.precision = precision
        self.algo = algo
        self.net = net.to(device).train()
        self.bc = bc.to(device).eval()
        for p in self.bc.parameters():
            p.requires_grad_(False)
        self.update_count = update
        self.micro_lanes = host_micro_lanes(cfg) if micro_lanes is None else micro_lanes
        # 032-R19 ⑥ for this learner only (not process switches: the inference servers share the
        # process): page-locked staging on the CUDA host only (the Linux ROCm host's GPU shares
        # its RAM, 060-R3: no page-locked buffers there), the spatial planes rounded to bf16 on the
        # host under bf16 autocast (their consumers round them the same way, `net.stage`).
        self.pinned = device.type == "cuda" and torch.version.hip is None
        self.spatial_bf16 = precision == "bf16"
        # The 032 §2.7 entity padding cut (`trim_entity_pad`). These three are attributes so
        # `bench rl --off` measures each alone and the tests compare without the cut.
        self.trim_pad = True
        self.opt = self.new_optimizer()
        self._collate = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rl-collate")
        # Per-phase times of an update (`bench rl --phases`): the device synchronised at each mark.
        self.phases = False
        self._ph: dict = defaultdict(float)
        self._mark = 0.0

    def new_optimizer(self) -> torch.optim.Optimizer:
        o = self.cfg["optimizer"]
        return torch.optim.Adam(self.net.parameters(), lr=o["lr"], betas=(o["beta1"], o["beta2"]), eps=o["eps"], weight_decay=o["weight_decay"], **adam_impl())

    def reset_to_bc(self) -> None:
        """042 Main Exploiter / League Exploiter reset to the BC policy (KL schedule restarts)."""
        with torch.no_grad():
            self.net.load_state_dict(self.bc.state_dict())
        self.opt = self.new_optimizer()
        self.update_count = 0

    def group_size(self, lanes: int) -> int:
        m = self.micro_lanes
        return max(1, lanes if m == 0 else min(m, lanes))

    def tick(self, phase: str) -> None:
        """Adds the time since the last mark to `phase` (`phases` only)."""
        if not self.phases:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        self._ph[phase] += now - self._mark
        self._mark = now

    def prepare(self, batch, start: int, end: int, t: int) -> dict:
        """Lanes `start..end` collated and staged for the copy (on the collate thread): the 032 §2.7
        entity padding cut (`trim_entity_pad`, dropout 0 in RL), the copied arrays page-locked
        (CUDA host) with the spatial planes in bf16 (bf16 autocast), core_state0 on the host and the
        outer segments' row indices from its counter (`phase_split`)."""
        net = self.net
        g = trim_entity_pad(batch.group(start, end, t, net.cfg.heads.max_select), net.cfg.variant.tokens() if self.trim_pad else 0)
        g = stage(g, self.device, self.precision, pinned=self.pinned, spatial_bf16=self.spatial_bf16, keys=STAGE_KEYS)
        s = state0_host(net, g)
        if net.outer is not None:
            g["phases"] = [self.pin(i) for i in phase_split(s[1][3], t, net.outer.segment)]
        g["state0_host"] = rebuild(s, iter([self.pin(p) for p in leaves(s)]))
        return g

    def pin(self, x: torch.Tensor) -> torch.Tensor:
        return x.pin_memory() if self.pinned else x

    def readback(self, logp: torch.Tensor, values: torch.Tensor):
        """The detached log-probs and values for the host targets, called before the BC forward is
        queued; the returned function gives them. CUDA host: an asynchronous copy into page-locked
        memory, waited for by an event after the BC forward is queued. Otherwise (CPU, Linux ROCm
        host) a plain copy now: it waits for the forward only, and the BC forward is queued right
        after, so the device runs it while the host computes the targets on either backend."""
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
        """Per-row coefficients of one lane group from the detached forward (on the host). The 4
        reward components (033-R3) run as 4·b rows of one recursion, component-major."""
        b, t = g["b"], g["t"]
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
        # Bootstrap after the trained steps: 0 after a terminal step, else the value there.
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
        else:
            adv4, vs4 = gae_valid(r4, ext, valid4, l["gamma"], self.cfg["ppo"]["gae_lambda"])
            for c, x in enumerate(comp(adv4)):
                pg += weights[c] * x
            rho_sum = float((ratio * valid).sum())
        vs = vs4.view(4, b, t).permute(1, 2, 0)
        coef = l["pg"] * pg + l["upgo"] * up if self.algo == "vtrace" else pg
        dev = self.device
        return {"coef": to_device(coef.reshape(-1), dev, self.pinned), "vs": to_device(vs.reshape(-1, 4), dev, self.pinned), "rho_sum": rho_sum, "adv_sum": float((pg * valid).sum())}

    def update(self, batch, weights) -> dict:
        """One update over `batch` (a `_core.SegmentBatch`, lanes x T); returns the statistics."""
        cfg, dev = self.cfg, self.device
        if self.phases:
            self._ph.clear()
            started = self._mark = time.perf_counter()
        t = cfg["seq_len"]
        lanes = batch.lanes
        g_size = self.group_size(lanes)
        groups = [(s, min(s + g_size, lanes)) for s in range(0, lanes, g_size)]
        coef_kl = kl_coef(cfg, self.update_count)
        epochs = 1 if self.algo == "vtrace" else max(1, cfg["ppo"]["epochs"])
        # The totals come from the batch itself; the lane groups are collated one ahead on a
        # worker thread (the Rust collate releases the GIL) while the previous one trains, as
        # in BC: at most two collated groups are alive (not every group of the batch at once).
        collate = lambda gi: self.prepare(batch, groups[gi][0], groups[gi][1], t)  # noqa: E731
        steps = int(sum(batch.trained))
        n = max(steps, 1)
        versions = [int(v) for v in batch.versions]
        # The loss parts of every group summed on the device in f64 (the sums of the host floats
        # they were before, bit for bit), read once after the optimizer step.
        parts = torch.zeros(PARTS, dtype=torch.float64, device=dev)
        rho_sum = adv_sum = 0.0
        norm = None
        # PPO keeps the first epoch's advantages and returns (computed before any change).
        ppo_targets: list = [None] * len(groups)
        apart = self.net.win_open_params()
        params = list(self.net.parameters())
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
                    part, rho, adv = self.group_loss(g, n, weights, coef_kl, ppo_targets, gi)
                    del g
                    if epoch == 0:
                        parts += part
                        rho_sum += rho
                        adv_sum += adv
            finally:
                # On an error: no collate job of this batch outlives the update (it would pin
                # memory and read the batch after the caller moved on).
                nxt.exception()
            # 042 optimizer table / 041-R3: global L2 clip after the lane-group sum; 042-R13: the
            # win_open head apart (041-R19).
            clipped = clip_global_norm(params, cfg["optimizer"]["grad_clip"], apart)
            if epoch == 0:
                norm = clipped
            for pg in self.opt.param_groups:
                pg["lr"] = cfg["optimizer"]["lr"]
            self.opt.step()
            self.tick("optimizer")
        nan = torch.full((1,), float("nan"), dtype=torch.float64, device=dev)
        vals = torch.cat([parts, nan if norm is None else norm.detach().double().reshape(1)]).tolist()
        self.tick("stats")
        stats = {
            "update": self.update_count + 1, "steps": steps, "kl_coef": coef_kl, "policy": vals[0], "value": vals[1], "win_open": vals[3], "kl": vals[2],
            "kl_heads": vals[4:10], "entropy_heads": vals[10:16], "rho_mean": rho_sum / n, "adv_mean": adv_sum / n,
            "lag_mean": sum(max(self.update_count - v, 0) for v in versions) / max(len(versions), 1),
            # The global gradient norm before the clip (first epoch; the 042 lane-group equivalence
            # check). Not in train.csv; null when not finite (the stats are JSON for the Rust loop).
            "grad_norm": vals[16] if math.isfinite(vals[16]) else None,
        }
        e = cfg["entropy"]
        bonus = sum(e[h] * stats["entropy_heads"][i] for i, h in enumerate(HEADS))
        stats["loss"] = stats["policy"] + stats["value"] + stats["win_open"] + stats["kl"] * coef_kl - bonus
        if self.phases:
            self._ph["python_total"] = time.perf_counter() - started
            stats["phases"] = {k: round(x, 4) for k, x in self._ph.items()}
        self.update_count += 1
        return stats

    def group_loss(self, g: dict, n: int, weights, coef_kl: float, ppo_targets: list, gi: int) -> tuple[torch.Tensor, float, float]:
        """Forward, targets, loss and backward of one lane group (divided by the batch's trained
        steps `n`); returns the detached loss parts (a device tensor, `PARTS`) and the host sums
        of ρ and of the advantage."""
        cfg, dev, net, pin = self.cfg, self.device, self.net, self.pinned
        t = g["t"]
        obs, labels = to_obs(g, dev, pinned=pin), net.labels(g, dev, pinned=pin)
        priv = to_device(g["privileged"], dev, pin).float()
        mask = to_device(g["mask"], dev, pin).float()
        s0 = state0_of(net, g, dev, pin)
        phases = [to_device(i, dev, pin) for i in g["phases"]] if "phases" in g else None
        self.tick("h2d")
        with autocast(dev, self.precision):
            # 032 §2.7: RL segments start at any decision, so an outer model runs phased.
            out = net.forward_seq(obs, labels, priv, s0, t, phased=True, phases=phases)
        logp = sum(out["logp"][k].float() for k in HEADS)
        values = out["value"].float()
        tg = ppo_targets[gi]
        read = self.readback(logp, values) if tg is None else None
        self.tick("forward")
        e = cfg["entropy"]
        ent_terms = [(out["entropy"][h].float() * mask).sum() / n for h in HEADS]
        entropy = sum(e[h] * term for h, term in zip(HEADS, ent_terms))
        # KL(π ‖ π_BC) with the frozen BC net on the same inputs (AMB-042-9). Queued before the
        # targets wait for the read-back, so the device runs it while the host computes them.
        with torch.no_grad(), autocast(dev, self.precision):
            bc = self.bc.forward_seq(obs, labels, priv, s0, t, phased=True, phases=phases)
        self.tick("bc_forward")
        kl = kl_per_head({k: (v.float() if v.is_floating_point() else v) for k, v in out["dists"].items()}, {k: (v.float() if v.is_floating_point() else v) for k, v in bc["dists"].items()})
        kl_terms = [(kl[h] * mask).sum() / n for h in HEADS]
        kl_total = sum(kl_terms)
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
        loss = policy + value - entropy + kl_total * coef_kl
        if win_open is not None:
            loss = loss + win_open
        self.tick("loss")
        loss.backward()
        self.tick("backward")
        zero = torch.zeros((), device=dev)
        part = torch.stack([policy.detach(), value.detach(), kl_total.detach(), zero if win_open is None else win_open.detach(), *[x.detach() for x in kl_terms], *[x.detach() for x in ent_terms]]).double()
        return part, tg["rho_sum"], tg["adv_sum"]


def frozen_copy(net):
    """A copy of `net` for the BC reference / reset (shares no tensors)."""
    return copy.deepcopy(net)


def stats_json(stats: dict) -> str:
    return json.dumps(stats)
