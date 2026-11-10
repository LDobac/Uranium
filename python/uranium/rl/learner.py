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
core_state0 and the outer segments' row indices built on the host), so the training thread's copies do not wait for the GPU there; the
host computes the targets from the detached log-probs and values while the device runs the BC
forward (queued before the targets on every backend), and the targets of the 4 reward
components are one recursion; the statistics stay on the device until one read per update. The
learner as it was before is `uranium.rl.reference` (equivalence tests, `bench rl`).

AMB-042-17: the KL schedule is a run setting counted from the run's update 0 and each head's KL
term has a multiplier `[kl.head_scale]` (042-R9; all 1.0 = the earlier loss, bit for bit); the
statistics add the 042-R2 진단 로그 (per-channel advantage RMS and value loss, per-step KL p99 /
max); 042-R16 self-imitation buffer lanes (`sil_returns`, last in the batch) train the "sil" loss
apart or ride along as ordinary lanes ("replay"). Without buffer lanes the update is as before.

042-R21 (model 2): with the 032-R24 aux modules the loss adds λ_priv L_priv + λ_jepa L_jepa (`[aux]`,
queue lanes only, batch-wide divisors from the host), and `[monitor] spatial_every` adds the 032-R25
dead-channel statistics (a separate no_grad eval pass per lane group, read with the update's other
statistics). Without them the update and its statistics are as before (aux keys `None`)."""

import copy
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch

from uranium.agent.infer import leaves, rebuild
from uranium.bc.clip import clip_global_norm
from uranium.device import adam_impl, autocast
from uranium.model import to_obs
from uranium.model.auxiliary import jepa_mean, jepa_moments, jepa_std, jepa_sums, priv_sum
from uranium.model.config import AUX_JEPA_SLOTS, check_aux
from uranium.model.monitor import SPEC_COLUMNS, ReluMonitor, channel_counts_of, packed, relu_stats, spatial_alarm, spatial_due, spatial_every
from uranium.model.net import LABEL_KEYS, OBS_KEYS, phase_split, stage, to_device, trim_entity_pad
from uranium.rl.returns import gae_valid, upgo_ratio, vtrace_ratio

HEADS = ("action", "delay", "queued", "selected", "target_unit", "target_loc")
# 042 / 060-R3: this host's lane group size (001-R9 `.env`), over the config's `micro_lanes`.
MICRO_LANES_ENV = "URANIUM_RL_MICRO_LANES"
# The arrays of a lane group the learner copies to the device (staged on the collate thread).
STAGE_KEYS = OBS_KEYS + LABEL_KEYS + ("privileged", "mask", "behaviour")
# Per lane group, the loss parts summed on the device (042-R2): policy, value, KL, win_open,
# KL per head, entropy per head, the value loss per channel (진단 로그, AMB-042-17), then the aux
# losses before λ (042-R21 ③: aux_priv, aux_jepa, one per configured k; 0 without the modules).
BASE_PARTS = 20
AUX_PARTS = 2 + AUX_JEPA_SLOTS
PARTS = BASE_PARTS + AUX_PARTS
# 042-R16 "sil" lane-group parts: the weighted loss, steps with A+ > 0, trained steps.
SIL_PARTS = 3


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
    """`max(floor, init * 0.5^(update / half_life))` (042 KL schedule, 042-R9); `update` = the
    learner's updates in this run (a new run, `--init` included, starts at 0, AMB-042-17)."""
    k = cfg["kl"]
    return max(k["floor"], k["init"] * 0.5 ** (update / k["half_life_updates"]))


def kl_head_scale(cfg: dict) -> list | None:
    """`[kl.head_scale]` in `HEADS` order (042-R9, AMB-042-17); `None` when every multiplier is
    1.0 (absent included): the KL term is then the plain sum, as before, bit for bit."""
    s = cfg["kl"].get("head_scale") or {}
    scale = [float(s.get(h, 1.0)) for h in HEADS]
    return None if all(x == 1.0 for x in scale) else scale


def scaled_kl(kl_terms: list, kl_total, scale: list | None):
    """The KL term the loss multiplies by the schedule coefficient: Σ_h scale_h · KL_h, or
    `kl_total` (the unscaled sum) without multipliers."""
    return kl_total if scale is None else sum(s * x for s, x in zip(scale, kl_terms))


def sil_settings(cfg: dict) -> dict:
    """`[sil]` (042-R16) with the defaults of an absent table (off)."""
    return {"enabled": False, "mode": "sil", "w_pg": 0.05, "w_v": 0.01, **(cfg.get("sil") or {})}


def sil_lane_count(batch) -> int:
    """042-R16 buffer lanes of `batch` (`sil_returns`, `None` = a queue lane); they must be the
    last lanes. A batch without the attribute has none."""
    rets = getattr(batch, "sil_returns", None)
    if rets is None:
        return 0
    rets = list(rets)
    k = sum(r is not None for r in rets)
    if any(r is not None for r in rets[:len(rets) - k]):
        raise ValueError("042-R16: self-imitation buffer lanes must come after the queue lanes")
    return k


def finite_or_none(x: float):
    """A statistic for the Rust run loop's JSON (NaN / inf is not JSON): `None` when not finite."""
    return x if math.isfinite(x) else None


def kl_tail(kl_steps: list, device) -> torch.Tensor:
    """042-R2 진단 로그: [p99, max] of the per-step KL over the trained steps (NaN elsewhere in
    `kl_steps`), on the device (read with the other statistics); not finite (NaN / -inf, `None` in
    the statistics) without a trained step."""
    if not kl_steps:
        return torch.full((2,), float("nan"), dtype=torch.float64, device=device)
    x = torch.cat(kl_steps).float()
    hi = torch.nan_to_num(x, nan=float("-inf")).max()
    return torch.stack([torch.nanquantile(x, 0.99), hi]).double()


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


def component_weights(g: dict, weights) -> list:
    """The 033 weight of each reward component for the advantage of a lane group: a scalar, but
    r_value (component 3) per lane [b, 1] when some lane's game gives it no policy weight (042
    "r_value의 게임별 적용", AMB-042-14: `r_value_policy` false = a model opponent with
    `r_value_vs_model = false`), and the build / units channels (components 1, 2) per lane [b, 1]
    when some lane's game scales them (033-R11 "scout", AMB-042-16: `pseudo_scale` [b, 2] =
    `[reward.scout]` / `[reward.survive]` `vs_model` in games against a model). With every lane on
    and every multiplier 1, the scalars as before (same results bit for bit). Only the policy
    gradient (V-trace PG, UPGO, PPO advantage) uses these; the value targets do not."""
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
    """042-R2 진단 로그 (AMB-042-17), host sums over the trained steps of a lane group: the
    combined policy-gradient advantage squared (`pg`, the value `adv_mean` averages) and each
    channel's (win, build, units, value) PG advantage squared before the 033 weights (V-trace: ρ
    (r + γ v_s' − V) per channel; PPO: the channel's GAE)."""
    return {"adv_sq": float((pg * pg * valid).sum()), "adv_sq_c": [float((x * x * valid).sum()) for x in chan]}


def sil_positive(g: dict, v: torch.Tensor, valid: torch.Tensor) -> dict:
    """042-R16 `sil_pos_frac` of buffer lanes trained as ordinary lanes ("replay"): trained steps
    of lanes with a finite `sil_return` R, and those with R − V_win > 0 (`v` [b, t, 4] host)."""
    ret = g.get("sil_return")
    if ret is None:
        return {"sil_pos": 0.0, "sil_steps": 0.0}
    r = torch.as_tensor(ret, dtype=torch.float32).reshape(-1)
    sel = valid & torch.isfinite(r)[:, None]
    pos = sel & ((r[:, None] - v[:, :, 0]) > 0)
    return {"sil_pos": float(pos.sum()), "sil_steps": float(sel.sum())}


def diagnostics(adv_sq: float, adv_sq_c: list, value_ch: list, kl_q: list, n: int) -> dict:
    """042-R2 진단 로그 (AMB-042-17) of an update over `n` trained steps: RMS of the combined and
    per-channel advantages, the per-channel value loss 0.5 (V − target)^2 mean (before the value
    loss weight), the per-step KL p99 and maximum (`None` without a trained step)."""
    return {
        "adv_rms": math.sqrt(adv_sq / n), "adv_rms_channels": [math.sqrt(x / n) for x in adv_sq_c],
        "value_loss_channels": list(value_ch), "kl_p99": finite_or_none(kl_q[0]), "kl_max": finite_or_none(kl_q[1]),
    }


def sil_stats(sil: dict, parts: list, pos: float, pos_steps: float) -> dict:
    """042-R16 statistics: `None` when off. "sil": the weighted loss of the buffer lanes (0 without
    them) and their share of steps with A+ > 0; "replay": no separate loss, the share from the
    ordinary targets."""
    if not sil["enabled"]:
        return {"sil_loss": None, "sil_pos_frac": None}
    if sil["mode"] == "sil":
        loss, pos, pos_steps = parts
        return {"sil_loss": loss, "sil_pos_frac": pos / pos_steps if pos_steps > 0 else None}
    return {"sil_loss": None, "sil_pos_frac": pos / pos_steps if pos_steps > 0 else None}


def aux_stats(weights: dict, ks: list, vals: list | None = None, counts: dict | None = None, tgt_std: float | None = None) -> dict:
    """042-R21 ③ statistics (before λ, batch means): `aux_priv`, `aux_jepa` (mean over k),
    `jepa_tgt_std`, `aux_jepa_slots` (the configured k in order) and `spatial` (the 032-R25
    statistics of a `[monitor]` update, set by the caller); `jepa_tgt_std_raw` (before the
    target's LayerNorm, not a train.csv column). `None` for a module the model does not have, a
    batch without a row for it (no privileged summary, a k without a pair) or a slot past the k
    list; all `None` without aux modules (the reference learner's statistics too)."""
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
    """042-R21 ①: the whole batch's aux divisors on the host (no collate, no sync), over the
    queue lanes `0..queue` (042-R16 buffer lanes take no aux): `priv` trained steps with a
    privileged summary, `pairs` per k (source t a trained step, target t+k a real step of the same
    segment, the bootstrap step included), `real` real steps (the JEPA target spread)."""
    trained = [int(x) for x in batch.trained][:queue]
    real = [min(int(x), t) for x in batch.real_steps][:queue]
    priv = [int(x) for x in batch.priv_trained][:queue]
    return {"priv": sum(priv), "real": sum(real), "pairs": [sum(max(0, min(tr, r - k)) for tr, r in zip(trained, real)) for k in ks]}


def value_channels(values: torch.Tensor, vs: torch.Tensor, mask: torch.Tensor, n: int) -> torch.Tensor:
    """042-R2 진단 로그: each channel's Σ 0.5 (V − target)^2 over the trained rows / `n` [4]."""
    return ((values - vs) ** 2 * mask[:, None]).sum(0) * (0.5 / n)


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
        # 042-R21 ①: `[aux]` keys and the trained model's aux modules must match (start error).
        # The BC reference's aux modules never run (`forward_seq(aux=False)`, 042-R21 ②).
        self.aux = check_aux(net.cfg.variant, cfg.get("aux"))
        self.ks = list(net.cfg.variant.aux_jepa)
        # 042-R21 ③ / 041-R25: dead-channel statistics every `spatial_every` updates (0 = never).
        self.monitor_every = spatial_every(cfg)
        self.device = device
        self.precision = precision
        self.algo = algo
        self.net = net.to(device).train()
        self.bc = bc.to(device).eval()
        for p in self.bc.parameters():
            p.requires_grad_(False)
        self.update_count = update
        # 042-R9 (AMB-042-17): per-head KL multipliers, None = all 1.0.
        self.kl_scale = kl_head_scale(cfg)
        self.micro_lanes = host_micro_lanes(cfg) if micro_lanes is None else micro_lanes
        # 032-R19 ⑥ for this learner only (not process switches: the inference servers share the
        # process): page-locked staging on the CUDA host only (the Linux ROCm host's GPU shares
        # its RAM, 060-R3: no page-locked buffers there), the spatial planes rounded to bf16 on the
        # host under bf16 autocast (their consumers round them the same way, `net.stage`).
        self.pinned = device.type == "cuda" and torch.version.hip is None
        self.spatial_bf16 = precision == "bf16"
        # The 032 §2.7 entity padding cut (`trim_entity_pad`): off. Equal only within float
        # rounding (shorter reductions) and no measured gain on the CUDA host (`bench rl`,
        # 2026-10-31, groups 1 and 4), so not used (032-R19: only changes with a measured gain);
        # `bench rl --trim` measures it again. These three are attributes so `bench rl` sets them.
        self.trim_pad = False
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
        """Lanes `start..end` collated and staged for the copy (on the collate thread): with
        `trim_pad` the 032 §2.7 entity padding cut (`trim_entity_pad`, dropout 0 in RL), the copied arrays page-locked
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
        reward components (033-R3) run as 4·b rows of one recursion, component-major. The value
        targets of every component are kept whatever its policy weight (AMB-042-14)."""
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
        """One update over `batch` (a `_core.SegmentBatch`, lanes x T); returns the statistics."""
        cfg, dev = self.cfg, self.device
        if self.phases:
            self._ph.clear()
            started = self._mark = time.perf_counter()
        t = cfg["seq_len"]
        lanes = batch.lanes
        g_size = self.group_size(lanes)
        # 042-R16: buffer lanes (last) train only the "sil" loss in their own lane groups; in
        # "replay" they are ordinary lanes. Without buffer lanes: the earlier groups.
        sil = sil_settings(cfg)
        k = sil_lane_count(batch)
        apart_lanes = k if sil["mode"] == "sil" else 0
        normal = lanes - apart_lanes
        groups = [(s, min(s + g_size, normal), False) for s in range(0, normal, g_size)]
        groups += [(s, min(s + g_size, lanes), True) for s in range(normal, lanes, g_size)]
        coef_kl = kl_coef(cfg, self.update_count)
        epochs = 1 if self.algo == "vtrace" else max(1, cfg["ppo"]["epochs"])
        # The totals come from the batch itself; the lane groups are collated one ahead on a
        # worker thread (the Rust collate releases the GIL) while the previous one trains, as
        # in BC: at most two collated groups are alive (not every group of the batch at once).
        collate = lambda gi: self.prepare(batch, groups[gi][0], groups[gi][1], t)  # noqa: E731
        trained = [int(x) for x in batch.trained]
        # The loss terms are means over the lanes that take them ("replay" buffer lanes included);
        # `steps` counts the queue lanes only, as the run loop's `trained_steps` (042-R2).
        n = max(sum(trained[:normal]), 1)
        steps = sum(trained[:lanes - k])
        n_sil = max(sum(trained[normal:]), 1)
        # Policy lag of the queue lanes (buffer lanes are older by design).
        versions = [int(v) for v in batch.versions][:lanes - k]
        # The loss parts of every group summed on the device in f64 (the sums of the host floats
        # they were before, bit for bit), read once after the optimizer step.
        parts = torch.zeros(PARTS, dtype=torch.float64, device=dev)
        sil_parts = torch.zeros(SIL_PARTS, dtype=torch.float64, device=dev)
        host = defaultdict(float)
        adv_sq_c = [0.0] * 4
        kl_steps: list = []
        norm = None
        # PPO keeps the first epoch's advantages and returns (computed before any change).
        ppo_targets: list = [None] * len(groups)
        apart = self.net.win_open_params()
        params = list(self.net.parameters())
        # 042-R21: aux divisors over the queue lanes (buffer lanes take no aux), the JEPA target
        # sums and the update's dead-channel monitor (first epoch, the weights before the step).
        aux = rl_aux_counts(batch, t, lanes - k, self.ks) if self.aux else None
        net_cfg = self.net.cfg
        moments = torch.zeros(4, net_cfg.entity.d_model + net_cfg.spatial.pool, dtype=torch.float64, device=dev) if "jepa" in self.aux else None
        mon = ReluMonitor(self.net, points="spatial") if spatial_due(self.monitor_every, self.update_count + 1) else None
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
                    if groups[gi][2]:
                        part = self.sil_group_loss(g, n_sil, sil)
                        if epoch == 0:
                            sil_parts += part
                        del g
                        continue
                    first = epoch == 0
                    part, tg, kl_step = self.group_loss(g, n, weights, coef_kl, ppo_targets, gi, aux, moments if first else None, mon if first else None)
                    del g
                    if epoch == 0:
                        parts += part
                        for key in ("rho_sum", "adv_sum", "adv_sq", "sil_pos", "sil_steps"):
                            host[key] += tg[key]
                        adv_sq_c = [a + x for a, x in zip(adv_sq_c, tg["adv_sq_c"])]
                        kl_steps.append(kl_step)
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
        # One read of every statistic (032-R19): the parts, the gradient norm, KL p99 / max, SIL,
        # then (042-R21 ③) the JEPA target spread and the monitor's statistics when taken.
        extra, channels = [], None
        if moments is not None:
            extra.append(torch.stack([jepa_std(moments[:2], aux["real"]), jepa_std(moments[2:], aux["real"])]))
        if mon is not None:
            st = mon.stats()
            channels = channel_counts_of(st)
            extra.append(packed(st))
        vals = torch.cat([parts, nan if norm is None else norm.detach().double().reshape(1), kl_tail(kl_steps, dev), sil_parts, *extra]).tolist()
        tail = vals[PARTS + 3 + SIL_PARTS:]
        self.tick("stats")
        stats = {
            "update": self.update_count + 1, "steps": steps, "kl_coef": coef_kl, "policy": vals[0], "value": vals[1], "win_open": vals[3], "kl": vals[2],
            "kl_heads": vals[4:10], "entropy_heads": vals[10:16], "rho_mean": host["rho_sum"] / n, "adv_mean": host["adv_sum"] / n,
            "lag_mean": sum(max(self.update_count - v, 0) for v in versions) / max(len(versions), 1),
            # The global gradient norm before the clip (first epoch; the 042 lane-group equivalence
            # check). Not in train.csv; null when not finite (the stats are JSON for the Rust loop).
            "grad_norm": finite_or_none(vals[PARTS]),
        }
        stats.update(diagnostics(host["adv_sq"], adv_sq_c, vals[16:BASE_PARTS], vals[PARTS + 1:PARTS + 3], n))
        stats.update(sil_stats(sil, vals[PARTS + 3:PARTS + 3 + SIL_PARTS], host["sil_pos"], host["sil_steps"]))
        stats.update(aux_stats(self.aux, self.ks, vals[BASE_PARTS:PARTS], aux, tail[:2] if moments is not None else (None, None)))
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
        # 042-R21 ①: the aux terms the loss added (λ > 0).
        for key, name in (("priv", "aux_priv"), ("jepa", "aux_jepa")):
            if self.aux.get(key, 0.0) > 0 and stats[name] is not None:
                stats["loss"] += self.aux[key] * stats[name]
        if self.phases:
            self._ph["python_total"] = time.perf_counter() - started
            stats["phases"] = {k: round(x, 4) for k, x in self._ph.items()}
        self.update_count += 1
        return stats

    def group_loss(self, g: dict, n: int, weights, coef_kl: float, ppo_targets: list, gi: int, aux: dict | None = None, moments: torch.Tensor | None = None,
                   mon: ReluMonitor | None = None) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """Forward, targets, loss and backward of one lane group (divided by the batch's trained
        steps `n`); returns the detached loss parts (a device tensor, `PARTS`), the targets' host
        sums (ρ, advantage, the 진단 로그 squares, `sil_positive`) and the per-step KL (NaN on
        untrained rows, a device tensor). `aux`: the batch's aux divisors (`rl_aux_counts`) when
        the model has an aux module (042-R21 ①); `moments`: the JEPA target sums it adds to;
        `mon`: the update's dead-channel monitor, run on this group after its backward."""
        cfg, dev, net, pin = self.cfg, self.device, self.net, self.pinned
        t = g["t"]
        obs, labels = to_obs(g, dev, pinned=pin), net.labels(g, dev, pinned=pin)
        priv = to_device(g["privileged"], dev, pin).float()
        mask = to_device(g["mask"], dev, pin).float()
        s0 = state0_of(net, g, dev, pin)
        phases = [to_device(i, dev, pin) for i in g["phases"]] if "phases" in g else None
        self.tick("h2d")
        with autocast(dev, self.precision):
            # 032 §2.7: RL segments start at any decision, so an outer model runs phased. 032-R24:
            # the aux outputs only with an aux module (the call is as before without one).
            out = net.forward_seq(obs, labels, priv, s0, t, phased=True, phases=phases, **({"aux": True} if aux else {}))
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
        # 042-R2 진단 로그: the per-step KL (used heads summed, before the head multipliers).
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
        # 042-R9 (AMB-042-17): head multipliers on the KL terms (none = the plain sum).
        loss = policy + value - entropy + scaled_kl(kl_terms, kl_total, self.kl_scale) * coef_kl
        if win_open is not None:
            loss = loss + win_open
        # 042-R21 ①: queue lanes only (a 042-R16 buffer lane riding as an ordinary lane, "replay",
        # takes no aux loss and is not measured).
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
            # 042-R21 ③ / 032-AC15: eval mode, no gradient, no RNG, the group's s0 and phases, the
            # learner's autocast; the returned state is dropped. Rows: the queue lanes' real steps.
            with mon:
                relu_stats(net, obs, labels, priv, s0, t, rows=real & lane_rows, points="spatial", phased=True, phases=phases, ctx=autocast(dev, self.precision), monitor=mon)
            self.tick("monitor")
        zero = torch.zeros((), device=dev)
        value_ch = value_channels(values.detach(), vs, mask, n)
        part = torch.cat([torch.stack([policy.detach(), value.detach(), kl_total.detach(), zero if win_open is None else win_open.detach(), *[x.detach() for x in kl_terms], *[x.detach() for x in ent_terms]]), value_ch, aux_part.detach()]).double()
        return part, tg, kl_step

    def aux_loss(self, out: dict, g: dict, priv, mask, real, lane_rows, t: int, aux: dict, moments, loss):
        """042-R21 ①: λ_priv · L_priv + λ_jepa · L_jepa added to `loss` (a weight of 0 adds
        nothing). Rows (main-session interim decision 2026-11-09, U3 review): L_priv the trained
        steps (`mask` > 0, the bootstrap row out) with a privileged summary; L_jepa the pairs whose
        source t is a trained step and target t+k a real step of the same segment (the bootstrap
        step included; one segment = one game). Divided by the batch's counts over the queue lanes.
        Returns the new loss and the aux parts [AUX_PARTS] (before λ)."""
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
        """042-R16 "sil" (Oh et al. 2018) on a lane group of buffer lanes, divided by the buffer
        lanes' trained steps `n_sil`: `w_pg · Σ[−log π(a|s) · A⁺] + w_v · Σ[0.5 · (A⁺)²]`,
        `A⁺ = max(R − V_win, 0)`, R = the lane's `sil_return`, log π the used heads' joint log-prob
        (032-R15), V_win the win head (privileged input as usual). The policy term holds A⁺ fixed;
        the value term's gradient flows into V_win. These lanes take no other loss term. Returns
        [loss, steps with A⁺ > 0, trained steps] (device, f64)."""
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
    """A copy of `net` for the BC reference / reset (shares no tensors)."""
    return copy.deepcopy(net)


def stats_json(stats: dict) -> str:
    return json.dumps(stats)
