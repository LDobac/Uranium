"""RL learner of one agent (042): one gradient update over a batch of segments. Targets
(V-trace + UPGO, or GAE for the PPO baseline) come from the detached forward; the loss is

  policy (V-trace PG + UPGO, or PPO clip) + value (per reward component, 033-R3)
  - Σ_h entropy_h (used heads, 032-R10) + kl_coef · KL(π ‖ π_BC) (exact, per head, AMB-042-9)
  + win_open (042-R13, regresses the win v-trace target, not in the advantage)

KL re-runs the frozen BC network on the same inputs and taken actions, so both distributions are
the conditionals along the taken action (042, AMB-042-4). Lanes run in groups of `micro_lanes`
with gradients summed before one optimizer step, like BC (041). The same math as the Burn
learner it replaces."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor

import torch

from uranium.agent.infer import leaves, rebuild
from uranium.bc.clip import clip_global_norm
from uranium.device import adam_impl, autocast
from uranium.model import to_obs
from uranium.rl.returns import gae, upgo, vtrace

HEADS = ("action", "delay", "queued", "selected", "target_unit", "target_loc")


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


def state0_of(net, g: dict, device):
    """Core state per lane (042 core_state0); a lane without one starts from zero (032-R5)."""
    s = net.core.init_state(g["b"], device)
    parts = leaves(s)
    for lane, st in enumerate(g["state0"]):
        if st is None:
            continue
        for leaf, (shape, v) in zip(parts, st):
            leaf[lane] = torch.as_tensor(v).reshape(shape)[0].to(device, leaf.dtype)
    return rebuild(s, iter(parts))


class Learner:
    """One learning agent: the trained net, the frozen BC reference, Adam (042 optimizer table)."""

    def __init__(self, cfg: dict, net, bc, device: torch.device, precision: str, algo: str, update: int = 0):
        self.cfg = cfg
        self.device = device
        self.precision = precision
        self.algo = algo
        self.net = net.to(device).train()
        self.bc = bc.to(device).eval()
        for p in self.bc.parameters():
            p.requires_grad_(False)
        self.update_count = update
        self.opt = self.new_optimizer()
        self._collate = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rl-collate")

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
        m = self.cfg["micro_lanes"]
        return max(1, lanes if m == 0 else min(m, lanes))

    def targets(self, g: dict, logp: torch.Tensor, values: torch.Tensor, weights) -> dict:
        """Per-row coefficients of one lane group from the detached forward (on the host)."""
        b, t = g["b"], g["t"]
        l = self.cfg["loss"]
        length = torch.as_tensor(g["trained"], dtype=torch.long)
        terminal = torch.as_tensor(g["terminal"], dtype=torch.bool)
        behaviour = torch.as_tensor(g["behaviour"]).float().view(b, t)
        rewards = torch.as_tensor(g["rewards"]).float().view(b, t, 4)
        lp = logp.detach().float().cpu().view(b, t)
        v = values.detach().float().cpu().view(b, t, 4)
        log_rho = lp - behaviour
        valid = torch.arange(t)[None, :] < length[:, None]
        pg = torch.zeros(b, t)
        up = torch.zeros(b, t)
        vs = torch.zeros(b, t, 4)
        rho_sum = 0.0
        for c in range(4):
            ext = torch.cat([v[:, :, c], torch.zeros(b, 1)], 1)
            # Bootstrap after the trained steps: 0 after a terminal step, else the value there.
            boot = torch.where(terminal, torch.zeros(b), ext.gather(1, length[:, None])[:, 0])
            ext = ext.scatter(1, length[:, None], boot[:, None])
            if self.algo == "vtrace":
                vt_vs, vt_pg, rho = vtrace(rewards[:, :, c], ext, log_rho, length, l["gamma"], l["td_lambda"], l["vtrace_rho_bar"], l["vtrace_c_bar"])
                pg += weights[c] * vt_pg
                up += weights[c] * upgo(rewards[:, :, c], ext, log_rho, length, l["gamma"], l["vtrace_rho_bar"])
                vs[:, :, c] = vt_vs
                if c == 0:
                    rho_sum += float(rho.sum())
            else:
                adv, ret = gae(rewards[:, :, c], ext, length, l["gamma"], self.cfg["ppo"]["gae_lambda"])
                pg += weights[c] * adv
                vs[:, :, c] = ret
                if c == 0:
                    rho_sum += float((log_rho.exp() * valid).sum())
        coef = l["pg"] * pg + l["upgo"] * up if self.algo == "vtrace" else pg
        dev = self.device
        return {"coef": coef.reshape(-1).to(dev), "vs": vs.reshape(-1, 4).to(dev), "rho_sum": rho_sum, "adv_sum": float((pg * valid).sum())}

    def update(self, batch, weights) -> dict:
        """One update over `batch` (a `_core.SegmentBatch`, lanes x T); returns the statistics."""
        cfg, dev = self.cfg, self.device
        t = cfg["seq_len"]
        lanes = batch.lanes
        g_size = self.group_size(lanes)
        groups = [(s, min(s + g_size, lanes)) for s in range(0, lanes, g_size)]
        coef_kl = kl_coef(cfg, self.update_count)
        epochs = 1 if self.algo == "vtrace" else max(1, cfg["ppo"]["epochs"])
        # The totals come from the batch itself; the lane groups are collated one ahead on a
        # worker thread (the Rust collate releases the GIL) while the previous one trains, as
        # in BC: at most two collated groups are alive (not every group of the batch at once).
        collate = lambda gi: batch.group(groups[gi][0], groups[gi][1], t, self.net.cfg.heads.max_select)  # noqa: E731
        steps = int(sum(batch.trained))
        n = max(steps, 1)
        versions = [int(v) for v in batch.versions]
        stats = {
            "update": self.update_count + 1, "steps": steps, "kl_coef": coef_kl, "policy": 0.0, "value": 0.0, "win_open": 0.0, "kl": 0.0,
            "kl_heads": [0.0] * 6, "entropy_heads": [0.0] * 6, "rho_mean": 0.0, "adv_mean": 0.0,
            "lag_mean": sum(max(self.update_count - v, 0) for v in versions) / max(len(versions), 1),
        }
        # PPO keeps the first epoch's advantages and returns (computed before any change).
        ppo_targets: list = [None] * len(groups)
        apart = self.net.win_open_params()
        params = list(self.net.parameters())
        for epoch in range(epochs):
            self.opt.zero_grad(set_to_none=True)
            nxt = self._collate.submit(collate, 0)
            for gi in range(len(groups)):
                g = nxt.result()
                if gi + 1 < len(groups):
                    nxt = self._collate.submit(collate, gi + 1)
                part = self.group_loss(g, n, weights, coef_kl, ppo_targets, gi)
                del g
                if epoch == 0:
                    for k in ("policy", "value", "win_open", "kl"):
                        stats[k] += part[k]
                    stats["rho_mean"] += part["rho_sum"]
                    stats["adv_mean"] += part["adv_sum"]
                    for h in range(6):
                        stats["kl_heads"][h] += part["kl_heads"][h]
                        stats["entropy_heads"][h] += part["entropy_heads"][h]
            # 042 optimizer table / 041-R3: global L2 clip after the lane-group sum; 042-R13: the
            # win_open head apart (041-R19).
            clip_global_norm(params, cfg["optimizer"]["grad_clip"], apart)
            for pg in self.opt.param_groups:
                pg["lr"] = cfg["optimizer"]["lr"]
            self.opt.step()
        e = cfg["entropy"]
        bonus = sum(e[h] * stats["entropy_heads"][i] for i, h in enumerate(HEADS))
        stats["loss"] = stats["policy"] + stats["value"] + stats["win_open"] + stats["kl"] * coef_kl - bonus
        stats["rho_mean"] /= n
        stats["adv_mean"] /= n
        self.update_count += 1
        return stats

    def group_loss(self, g: dict, n: int, weights, coef_kl: float, ppo_targets: list, gi: int) -> dict:
        """Forward, targets, loss and backward of one lane group (divided by the batch's trained
        steps `n`); returns the detached loss parts."""
        cfg, dev, net = self.cfg, self.device, self.net
        t = g["t"]
        obs, labels = to_obs(g, dev), net.labels(g, dev)
        priv = torch.as_tensor(g["privileged"]).to(dev, torch.float32)
        mask = torch.as_tensor(g["mask"]).to(dev, torch.float32)
        s0 = state0_of(net, g, dev)
        with autocast(dev, self.precision):
            out = net.forward_seq(obs, labels, priv, s0, t)
        logp = sum(out["logp"][k].float() for k in HEADS)
        values = out["value"].float()
        tg = ppo_targets[gi]
        if tg is None:
            tg = self.targets(g, logp, values, weights)
            if self.algo == "ppo":
                ppo_targets[gi] = tg
        coef = tg["coef"]
        if self.algo == "vtrace":
            policy = -(logp * coef)
        else:
            clip = cfg["ppo"]["clip"]
            old = torch.as_tensor(g["behaviour"]).to(dev, torch.float32)
            ratio = (logp - old).exp()
            policy = -torch.minimum(ratio * coef, ratio.clamp(1 - clip, 1 + clip) * coef)
        policy = (policy * mask).sum() / n
        vs = tg["vs"]
        value = ((values - vs) ** 2).sum(1).mul(mask).sum() * (0.5 * cfg["loss"]["value"] / n)
        win_open = None
        if out["win_open"] is not None:
            win_open = ((out["win_open"].float() - vs[:, 0]) ** 2 * mask).sum() * (0.5 * cfg["loss"]["win_open"] / n)

        e = cfg["entropy"]
        ent_terms = [(out["entropy"][h].float() * mask).sum() / n for h in HEADS]
        entropy = sum(e[h] * term for h, term in zip(HEADS, ent_terms))
        # KL(π ‖ π_BC) with the frozen BC net on the same inputs (AMB-042-9).
        with torch.no_grad(), autocast(dev, self.precision):
            bc = self.bc.forward_seq(obs, labels, priv, s0, t)
        kl = kl_per_head({k: (v.float() if v.is_floating_point() else v) for k, v in out["dists"].items()}, {k: (v.float() if v.is_floating_point() else v) for k, v in bc["dists"].items()})
        kl_terms = [(kl[h] * mask).sum() / n for h in HEADS]
        kl_total = sum(kl_terms)
        loss = policy + value - entropy + kl_total * coef_kl
        if win_open is not None:
            loss = loss + win_open
        loss.backward()
        vals = torch.stack([policy.detach(), value.detach(), kl_total.detach(), torch.zeros((), device=dev) if win_open is None else win_open.detach(), *[x.detach() for x in kl_terms], *[x.detach() for x in ent_terms]]).tolist()
        return {"policy": vals[0], "value": vals[1], "kl": vals[2], "win_open": vals[3], "kl_heads": vals[4:10], "entropy_heads": vals[10:16], "rho_sum": tg["rho_sum"], "adv_sum": tg["adv_sum"]}


def frozen_copy(net):
    """A copy of `net` for the BC reference / reset (shares no tensors)."""
    return copy.deepcopy(net)


def stats_json(stats: dict) -> str:
    return json.dumps(stats)
