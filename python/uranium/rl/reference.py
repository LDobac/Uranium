"""The RL learner as it was before the 032-R19 ⑥ optimisations (042 learner optimisation, commit
b9da476): the reference the optimised `Learner` is compared with, bit for bit on CPU where the
operations are the same (tests/python/test_rl_opt.py), and the baseline of `python -m uranium bench
rl --learner reference`. Same loss, targets and update; per lane group it reads the targets'
inputs and the loss parts back (host syncs), copies core_state0 per lane, the outer segments' rows
come from the device (`nonzero`), the copies are pageable unless `URANIUM_H2D=pinned`, and the
targets run one reward component at a time. Only additions: the phase marks of `bench rl
--phases` and `grad_norm` in the statistics. Not used by training."""

import torch

from uranium.agent.infer import leaves, rebuild
from uranium.bc.clip import clip_global_norm
from uranium.device import autocast
from uranium.model import to_obs
from uranium.rl.learner import HEADS, Learner, kl_coef, kl_per_head


def _valid(length: torch.Tensor, t: int) -> torch.Tensor:
    return torch.arange(t, device=length.device)[None, :] < length[:, None]


def vtrace(rewards, values, log_rho, length, gamma: float, lam: float, rho_bar: float, c_bar: float):
    """(vs [L,T], pg_adv [L,T], rho [L,T]); c_t = λ min(c̄, π/μ)."""
    l, t = rewards.shape
    valid = _valid(length, t)
    ratio = log_rho.exp()
    rho = torch.clamp(ratio, max=rho_bar) * valid
    c = lam * torch.clamp(ratio, max=c_bar)
    vs = values.clone()  # positions >= length keep the value (vs[length] = bootstrap)
    for i in range(t - 1, -1, -1):
        delta = rho[:, i] * (rewards[:, i] + gamma * values[:, i + 1] - values[:, i])
        step = values[:, i] + delta + gamma * c[:, i] * (vs[:, i + 1] - values[:, i + 1])
        vs[:, i] = torch.where(valid[:, i], step, values[:, i])
    pg = rho * (rewards + gamma * vs[:, 1:] - values[:, :-1])
    return vs[:, :-1] * valid, pg * valid, rho


def upgo(rewards, values, log_rho, length, gamma: float, rho_bar: float):
    """UPGO advantage [L,T]: the return follows the trajectory while Q_{t+1} >= V_{t+1}."""
    l, t = rewards.shape
    valid = _valid(length, t)
    g = values.clone()
    for i in range(t - 1, -1, -1):
        last = (i + 1) >= length  # the next position is the bootstrap
        if i + 1 < t:
            q_next = rewards[:, i + 1] + gamma * values[:, i + 2]
            follow = torch.where(q_next >= values[:, i + 1], g[:, i + 1], values[:, i + 1])
        else:
            follow = values[:, t]
        boot = values.gather(1, length.clamp(max=t)[:, None])[:, 0]
        nxt = torch.where(last, boot, follow)
        g[:, i] = torch.where(valid[:, i], rewards[:, i] + gamma * nxt, values[:, i])
    w = torch.clamp(log_rho.exp(), max=rho_bar)
    return w * (g[:, :-1] - values[:, :-1]) * valid


def gae(rewards, values, length, gamma: float, lam: float):
    """(advantages [L,T], returns [L,T])."""
    l, t = rewards.shape
    valid = _valid(length, t)
    adv = torch.zeros_like(rewards)
    acc = torch.zeros(l, dtype=rewards.dtype, device=rewards.device)
    for i in range(t - 1, -1, -1):
        delta = rewards[:, i] + gamma * values[:, i + 1] - values[:, i]
        acc = torch.where(valid[:, i], delta + gamma * lam * acc, torch.zeros_like(acc))
        adv[:, i] = acc
    return adv * valid, (adv + values[:, :-1]) * valid


def state0_of(net, g: dict, device):
    """Core state per lane (042 core_state0); a lane without one starts from zero (032-R5)."""
    s = net.init_state(g["b"], device)
    parts = leaves(s)
    for lane, st in enumerate(g["state0"]):
        if st is None:
            continue
        for leaf, (shape, v) in zip(parts, st):
            leaf[lane] = torch.as_tensor(v).reshape(shape)[0].to(device, leaf.dtype)
    return rebuild(s, iter(parts))


class ReferenceLearner(Learner):
    """`Learner` with the update of commit b9da476."""

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
        import time

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
        collate = lambda gi: batch.group(groups[gi][0], groups[gi][1], t, self.net.cfg.heads.max_select)  # noqa: E731
        steps = int(sum(batch.trained))
        n = max(steps, 1)
        versions = [int(v) for v in batch.versions]
        stats = {
            "update": self.update_count + 1, "steps": steps, "kl_coef": coef_kl, "policy": 0.0, "value": 0.0, "win_open": 0.0, "kl": 0.0,
            "kl_heads": [0.0] * 6, "entropy_heads": [0.0] * 6, "rho_mean": 0.0, "adv_mean": 0.0,
            "lag_mean": sum(max(self.update_count - v, 0) for v in versions) / max(len(versions), 1),
        }
        ppo_targets: list = [None] * len(groups)
        apart = self.net.win_open_params()
        params = list(self.net.parameters())
        norm = None
        for epoch in range(epochs):
            self.opt.zero_grad(set_to_none=True)
            nxt = self._collate.submit(collate, 0)
            for gi in range(len(groups)):
                self.tick("other")
                g = nxt.result()
                self.tick("collate_wait")
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
            clipped = clip_global_norm(params, cfg["optimizer"]["grad_clip"], apart)
            if epoch == 0:
                norm = clipped
            for pg in self.opt.param_groups:
                pg["lr"] = cfg["optimizer"]["lr"]
            self.opt.step()
            self.tick("optimizer")
        e = cfg["entropy"]
        bonus = sum(e[h] * stats["entropy_heads"][i] for i, h in enumerate(HEADS))
        stats["loss"] = stats["policy"] + stats["value"] + stats["win_open"] + stats["kl"] * coef_kl - bonus
        stats["rho_mean"] /= n
        stats["adv_mean"] /= n
        stats["grad_norm"] = None if norm is None else float(norm)
        self.tick("stats")
        if self.phases:
            self._ph["python_total"] = time.perf_counter() - started
            stats["phases"] = {k: round(x, 4) for k, x in self._ph.items()}
        self.update_count += 1
        return stats

    def group_loss(self, g: dict, n: int, weights, coef_kl: float, ppo_targets: list, gi: int) -> dict:
        cfg, dev, net = self.cfg, self.device, self.net
        t = g["t"]
        obs, labels = to_obs(g, dev), net.labels(g, dev)
        priv = torch.as_tensor(g["privileged"]).to(dev, torch.float32)
        mask = torch.as_tensor(g["mask"]).to(dev, torch.float32)
        s0 = state0_of(net, g, dev)
        self.tick("h2d")
        with autocast(dev, self.precision):
            out = net.forward_seq(obs, labels, priv, s0, t, phased=True)
        logp = sum(out["logp"][k].float() for k in HEADS)
        values = out["value"].float()
        self.tick("forward")
        tg = ppo_targets[gi]
        if tg is None:
            tg = self.targets(g, logp, values, weights)
            if self.algo == "ppo":
                ppo_targets[gi] = tg
        self.tick("targets")
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
        with torch.no_grad(), autocast(dev, self.precision):
            bc = self.bc.forward_seq(obs, labels, priv, s0, t, phased=True)
        self.tick("bc_forward")
        kl = kl_per_head({k: (v.float() if v.is_floating_point() else v) for k, v in out["dists"].items()}, {k: (v.float() if v.is_floating_point() else v) for k, v in bc["dists"].items()})
        kl_terms = [(kl[h] * mask).sum() / n for h in HEADS]
        kl_total = sum(kl_terms)
        self.tick("kl")
        loss = policy + value - entropy + kl_total * coef_kl
        if win_open is not None:
            loss = loss + win_open
        self.tick("loss")
        loss.backward()
        self.tick("backward")
        vals = torch.stack([policy.detach(), value.detach(), kl_total.detach(), torch.zeros((), device=dev) if win_open is None else win_open.detach(), *[x.detach() for x in kl_terms], *[x.detach() for x in ent_terms]]).tolist()
        return {"policy": vals[0], "value": vals[1], "kl": vals[2], "win_open": vals[3], "kl_heads": vals[4:10], "entropy_heads": vals[10:16], "rho_sum": tg["rho_sum"], "adv_sum": tg["adv_sum"]}
