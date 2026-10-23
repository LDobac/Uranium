"""Return and advantage targets (042): V-trace (IMPALA), UPGO (AlphaStar) and GAE (PPO baseline),
batched over lanes: tensors [L, T] (rewards, log_rho) and [L, T+1] (values), `length` [L] the
trained steps of each lane, `values[l, length[l]]` the bootstrap value (0 after a terminal step).
Positions at or after a lane's length are ignored (outputs 0 there). The same recursions as the
Rust reference `uranium_league::returns`, which the tests compare against."""

import torch


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
