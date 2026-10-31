"""Return and advantage targets (042): V-trace (IMPALA), UPGO (AlphaStar) and GAE (PPO baseline),
batched over lanes: tensors [L, T] (rewards, log_rho) and [L, T+1] (values), `length` [L] the
trained steps of each lane, `values[l, length[l]]` the bootstrap value (0 after a terminal step).
Positions at or after a lane's length are ignored (outputs 0 there). The same recursions as the
Rust reference `uranium_league::returns`, which the tests compare against.

032-R19 ⑥ (042 learner optimisation): the terms that do not depend on the recursion are computed
for all steps at once before the backward loop (the same element operations in the same order, so
the same values bit for bit), and the learner runs the 4 reward components as 4·L rows of one call
(`*_ratio`: the importance ratio π/μ = exp(log_rho) is passed in, computed once per lane group)."""

import torch


def _valid(length: torch.Tensor, t: int) -> torch.Tensor:
    return torch.arange(t, device=length.device)[None, :] < length[:, None]


def vtrace(rewards, values, log_rho, length, gamma: float, lam: float, rho_bar: float, c_bar: float):
    """(vs [L,T], pg_adv [L,T], rho [L,T]); c_t = λ min(c̄, π/μ)."""
    return vtrace_ratio(rewards, values, log_rho.exp(), _valid(length, rewards.shape[1]), gamma, lam, rho_bar, c_bar)


def vtrace_ratio(rewards, values, ratio, valid, gamma: float, lam: float, rho_bar: float, c_bar: float):
    """`vtrace` from the ratio π/μ [L,T] and the trained-step mask `valid` [L,T]."""
    t = rewards.shape[1]
    rho = torch.clamp(ratio, max=rho_bar) * valid
    c = lam * torch.clamp(ratio, max=c_bar)
    # values_i + ρ_i (r_i + γ V_{i+1} - V_i) and γ c_i: the step's terms without vs_{i+1}.
    base = values[:, :-1] + rho * (rewards + gamma * values[:, 1:] - values[:, :-1])
    gc = gamma * c
    vs = values.clone()  # positions >= length keep the value (vs[length] = bootstrap)
    for i in range(t - 1, -1, -1):
        step = base[:, i] + gc[:, i] * (vs[:, i + 1] - values[:, i + 1])
        vs[:, i] = torch.where(valid[:, i], step, values[:, i])
    pg = rho * (rewards + gamma * vs[:, 1:] - values[:, :-1])
    return vs[:, :-1] * valid, pg * valid, rho


def upgo(rewards, values, log_rho, length, gamma: float, rho_bar: float):
    """UPGO advantage [L,T]: the return follows the trajectory while Q_{t+1} >= V_{t+1}."""
    return upgo_ratio(rewards, values, log_rho.exp(), length, _valid(length, rewards.shape[1]), gamma, rho_bar)


def upgo_ratio(rewards, values, ratio, length, valid, gamma: float, rho_bar: float):
    """`upgo` from the ratio π/μ [L,T] and the trained-step mask `valid` [L,T]."""
    t = rewards.shape[1]
    # The next position is the bootstrap (step i: i + 1 >= length), and its value.
    last = torch.arange(1, t + 1, device=length.device)[None, :] >= length[:, None]
    boot = values.gather(1, length.clamp(max=t)[:, None])[:, 0]
    # Step i < T-1 follows the return when Q_{i+1} = r_{i+1} + γ V_{i+2} >= V_{i+1}.
    up = (rewards[:, 1:] + gamma * values[:, 2:]) >= values[:, 1:-1]
    g = values.clone()
    for i in range(t - 1, -1, -1):
        follow = torch.where(up[:, i], g[:, i + 1], values[:, i + 1]) if i + 1 < t else values[:, t]
        nxt = torch.where(last[:, i], boot, follow)
        g[:, i] = torch.where(valid[:, i], rewards[:, i] + gamma * nxt, values[:, i])
    w = torch.clamp(ratio, max=rho_bar)
    return w * (g[:, :-1] - values[:, :-1]) * valid


def gae(rewards, values, length, gamma: float, lam: float):
    """(advantages [L,T], returns [L,T])."""
    return gae_valid(rewards, values, _valid(length, rewards.shape[1]), gamma, lam)


def gae_valid(rewards, values, valid, gamma: float, lam: float):
    """`gae` from the trained-step mask `valid` [L,T]."""
    l, t = rewards.shape
    delta = rewards + gamma * values[:, 1:] - values[:, :-1]
    gl = gamma * lam
    adv = torch.zeros_like(rewards)
    zero = torch.zeros(l, dtype=rewards.dtype, device=rewards.device)
    acc = zero
    for i in range(t - 1, -1, -1):
        acc = torch.where(valid[:, i], delta[:, i] + gl * acc, zero)
        adv[:, i] = acc
    return adv * valid, (adv + values[:, :-1]) * valid
