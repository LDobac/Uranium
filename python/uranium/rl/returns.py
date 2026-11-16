import torch


def _valid(length: torch.Tensor, t: int) -> torch.Tensor:
    return torch.arange(t, device=length.device)[None, :] < length[:, None]


def vtrace(rewards, values, log_rho, length, gamma: float, lam: float, rho_bar: float, c_bar: float):
    return vtrace_ratio(rewards, values, log_rho.exp(), _valid(length, rewards.shape[1]), gamma, lam, rho_bar, c_bar)


def vtrace_ratio(rewards, values, ratio, valid, gamma: float, lam: float, rho_bar: float, c_bar: float):
    t = rewards.shape[1]
    rho = torch.clamp(ratio, max=rho_bar) * valid
    c = lam * torch.clamp(ratio, max=c_bar)
    base = values[:, :-1] + rho * (rewards + gamma * values[:, 1:] - values[:, :-1])
    gc = gamma * c
    vs = values.clone()
    for i in range(t - 1, -1, -1):
        step = base[:, i] + gc[:, i] * (vs[:, i + 1] - values[:, i + 1])
        vs[:, i] = torch.where(valid[:, i], step, values[:, i])
    pg = rho * (rewards + gamma * vs[:, 1:] - values[:, :-1])
    return vs[:, :-1] * valid, pg * valid, rho


def upgo(rewards, values, log_rho, length, gamma: float, rho_bar: float):
    return upgo_ratio(rewards, values, log_rho.exp(), length, _valid(length, rewards.shape[1]), gamma, rho_bar)


def upgo_ratio(rewards, values, ratio, length, valid, gamma: float, rho_bar: float):
    t = rewards.shape[1]
    last = torch.arange(1, t + 1, device=length.device)[None, :] >= length[:, None]
    boot = values.gather(1, length.clamp(max=t)[:, None])[:, 0]
    up = (rewards[:, 1:] + gamma * values[:, 2:]) >= values[:, 1:-1]
    g = values.clone()
    for i in range(t - 1, -1, -1):
        follow = torch.where(up[:, i], g[:, i + 1], values[:, i + 1]) if i + 1 < t else values[:, t]
        nxt = torch.where(last[:, i], boot, follow)
        g[:, i] = torch.where(valid[:, i], rewards[:, i] + gamma * nxt, values[:, i])
    w = torch.clamp(ratio, max=rho_bar)
    return w * (g[:, :-1] - values[:, :-1]) * valid


def gae(rewards, values, length, gamma: float, lam: float):
    return gae_valid(rewards, values, _valid(length, rewards.shape[1]), gamma, lam)


def gae_valid(rewards, values, valid, gamma: float, lam: float):
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
