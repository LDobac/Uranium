import torch
import torch.nn as nn
import torch.nn.functional as F

from uranium.model.value import fixed_linear

AUX_PRIV_HIDDEN = 256
AUX_JEPA_HIDDEN = 512


class AuxMlp(nn.Module):
    def __init__(self, d_in: int, hidden: int, d_out: int, seed: int):
        super().__init__()
        self.l1 = fixed_linear(d_in, hidden, seed)
        self.l2 = fixed_linear(hidden, d_out, seed + 1)

    def forward(self, h):
        return self.l2(F.relu(self.l1(h)))


def aux_priv_module(d_core: int, privileged: int, seed: int) -> AuxMlp:
    return AuxMlp(d_core, AUX_PRIV_HIDDEN, privileged, seed)


def aux_jepa_modules(d_core: int, d_y: int, ks: list, seed_of) -> nn.ModuleDict:
    return nn.ModuleDict({str(k): AuxMlp(d_core, AUX_JEPA_HIDDEN, d_y, seed_of(k)) for k in ks})


def priv_sum(pred: torch.Tensor, target: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    err = (pred.float() - target.float()).square().mean(1)
    return (err * rows).sum()


def jepa_target(y: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(y.float(), (y.shape[1],))


def jepa_sums(preds: dict, y: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor, ks: list) -> list:
    b, t = src.shape
    yn = jepa_target(y).view(b, t, -1)
    out = []
    for k in ks:
        if k >= t:
            out.append(yn.new_zeros(()))
            continue
        p = preds[k].float().view(b, t, -1)[:, :t - k]
        cos = F.cosine_similarity(p, yn[:, k:], dim=2)
        valid = (src[:, :t - k] & tgt[:, k:]).float()
        out.append(((1.0 - cos) * valid).sum())
    return out


def jepa_moments(y: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    r = rows.double()[:, None]
    yn = jepa_target(y.detach()).double() * r
    yr = y.detach().double() * r
    return torch.stack([yn.sum(0), (yn * yn).sum(0), yr.sum(0), (yr * yr).sum(0)])


def jepa_std(moments: torch.Tensor, n: int) -> torch.Tensor:
    if n <= 0:
        return torch.full((), float("nan"), dtype=torch.float64, device=moments.device)
    mean = moments[0] / n
    return (moments[1] / n - mean * mean).clamp_min(0.0).sqrt().mean()


def jepa_mean(sums: list, counts: list) -> tuple:
    per = [s / c if c > 0 else s * 0.0 for s, c in zip(sums, counts)]
    live = [x for x, c in zip(per, counts) if c > 0]
    total = sum(live) / len(live) if live else sums[0] * 0.0
    return total, per
