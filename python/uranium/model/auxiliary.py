"""032-R24 training-only aux modules and the 041-R23 / 042-R21 aux losses.

Modules (AMB-032-13 Resolved, 2026-11-09): `aux_priv` = MLP(core.hidden -> 256 -> P) predicting the
step's privileged vector, one `aux_jepa` predictor g_k = MLP(core.hidden -> 512 -> d_model +
spatial.pool) per k predicting the encoder summary [e_pool, s_pool] of step t+k. Hidden ReLU.
Built by `Variants` last, under `fork_rng`, each layer from its own fixed-seed generator
(`fixed_linear`, seeds in variants.py), so the shared modules keep their initial weights.

Losses (both trainers; the caller gives the batch-wide divisors, so the lane groups' sums are the
loss of one pass over every lane, 041-R3):
- `priv_sum`: Σ over the masked rows of the mean squared error over the P dimensions (privileged
  dimension weights all 1).
- `jepa_sums`: per k, Σ over the valid pairs (t, t+k) of the same lane (same chunk / segment, so
  the same game) of 1 − cos(g_k(h_t), LN(y_{t+k})), LN a LayerNorm without parameters over the
  dimensions and y the detached `jepa_target` (stop-grad, before `core_add`).
- `jepa_moments`: per dimension Σ LN(y) and Σ LN(y)² over the target rows, for `jepa_tgt_std`
  (the mean over dimensions of the per-dimension standard deviation, the collapse watch).

Everything stays on the device (masks multiply, no indexing, no host sync, 032-R19)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from uranium.model.value import fixed_linear

AUX_PRIV_HIDDEN = 256
AUX_JEPA_HIDDEN = 512


class AuxMlp(nn.Module):
    """Linear -> ReLU -> Linear, each Linear from its own generator (`seed`, `seed + 1`)."""

    def __init__(self, d_in: int, hidden: int, d_out: int, seed: int):
        super().__init__()
        self.l1 = fixed_linear(d_in, hidden, seed)
        self.l2 = fixed_linear(hidden, d_out, seed + 1)

    def forward(self, h):
        return self.l2(F.relu(self.l1(h)))


def aux_priv_module(d_core: int, privileged: int, seed: int) -> AuxMlp:
    """032-R24 ①: MLP(512 -> 256 -> P) by default."""
    return AuxMlp(d_core, AUX_PRIV_HIDDEN, privileged, seed)


def aux_jepa_modules(d_core: int, d_y: int, ks: list, seed_of) -> nn.ModuleDict:
    """032-R24 ②: one MLP(512 -> 512 -> 512) per k, keyed `str(k)` in the `[model] aux_jepa`
    order, each from `seed_of(k)` (keyed by k, so a k starts from the same weights whatever the
    other k are)."""
    return nn.ModuleDict({str(k): AuxMlp(d_core, AUX_JEPA_HIDDEN, d_y, seed_of(k)) for k in ks})


def priv_sum(pred: torch.Tensor, target: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Σ_rows mean_d (pred − target)² over `rows` [r] (f32 0 / 1), f32."""
    err = (pred.float() - target.float()).square().mean(1)
    return (err * rows).sum()


def jepa_target(y: torch.Tensor) -> torch.Tensor:
    """The normalised target: LayerNorm over the dimensions, no learned parameters (f32)."""
    return F.layer_norm(y.float(), (y.shape[1],))


def jepa_sums(preds: dict, y: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor, ks: list) -> list:
    """Per k of `ks`: Σ (1 − cos(g_k(h_t), LN(y_{t+k}))) over the pairs whose source row t is in
    `src` [b, t] and target row t+k in `tgt` [b, t] (bool), inside each row of the batch (a lane's
    chunk / segment); a k ≥ T has no pair (0). `y` [b*t, d] is the detached target (b-major)."""
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
    """[2, d] f64: Σ LN(y) and Σ LN(y)² over `rows` [r] (0 / 1) per dimension."""
    yn = jepa_target(y.detach()).double() * rows.double()[:, None]
    return torch.stack([yn.sum(0), (yn * yn).sum(0)])


def jepa_std(moments: torch.Tensor, n: int) -> torch.Tensor:
    """`jepa_tgt_std`: mean over dimensions of the per-dimension standard deviation (population)
    from `jepa_moments` summed over `n` rows; NaN without a row."""
    if n <= 0:
        return torch.full((), float("nan"), dtype=torch.float64, device=moments.device)
    mean = moments[0] / n
    return (moments[1] / n - mean * mean).clamp_min(0.0).sqrt().mean()


def jepa_mean(sums: list, counts: list) -> tuple:
    """(L_jepa, per-k losses): each k's sum over its batch-wide pair count, and their mean over
    the k with at least one pair in the batch (a k with none, e.g. k ≥ T, is left out of the mean
    and its own loss is 0). The counts are host integers (no sync)."""
    per = [s / c if c > 0 else s * 0.0 for s, c in zip(sums, counts)]
    live = [x for x, c in zip(per, counts) if c > 0]
    total = sum(live) / len(live) if live else sums[0] * 0.0
    return total, per
