"""041-R3 gradient clip: one global L2 norm over all parameter gradients (not per tensor); a
non-finite norm zeroes every gradient of the update. 041-R19: the win_open head is clipped on its
own with the same max norm, so the other parameters update as without it. No host sync (the
same steps as the Burn trainer's `clip.rs`)."""

import torch

# Added to the norm before dividing (as torch `clip_grad_norm_`), so a zero gradient stays zero.
CLIP_EPS = 1e-6


def _clip(params: list[torch.Tensor], max_norm: float) -> torch.Tensor | None:
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return None
    norm = torch.stack([g.float().square().sum() for g in grads]).sum().sqrt()
    bad = ~torch.isfinite(norm)
    coef = torch.clamp(max_norm / (norm + CLIP_EPS), max=1.0).masked_fill(bad, 0.0)
    for g in grads:
        g.copy_((g.float() * coef).masked_fill(bad, 0.0).to(g.dtype))
    return norm


def clip_global_norm(params, max_norm: float, apart=()) -> torch.Tensor | None:
    """Clips the gradients of `params` in place; `apart` (041-R19) are clipped on their own.
    Returns the norm of the rest before clipping (a device tensor, not read back)."""
    apart_ids = {id(p) for p in apart}
    norm = _clip([p for p in params if id(p) not in apart_ids], max_norm)
    if apart:
        _clip(list(apart), max_norm)
    return norm
