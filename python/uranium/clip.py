import torch

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
    apart_ids = {id(p) for p in apart}
    norm = _clip([p for p in params if id(p) not in apart_ids], max_norm)
    if apart:
        _clip(list(apart), max_norm)
    return norm
