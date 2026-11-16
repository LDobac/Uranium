import torch


def leaves(s) -> list[torch.Tensor]:
    if isinstance(s, torch.Tensor):
        return [s]
    return [t for x in s for t in leaves(x)]


def rebuild(s, it):
    if isinstance(s, torch.Tensor):
        return next(it)
    out = [rebuild(x, it) for x in s]
    return tuple(out) if isinstance(s, tuple) else out


def select_rows(s, rows: torch.Tensor):
    return rebuild(s, iter([t.index_select(0, rows) for t in leaves(s)]))


def scatter_rows(s, rows: torch.Tensor, sub):
    return rebuild(s, iter([t.index_copy(0, rows, u.to(t.dtype)) for t, u in zip(leaves(s), leaves(sub))]))


def zero_rows(s, rows: torch.Tensor):
    return rebuild(s, iter([t.index_fill(0, rows, 0) for t in leaves(s)]))
