"""044-R3 ② sampling noise of the policy labeller: the Gumbel-max uniforms of every head draw come
from a counter-based hash of (view seed, step frame, head call, column), so a step's draw depends
only on its own identity, never on the batch it is computed in: not on the lane count, which games
share a forward (`micro_lanes`), the entity padding (`pad_entities`, the group's entity count N
sets the selection and target widths) or how a view is cut into `seq_len` chunks. Entity columns
are the entity's index in the step's own observation and the selection's END has its own column
id, so a wider padded group adds columns without moving any other column's value.

The hash is exact integer arithmetic on int64 tensors that never overflows (32-bit values,
multiplications split in 16-bit halves), so the noise is the same on every device; it is the
`lowbias32` integer mix (C. Wellons). Uniforms are the top 24 bits + 1/2 over 2^24, clamped to the
(1e-6, 1 - 1e-6) range of `gumbel_argmax`."""

import torch

M32 = 0xFFFFFFFF
# Domain salts of the hash inputs (any distinct 32-bit constants; fixed: changing one changes labels).
SALT_SEED, SALT_CALL, SALT_COL = 0x243F6A88, 0x13198A2E, 0xA4093822
# Column id of the selection's END (entity columns are 0..N-1, N < this).
END_COLUMN = 0x000FFFFF


def _mul32(x: torch.Tensor, c: int) -> torch.Tensor:
    """(x * c) mod 2^32 for x in [0, 2^32): two products below 2^49, no int64 overflow."""
    lo, hi = c & 0xFFFF, c >> 16
    return (x * lo + (((x * hi) & 0xFFFF) << 16)) & M32


def mix32(x: torch.Tensor) -> torch.Tensor:
    """lowbias32 on int64 tensors holding 32-bit values."""
    x = x ^ (x >> 16)
    x = _mul32(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = _mul32(x, 0x846CA68B)
    return x ^ (x >> 16)


def row_keys(seeds: list[int], frames: list[int], device) -> torch.Tensor:
    """Per row the 32-bit key of (63-bit view seed, step frame), int64 on `device`."""
    lo = torch.tensor([s & M32 for s in seeds], dtype=torch.int64, device=device)
    hi = torch.tensor([(s >> 32) & M32 for s in seeds], dtype=torch.int64, device=device)
    f = torch.tensor([f & M32 for f in frames], dtype=torch.int64, device=device)
    return mix32(mix32(mix32(lo ^ SALT_SEED) ^ hi) ^ f)


class RowNoise:
    """The noise source of one `Heads.forward` over rows with the given keys (`row_keys`), passed
    as its `generator` (`gumbel_argmax` calls `uniform`). The heads draw in a fixed order: ActionType,
    Delay, Queued, `max_select` selection steps (width N + 1, END last), TargetUnit (width N),
    Location; the call index is part of the hash, and the selection / target widths are checked
    against `n` so a change of that order fails instead of shifting the noise."""

    def __init__(self, keys: torch.Tensor, n: int, max_select: int):
        self.keys, self.n, self.max_select = keys, n, max_select
        self.call = 0

    def uniform(self, shape, device) -> torch.Tensor:
        rows, width = shape
        c = self.call
        self.call += 1
        s = self.max_select
        cols = torch.arange(width, dtype=torch.int64, device=device)
        if 3 <= c < 3 + s:
            if width != self.n + 1:
                raise ValueError(f"044-R3 noise: selection draw {c - 3} has width {width}, expected N + 1 = {self.n + 1}")
            cols[-1] = END_COLUMN
        elif c == 3 + s:
            if width != self.n:
                raise ValueError(f"044-R3 noise: target unit draw has width {width}, expected N = {self.n}")
        elif c > 4 + s:
            raise ValueError(f"044-R3 noise: draw {c} past the heads' {5 + s} draws")
        if rows != self.keys.shape[0]:
            raise ValueError(f"044-R3 noise: {rows} rows, keys for {self.keys.shape[0]}")
        call = mix32(self.keys.to(device) ^ int(mix32(torch.tensor([c ^ SALT_CALL]))[0]))
        x = mix32(call[:, None] ^ mix32(cols ^ SALT_COL)[None, :])
        u = ((x >> 8).to(torch.float32) + 0.5) / float(1 << 24)
        return u.clamp(1e-6, 1.0 - 1e-6)
