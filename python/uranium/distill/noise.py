import torch

M32 = 0xFFFFFFFF
SALT_SEED, SALT_CALL, SALT_COL = 0x243F6A88, 0x13198A2E, 0xA4093822
END_COLUMN = 0x000FFFFF


def _mul32(x: torch.Tensor, c: int) -> torch.Tensor:
    lo, hi = c & 0xFFFF, c >> 16
    return (x * lo + (((x * hi) & 0xFFFF) << 16)) & M32


def mix32(x: torch.Tensor) -> torch.Tensor:
    x = x ^ (x >> 16)
    x = _mul32(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = _mul32(x, 0x846CA68B)
    return x ^ (x >> 16)


def row_keys(seeds: list[int], frames: list[int], device) -> torch.Tensor:
    lo = torch.tensor([s & M32 for s in seeds], dtype=torch.int64, device=device)
    hi = torch.tensor([(s >> 32) & M32 for s in seeds], dtype=torch.int64, device=device)
    f = torch.tensor([f & M32 for f in frames], dtype=torch.int64, device=device)
    return mix32(mix32(mix32(lo ^ SALT_SEED) ^ hi) ^ f)


class RowNoise:
    def __init__(self, keys: torch.Tensor, n: int, max_select: int, actions: int, delay_buckets: int, cells: int):
        self.keys, self.n, self.max_select = keys, n, max_select
        self.widths = [actions, delay_buckets, 2, *([n + 1] * max_select), n, cells]
        self.call = 0

    @property
    def draws(self) -> int:
        return len(self.widths)

    def finished(self) -> None:
        if self.call != self.draws:
            raise ValueError(f"044-R3 noise: the heads made {self.call} draws, expected {self.draws}")

    def uniform(self, shape, device) -> torch.Tensor:
        rows, width = shape
        c = self.call
        self.call += 1
        s = self.max_select
        if c >= self.draws:
            raise ValueError(f"044-R3 noise: draw {c} past the heads' {self.draws} draws")
        if width != self.widths[c]:
            raise ValueError(f"044-R3 noise: draw {c} has width {width}, expected {self.widths[c]} (head order or widths changed)")
        cols = torch.arange(width, dtype=torch.int64, device=device)
        if 3 <= c < 3 + s:
            cols[-1] = END_COLUMN
        if rows != self.keys.shape[0]:
            raise ValueError(f"044-R3 noise: {rows} rows, keys for {self.keys.shape[0]}")
        call = mix32(self.keys.to(device) ^ int(mix32(torch.tensor([c ^ SALT_CALL]))[0]))
        x = mix32(call[:, None] ^ mix32(cols ^ SALT_COL)[None, :])
        u = ((x >> 8).to(torch.float32) + 0.5) / float(1 << 24)
        return u.clamp(1e-6, 1.0 - 1e-6)
