"""ValueHeads (032 §2.6): win (tanh), build, units, value (033-R3). Privileged inputs enter here
only (032-R11); without them (inference) the privileged branch is a zero vector."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from uranium.model.config import InputDims, ValueConfig
from uranium.model.layers import Linear, Mlp

VALUE_WIN, VALUE_BUILD, VALUE_UNITS, VALUE_VALUE = 0, 1, 2, 3

# 032-R18 ③: seeds of the win_open layers' own generator (not the global RNG).
OPEN_SEEDS = (0x3218_0001, 0x3218_0002, 0x3218_0003)


def fixed_linear(d_in: int, d_out: int, seed: int) -> nn.Linear:
    """032-R18 ③: a Linear initialised U(-k, k), k = 1/sqrt(d_in), from its own fixed-seed
    generator, so building the head does not move the other modules' initial weights."""
    lin = Linear(d_in, d_out)
    g = torch.Generator().manual_seed(seed)
    k = 1.0 / math.sqrt(d_in)
    with torch.no_grad():
        lin.weight.copy_(torch.empty(d_out, d_in).uniform_(-k, k, generator=g))
        lin.bias.copy_(torch.empty(d_out).uniform_(-k, k, generator=g))
    return lin


class WinOpenHead(nn.Module):
    """032-R18: own MLP over detached h, z and race embeddings (no gradient reaches the policy,
    encoders, core or other value heads), no privileged input."""

    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.l1 = fixed_linear(d_in, hidden, OPEN_SEEDS[0])
        self.l2 = fixed_linear(hidden, hidden, OPEN_SEEDS[1])
        self.out = fixed_linear(hidden, 1, OPEN_SEEDS[2])


class ValueHeads(nn.Module):
    def __init__(self, inp: InputDims, c: ValueConfig, d_core: int, d_z: int, d_race: int):
        super().__init__()
        self.privileged = Linear(inp.privileged, c.privileged)
        self.shared = Mlp(d_core + d_z + d_race + c.privileged, c.hidden, c.hidden)
        self.out = Linear(c.hidden, 4)
        self.priv_dim = c.privileged
        # Built last (032-R18 ③), only when the config asks for it (032-R18 ②).
        self.open = WinOpenHead(d_core + d_z + d_race, c.hidden) if c.win_open else None

    def forward(self, h, z, race, priv):
        """-> [B,4] in the order win, build, units, value."""
        b = h.shape[0]
        p = F.relu(self.privileged(priv)) if priv is not None else torch.zeros(b, self.priv_dim, dtype=h.dtype, device=h.device)
        x = self.out(F.relu(self.shared(torch.cat([h, z.to(h.dtype), race.to(h.dtype), p.to(h.dtype)], 1))))
        return torch.cat([torch.tanh(x[:, :1]), x[:, 1:]], 1)

    def win_open(self, h, z, race):
        """032-R18: [B] in [-1, 1], None without the head. No privileged input."""
        if self.open is None:
            return None
        x = torch.cat([h.detach(), z.detach().to(h.dtype), race.detach().to(h.dtype)], 1)
        x = F.relu(self.open.l2(F.relu(self.open.l1(x))))
        return torch.tanh(self.open.out(x)).view(-1)
