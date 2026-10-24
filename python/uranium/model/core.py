"""Temporal core (032 §2.4): `LstmCore` and `GtrxlCore` behind one interface (032-AC6). State is
reset per episode by starting from `init_state` (032-R5). Same math as the Burn cores."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from uranium.model.config import CoreConfig
from uranium.model.layers import Attention, FeedForward, Linear

# ROCm: MIOpen's fused LSTM backward fails on the Linux ROCm host (2026-10-23), so `nn.LSTM`
# runs PyTorch's own kernels there; CUDA keeps cuDNN (032 §6 note).
RNN_DNN = torch.version.hip is None


class LstmCore(nn.Module):
    """`Lstm(d_in -> hidden)`, one layer, sigmoid gates, tanh cell and output (032 §2.4). State =
    (cell, hidden), each [B, hidden]."""

    kind = "Lstm"

    def __init__(self, d_in: int, c: CoreConfig):
        super().__init__()
        self.lstm = nn.LSTM(d_in, c.hidden, batch_first=True)
        self.hidden = c.hidden

    def init_state(self, b: int, device):
        z = torch.zeros(b, self.hidden, device=device)
        return (z, z.clone())

    def forward_seq(self, x, s):
        c0, h0 = s
        with torch.backends.cudnn.flags(enabled=RNN_DNN):
            out, (hn, cn) = self.lstm(x, (h0[None].to(x.dtype), c0[None].to(x.dtype)))
        return out, (cn[0], hn[0])

    def step(self, x, s):
        out, s = self.forward_seq(x[:, None], s)
        return out[:, 0], s

    @staticmethod
    def detach_state(s):
        return (s[0].detach(), s[1].detach())

    @staticmethod
    def mask_state(s, keep):
        """Zero the rows where `keep` [B] is false (episode start in that lane, 032-R5)."""
        k = keep.to(s[0].dtype)[:, None]
        return (s[0] * k, s[1] * k)


GATE_BIAS = 2.0  # GTrXL gate bias b_g: starts close to the identity map


class GruGate(nn.Module):
    """GRU-type gating residual of GTrXL: g = (1-z)*x + z*h."""

    def __init__(self, d: int):
        super().__init__()
        self.wr = Linear(d, d)
        self.ur = Linear(d, d, bias=False)
        self.wz = Linear(d, d)
        self.uz = Linear(d, d, bias=False)
        self.wg = Linear(d, d)
        self.ug = Linear(d, d, bias=False)

    def forward(self, x, y):
        r = torch.sigmoid(self.wr(y) + self.ur(x))
        z = torch.sigmoid(self.wz(y) + self.uz(x) - GATE_BIAS)
        h = torch.tanh(self.wg(y) + self.ug(r * x))
        return x * (1.0 - z) + z * h


class GtrxlLayer(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, heads)
        self.gate1 = GruGate(d)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = FeedForward(d, 4 * d, 0.0)
        self.gate2 = GruGate(d)


class GtrxlCore(nn.Module):
    """Causal self-attention over the last M core inputs, 2 layers (032 §2.4). State = (memory:
    per layer the last M layer inputs [B,M,d], valid [B,M] bool)."""

    kind = "Gtrxl"

    def __init__(self, d_in: int, c: CoreConfig):
        super().__init__()
        d = c.hidden
        self.input = Linear(d_in, d)
        self.layers = nn.ModuleList(GtrxlLayer(d, c.gtrxl_heads) for _ in range(c.gtrxl_layers))
        # [M+1, layers*heads] relative position bias (distance 0..=M).
        self.rel_table = nn.Parameter(torch.randn(c.gtrxl_memory + 1, c.gtrxl_layers * c.gtrxl_heads) * 0.02)
        self.hidden, self.heads, self.memory = d, c.gtrxl_heads, c.gtrxl_memory

    def init_state(self, b: int, device):
        mem = [torch.zeros(b, self.memory, self.hidden, device=device) for _ in self.layers]
        return (mem, torch.zeros(b, self.memory, dtype=torch.bool, device=device))

    def relative(self, t: int, device):
        """Relative distance index [T, M+T] and the causal window mask [T, M+T]."""
        m = self.memory
        rel = (m + torch.arange(t, device=device)[:, None]) - torch.arange(m + t, device=device)[None, :]
        inside = (rel >= 0) & (rel <= m)
        return torch.where(inside, rel, torch.zeros_like(rel)), inside

    def forward_seq(self, x, s0):
        mem0, valid0 = s0
        b, t, _ = x.shape
        m, h = self.memory, self.heads
        idx, window = self.relative(t, x.device)
        bias_all = F.embedding(idx, self.rel_table).permute(2, 0, 1)  # [L*H, T, M+T]
        valid_ext = torch.cat([valid0, torch.ones(b, t, dtype=torch.bool, device=x.device)], 1)
        key_ok = window[None] & valid_ext[:, None, :]
        x = self.input(x)
        memory = []
        for l, layer in enumerate(self.layers):
            keys_in = torch.cat([mem0[l].to(x.dtype), x], 1)
            memory.append(keys_in[:, t:t + m])
            y = layer.ln1(keys_in)
            bias = bias_all[l * h:(l + 1) * h][None].expand(b, h, t, m + t)
            a = layer.attn(y[:, m:], y, bias, key_ok)
            x1 = layer.gate1(x, F.relu(a))
            x = layer.gate2(x1, F.relu(layer.ffn(layer.ln2(x1))))
        return x, (memory, valid_ext[:, t:t + m])

    def step(self, x, s):
        out, s = self.forward_seq(x[:, None], s)
        return out[:, 0], s

    @staticmethod
    def detach_state(s):
        # Stored in f32 like `init_state` (the memory is cast to the compute dtype where it is
        # used, so the values are the same): the state's dtype does not change between batches,
        # which torch.compile would recompile for.
        return ([t.detach().float() for t in s[0]], s[1])

    @staticmethod
    def mask_state(s, keep):
        k = keep.to(s[0][0].dtype)[:, None, None]
        return ([t * k for t in s[0]], s[1] & keep[:, None])


def build_core(d_in: int, c: CoreConfig) -> nn.Module:
    if c.kind == "Lstm":
        return LstmCore(d_in, c)
    if c.kind == "Gtrxl":
        return GtrxlCore(d_in, c)
    raise ValueError(f"core kind {c.kind!r}")
