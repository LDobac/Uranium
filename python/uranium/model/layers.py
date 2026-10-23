"""Building blocks shared by the encoders, core and heads (032 §2): the same math as the Burn
`layers.rs`, with the same parameter names (032-AC10)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MASKED = -1e9  # 032-R4: masked logits


class Mlp(nn.Module):
    """Linear -> ReLU -> Linear."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.l1 = nn.Linear(d_in, d_hidden)
        self.l2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))


class ResMlp(nn.Module):
    """Pre-activation residual MLP block: x + L2(relu(L1(relu(x))))."""

    def __init__(self, d: int):
        super().__init__()
        self.l1 = nn.Linear(d, d)
        self.l2 = nn.Linear(d, d)

    def forward(self, x):
        return x + self.l2(F.relu(self.l1(F.relu(x))))


def conv3x3(c_in: int, c_out: int, stride: int) -> nn.Conv2d:
    return nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1)


class ResBlock(nn.Module):
    """032 §2.2 ResBlock(2 x Conv3x3)."""

    def __init__(self, c: int):
        super().__init__()
        self.c1 = conv3x3(c, c, 1)
        self.c2 = conv3x3(c, c, 1)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class GruCell(nn.Module):
    """GRU cell (Selected head query update, 032 §2.5 #4)."""

    def __init__(self, d_in: int, d: int):
        super().__init__()
        self.x = nn.Linear(d_in, 3 * d)
        self.h = nn.Linear(d, 3 * d)

    def forward(self, x, h):
        gx = self.x(x).chunk(3, 1)
        gh = self.h(h).chunk(3, 1)
        r = torch.sigmoid(gx[0] + gh[0])
        z = torch.sigmoid(gx[1] + gh[1])
        cand = torch.tanh(gx[2] + r * gh[2])
        return z * h + (1.0 - z) * cand


class Attention(nn.Module):
    """Multi-head attention with an additive logit bias and a key mask (the distance bias, 032
    §2.1, and the GTrXL relative positions need the bias)."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0, "d_model must divide by heads"
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.heads = heads

    def forward(self, x_q, x_kv, bias, key_ok):
        """`x_q` [B,Tq,d], `x_kv` [B,Tk,d], `bias` [B or 1,H,Tq,Tk] or None,
        `key_ok` [B,Tq or 1,Tk] (True = attend)."""
        b, tq, d = x_q.shape
        tk = x_kv.shape[1]
        h, dh = self.heads, d // self.heads
        q = self.q(x_q).view(b, tq, h, dh).transpose(1, 2)
        k = self.k(x_kv).view(b, tk, h, dh).transpose(1, 2)
        v = self.v(x_kv).view(b, tk, h, dh).transpose(1, 2)
        scores = q @ k.transpose(2, 3) / math.sqrt(dh)
        if bias is not None:
            scores = scores + bias
        attn = torch.softmax(scores.masked_fill(~key_ok[:, None], MASKED), 3)
        return self.o((attn @ v).transpose(1, 2).reshape(b, tq, d))


class FeedForward(nn.Module):
    """Pre-LN transformer feed-forward with GELU (exact erf form, as Burn `gelu`)."""

    def __init__(self, d: int, ffn: int, dropout: float):
        super().__init__()
        self.l1 = nn.Linear(d, ffn)
        self.l2 = nn.Linear(ffn, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.l2(self.dropout(F.gelu(self.l1(x))))


def masked_mean(x, mask):
    """Mean of `x` [B,N,D] over rows where `mask` [B,N] is true (zero when none)."""
    m = mask.to(x.dtype)[..., None]
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


def log_softmax_f32(logits):
    """Log-softmax over the last dim in f32 whatever the compute dtype (032-R8)."""
    return F.log_softmax(logits.float(), 1)


def masked_log_softmax(logits, allowed):
    """Masked log-softmax over the last dim of [B,K] in f32 (032-R4, 032-R8)."""
    return F.log_softmax(logits.float().masked_fill(~allowed, MASKED), 1)


def gumbel_argmax(logp, temp: float, generator: torch.Generator | None = None):
    """Gumbel-max sample from log-probs [B,K] (032-R9). `temp == 0` is greedy. Masked entries
    (-1e9) are never chosen because the noise is bounded."""
    if temp <= 0:
        return logp.argmax(1)
    u = torch.empty(logp.shape, dtype=torch.float32, device=logp.device).uniform_(1e-6, 1.0 - 1e-6, generator=generator)
    g = -torch.log(-torch.log(u))
    return (logp / temp + g).argmax(1)


def tempered(logp, temp: float):
    """Re-normalised log-probs at a sampling temperature (behaviour log-probs, 042)."""
    if temp <= 0 or abs(temp - 1.0) < 1e-6:
        return logp
    return F.log_softmax(logp / temp, 1)


def pick(logp, choice):
    """log p(choice) from log-probs [B,K] and choice [B] -> [B]."""
    return logp.gather(1, choice[:, None]).squeeze(1)


def entropy(logp):
    """Entropy of the distribution given by log-probs [B,K] -> [B]; masked entries add 0."""
    return -(logp.exp() * logp).sum(1)
