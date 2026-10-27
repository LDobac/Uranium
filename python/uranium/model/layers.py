"""Building blocks shared by the encoders, core and heads (032 §2): the same math as the Burn
`layers.rs`, with the same parameter names (032-AC10)."""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

MASKED = -1e9  # 032-R4: masked logits


def env_choice(key: str, choices: tuple, default: str) -> str:
    """An implementation switch from the environment (AMB-000-4 step 2: same math, chosen per
    device); an unknown value is an error."""
    v = os.environ.get(key, "") or default
    if v not in choices:
        raise ValueError(f"{key}={v!r}: expected one of {choices}")
    return v


# AMB-000-4 step 2: `URANIUM_LINEAR_WGRAD` - a Linear layer's weight gradient grad^T @ x from the
# vendor GEMM (`vendor`) or split over the rows in pieces of SPLITK_CHUNK, as one batched GEMM and a
# sum (`splitk`, the same sum in another order). The rows are batch x tokens (12,288 for one lane
# group of 64 steps x 192 entities); rocBLAS on the Linux ROCm host is slow for that long inner
# dimension (256x256 layer: 5.4 -> 0.35 ms).
LINEAR_WGRAD_MODES = ("vendor", "splitk")
SPLITK_CHUNK = 512


def splitk_mm(a, b, chunk: int | None = None):
    """a^T @ b for a [K, M], b [K, N] with K split in pieces of `chunk` (default SPLITK_CHUNK)."""
    chunk = chunk or SPLITK_CHUNK
    k = a.shape[0]
    p = k // chunk
    if p < 2:
        return a.t() @ b
    out = torch.bmm(a[:p * chunk].view(p, chunk, -1).transpose(1, 2), b[:p * chunk].view(p, chunk, -1)).sum(0)
    if k > p * chunk:
        out = out + a[p * chunk:].t() @ b[p * chunk:]
    return out


class LinearSplitK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        ctx.save_for_backward(x, w)
        ctx.has_bias = b is not None
        return F.linear(x, w, b)

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        g2 = g.reshape(-1, g.shape[-1])
        gx = (g @ w) if ctx.needs_input_grad[0] else None
        gw = splitk_mm(g2, x.reshape(-1, x.shape[-1])) if ctx.needs_input_grad[1] else None
        gb = g2.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return gx, gw, gb


class Linear(nn.Linear):
    """`nn.Linear` (same parameters) whose weight gradient follows `URANIUM_LINEAR_WGRAD` (f32
    training only: under autocast or without gradients the library path runs)."""

    def forward(self, x):
        if (
            torch.is_grad_enabled()
            and self.weight.requires_grad
            and x.dtype == torch.float32
            and not torch.is_autocast_enabled(x.device.type)
            and env_choice("URANIUM_LINEAR_WGRAD", LINEAR_WGRAD_MODES, "vendor") == "splitk"
        ):
            return LinearSplitK.apply(x, self.weight, self.bias)
        return super().forward(x)


class Mlp(nn.Module):
    """Linear -> ReLU -> Linear."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.l1 = Linear(d_in, d_hidden)
        self.l2 = Linear(d_hidden, d_out)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))


class ResMlp(nn.Module):
    """Pre-activation residual MLP block: x + L2(relu(L1(relu(x))))."""

    def __init__(self, d: int):
        super().__init__()
        self.l1 = Linear(d, d)
        self.l2 = Linear(d, d)

    def forward(self, x):
        return x + self.l2(F.relu(self.l1(F.relu(x))))


def conv3x3(c_in: int, c_out: int, stride: int) -> nn.Conv2d:
    return nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1)


# 032-R19: `URANIUM_CHANNELS_LAST` - the activations of the spatial encoder and the location head
# in NCHW (`off`) or channels-last (`on`) memory order. Same values; cuDNN runs its NHWC kernels
# without converting every conv's input and output (CUDA host, one update: about 180 ms of
# layout kernels).
CHANNELS_LAST_MODES = ("off", "on")


STEM_PAD_MODES = ("off", "on")


def stem_pad(conv: nn.Conv2d, x) -> int:
    """Zero input channels the stem conv gets under `URANIUM_STEM_PAD=on` (to a multiple of 8;
    0 when it already is, the switch is off, or the call takes the bmm weight-gradient path,
    which `conv_wgrad` keeps)."""
    if env_choice("URANIUM_STEM_PAD", STEM_PAD_MODES, "off") != "on" or bmm_wgrad_active(conv, x):
        return 0
    return -conv.in_channels % 8


def conv_layout(x: torch.Tensor) -> torch.Tensor:
    """`x` [B,C,H,W] in the memory order `URANIUM_CHANNELS_LAST` picks (a no-op when it is already)."""
    if env_choice("URANIUM_CHANNELS_LAST", CHANNELS_LAST_MODES, "off") == "on":
        return x.contiguous(memory_format=torch.channels_last)
    return x


# AMB-000-4 step 2: `URANIUM_CONV_WGRAD` - the weight gradient of the spatial encoder's stem and
# stride-2 convs from the vendor library (`vendor`) or from per-sample batched GEMMs over the
# unfolded input (`bmm`, the same sum). MIOpen's weight gradient on the Linux ROCm host is slow for
# these shapes (64 rows: stem 67 -> 37 ms, 32->64 s2 @128 35 -> 9 ms); forward and input gradient
# stay with the library. ResBlocks of at most RES_BMM_MAX_CH channels take it too (64 @ 64^2:
# 14.4 -> 12.2 ms, 64 @ 32^2: 5.2 -> 3.7 ms); with 128 channels MIOpen is faster.
CONV_WGRAD_MODES = ("vendor", "bmm")
WGRAD_CHUNK = 8
RES_BMM_MAX_CH = 64


def bmm_weight_grad(x, gy, stride: int):
    """d(conv3x3 pad 1)/d(weight) [C_out, C_in, 3, 3] summed over samples in chunks."""
    r, ci = x.shape[:2]
    co = gy.shape[1]
    out = torch.zeros(co, ci * 9, dtype=torch.float32, device=x.device)
    for s in range(0, r, WGRAD_CHUNK):
        cols = F.unfold(x[s:s + WGRAD_CHUNK], 3, padding=1, stride=stride)
        out += torch.bmm(gy[s:s + WGRAD_CHUNK].flatten(2), cols.transpose(1, 2)).sum(0).float()
    return out.view(co, ci, 3, 3)


class ConvBmmWgrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, stride: int):
        ctx.save_for_backward(x, w)
        ctx.stride = stride
        return F.conv2d(x, w, b, stride, 1)

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        s = ctx.stride
        gx = gw = gb = None
        if ctx.needs_input_grad[0]:
            gx = torch.ops.aten.convolution_backward(gy, x, w, None, [s, s], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False])[0]
        if ctx.needs_input_grad[1]:
            gw = bmm_weight_grad(x, gy, s).to(w.dtype)
        if ctx.needs_input_grad[2]:
            gb = gy.sum((0, 2, 3)).to(w.dtype)
        return gx, gw, gb, None


def bmm_wgrad_fits(conv: nn.Conv2d) -> bool:
    """The batched-GEMM weight gradient is written for a 3x3, padding 1, dilation 1, ungrouped
    conv with stride 1 or 2; any other conv keeps the library path."""
    return conv.kernel_size == (3, 3) and conv.padding == (1, 1) and conv.dilation == (1, 1) and conv.groups == 1 and conv.stride[0] == conv.stride[1] in (1, 2) and conv.padding_mode == "zeros"


def bmm_wgrad_active(conv: nn.Conv2d, x) -> bool:
    """`conv_wgrad` takes the bmm weight-gradient path for this call (f32 only: under autocast the
    library path runs; convs other than 3x3 pad 1 too, `bmm_wgrad_fits`)."""
    return env_choice("URANIUM_CONV_WGRAD", CONV_WGRAD_MODES, "vendor") == "bmm" and x.dtype == torch.float32 and not torch.is_autocast_enabled(x.device.type) and bmm_wgrad_fits(conv)


def conv_wgrad(conv: nn.Conv2d, x):
    """`conv(x)`, its weight gradient as `URANIUM_CONV_WGRAD` picks (`bmm_wgrad_active`)."""
    if bmm_wgrad_active(conv, x):
        return ConvBmmWgrad.apply(x, conv.weight, conv.bias, conv.stride[0])
    return conv(x)


class ResBlock(nn.Module):
    """032 §2.2 ResBlock(2 x Conv3x3)."""

    def __init__(self, c: int):
        super().__init__()
        self.c1 = conv3x3(c, c, 1)
        self.c2 = conv3x3(c, c, 1)

    def forward(self, x):
        if x.shape[1] <= RES_BMM_MAX_CH:
            return F.relu(x + conv_wgrad(self.c2, F.relu(conv_wgrad(self.c1, x))))
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class GruCell(nn.Module):
    """GRU cell (Selected head query update, 032 §2.5 #4)."""

    def __init__(self, d_in: int, d: int):
        super().__init__()
        self.x = Linear(d_in, 3 * d)
        self.h = Linear(d, 3 * d)

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
        self.q = Linear(d, d)
        self.k = Linear(d, d)
        self.v = Linear(d, d)
        self.o = Linear(d, d)
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

    def forward_fused(self, x, bucket_u8, table, key_ok):
        """Self-attention of `x` [B,N,d] with `bias = table[bucket]` through the fused kernel
        (`uranium.kernels.attention`, the same math as `forward`); `key_ok` [B,N]."""
        from uranium.kernels.attention import fused

        b, n, d = x.shape
        h, dh = self.heads, d // self.heads
        q = self.q(x).view(b, n, h, dh).transpose(1, 2)
        k = self.k(x).view(b, n, h, dh).transpose(1, 2)
        v = self.v(x).view(b, n, h, dh).transpose(1, 2)
        return self.o(fused(q, k, v, bucket_u8, table, key_ok).transpose(1, 2).reshape(b, n, d))


class FeedForward(nn.Module):
    """Pre-LN transformer feed-forward with GELU (exact erf form, as Burn `gelu`)."""

    def __init__(self, d: int, ffn: int, dropout: float):
        super().__init__()
        self.l1 = Linear(d, ffn)
        self.l2 = Linear(ffn, d)
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
