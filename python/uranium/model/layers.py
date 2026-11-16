import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from uranium.env import env_choice
from uranium.kernels import MASKED

LINEAR_WGRAD_MODES = ("vendor", "splitk")
SPLITK_CHUNK = 512


def splitk_mm(a, b, chunk: int | None = None):
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
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.l1 = Linear(d_in, d_hidden)
        self.l2 = Linear(d_hidden, d_out)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))


class ResMlp(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.l1 = Linear(d, d)
        self.l2 = Linear(d, d)

    def forward(self, x):
        return x + self.l2(F.relu(self.l1(F.relu(x))))


def conv3x3(c_in: int, c_out: int, stride: int) -> nn.Conv2d:
    return nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1)


CHANNELS_LAST_MODES = ("off", "on")


STEM_PAD_MODES = ("off", "on")


def stem_pad(conv: nn.Conv2d, x, on: bool | None = None) -> int:
    if on is None:
        on = env_choice("URANIUM_STEM_PAD", STEM_PAD_MODES, "off") == "on"
    if not on or bmm_wgrad_active(conv, x):
        return 0
    return -conv.in_channels % 8


def conv_layout(x: torch.Tensor, on: bool | None = None) -> torch.Tensor:
    if on is None:
        on = env_choice("URANIUM_CHANNELS_LAST", CHANNELS_LAST_MODES, "off") == "on"
    if on:
        return x.contiguous(memory_format=torch.channels_last)
    return x


CONV_WGRAD_MODES = ("vendor", "bmm")
WGRAD_CHUNK = 8
RES_BMM_MAX_CH = 64


def bmm_weight_grad(x, gy, stride: int):
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
    return conv.kernel_size == (3, 3) and conv.padding == (1, 1) and conv.dilation == (1, 1) and conv.groups == 1 and conv.stride[0] == conv.stride[1] in (1, 2) and conv.padding_mode == "zeros"


def bmm_wgrad_active(conv: nn.Conv2d, x) -> bool:
    return env_choice("URANIUM_CONV_WGRAD", CONV_WGRAD_MODES, "vendor") == "bmm" and x.dtype == torch.float32 and not torch.is_autocast_enabled(x.device.type) and bmm_wgrad_fits(conv)


def conv_wgrad(conv: nn.Conv2d, x):
    if bmm_wgrad_active(conv, x):
        return ConvBmmWgrad.apply(x, conv.weight, conv.bias, conv.stride[0])
    return conv(x)


class ResBlock(nn.Module):
    def __init__(self, c: int, preact: bool = False, nonneg_input: bool = False):
        super().__init__()
        self.c1 = conv3x3(c, c, 1)
        self.c2 = conv3x3(c, c, 1)
        self.preact = preact
        self.nonneg_input = nonneg_input

    def inner_act(self, x):
        if self.preact and not self.nonneg_input:
            x = F.relu(x)
        if x.shape[1] <= RES_BMM_MAX_CH:
            return F.relu(conv_wgrad(self.c1, x))
        return F.relu(self.c1(x))

    def forward(self, x):
        if x.shape[1] <= RES_BMM_MAX_CH:
            y = x + conv_wgrad(self.c2, self.inner_act(x))
        else:
            y = x + self.c2(self.inner_act(x))
        return y if self.preact else F.relu(y)


class GruCell(nn.Module):
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
    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0, "d_model must divide by heads"
        self.q = Linear(d, d)
        self.k = Linear(d, d)
        self.v = Linear(d, d)
        self.o = Linear(d, d)
        self.heads = heads

    def forward(self, x_q, x_kv, bias, key_ok):
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
        from uranium.kernels.attention import fused

        b, n, d = x.shape
        h, dh = self.heads, d // self.heads
        q = self.q(x).view(b, n, h, dh).transpose(1, 2)
        k = self.k(x).view(b, n, h, dh).transpose(1, 2)
        v = self.v(x).view(b, n, h, dh).transpose(1, 2)
        return self.o(fused(q, k, v, bucket_u8, table, key_ok).transpose(1, 2).reshape(b, n, d))


class FeedForward(nn.Module):
    def __init__(self, d: int, ffn: int, dropout: float):
        super().__init__()
        self.l1 = Linear(d, ffn)
        self.l2 = Linear(ffn, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.l2(self.dropout(F.gelu(self.l1(x))))


def masked_mean(x, mask):
    m = mask.to(x.dtype)[..., None]
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


def log_softmax_f32(logits):
    return F.log_softmax(logits.float(), 1)


def masked_log_softmax(logits, allowed):
    return F.log_softmax(logits.float().masked_fill(~allowed, MASKED), 1)


def gumbel_argmax(logp, temp: float, generator=None):
    if temp <= 0:
        return logp.argmax(1)
    if generator is not None and not isinstance(generator, torch.Generator):
        u = generator.uniform(logp.shape, logp.device)
    else:
        u = torch.empty(logp.shape, dtype=torch.float32, device=logp.device).uniform_(1e-6, 1.0 - 1e-6, generator=generator)
    g = -torch.log(-torch.log(u))
    return (logp / temp + g).argmax(1)


def tempered(logp, temp: float):
    if temp <= 0 or abs(temp - 1.0) < 1e-6:
        return logp
    return F.log_softmax(logp / temp, 1)


def pick(logp, choice):
    return logp.gather(1, choice[:, None]).squeeze(1)


def entropy(logp):
    return -(logp.exp() * logp).sum(1)
