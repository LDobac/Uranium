"""Entity attention with the gathered distance bias (032 §2.1, 032-R13): the fused kernel
(`attn_bias.cu`) and its PyTorch reference, which is the math of `layers.Attention` with
`bias = table[bucket]` (060-R1)."""

import math

import torch

from uranium.model.layers import MASKED

HEAD_DIM = 32


def reference(q, k, v, bucket, table, key_ok, scale: float):
    """q, k, v [B,H,N,dh]; bucket [B,N,N] int; table [NB,H]; key_ok [B,N] bool."""
    bias = table[bucket.long()].permute(0, 3, 1, 2)
    s = (q @ k.transpose(2, 3)) * scale + bias
    p = torch.softmax(s.masked_fill(~key_ok[:, None, None, :], MASKED), 3)
    return p @ v


class FusedBiasAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bucket, table, key_ok, scale: float):
        from uranium.kernels import load

        mod = load("attn_bias")
        out, stats = mod.fwd(q, k, v, bucket, table, key_ok, scale)
        ctx.save_for_backward(q, k, v, bucket, table, key_ok, out, stats)
        ctx.scale = scale
        ctx.mod = mod
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, bucket, table, key_ok, out, stats = ctx.saved_tensors
        dout = dout.contiguous()
        delta = (dout.float() * out.float()).sum(3).contiguous()
        dq, dk, dv, dtable = ctx.mod.bwd(q, k, v, bucket, table, key_ok, dout, stats, delta, ctx.scale)
        return dq, dk, dv, None, dtable.to(table.dtype), None, None


def available(device) -> bool:
    """The fused kernel can run here (GPU, built)."""
    if torch.device(device).type != "cuda":
        return False
    from uranium.kernels import load

    return load("attn_bias") is not None


def fused(q, k, v, bucket_u8, table, key_ok):
    """q, k, v [B,H,N,32] (same dtype), bucket_u8 [B,N,N] uint8, table [NB,H] f32, key_ok [B,N]."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    return FusedBiasAttention.apply(q.contiguous(), k.contiguous(), v.contiguous(), bucket_u8, table.float().contiguous(), key_ok.contiguous(), scale)
