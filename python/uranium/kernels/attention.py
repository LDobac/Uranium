"""Entity attention with the gathered distance bias (032 §2.1, 032-R13): the fused kernel
(`attn_bias.cu`) and its PyTorch reference, which is the math of `layers.Attention` with
`bias = table[bucket]` (060-R1). The kernel is registered as the custom ops
`uranium::attn_bias_fwd` / `uranium::attn_bias_bwd`, so torch.compile keeps it as one node instead
of breaking the graph at the extension call."""

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


def _mod():
    from uranium.kernels import load

    return load("attn_bias")


@torch.library.custom_op("uranium::attn_bias_fwd", mutates_args=())
def attn_bias_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bucket: torch.Tensor, table: torch.Tensor, key_ok: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    out, stats = _mod().fwd(q, k, v, bucket, table, key_ok, scale)
    return out, stats


@attn_bias_fwd.register_fake
def _(q, k, v, bucket, table, key_ok, scale):
    b, h, n, _ = q.shape
    return torch.empty_like(q), q.new_empty((b, h, n, 2), dtype=torch.float32)


@torch.library.custom_op("uranium::attn_bias_bwd", mutates_args=())
def attn_bias_bwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bucket: torch.Tensor, table: torch.Tensor, key_ok: torch.Tensor, dout: torch.Tensor, stats: torch.Tensor, delta: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv, dtable = _mod().bwd(q, k, v, bucket, table, key_ok, dout, stats, delta, scale)
    return dq, dk, dv, dtable


@attn_bias_bwd.register_fake
def _(q, k, v, bucket, table, key_ok, dout, stats, delta, scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), torch.empty_like(table)


def _setup(ctx, inputs, output):
    q, k, v, bucket, table, key_ok, scale = inputs
    out, stats = output
    ctx.save_for_backward(q, k, v, bucket, table, key_ok, out, stats)
    ctx.scale = scale


def _backward(ctx, dout, _dstats):
    q, k, v, bucket, table, key_ok, out, stats = ctx.saved_tensors
    dout = dout.contiguous()
    delta = (dout.float() * out.float()).sum(3).contiguous()
    dq, dk, dv, dtable = attn_bias_bwd(q, k, v, bucket, table, key_ok, dout, stats, delta, ctx.scale)
    return dq, dk, dv, None, dtable.to(table.dtype), None, None


attn_bias_fwd.register_autograd(_backward, setup_context=_setup)


def available(device) -> bool:
    """The fused kernel can run here (GPU, built)."""
    if torch.device(device).type != "cuda":
        return False
    return _mod() is not None


def fused(q, k, v, bucket_u8, table, key_ok):
    """q, k, v [B,H,N,32] (same dtype), bucket_u8 [B,N,N] uint8, table [NB,H] f32, key_ok [B,N]."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    return attn_bias_fwd(q.contiguous(), k.contiguous(), v.contiguous(), bucket_u8, table.float().contiguous(), key_ok.contiguous(), scale)[0]
