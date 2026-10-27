"""Encoders (032 §2.1–2.3): entity transformer with a distance bias, spatial U-Net encoder with
the entity scatter, scalar and strategy (z) encoder. Same math and parameter names as Burn."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from uranium.model.config import EntityConfig, InputDims, ScalarConfig, SpatialConfig
from uranium.model.layers import Attention, FeedForward, Linear, Mlp, ResBlock, conv3x3, conv_layout, conv_wgrad, env_choice, masked_mean, stem_pad

# AMB-000-4 step 2: how the distance bias is gathered and its table gradient summed
# (`URANIUM_BIAS_GRAD`); same values either way:
#   embedding - F.embedding to [B,N,N,LH], used through a permuted view (CUDA's fastest)
#   index_add - as embedding, the backward one index_add_ (lost on real data: atomic contention on
#               the few buckets most pairs share, reports/pytorch_opt.md 2.5)
#   table_t   - indexing the transposed table, so each head's [N,N] plane is contiguous (Linux ROCm
#               host, encoder fwd+bwd 64 rows x N512: 1129 -> 872 ms)
BIAS_GRAD_MODES = ("embedding", "index_add", "table_t")
# `URANIUM_ENTITY_ATTN`: the entity attention in PyTorch (`torch`) or through the fused kernel
# with the distance bias (`fused`, uranium/kernels/attn_bias.cu, 060-R1). Same math.
ENTITY_ATTN_MODES = ("torch", "fused")


class GatherIndexAdd(torch.autograd.Function):
    """`table[idx]` whose backward accumulates the rows with `index_add_`."""

    @staticmethod
    def forward(ctx, idx, table):
        ctx.save_for_backward(idx)
        ctx.rows = table.shape[0]
        return F.embedding(idx, table)

    @staticmethod
    def backward(ctx, g):
        (idx,) = ctx.saved_tensors
        d = g.shape[-1]
        gt = torch.zeros(ctx.rows, d, dtype=g.dtype, device=g.device).index_add_(0, idx.reshape(-1), g.reshape(-1, d))
        return None, gt


class EntityLayer(nn.Module):
    def __init__(self, c: EntityConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model)
        self.attn = Attention(c.d_model, c.heads)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.ffn = FeedForward(c.d_model, c.ffn, c.dropout)


class EntityEncoder(nn.Module):
    """032 §2.1: embeddings + pre-LN transformer, attention logits + bias[l, h, bucket(i, j)]."""

    def __init__(self, inp: InputDims, c: EntityConfig):
        super().__init__()
        d = c.d_model
        self.unit_type = nn.Embedding(inp.unit_types, c.unit_type_emb)
        self.owner = nn.Embedding(inp.owners, c.owner_emb)
        self.order = nn.Embedding(inp.orders, c.order_emb)
        self.last_action = nn.Embedding(inp.actions, c.last_action_emb)
        self.input = Linear(c.unit_type_emb + c.owner_emb + c.order_emb + c.last_action_emb + inp.entity_cont, d)
        self.layers = nn.ModuleList(EntityLayer(c) for _ in range(c.layers))
        self.ln_out = nn.LayerNorm(d)
        # [buckets, layers * heads]: bias[l, h, bucket] (032 §2.1).
        self.bias_table = nn.Parameter(torch.randn(2 * c.distance_buckets, c.layers * c.heads) * 0.02)
        self.pool = Linear(2 * d, d)
        self.heads = c.heads
        self.dist_buckets = c.distance_buckets

    def forward(self, obs: dict, add=None, tokens=None, tok_bias=None, pool=None):
        """-> (E [B,N,d], e_pool [B,d]). The 032 §2.7 variants: `add` [B,N,d] is added to the
        input projection (`terrain`), `tokens` [B,K,d] join the transformer with their own bias
        rows `tok_bias` [K, layers*heads] and are dropped from E (`ctx_token`, `outer`), `pool`
        replaces the mean pool (`attn_pool`)."""
        cat = obs["entity_cat"]
        x = torch.cat([self.unit_type(cat[..., 0]), self.owner(cat[..., 1]), self.order(cat[..., 2]), self.last_action(cat[..., 3]), obs["entity_cont"]], 2)
        x = self.input(x)
        if add is not None:
            x = x + add.to(x.dtype)
        mask = obs["entity_mask"]
        # The entity count of x itself: dropping the tokens below then needs no relation between
        # two inputs' sizes, which torch.compile guarded as a bound on N (a recompile when a
        # larger N came, CUDA host 2026-10-25).
        n = x.shape[1]
        table = self.bias_table
        if tokens is not None:
            x = torch.cat([x, tokens.to(x.dtype)], 1)
            mask = torch.cat([mask, mask.new_ones(mask.shape[0], tokens.shape[1])], 1)
            table = torch.cat([self.bias_table, tok_bias.to(self.bias_table.dtype)], 0)
        idx = self.buckets(obs, 0 if tokens is None else tokens.shape[1])
        if self.use_fused(x, table):
            # AMB-000-4 step 2: the fused kernel reads the bucket ids (computed here, uint8) and
            # the layer's table columns; no [B, L*H, N, N] bias is built.
            bucket = idx.to(torch.uint8)
            for l, layer in enumerate(self.layers):
                y = layer.ln1(x)
                x = x + layer.attn.forward_fused(y, bucket, table[:, l * self.heads:(l + 1) * self.heads], mask)
                x = x + layer.ffn(layer.ln2(x))
        else:
            bias = self.distance_bias(idx, table)
            key_ok = mask[:, None, :]
            for l, layer in enumerate(self.layers):
                y = layer.ln1(x)
                x = x + layer.attn(y, y, bias[:, l * self.heads:(l + 1) * self.heads], key_ok)
                x = x + layer.ffn(layer.ln2(x))
        e = self.ln_out(x)
        if tokens is not None:
            e = e[:, :n]
        mask = obs["entity_mask"]
        own = (cat[..., 1] == 0) & mask
        enemy = (cat[..., 1] == 1) & mask
        if pool is not None:
            return e, pool(e, own, enemy)
        pool = self.pool(torch.cat([masked_mean(e, own), masked_mean(e, enemy)], 1))
        return e, pool

    def use_fused(self, x, table) -> bool:
        """`URANIUM_ENTITY_ATTN=fused` and the kernel can run here (GPU, built, head dim 32,
        at most 64 buckets); otherwise the PyTorch path."""
        if env_choice("URANIUM_ENTITY_ATTN", ENTITY_ATTN_MODES, "torch") != "fused" or x.device.type != "cuda":
            return False
        from uranium.kernels.attention import HEAD_DIM, available

        return x.shape[-1] // self.heads == HEAD_DIM and table.shape[0] <= 64 and available(x.device)

    def distance_bias(self, idx, table):
        """[B, layers*heads, N, N] from the bucket ids [B,N,N]. The table rows gathered by bucket:
        the same values as Burn's one-hot @ table, with a gather gradient (032-R13, 2026-10-23)."""
        mode = env_choice("URANIUM_BIAS_GRAD", BIAS_GRAD_MODES, "embedding")
        if mode == "table_t":
            return table.t()[:, idx].permute(1, 0, 2, 3)
        if mode == "index_add":
            return GatherIndexAdd.apply(idx, table).permute(0, 3, 1, 2)
        return F.embedding(idx, table).permute(0, 3, 1, 2)

    def buckets(self, obs: dict, tokens: int = 0):
        """bucket(i,j) = same_owner * D + clamp(floor(log2(1 + pixel distance)), 0, D-1) (032 §2.1),
        in f32 (squared pixel distances overflow f16, 032-R8). With `tokens` extra tokens
        (032 §2.7) the ids are [B,N+K,N+K]: a pair with token k (as key of an entity, or as the
        query) takes the token's own bucket 2D + k."""
        pos = obs["entity_pos"].float()
        dist = (pos[:, :, None, :] - pos[:, None, :, :]).pow(2).sum(3).sqrt()
        log_d = torch.log(dist + 1.0) / math.log(2)
        dist_bucket = log_d.floor().clamp(0, self.dist_buckets - 1).long()
        owner = obs["entity_cat"][..., 1]
        same = owner[:, :, None] == owner[:, None, :]
        idx = same.long() * self.dist_buckets + dist_bucket
        if tokens == 0:
            return idx
        b, n, _ = idx.shape
        tok = 2 * self.dist_buckets + torch.arange(tokens, device=idx.device)
        top = torch.cat([idx, tok.view(1, 1, tokens).expand(b, n, tokens)], 2)
        bottom = tok.view(1, tokens, 1).expand(b, tokens, n + tokens)
        return torch.cat([top, bottom], 1)


class Down(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = conv3x3(c_in, c_out, 2)
        self.res = ResBlock(c_out)


class SpatialEncoder(nn.Module):
    """032 §2.2: entity scatter-mean + stem + U-Net encoder with skips."""

    def __init__(self, inp: InputDims, c: SpatialConfig, d_entity: int):
        super().__init__()
        c_in = c.stem_channels
        downs = []
        for c_out in c.stages:
            downs.append(Down(c_in, c_out))
            c_in = c_out
        side = inp.map_size // 16
        self.scatter = Linear(d_entity, c.scatter_channels)
        self.stem = conv3x3(inp.spatial_channels + c.scatter_channels, c.stem_channels, 1)
        self.downs = nn.ModuleList(downs)
        self.bottleneck = nn.ModuleList(ResBlock(c_in) for _ in range(2))
        self.pool = Linear(c_in * side * side, c.pool)

    def forward(self, obs: dict, e, film=None):
        """`film` = (γ, β) [B, stem channels] each: the stem output becomes x·(1+γ) + β (032 §2.7
        `outer`)."""
        spatial = obs["spatial"]
        h, w = spatial.shape[2], spatial.shape[3]
        parts = [spatial, self.scatter_mean(obs, e, h, w).to(spatial.dtype)]
        pad = stem_pad(self.stem, spatial)
        if pad:
            # 032-R19 `URANIUM_STEM_PAD`: zero input channels up to a multiple of 8, written by
            # this cat, and zero weight columns for them (0 x w = 0: same conv). cuDNN's NHWC
            # kernels want 8-channel multiples and otherwise pad in a pass of their own.
            parts.append(spatial.new_zeros(spatial.shape[0], pad, h, w))
            x = F.conv2d(conv_layout(torch.cat(parts, 1)), F.pad(self.stem.weight, (0, 0, 0, 0, 0, pad)), self.stem.bias, self.stem.stride, self.stem.padding)
        else:
            x = conv_wgrad(self.stem, conv_layout(torch.cat(parts, 1)))
        x = F.relu(x)
        if film is not None:
            g, b = film
            x = x * (1.0 + g.to(x.dtype)[:, :, None, None]) + b.to(x.dtype)[:, :, None, None]
        outs = []
        for d in self.downs:
            x = d.res(F.relu(conv_wgrad(d.conv, x)))
            outs.append(x)
        for r in self.bottleneck:
            x = r(x)
        return {"s8": x, "skips": outs[:3], "pool": self.pool(x.flatten(1))}

    def scatter_mean(self, obs: dict, e, h: int, w: int):
        """Linear(E) scatter-added to `entity_cell` and divided by the per-cell count -> [B,c,H,W]."""
        b, n, _ = e.shape
        v = self.scatter(e)
        m = obs["entity_mask"].to(v.dtype)[..., None]
        v = v * m
        c = v.shape[2]
        cells = obs["entity_cell"].long()[..., None]
        total = torch.zeros(b, h * w, c, dtype=v.dtype, device=v.device).scatter_add(1, cells.expand(b, n, c), v)
        count = torch.zeros(b, h * w, 1, dtype=v.dtype, device=v.device).scatter_add(1, cells, m)
        return (total / count.clamp_min(1.0)).transpose(1, 2).reshape(b, c, h, w)


class ScalarEncoder(nn.Module):
    """032 §2.3: continuous scalars, categorical embeddings and the strategy condition z (030 §5);
    `z_present == false` zeroes z."""

    def __init__(self, inp: InputDims, c: ScalarConfig):
        super().__init__()
        self.cont = Mlp(inp.scalar, c.hidden, c.hidden)
        self.race = nn.Embedding(inp.races, c.race_emb)
        self.map_family = nn.Embedding(inp.map_families, c.map_family_emb)
        self.start_loc = nn.Embedding(inp.start_locs, c.start_loc_emb)
        self.z_build = nn.Embedding(inp.actions, c.z_build_emb)
        self.z_build_pos = nn.Embedding(inp.z_build_len, c.z_build_emb)
        self.z_units = Linear(inp.z_units, c.z_units_emb)
        self.z_out = Linear(c.z_build_emb + c.z_units_emb, c.z_out)
        self.out = Linear(c.hidden + c.race_emb + c.map_family_emb + c.start_loc_emb + c.z_out, c.hidden)

    def forward(self, obs: dict):
        sc = obs["scalar_cat"]
        race = self.race(sc[:, 0])
        map_family = self.map_family(sc[:, 1])
        start_loc = self.start_loc(sc[:, 2])
        zb_ids = obs["z_build"]
        b, length = zb_ids.shape
        pos = torch.arange(length, device=zb_ids.device)[None].expand(b, length)
        zb = masked_mean(self.z_build(zb_ids) + self.z_build_pos(pos), zb_ids != 0)
        zu = self.z_units(obs["z_units"])
        z = F.relu(self.z_out(torch.cat([zb, zu], 1))) * obs["z_present"].to(zu.dtype)[:, None]
        cont = F.relu(self.cont(obs["scalar"]))
        c = F.relu(self.out(torch.cat([cont, race, map_family, start_loc, z], 1)))
        return {"c": c, "z": z, "race": race}
