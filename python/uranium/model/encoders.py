import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from uranium.env import env_choice
from uranium.kernels import ENTITY_ATTN_MODES
from uranium.model.config import EntityConfig, InputDims, ScalarConfig, SpatialConfig
from uranium.model.layers import Attention, FeedForward, Linear, Mlp, ResBlock, conv3x3, conv_layout, conv_wgrad, masked_mean, stem_pad

BIAS_GRAD_MODES = ("embedding", "index_add", "table_t")


class GatherIndexAdd(torch.autograd.Function):
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
        self.bias_table = nn.Parameter(torch.randn(2 * c.distance_buckets, c.layers * c.heads) * 0.02)
        self.pool = Linear(2 * d, d)
        self.heads = c.heads
        self.dist_buckets = c.distance_buckets

    def forward(self, obs: dict, add=None, tokens=None, tok_bias=None, pool=None):
        cat = obs["entity_cat"]
        x = torch.cat([self.unit_type(cat[..., 0]), self.owner(cat[..., 1]), self.order(cat[..., 2]), self.last_action(cat[..., 3]), obs["entity_cont"]], 2)
        x = self.input(x)
        if add is not None:
            x = x + add.to(x.dtype)
        mask = obs["entity_mask"]
        n = x.shape[1]
        table = self.bias_table
        if tokens is not None:
            x = torch.cat([x, tokens.to(x.dtype)], 1)
            mask = torch.cat([mask, mask.new_ones(mask.shape[0], tokens.shape[1])], 1)
            table = torch.cat([self.bias_table, tok_bias.to(self.bias_table.dtype)], 0)
        idx = self.buckets(obs, 0 if tokens is None else tokens.shape[1])
        if self.use_fused(x, table):
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
        if env_choice("URANIUM_ENTITY_ATTN", ENTITY_ATTN_MODES, "torch") != "fused" or x.device.type != "cuda":
            return False
        from uranium.kernels.attention import HEAD_DIM, available

        return x.shape[-1] // self.heads == HEAD_DIM and table.shape[0] <= 64 and available(x.device)

    def distance_bias(self, idx, table):
        mode = env_choice("URANIUM_BIAS_GRAD", BIAS_GRAD_MODES, "embedding")
        if mode == "table_t":
            return table.t()[:, idx].permute(1, 0, 2, 3)
        if mode == "index_add":
            return GatherIndexAdd.apply(idx, table).permute(0, 3, 1, 2)
        return F.embedding(idx, table).permute(0, 3, 1, 2)

    def buckets(self, obs: dict, tokens: int = 0):
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
    def __init__(self, c_in: int, c_out: int, preact: bool = False):
        super().__init__()
        self.conv = conv3x3(c_in, c_out, 2)
        self.res = ResBlock(c_out, preact, nonneg_input=True)


class SpatialEncoder(nn.Module):
    def __init__(self, inp: InputDims, c: SpatialConfig, d_entity: int, preact: bool = False):
        super().__init__()
        c_in = c.stem_channels
        downs = []
        for c_out in c.stages:
            downs.append(Down(c_in, c_out, preact))
            c_in = c_out
        side = inp.map_size // 16
        self.scatter = Linear(d_entity, c.scatter_channels)
        self.stem = conv3x3(inp.spatial_channels + c.scatter_channels, c.stem_channels, 1)
        self.downs = nn.ModuleList(downs)
        self.bottleneck = nn.ModuleList(ResBlock(c_in, preact) for _ in range(2))
        self.pool = Linear(c_in * side * side, c.pool)
        self.channels_last: bool | None = None
        self.stem_pad: bool | None = None

    def forward(self, obs: dict, e, film=None):
        spatial = obs["spatial"]
        h, w = spatial.shape[2], spatial.shape[3]
        parts = [spatial, self.scatter_mean(obs, e, h, w).to(spatial.dtype)]
        pad = stem_pad(self.stem, spatial, self.stem_pad)
        if pad:
            parts.append(spatial.new_zeros(spatial.shape[0], pad, h, w))
            x = F.conv2d(conv_layout(torch.cat(parts, 1), self.channels_last), F.pad(self.stem.weight, (0, 0, 0, 0, 0, pad)), self.stem.bias, self.stem.stride, self.stem.padding)
        else:
            x = conv_wgrad(self.stem, conv_layout(torch.cat(parts, 1), self.channels_last))
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
