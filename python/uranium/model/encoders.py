"""Encoders (032 §2.1–2.3): entity transformer with a distance bias, spatial U-Net encoder with
the entity scatter, scalar and strategy (z) encoder. Same math and parameter names as Burn."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from uranium.model.config import EntityConfig, InputDims, ScalarConfig, SpatialConfig
from uranium.model.layers import Attention, FeedForward, Mlp, ResBlock, conv3x3, masked_mean


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
        self.input = nn.Linear(c.unit_type_emb + c.owner_emb + c.order_emb + c.last_action_emb + inp.entity_cont, d)
        self.layers = nn.ModuleList(EntityLayer(c) for _ in range(c.layers))
        self.ln_out = nn.LayerNorm(d)
        # [buckets, layers * heads]: bias[l, h, bucket] (032 §2.1).
        self.bias_table = nn.Parameter(torch.randn(2 * c.distance_buckets, c.layers * c.heads) * 0.02)
        self.pool = nn.Linear(2 * d, d)
        self.heads = c.heads
        self.dist_buckets = c.distance_buckets

    def forward(self, obs: dict):
        cat = obs["entity_cat"]
        x = torch.cat([self.unit_type(cat[..., 0]), self.owner(cat[..., 1]), self.order(cat[..., 2]), self.last_action(cat[..., 3]), obs["entity_cont"]], 2)
        x = self.input(x)
        bias = self.distance_bias(obs)
        mask = obs["entity_mask"]
        key_ok = mask[:, None, :]
        for l, layer in enumerate(self.layers):
            y = layer.ln1(x)
            x = x + layer.attn(y, y, bias[:, l * self.heads:(l + 1) * self.heads], key_ok)
            x = x + layer.ffn(layer.ln2(x))
        e = self.ln_out(x)
        own = (cat[..., 1] == 0) & mask
        enemy = (cat[..., 1] == 1) & mask
        pool = self.pool(torch.cat([masked_mean(e, own), masked_mean(e, enemy)], 1))
        return e, pool

    def distance_bias(self, obs: dict):
        """[B, layers*heads, N, N]. The table rows gathered by bucket: the same values as Burn's
        one-hot @ table, with a gather gradient (032-R13, 2026-10-23)."""
        return F.embedding(self.buckets(obs), self.bias_table).permute(0, 3, 1, 2)

    def buckets(self, obs: dict):
        """bucket(i,j) = same_owner * D + clamp(floor(log2(1 + pixel distance)), 0, D-1) (032 §2.1),
        in f32 (squared pixel distances overflow f16, 032-R8)."""
        pos = obs["entity_pos"].float()
        dist = (pos[:, :, None, :] - pos[:, None, :, :]).pow(2).sum(3).sqrt()
        log_d = torch.log(dist + 1.0) / math.log(2)
        dist_bucket = log_d.floor().clamp(0, self.dist_buckets - 1).long()
        owner = obs["entity_cat"][..., 1]
        same = owner[:, :, None] == owner[:, None, :]
        return same.long() * self.dist_buckets + dist_bucket


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
        self.scatter = nn.Linear(d_entity, c.scatter_channels)
        self.stem = conv3x3(inp.spatial_channels + c.scatter_channels, c.stem_channels, 1)
        self.downs = nn.ModuleList(downs)
        self.bottleneck = nn.ModuleList(ResBlock(c_in) for _ in range(2))
        self.pool = nn.Linear(c_in * side * side, c.pool)

    def forward(self, obs: dict, e):
        spatial = obs["spatial"]
        h, w = spatial.shape[2], spatial.shape[3]
        x = F.relu(self.stem(torch.cat([spatial, self.scatter_mean(obs, e, h, w).to(spatial.dtype)], 1)))
        outs = []
        for d in self.downs:
            x = d.res(F.relu(d.conv(x)))
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
        self.z_units = nn.Linear(inp.z_units, c.z_units_emb)
        self.z_out = nn.Linear(c.z_build_emb + c.z_units_emb, c.z_out)
        self.out = nn.Linear(c.hidden + c.race_emb + c.map_family_emb + c.start_loc_emb + c.z_out, c.hidden)

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
