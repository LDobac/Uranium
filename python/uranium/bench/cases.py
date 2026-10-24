"""Bench cases (060-R2). Kernel cases keep the names of the 2026-10-23 Burn bench so the reports
compare; they run the model's own modules where one exists (core, attention, location head ups,
pointer loop), so an optimisation of the model shows up here. Each case times forward (no grad) and
forward+backward, and checks the forward output against CPU f32 on the same inputs and weights."""

import os
import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from uranium.bench.inputs import random_obs, sampled_labels
from uranium.bench.timing import Timing, measure, peak_gib, reset_peak
from uranium.device import autocast
from uranium.model import UraniumNet, action_columns, input_dims
from uranium.model.config import CoreConfig, UraniumNetConfig, VariantConfig
from uranium.model.core import LstmCore
from uranium.model.heads import gather_cap
from uranium.model.layers import Attention, FeedForward, gumbel_argmax, masked_log_softmax

CPU = torch.device("cpu")
HEADS = ("action", "delay", "queued", "selected", "target_unit", "target_loc")


@dataclass
class Ctx:
    device: torch.device
    precision: str
    accuracy: bool = True
    budget_s: float = 10.0


@dataclass
class Row:
    case: str
    shape: str
    fwd: Timing | None = None
    fwd_bwd: Timing | None = None
    gflops: float | None = None
    max_abs: float | None = None
    max_rel: float | None = None
    peak_gib: float | None = None
    note: str = ""
    status: str = "ok"
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("case", "shape", "gflops", "max_abs", "max_rel", "peak_gib", "note", "status")}
        d["fwd"] = self.fwd.as_dict() if self.fwd else None
        d["fwd_bwd"] = self.fwd_bwd.as_dict() if self.fwd_bwd else None
        return d | self.extra


def seeded(seed: int) -> torch.Generator:
    g = torch.Generator(device=CPU)
    g.manual_seed(seed)
    return g


def op_case(ctx: Ctx, name: str, shape: str, make, flops: float | None = None, backward: bool = True, accuracy: bool = True, note: str = "") -> Row:
    """`make(device)` -> (leaf tensors and modules that take gradients, `f()` -> output tensor),
    built deterministically so the CPU and device copies hold the same values."""
    row = Row(name, shape, note=note)
    leaves, f = make(ctx.device)
    params = [p for x in leaves for p in (x.parameters() if isinstance(x, nn.Module) else [x])]

    def fwd():
        with torch.no_grad(), autocast(ctx.device, ctx.precision):
            f()

    def fwd_bwd():
        for p in params:
            p.grad = None
        with autocast(ctx.device, ctx.precision):
            out = f()
        out.float().sum().backward()

    reset_peak(ctx.device)
    row.fwd = measure(fwd, ctx.device, budget_s=ctx.budget_s)
    if backward and params:
        row.fwd_bwd = measure(fwd_bwd, ctx.device, budget_s=ctx.budget_s)
    row.peak_gib = peak_gib(ctx.device)
    if flops:
        row.gflops = flops / (row.fwd.median_ms * 1e-3) / 1e9
    if ctx.accuracy and accuracy and ctx.device.type != "cpu":
        _, fc = make(CPU)
        with torch.no_grad():
            ref = fc().float()
            with autocast(ctx.device, ctx.precision):
                got = f().float().cpu()
        diff = (got - ref).abs().max().item()
        row.max_abs, row.max_rel = diff, diff / max(ref.abs().max().item(), 1e-12)
    return row


# --- kernel cases ---------------------------------------------------------------------------

def matmul(ctx: Ctx, dtype: torch.dtype, n: int) -> Row:
    def make(dev):
        g = seeded(1)
        a, b = torch.randn(n, n, generator=g), torch.randn(n, n, generator=g)
        a, b = a.to(dev, dtype if dev.type != "cpu" else torch.float32), b.to(dev, dtype if dev.type != "cpu" else torch.float32)
        return [], lambda: a @ b

    name = {torch.float32: "matmul_f32", torch.float16: "matmul_f16", torch.bfloat16: "matmul_bf16"}[dtype]
    inner = Ctx(ctx.device, "f32", ctx.accuracy, ctx.budget_s)
    return op_case(inner, name, f"{n}x{n} @ {n}x{n}", make, flops=2 * n**3, backward=False, accuracy=n == 1024)


def conv2d_stem(ctx: Ctx) -> Row:
    inp = input_dims()
    c_in = inp.spatial_channels + UraniumNetConfig().spatial.scatter_channels

    def make(dev):
        torch.manual_seed(2)
        conv = nn.Conv2d(c_in, 32, 3, padding=1)
        x = torch.randn(16, c_in, 128, 128, generator=seeded(2)).to(dev).requires_grad_()
        conv = conv.to(dev)
        return [conv, x], lambda: conv(x)

    return op_case(ctx, "conv2d_3x3_128sq", f"[16,{c_in},128,128] -> 32ch", make, flops=2 * 16 * 128 * 128 * 32 * c_in * 9)


def location_ups(ctx: Ctx, rows: int) -> Row:
    """The four ConvTranspose2d of `LocationHead` (032 §2.5 #6) on `rows` rows."""
    cfg = UraniumNetConfig()
    c1, c2, c3, c4 = cfg.spatial.stages
    layers = [(cfg.heads.loc_ar_channels + c4, c3, 8), (2 * c3, c2, 16), (2 * c2, c1, 32), (2 * c1, 32, 64)]

    def make(dev):
        torch.manual_seed(3)
        ups = nn.ModuleList(nn.ConvTranspose2d(i, o, 4, stride=2, padding=1) for i, o, _ in layers).to(dev)
        g = seeded(3)
        xs = [torch.randn(rows, i, s, s, generator=g).to(dev).requires_grad_() for i, _, s in layers]
        return [ups, *xs], lambda: torch.cat([F.relu(u(x)).flatten(1) for u, x in zip(ups, xs)], 1)

    return op_case(ctx, f"conv_transpose_8_to_128_r{rows}", f"4 ups, {rows} rows, 8²->128²", make)


def lstm_seq(ctx: Ctx) -> Row:
    cfg = UraniumNetConfig()
    d_in = cfg.entity.d_model + cfg.spatial.pool + cfg.scalar.hidden

    def make(dev):
        torch.manual_seed(4)
        core = LstmCore(d_in, CoreConfig()).to(dev)
        x = torch.randn(32, 64, d_in, generator=seeded(4)).to(dev).requires_grad_()
        s = core.init_state(32, dev)
        return [core, x], lambda: core.forward_seq(x, s)[0]

    return op_case(ctx, "lstm_seq_b32_t64", f"[32,64,{d_in}] -> 512", make)


def lstm_step(ctx: Ctx) -> Row:
    cfg = UraniumNetConfig()
    d_in = cfg.entity.d_model + cfg.spatial.pool + cfg.scalar.hidden

    def make(dev):
        torch.manual_seed(5)
        core = LstmCore(d_in, CoreConfig()).to(dev)
        x = torch.randn(16, d_in, generator=seeded(5)).to(dev)
        s = core.init_state(16, dev)
        return [], lambda: core.step(x, s)[0]

    return op_case(ctx, "lstm_step_b16", f"[16,{d_in}] -> 512 (inference step)", make, backward=False)


class EncoderStack(nn.Module):
    """Pre-LN attention + FFN layers of the entity encoder (032 §2.1), optional distance bias."""

    def __init__(self, d: int, heads: int, ffn: int, layers: int, buckets: int):
        super().__init__()
        self.ln1 = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.attn = nn.ModuleList(Attention(d, heads) for _ in range(layers))
        self.ln2 = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.ffn = nn.ModuleList(FeedForward(d, ffn, 0.0) for _ in range(layers))
        self.table = nn.Parameter(torch.randn(buckets, layers * heads) * 0.02)
        self.heads = heads

    def forward(self, x, key_ok, bucket=None):
        bias = None if bucket is None else F.embedding(bucket, self.table).permute(0, 3, 1, 2)
        for l in range(len(self.attn)):
            y = self.ln1[l](x)
            b = None if bias is None else bias[:, l * self.heads:(l + 1) * self.heads]
            x = x + self.attn[l](y, y, b, key_ok)
            x = x + self.ffn[l](self.ln2[l](x))
        return x


def transformer(ctx: Ctx, n: int, with_bias: bool) -> Row:
    e = UraniumNetConfig().entity

    def make(dev):
        torch.manual_seed(6)
        m = EncoderStack(e.d_model, e.heads, e.ffn, e.layers, 2 * e.distance_buckets).to(dev)
        g = seeded(6)
        x = torch.randn(16, n, e.d_model, generator=g).to(dev).requires_grad_()
        key_ok = torch.ones(16, 1, n, dtype=torch.bool)
        key_ok[:, :, n - n // 8:] = False
        bucket = torch.randint(0, 2 * e.distance_buckets, (16, n, n), generator=g).to(dev) if with_bias else None
        key_ok = key_ok.to(dev)
        return [m, x], lambda: m(x, key_ok, bucket)

    name = f"attention_distance_bias_n{n}" if with_bias else f"transformer_n{n}"
    return op_case(ctx, name, f"[16,{n},{e.d_model}] L{e.layers} h{e.heads} ff{e.ffn}" + (" + bias[bucket]" if with_bias else ""), make)


def scatter_entities(ctx: Ctx) -> Row:
    from uranium.model.encoders import SpatialEncoder

    cfg = UraniumNetConfig()
    inp = input_dims()

    def make(dev):
        torch.manual_seed(7)
        enc = SpatialEncoder(inp, cfg.spatial, cfg.entity.d_model).to(dev)
        g = seeded(7)
        e = torch.randn(16, 256, cfg.entity.d_model, generator=g).to(dev).requires_grad_()
        obs = {"entity_mask": torch.ones(16, 256, dtype=torch.bool).to(dev), "entity_cell": torch.randint(0, 128 * 128, (16, 256), generator=g).to(dev)}
        return [enc.scatter, e], lambda: enc.scatter_mean(obs, e, 128, 128)

    return op_case(ctx, "scatter_add_entities_to_128sq", f"[16,256,{cfg.entity.d_model}] -> [16,{cfg.spatial.scatter_channels},128,128]", make)


def gather_mask(ctx: Ctx) -> Row:
    inp = input_dims()

    def make(dev):
        g = seeded(8)
        cap = (torch.rand(16, 256, inp.caps, generator=g) < 0.5).to(dev)
        idx = torch.randint(0, inp.caps, (16,), generator=g).to(dev)
        return [], lambda: gather_cap(cap, idx).float()

    return op_case(ctx, "gather_unit_cap_mask", f"unit_cap[16,256,{inp.caps}] gather by action", make, backward=False)


def gumbel(ctx: Ctx) -> Row:
    def make(dev):
        logp = torch.randn(16, 128 * 128, generator=seeded(9)).log_softmax(1).to(dev)
        gen = torch.Generator(device=dev)
        return [], lambda: gumbel_argmax(logp, 1.0, gen.manual_seed(9)).float()

    return op_case(ctx, "gumbel_max_128sq", "logits[16,16384] -> sample", make, backward=False, accuracy=False)


def masked_softmax(ctx: Ctx) -> Row:
    def make(dev):
        g = seeded(10)
        x = torch.randn(16, 128 * 128, generator=g).to(dev).requires_grad_()
        ok = (torch.rand(16, 128 * 128, generator=g) < 0.5).to(dev)
        return [x], lambda: masked_log_softmax(x, ok)

    return op_case(ctx, "softmax_masked_128sq", "[16,16384] masked log-softmax", make)


def pointer_loop(ctx: Ctx, net_cache: dict) -> Row:
    """`Heads.selected`: the 12-step pointer network (032 §2.5 #4)."""
    def make(dev):
        net = model(net_cache, "Gtrxl", dev)
        g = seeded(11)
        b, n = 16, 256
        obs = random_obs(b, n, net.cfg.input, dev, g)
        e = torch.randn(b, n, net.cfg.entity.d_model, generator=g).to(dev).requires_grad_()
        ar = torch.randn(b, net.cfg.heads.ar, generator=g).to(dev).requires_grad_()
        actions = torch.randint(0, net.cfg.input.actions, (b,), generator=g).to(dev)
        info = net.table.info(actions)
        return [net.heads.sel_key, net.heads.sel_query, net.heads.sel_gru, e, ar], lambda: net.heads.selected(ar, e, obs, info, None, 0.0, None)[0]["dists"]

    return op_case(ctx, "pointer_gru_12step", "keys[16,256], 12 steps", make, accuracy=False)


# --- model cases ----------------------------------------------------------------------------

# 032-R20: the comparison runs' structure switches (`[model]` of config/bc_arch_abl_*_pt.toml).
VARIANTS = {
    "terrain": VariantConfig(terrain=True),
    "ctx": VariantConfig(ctx_token=True),
    "pool": VariantConfig(attn_pool=True),
    "outer": VariantConfig(outer=32),
    "all": VariantConfig(terrain=True, ctx_token=True, attn_pool=True, outer=32),
}


def model(cache: dict, kind: str, dev, seed: int = 0, variant: str | None = None) -> UraniumNet:
    key = (kind, str(dev), variant)
    if key not in cache:
        cfg = UraniumNetConfig()
        cfg.input = input_dims()
        cfg.core.kind = kind
        if variant is not None:
            cfg.variant = VARIANTS[variant]
        torch.manual_seed(seed)
        cache[key] = UraniumNet(cfg, action_columns()).to(dev)
    return cache[key]


def net_act(ctx: Ctx, cache: dict, kind: str, b: int, n: int, name: str | None = None, variant: str | None = None) -> Row:
    net = model(cache, kind, ctx.device, variant=variant).eval()
    obs = random_obs(b, n, net.cfg.input, ctx.device, seeded(20))
    state = net.init_state(b, ctx.device)
    gen = torch.Generator(device=ctx.device)

    def act():
        with torch.no_grad(), autocast(ctx.device, ctx.precision):
            out = net.act(obs, state, 1.0, gen.manual_seed(20))
        out["action"]["action"].cpu()  # the broker reads the actions back each batch

    reset_peak(ctx.device)
    row = Row(name or f"net_act_{kind.lower()}_b{b}_n{n}", f"B={b} N={n}", fwd=measure(act, ctx.device, budget_s=ctx.budget_s))
    row.peak_gib = peak_gib(ctx.device)
    params = sum(p.numel() for p in net.parameters())
    row.note = f"p95 {row.fwd.p95_ms:.1f} ms (032-AC2 <= 40 ms at B16 N256), {params} params"
    return row


def net_train(ctx: Ctx, cache: dict, kind: str, b: int, t: int, n: int, variant: str | None = None) -> Row:
    """One lane group's forward + loss + backward as `BcTrainer.group_loss` (041), synthetic
    data; peak device memory is the 060-R3 figure for this shape. The entity padding is not cut
    for a variant's extra tokens (`trim_entity_pad`): N + K here, one tile more than training."""
    net = model(cache, kind, ctx.device, variant=variant).train()
    chunk = int(os.environ.get("URANIUM_BC_ENTITY_CHUNK", "0") or 0)
    obs = random_obs(b * t, n, net.cfg.input, ctx.device, seeded(30))
    labels = sampled_labels(net, obs, 30)
    net.train()
    priv = torch.randn(b * t, net.cfg.input.privileged, generator=seeded(31)).to(ctx.device)
    w = torch.ones(b * t, device=ctx.device)
    params = list(net.parameters())

    def step():
        for p in params:
            p.grad = None
        with autocast(ctx.device, ctx.precision):
            out = net.forward_seq(obs, labels, priv, net.init_state(b, ctx.device), t, chunk)
        loss = sum((-out["logp"][k].float() * w).sum() for k in HEADS) / w.sum() + out["value"][:, 0].float().square().mean()
        loss.backward()

    reset_peak(ctx.device)
    name = f"net_train_{kind.lower()}_b{b}_t{t}_n{n}" + (f"_chunk{chunk}" if chunk else "") + (f"_{variant}" if variant else "")
    row = Row(name, f"B={b} T={t} N={n}", fwd_bwd=measure(step, ctx.device, warmup=2, min_reps=5, budget_s=ctx.budget_s * 2))
    row.peak_gib = peak_gib(ctx.device)
    row.note = f"{b * t / (row.fwd_bwd.median_ms * 1e-3):.1f} samples/s"
    return row


# --- suites ---------------------------------------------------------------------------------

def suite(name: str, ctx: Ctx, cache: dict):
    """(case name, thunk) pairs of a suite; names are filterable before anything runs."""
    if name == "kernels":
        out = [(f"matmul_{d}_{n}", lambda dt=dt, n=n: matmul(ctx, dt, n)) for d, dt in (("f32", torch.float32), ("f16", torch.float16), ("bf16", torch.bfloat16)) for n in (1024, 2048, 4096)]
        out += [
            ("conv2d_3x3_128sq", lambda: conv2d_stem(ctx)),
            ("conv_transpose_8_to_128_r16", lambda: location_ups(ctx, 16)),
            ("conv_transpose_8_to_128_r40", lambda: location_ups(ctx, 40)),
            ("lstm_seq_b32_t64", lambda: lstm_seq(ctx)),
            ("lstm_step_b16", lambda: lstm_step(ctx)),
            ("transformer_n256", lambda: transformer(ctx, 256, False)),
            ("transformer_n512", lambda: transformer(ctx, 512, False)),
            ("attention_distance_bias_n256", lambda: transformer(ctx, 256, True)),
            ("attention_distance_bias_n512", lambda: transformer(ctx, 512, True)),
            ("scatter_add_entities_to_128sq", lambda: scatter_entities(ctx)),
            ("gather_unit_cap_mask", lambda: gather_mask(ctx)),
            ("softmax_masked_128sq", lambda: masked_softmax(ctx)),
            ("gumbel_max_128sq", lambda: gumbel(ctx)),
            ("pointer_gru_12step", lambda: pointer_loop(ctx, cache)),
        ]
        return out
    if name == "model":
        return [
            ("net_act_lstm", lambda: net_act(ctx, cache, "Lstm", 16, 256, "net_act_lstm")),
            ("net_act_gtrxl", lambda: net_act(ctx, cache, "Gtrxl", 16, 256, "net_act_gtrxl")),
            ("net_train_gtrxl_b2_t16_n256", lambda: net_train(ctx, cache, "Gtrxl", 2, 16, 256)),
            ("net_train_lstm_b1_t64_n512", lambda: net_train(ctx, cache, "Lstm", 1, 64, 512)),
            ("net_train_gtrxl_b1_t64_n512", lambda: net_train(ctx, cache, "Gtrxl", 1, 64, 512)),
        ]
    if name == "infer":
        return [(f"net_act_gtrxl_b{b}_n{n}", lambda b=b, n=n: net_act(ctx, cache, "Gtrxl", b, n)) for n in (128, 256, 512) for b in (1, 4, 16, 32)]
    if name == "variants":
        # 032-R20: inference and the worst training shape (060-R3) of each comparison variant.
        out = []
        for v in VARIANTS:
            out.append((f"net_act_gtrxl_{v}", lambda v=v: net_act(ctx, cache, "Gtrxl", 16, 256, f"net_act_gtrxl_{v}", v)))
            out.append((f"net_train_gtrxl_b1_t64_n512_{v}", lambda v=v: net_train(ctx, cache, "Gtrxl", 1, 64, 512, v)))
        return out
    raise ValueError(f"suite {name!r}: expected kernels, model, infer or variants")


def run_suites(names: list[str], ctx: Ctx, pattern: str | None, log=print) -> list[Row]:
    cache: dict = {}
    rows = []
    rx = re.compile(pattern) if pattern else None
    for s in names:
        for case, thunk in suite(s, ctx, cache):
            if rx and not rx.search(case):
                continue
            try:
                row = thunk()
            except Exception as e:  # noqa: BLE001 - a failing case is a reported result
                row = Row(case, "", status="failed", note=str(e).splitlines()[0][:160])
                if ctx.device.type == "cuda":
                    torch.cuda.empty_cache()
            rows.append(row)
            log(f"[bench] {row.case}: " + ", ".join(f"{k} {v.median_ms:.3f} ms" for k, v in (("fwd", row.fwd), ("fwd+bwd", row.fwd_bwd)) if v) + (f" [{row.status}: {row.note}]" if row.status != "ok" else ""))
    return rows
