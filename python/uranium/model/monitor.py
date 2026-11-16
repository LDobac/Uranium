import contextlib
import math

import torch
import torch.nn.functional as F

from uranium.model.layers import ResMlp, conv_layout

ROWS_KEY = "_monitor_rows"
CHUNK_ELEMENTS = 1 << 24
CONST_THRESHOLD = 1e-4
POINTS = ("spatial", "all")

SPEC_LAYERS = (
    ("stem", "spatial.stem"),
    *((f"c{i}", f"spatial.down{i}.conv") for i in range(4)),
    *((f"r{i}", f"spatial.down{i}.inner") for i in range(4)),
    ("r4", "spatial.bottleneck0.inner"),
    ("r5", "spatial.bottleneck1.inner"),
    *((f"d{i}", f"spatial.down{i}.res") for i in range(4)),
    ("b0", "spatial.bottleneck0"),
    ("b1", "spatial.bottleneck1"),
)
SPEC_COLUMNS = (tuple(f"sp_pos_{name}" for name, _ in SPEC_LAYERS) + tuple(f"sp_dead_{name}" for name, _ in SPEC_LAYERS)
                + ("sp_const_s8", "sp_const_pool"))
CONST_POINTS = ("spatial.s8.const", "spatial.pool.const")


class _Acc:
    __slots__ = ("pos", "n", "ch_pos", "channels")

    def __init__(self):
        self.pos = self.n = self.ch_pos = None
        self.channels = 0

    def add(self, pos, n, ch_pos):
        if self.pos is None:
            self.pos, self.n, self.ch_pos = pos, n, ch_pos
            self.channels = ch_pos.numel()
        else:
            self.pos, self.n, self.ch_pos = self.pos + pos, self.n + n, self.ch_pos + ch_pos

    def result(self) -> dict:
        return {
            "pos_frac": self.pos / self.n.clamp_min(1.0),
            "never_active": (self.ch_pos == 0).sum(),
            "channels": self.channels,
            "elements": self.n,
        }


class _Moments:
    __slots__ = ("n", "s", "ss", "dtype")

    def __init__(self):
        self.n = self.s = self.ss = None
        self.dtype = None

    def add(self, n, s, ss, dtype):
        if self.n is None:
            self.n, self.s, self.ss, self.dtype = n, s, ss, str(dtype).replace("torch.", "")
        else:
            self.n, self.s, self.ss = self.n + n, self.s + s, self.ss + ss

    def result(self, threshold: float) -> dict:
        n = self.n.clamp_min(1.0)
        mean = self.s / n
        std = (self.ss / n - mean * mean).clamp_min(0.0).sqrt()
        std = torch.where(self.n > 0, std, torch.full_like(std, math.nan))
        return {"const": (std < threshold).sum(), "channels": int(std.numel()), "threshold": threshold, "dtype": self.dtype, "std": std}


def _keep_of(x, rows, mask):
    keep = None
    if rows is not None:
        keep = rows.reshape(-1, *([1] * (x.dim() - 1))).to(torch.bool)
    if mask is not None:
        m = mask.reshape(*mask.shape, *([1] * (x.dim() - mask.dim()))).to(torch.bool)
        keep = m if keep is None else keep & m
    return keep


def _per(x, keep) -> int:
    per = 1
    for d in range(x.dim()):
        if keep.shape[d] == 1:
            per *= x.shape[d]
    return per


def channel_counts(x: torch.Tensor, rows: torch.Tensor | None = None, mask: torch.Tensor | None = None, channel_dim: int = -1):
    cd = channel_dim % x.dim()
    keep = _keep_of(x, rows, mask)
    dims = [d for d in range(x.dim()) if d != cd]
    if keep is None:
        n = torch.tensor(float(x.numel()), dtype=torch.float64, device=x.device)
    else:
        n = keep.sum().to(torch.float64) * _per(x, keep)
    step = max(1, CHUNK_ELEMENTS // max(1, x[:1].numel()))
    ch = None
    for s in range(0, x.shape[0], step):
        pos = x[s:s + step] > 0
        if keep is not None:
            pos = pos & keep[s:s + step]
        c = pos.sum(dim=dims)
        ch = c if ch is None else ch + c
    ch = torch.zeros(x.shape[cd], dtype=torch.float64, device=x.device) if ch is None else ch.to(torch.float64)
    return ch.sum(), n, ch


def channel_moments(x: torch.Tensor, rows: torch.Tensor | None = None, channel_dim: int = 1):
    cd = channel_dim % x.dim()
    keep = _keep_of(x, rows, None)
    dims = [d for d in range(x.dim()) if d != cd]
    c = x.shape[cd]
    per = x[:1].numel() // c
    if keep is None:
        n = torch.full((c,), float(x.shape[0] * per), dtype=torch.float64, device=x.device)
    else:
        n = (keep.sum().to(torch.float64) * per).expand(c).clone()
    step = max(1, (CHUNK_ELEMENTS // 8) // max(1, x[:1].numel()))
    s = torch.zeros(c, dtype=torch.float64, device=x.device)
    ss = torch.zeros(c, dtype=torch.float64, device=x.device)
    for i in range(0, x.shape[0], step):
        xs = x[i:i + step].to(torch.float64)
        if keep is not None:
            xs = xs * keep[i:i + step]
        s = s + xs.sum(dim=dims)
        ss = ss + (xs * xs).sum(dim=dims)
    return n, s, ss


def channel_stats(x: torch.Tensor, rows: torch.Tensor | None = None, mask: torch.Tensor | None = None, channel_dim: int = -1) -> dict:
    acc = _Acc()
    acc.add(*channel_counts(x, rows, mask, channel_dim))
    return acc.result()


def const_stats(x: torch.Tensor, rows: torch.Tensor | None = None, channel_dim: int = 1, threshold: float = CONST_THRESHOLD) -> dict:
    m = _Moments()
    m.add(*channel_moments(x, rows, channel_dim), x.dtype)
    return m.result(threshold)


class ReluMonitor:
    def __init__(self, net, inner: bool = True, points: str = "spatial", threshold: float = CONST_THRESHOLD):
        if points not in POINTS:
            raise ValueError(f"points {points!r}: expected one of {POINTS}")
        self.net = net
        self.inner = inner
        self.points = points
        self.threshold = threshold
        self.acc: dict[str, _Acc] = {}
        self.mom: dict[str, _Moments] = {}
        self.handles = []
        self.core_skipped = 0
        self._rows = self._ent_mask = self._sp_rows = self._head_rows = None
        self._enc_rows = 0

    def _add(self, name, x, rows=None, mask=None, channel_dim=-1):
        self.acc.setdefault(name, _Acc()).add(*channel_counts(x.detach(), rows, mask, channel_dim))

    def _moments(self, name, x, rows, channel_dim):
        x = x.detach()
        self.mom.setdefault(name, _Moments()).add(*channel_moments(x, rows, channel_dim), x.dtype)

    def reset(self):
        self.acc.clear()
        self.mom.clear()
        self.core_skipped = 0

    def expected(self) -> list[str]:
        sp = self.net.spatial
        out = ["spatial.stem"]
        for i in range(len(sp.downs)):
            out += [f"spatial.down{i}.conv", f"spatial.down{i}.res"] + ([f"spatial.down{i}.inner"] if self.inner else [])
        for j in range(len(sp.bottleneck)):
            out += [f"spatial.bottleneck{j}"] + ([f"spatial.bottleneck{j}.inner"] if self.inner else [])
        return out

    def stats(self) -> dict:
        missing = [k for k in self.expected() if k not in self.acc] + [k for k in CONST_POINTS if k not in self.mom]
        if missing:
            raise RuntimeError(f"ReluMonitor: no values for {missing}: run the eager module's forward with the monitor active (a torch.compile wrapper does not call the hooks)")
        out = {k: a.result() for k, a in self.acc.items()}
        out["spatial.s8"] = out[f"spatial.bottleneck{len(self.net.spatial.bottleneck) - 1}"]
        for k, m in self.mom.items():
            out[k] = m.result(self.threshold)
        return out

    def __enter__(self):
        self._register()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _hook(self, module, fn, pre=False, with_kwargs=False):
        if pre:
            self.handles.append(module.register_forward_pre_hook(fn, with_kwargs=with_kwargs))
        else:
            self.handles.append(module.register_forward_hook(fn, with_kwargs=with_kwargs))

    def _register(self):
        net = self.net
        if self.handles:
            return
        sp = net.spatial

        def spatial_pre(m, args, kwargs):
            obs, e = args[0], args[1]
            rows = obs.get(ROWS_KEY)
            self._sp_rows = rows
            with torch.no_grad():
                spatial = obs["spatial"]
                h, w = spatial.shape[2], spatial.shape[3]
                step = max(1, CHUNK_ELEMENTS // max(1, m.stem.out_channels * h * w))
                for s in range(0, spatial.shape[0], step):
                    sub = {k: obs[k][s:s + step] for k in ("entity_mask", "entity_cell")}
                    x = torch.cat([spatial[s:s + step], m.scatter_mean(sub, e[s:s + step], h, w).to(spatial.dtype)], 1)
                    self._add("spatial.stem", F.relu(m.stem(conv_layout(x, m.channels_last))), None if rows is None else rows[s:s + step], channel_dim=1)

        def spatial_out(m, args, kwargs, out):
            self._moments("spatial.s8.const", out["s8"], self._sp_rows, 1)
            self._moments("spatial.pool.const", out["pool"], self._sp_rows, 1)

        self._hook(sp, spatial_pre, pre=True, with_kwargs=True)
        self._hook(sp, spatial_out, with_kwargs=True)

        def res_pre(name, conv_point):
            def fn(m, args):
                x = args[0]
                if conv_point:
                    self._add(f"{name}.conv", x, self._sp_rows, channel_dim=1)
                if self.inner:
                    with torch.no_grad():
                        step = max(1, CHUNK_ELEMENTS // max(1, x[:1].numel()))
                        for s in range(0, x.shape[0], step):
                            r = None if self._sp_rows is None else self._sp_rows[s:s + step]
                            self._add(f"{name}.inner", m.inner_act(x[s:s + step]), r, channel_dim=1)
            return fn

        for i, d in enumerate(sp.downs):
            name = f"spatial.down{i}"
            self._hook(d.res, res_pre(name, True), pre=True)
            self._hook(d.res, lambda m, i_, o, name=name: self._add(f"{name}.res", o, self._sp_rows, channel_dim=1))
        for j, r in enumerate(sp.bottleneck):
            name = f"spatial.bottleneck{j}"
            if self.inner:
                self._hook(r, res_pre(name, False), pre=True)
            self._hook(r, lambda m, i_, o, name=name: self._add(name, o, self._sp_rows, channel_dim=1))
        if self.points == "all":
            self._register_all()

    def _register_all(self):
        net = self.net

        def entity_pre(m, args, kwargs):
            obs = args[0]
            self._rows = obs.get(ROWS_KEY)
            self._enc_rows = obs["entity_cat"].shape[0]
            tokens = kwargs.get("tokens", args[2] if len(args) > 2 else None)
            ent = obs["entity_mask"]
            if tokens is not None:
                ent = torch.cat([ent, ent.new_zeros(ent.shape[0], tokens.shape[1])], 1)
            self._ent_mask = ent

        self._hook(net.entity, entity_pre, pre=True, with_kwargs=True)
        for l, layer in enumerate(net.entity.layers):
            self._hook(layer.ffn.l1, lambda m, i, o, l=l: self._add(f"entity.L{l}.ffn", o, self._rows, self._ent_mask))

        hd = net.heads

        def heads_pre(m, args, kwargs):
            obs = args[6] if len(args) > 6 else kwargs["obs"]
            self._head_rows = obs.get(ROWS_KEY)

        self._hook(hd, heads_pre, pre=True, with_kwargs=True)
        self._hook(hd.type_in, lambda m, i, o: self._add("heads.type_in", o, self._head_rows))
        for k, r in enumerate(hd.type_res):
            if isinstance(r, ResMlp):
                self._hook(r.l1, lambda m, i, o, k=k: self._add(f"heads.type_res{k}.inner", o, self._head_rows))
            self._hook(r, lambda m, i, o, k=k: self._add(f"heads.type_res{k}.out", o, self._head_rows))

        for l, layer in enumerate(getattr(net.core, "layers", [])):
            self._hook(layer.attn, lambda m, i, o, l=l: self._core(f"core.L{l}.attn", o))
            self._hook(layer.ffn, lambda m, i, o, l=l: self._core(f"core.L{l}.ffn", o))
            self._hook(layer.ffn.l1, lambda m, i, o, l=l: self._core(f"core.L{l}.ffn_hidden", o))

    def _core(self, name, o):
        if o.shape[0] * o.shape[1] != self._enc_rows:
            self.core_skipped += 1
            return
        mask = None if self._rows is None else self._rows.reshape(o.shape[0], o.shape[1])
        self._add(name, o, mask=mask)


def relu_stats(net, obs: dict, labels: dict, priv, s0, t: int, rows: torch.Tensor | None = None, inner: bool = True, points: str = "spatial", phased: bool = False,
               phases: list | None = None, ctx=None, threshold: float = CONST_THRESHOLD, monitor: "ReluMonitor | None" = None) -> dict:
    was = net.training
    net.eval()
    try:
        o = dict(obs)
        if rows is not None:
            o[ROWS_KEY] = rows.to(obs["entity_mask"].device).bool()
        with torch.no_grad(), (ctx if ctx is not None else contextlib.nullcontext()):
            if monitor is not None:
                net.forward_seq(o, labels, priv, s0, t, 0, phased, phases)
                return None
            with ReluMonitor(net, inner, points, threshold) as mon:
                net.forward_seq(o, labels, priv, s0, t, 0, phased, phases)
            return mon.stats()
    finally:
        net.train(was)


def packed(stats: dict) -> torch.Tensor:
    vals = [stats[point]["pos_frac"].to(torch.float64) for _, point in SPEC_LAYERS]
    vals += [stats[point]["never_active"].to(torch.float64) for _, point in SPEC_LAYERS]
    vals += [stats[k]["const"].to(torch.float64) for k in CONST_POINTS]
    return torch.stack(vals)


ALARM_FROM = 600
ALARM_SHARE = 0.5
ALARM_LAYERS = ("stem", "c0", "c1", "c2", "c3", "r0", "r1", "r2", "r3", "r4", "r5")


def spatial_every(cfg: dict) -> int:
    return int((cfg.get("monitor") or {}).get("spatial_every") or 0)


def spatial_due(every: int, update: int) -> bool:
    return every > 0 and update % every == 0


def channel_counts_of(stats: dict) -> dict:
    out = {f"sp_dead_{name}": int(stats[point]["channels"]) for name, point in SPEC_LAYERS}
    out["sp_const_s8"] = int(stats["spatial.s8.const"]["channels"])
    out["sp_const_pool"] = int(stats["spatial.pool.const"]["channels"])
    return out


def spatial_alarm(update: int, record: dict, channels: dict) -> list[str]:
    if update < ALARM_FROM:
        return []
    keys = [f"sp_dead_{n}" for n in ALARM_LAYERS] + ["sp_const_s8", "sp_const_pool"]
    out = []
    for k in keys:
        v, c = record.get(k), channels.get(k, 0)
        if v is not None and c > 0 and v >= ALARM_SHARE * c:
            out.append(f"{k} = {v:g} of {c}")
    return out


def stats_to_host(stats: dict) -> dict:
    out = {}
    for k, v in stats.items():
        if "const" in v:
            out[k] = {"const": int(v["const"]), "channels": v["channels"], "threshold": v["threshold"], "dtype": v["dtype"],
                      "std_min": float(torch.nan_to_num(v["std"], nan=math.inf).min()) if v["std"].numel() else None}
        else:
            out[k] = {"pos_frac": float(v["pos_frac"]), "never_active": int(v["never_active"]), "channels": int(v["channels"]), "elements": int(v["elements"])}
    return out
