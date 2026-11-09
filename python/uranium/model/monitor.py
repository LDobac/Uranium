"""Dead-channel statistics (032-R25, 041-R25, 042-R21; reports/jepa_survey.md 10.4), for the
diagnostic `python -m uranium diag spatial` and the trainers' `[monitor] spatial_every` record.

Points (`ReluMonitor(points=...)`):
- `points="spatial"` (the trainers, 032-R25):
  - ① ReLU outputs: `spatial.stem` relu(stem conv), before the outer FiLM;
    `spatial.down{i}.conv` relu(stride-2 conv), the input of the stage's ResBlock;
    `spatial.down{i}.inner`, `spatial.bottleneck{j}.inner`: the ResBlocks' inner ReLU
    (`ResBlock.inner_act`, the same function the forward uses: relu(c1(relu(x))) under
    `spatial_preact`, 032-R22)
  - ② block outputs: `spatial.down{i}.res` (stage outputs, the first three are the skips),
    `spatial.bottleneck{j}`; `spatial.s8` repeats the last bottleneck block. Under
    `spatial_preact` the blocks have no final ReLU and negative outputs are normal: these points
    are then reference values only (032-R25 ②)
  - ③ `spatial.s8.const`, `spatial.pool.const`: the S8 channels / s_pool dimensions whose
    standard deviation over the measured rows is below `threshold` (AMB-041-14: 1e-4). Measured
    rows: for S8, real rows x H x W per channel; for s_pool, real rows per dimension. Counts, sums
    and sums of squares in f64 per call, so the spatial encoder may run in several calls (outer
    segments, RL phases). Under bf16 autocast the activations are bf16 (rounding steps near 1e-4
    of a value around 1 and up): `dtype` records it.
  Each ①② point: `pos_frac` (share > 0 of the measured elements) and `never_active` (channels
  never > 0 over all of them, the spec's `dead`). 041-R25 names: `SPEC_LAYERS`, `SPEC_COLUMNS`,
  `packed`.
- `points="all"` (the diagnostic) adds: `entity.L{l}.ffn` (the FFN hidden layer, GELU input > 0,
  real tokens only), `heads.type_in`, `heads.type_res{k}.inner` / `.out`, and per GTrXL layer
  `core.L{l}.attn` / `.ffn` (the ReLU before each gate) and `.ffn_hidden`. In the phased RL pass
  (`forward_seq(phased=True)`) the core runs over the whole sequence once per phase, so its points
  are skipped there (`core_skipped` counts the calls).

Rows: `ROWS_KEY` in the observation dict (bool [rows], b-major like every other key) limits the
counts to real steps (padding excluded); the model's own slicing of the observations (outer
segments, phases, entity chunks) cuts it with the rest. Without it every row counts.

Implementation: forward hooks on modules that are always called as modules (the encoders, the
ResBlocks, Linears, the FFN and attention modules), so the vendor / bmm conv switches do not hide a
point; the stem output is recomputed from the encoder's inputs in row chunks. Counts stay on the
device as tensors (no host sync until the caller reads them); scratch memory is bounded by row
chunks. The hooks are on the eager module's submodules: a torch.compile wrapper does not call them,
and `stats()` refuses when an expected spatial point is missing.

Trainer contract (041-R25 ④, 042-R21 ③, 032-AC15): every `[monitor] spatial_every` updates, per
lane group, in a separate pass: `relu_stats(net, obs, labels, priv, s0, t, rows=<real steps>,
points="spatial", ctx=<the trainer's autocast>)` under no_grad, eval mode (no dropout), with the
group's pre-update s0, on the eager module; the returned state is never stored, and no RNG is used,
so training with and without it is bit-identical on CPU. Sum the groups by passing one
`ReluMonitor` through every group (or add the `packed()` vectors' counts), then append
`packed(stats)` (one 1-D f64 tensor in `SPEC_COLUMNS` order) to the update's stats tensor so the
host reads it in the update's single sync (032-R19).
"""

import contextlib
import math

import torch
import torch.nn.functional as F

from uranium.model.layers import ResMlp, conv_layout

ROWS_KEY = "_monitor_rows"
CHUNK_ELEMENTS = 1 << 24  # counted in row chunks of at most this many elements (bounded scratch memory)
CONST_THRESHOLD = 1e-4  # AMB-041-14
POINTS = ("spatial", "all")

# 041-R25 layer names -> monitor points (the four-stage encoder with two bottleneck blocks, 032 §2.2)
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
# 042-R21 ③ train.csv columns, grouped: every sp_pos_<layer> in SPEC_LAYERS order, then every
# sp_dead_<layer> in the same order, then sp_const_s8, sp_const_pool (041-R25 BC log, same order)
SPEC_COLUMNS = (tuple(f"sp_pos_{name}" for name, _ in SPEC_LAYERS) + tuple(f"sp_dead_{name}" for name, _ in SPEC_LAYERS)
                + ("sp_const_s8", "sp_const_pool"))
CONST_POINTS = ("spatial.s8.const", "spatial.pool.const")


class _Acc:
    """Running counts of one point: positive elements, elements, per-channel positives."""

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
    """Per-channel count, sum and sum of squares (f64) for the `const` count."""

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
    """Elements of `x` per kept entry of `keep` (the dims `keep` broadcasts over)."""
    per = 1
    for d in range(x.dim()):
        if keep.shape[d] == 1:
            per *= x.shape[d]
    return per


def channel_counts(x: torch.Tensor, rows: torch.Tensor | None = None, mask: torch.Tensor | None = None, channel_dim: int = -1):
    """(positive elements, counted elements, per-channel positive counts) of `x`, all tensors on
    x's device (float64 / float64 / float64 [C]). `rows` [r] bool keeps rows (dim 0); `mask` bool
    over the leading dims of a token tensor (e.g. [r, n] for [r, n, C]) keeps tokens. Multiplies by
    the masks instead of indexing, so there is no host sync; counts row chunks (one byte per
    element of scratch, never a full-size wider copy)."""
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
        c = pos.sum(dim=dims)  # int64, accumulated without a widened copy
        ch = c if ch is None else ch + c
    ch = torch.zeros(x.shape[cd], dtype=torch.float64, device=x.device) if ch is None else ch.to(torch.float64)
    return ch.sum(), n, ch


def channel_moments(x: torch.Tensor, rows: torch.Tensor | None = None, channel_dim: int = 1):
    """(count [C], sum [C], sum of squares [C]) of `x` per channel over the kept rows and every
    other dim, f64, in row chunks (032-R25 ③)."""
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
    """`pos_frac` (share > 0), `never_active` (channels never > 0), `channels`, `elements` of one
    activation tensor (the "given the activations" entry point). Tensors except `channels`."""
    acc = _Acc()
    acc.add(*channel_counts(x, rows, mask, channel_dim))
    return acc.result()


def const_stats(x: torch.Tensor, rows: torch.Tensor | None = None, channel_dim: int = 1, threshold: float = CONST_THRESHOLD) -> dict:
    """032-R25 ③ of one tensor: channels whose standard deviation over the kept rows (and every
    other dim) is below `threshold`."""
    m = _Moments()
    m.add(*channel_moments(x, rows, channel_dim), x.dtype)
    return m.result(threshold)


class ReluMonitor:
    """Forward hooks on `net` (a `UraniumNet`, eager) that accumulate the points of the module
    docstring while it is active. `with ReluMonitor(net) as mon: net.forward_seq(...)`, then
    `mon.stats()`. The hooks are removed on exit; `reset()` clears the counts."""

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
        # Row / token masks of the latest call of each part (the core's rows follow the encoders').
        self._rows = self._ent_mask = self._sp_rows = self._head_rows = None
        self._enc_rows = 0

    # ------------------------------------------------------------------ bookkeeping
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
        """The spatial points this net must produce (absent ones mean the hooks did not run)."""
        sp = self.net.spatial
        out = ["spatial.stem"]
        for i in range(len(sp.downs)):
            out += [f"spatial.down{i}.conv", f"spatial.down{i}.res"] + ([f"spatial.down{i}.inner"] if self.inner else [])
        for j in range(len(sp.bottleneck)):
            out += [f"spatial.bottleneck{j}"] + ([f"spatial.bottleneck{j}.inner"] if self.inner else [])
        return out

    def stats(self) -> dict:
        """{point: {"pos_frac", "never_active", "channels", "elements"}} and the two `const`
        points {"const", "channels", "threshold", "dtype", "std"}; `spatial.s8` repeats the last
        bottleneck block. Raises if a spatial point is missing (a compiled wrapper, or no pass)."""
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

    # ------------------------------------------------------------------ hooks
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
                # recomputed in row chunks: a full-batch copy of the stem output is the largest
                # activation of the model
                step = max(1, CHUNK_ELEMENTS // max(1, m.stem.out_channels * h * w))
                for s in range(0, spatial.shape[0], step):
                    sub = {k: obs[k][s:s + step] for k in ("entity_mask", "entity_cell")}
                    x = torch.cat([spatial[s:s + step], m.scatter_mean(sub, e[s:s + step], h, w).to(spatial.dtype)], 1)
                    self._add("spatial.stem", F.relu(m.stem(conv_layout(x))), None if rows is None else rows[s:s + step], channel_dim=1)

        def spatial_out(m, args, kwargs, out):
            self._moments("spatial.s8.const", out["s8"], self._sp_rows, 1)
            self._moments("spatial.pool.const", out["pool"], self._sp_rows, 1)

        self._hook(sp, spatial_pre, pre=True, with_kwargs=True)
        self._hook(sp, spatial_out, with_kwargs=True)

        def res_pre(name, conv_point):
            def fn(m, args):
                x = args[0]
                if conv_point:
                    # the block input of a stage is relu(stride-2 conv)
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

        # Rows of each encoder call (the entity encoder runs first in every `encode`).
        def entity_pre(m, args, kwargs):
            obs = args[0]
            self._rows = obs.get(ROWS_KEY)
            self._enc_rows = obs["entity_cat"].shape[0]
            tokens = kwargs.get("tokens", args[2] if len(args) > 2 else None)
            ent = obs["entity_mask"]
            if tokens is not None:
                # the 032 §2.7 extra tokens (context, outer) are not entities: not counted
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
        """A core point over [b, t, d]: the steps are the latest encoder call's rows (b-major, the
        whole sequence or one outer segment). A phased pass runs the core over the whole sequence
        for a subset of rows (and again per phase): skipped."""
        if o.shape[0] * o.shape[1] != self._enc_rows:
            self.core_skipped += 1
            return
        mask = None if self._rows is None else self._rows.reshape(o.shape[0], o.shape[1])
        self._add(name, o, mask=mask)


def relu_stats(net, obs: dict, labels: dict, priv, s0, t: int, rows: torch.Tensor | None = None, inner: bool = True, points: str = "spatial", phased: bool = False,
               phases: list | None = None, ctx=None, threshold: float = CONST_THRESHOLD, monitor: "ReluMonitor | None" = None) -> dict:
    """One teacher-forced pass of `net` (eval mode, no gradients, no RNG; the mode is restored and
    the returned state is dropped) over a batch, returning the points' statistics
    (`ReluMonitor.stats`). `rows` [B*T] bool: real steps. `phased` / `phases`: as
    `forward_seq` (the RL learner). `ctx`: a context manager around the forward (the trainer's
    autocast). `monitor`: accumulate into this active monitor instead (several lane groups);
    then the caller reads `monitor.stats()` and the return value is None."""
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
    """041-R25 / 042-R21 ③ columns (`SPEC_COLUMNS` order) as one 1-D f64 tensor on the stats'
    device: every layer's pos (041-R25 layer order), then every layer's dead (same order), then
    const S8, const pool. The trainer appends it to the update's stats tensor and reads
    everything in one host transfer."""
    vals = [stats[point]["pos_frac"].to(torch.float64) for _, point in SPEC_LAYERS]
    vals += [stats[point]["never_active"].to(torch.float64) for _, point in SPEC_LAYERS]
    vals += [stats[k]["const"].to(torch.float64) for k in CONST_POINTS]
    return torch.stack(vals)


def stats_to_host(stats: dict) -> dict:
    """The statistics as Python numbers (one host sync per tensor; for reports)."""
    out = {}
    for k, v in stats.items():
        if "const" in v:
            out[k] = {"const": int(v["const"]), "channels": v["channels"], "threshold": v["threshold"], "dtype": v["dtype"],
                      "std_min": float(torch.nan_to_num(v["std"], nan=math.inf).min()) if v["std"].numel() else None}
        else:
            out[k] = {"pos_frac": float(v["pos_frac"]), "never_active": int(v["never_active"]), "channels": int(v["channels"]), "elements": int(v["elements"])}
    return out
