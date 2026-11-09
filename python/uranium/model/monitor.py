"""Dead-channel statistics of the ReLU-bearing layers (reports/jepa_survey.md 10.4; shared by the
diagnostic `python -m uranium diag spatial` and, later, the trainers).

Per point: the share of activations > 0 and the number of channels never > 0 over every row seen.
A ReLU output is > 0 exactly where its input is, so each point is read from the tensor the ReLU is
applied to (or its output); a GELU point (the transformer FFN hidden layer) counts its input > 0 too.

Points (absent modules are skipped, e.g. the LSTM core has none):
- `spatial.stem`: relu(stem conv), before the outer FiLM (recomputed from the encoder's inputs, so
  it is read whatever the stem's implementation switch)
- `spatial.down{i}.conv`: relu(stride-2 conv), the input of the stage's ResBlock
- `spatial.down{i}.inner`, `spatial.bottleneck{j}.inner`: relu(c1(x)) inside the ResBlock
  (recomputed from the block input; `inner=False` skips it)
- `spatial.down{i}.res`, `spatial.bottleneck{j}`: the ResBlock outputs (skip i, deep stages);
  `spatial.s8` is the last bottleneck output (the location head's deepest input and `sp_pool`'s)
- `entity.L{l}.ffn`: the entity transformer FFN hidden layer (GELU input > 0), real tokens only
- `heads.type_in`: relu(type_in(h)); `heads.type_res{k}.inner`: the ResMlp's relu(l1(.));
  `heads.type_res{k}.out`: the block output, which the next ReLU reads
- `core.L{l}.attn`, `core.L{l}.ffn`: relu(attention / FFN output) before the GTrXL gates;
  `core.L{l}.ffn_hidden`: the core FFN hidden layer (GELU input > 0)

Module forward hooks only, on modules that are always called as modules (the ResBlocks, Linears,
the FFN and attention modules), so the vendor / bmm conv switches do not hide a point. The counts
stay on the device as tensors: no host sync until the caller reads them. Use the eager module (not a
torch.compile wrapper): the hooks are on its submodules.

Rows: `ROWS_KEY` in the observation dict (bool [rows], b-major like every other key) limits the
counts to real rows (padding steps excluded); the model's own slicing of the observations (outer
segments, entity chunks) cuts it with the rest. Without it every row counts.
"""

import contextlib

import torch
import torch.nn.functional as F

from uranium.model.layers import ResMlp, conv_layout

ROWS_KEY = "_monitor_rows"


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


CHUNK_ELEMENTS = 1 << 24  # counted in row chunks of at most this many elements (bounded scratch memory)


def channel_counts(x: torch.Tensor, rows: torch.Tensor | None = None, mask: torch.Tensor | None = None, channel_dim: int = -1):
    """(positive elements, counted elements, per-channel positive counts) of `x`, all tensors on
    x's device (float64 / float64 / float64 [C]). `rows` [r] bool keeps rows (dim 0); `mask` bool
    over the leading dims of a token tensor (e.g. [r, n] for [r, n, C]) keeps tokens. Multiplies by
    the masks instead of indexing, so there is no host sync; counts row chunks (one byte per
    element of scratch, never a full-size wider copy)."""
    cd = channel_dim % x.dim()
    keep = None
    if rows is not None:
        keep = rows.reshape(-1, *([1] * (x.dim() - 1))).to(torch.bool)
    if mask is not None:
        m = mask.reshape(*mask.shape, *([1] * (x.dim() - mask.dim()))).to(torch.bool)
        keep = m if keep is None else keep & m
    dims = [d for d in range(x.dim()) if d != cd]
    if keep is None:
        n = torch.tensor(float(x.numel()), dtype=torch.float64, device=x.device)
    else:
        # elements per kept entry of `keep`: the product of the dims it broadcasts over
        per = 1
        for d in range(x.dim()):
            if keep.shape[d] == 1:
                per *= x.shape[d]
        n = keep.sum().to(torch.float64) * per
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


def channel_stats(x: torch.Tensor, rows: torch.Tensor | None = None, mask: torch.Tensor | None = None, channel_dim: int = -1) -> dict:
    """`pos_frac` (share > 0), `never_active` (channels never > 0), `channels`, `elements` of one
    activation tensor (the "given the activations" entry point). Tensors except `channels`."""
    acc = _Acc()
    acc.add(*channel_counts(x, rows, mask, channel_dim))
    return acc.result()


class ReluMonitor:
    """Forward hooks on `net` (a `UraniumNet`) that accumulate the points of the module docstring
    while it is active. `with ReluMonitor(net) as mon: net.forward_seq(...)`, then `mon.stats()`.
    The hooks are removed on exit; `reset()` clears the counts."""

    def __init__(self, net, inner: bool = True):
        self.net = net
        self.inner = inner
        self.acc: dict[str, _Acc] = {}
        self.handles = []
        # Row / token masks of the latest call of each part (the core's rows follow the encoders').
        self._rows = self._ent_mask = self._sp_rows = self._head_rows = None

    # ------------------------------------------------------------------ bookkeeping
    def _add(self, name, x, rows=None, mask=None, channel_dim=-1):
        self.acc.setdefault(name, _Acc()).add(*channel_counts(x.detach(), rows, mask, channel_dim))

    def reset(self):
        self.acc.clear()

    def stats(self) -> dict:
        """{point: {"pos_frac", "never_active", "channels", "elements"}}; `spatial.s8` repeats the
        last bottleneck block."""
        out = {k: a.result() for k, a in self.acc.items()}
        last = [k for k in out if k.startswith("spatial.bottleneck") and not k.endswith(".inner")]
        if last:
            out["spatial.s8"] = out[sorted(last, key=lambda k: int(k[len("spatial.bottleneck"):]))[-1]]
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
        # Rows of each encoder call (the entity encoder runs in every `encode`, first).
        def entity_pre(m, args, kwargs):
            obs = args[0]
            self._rows = obs.get(ROWS_KEY)
            tokens = kwargs.get("tokens", args[2] if len(args) > 2 else None)
            ent = obs["entity_mask"]
            if tokens is not None:
                # the 032 §2.7 extra tokens (context, outer) are not entities: not counted
                ent = torch.cat([ent, ent.new_zeros(ent.shape[0], tokens.shape[1])], 1)
            self._ent_mask = ent

        self._hook(net.entity, entity_pre, pre=True, with_kwargs=True)
        for l, layer in enumerate(net.entity.layers):
            self._hook(layer.ffn.l1, lambda m, i, o, l=l: self._add(f"entity.L{l}.ffn", o, self._rows, self._ent_mask))

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

        self._hook(sp, spatial_pre, pre=True, with_kwargs=True)

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
                            self._add(f"{name}.inner", F.relu(m.c1(x[s:s + step])), r, channel_dim=1)
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

        core = net.core
        for l, layer in enumerate(getattr(core, "layers", [])):
            self._hook(layer.attn, lambda m, i, o, l=l: self._add(f"core.L{l}.attn", o, mask=self._core_rows(o)))
            self._hook(layer.ffn, lambda m, i, o, l=l: self._add(f"core.L{l}.ffn", o, mask=self._core_rows(o)))
            self._hook(layer.ffn.l1, lambda m, i, o, l=l: self._add(f"core.L{l}.ffn_hidden", o, mask=self._core_rows(o)))

    def _core_rows(self, o):
        """The core's [b, t, d] steps as a [b, t] mask: the rows of the latest encoder call, which
        are the same rows in b-major order (the whole sequence, or one outer segment)."""
        return None if self._rows is None else self._rows.reshape(o.shape[0], o.shape[1])


@torch.no_grad()
def relu_stats(net, obs: dict, labels: dict, priv, s0, t: int, rows: torch.Tensor | None = None, inner: bool = True) -> dict:
    """One teacher-forced pass of `net` (eval mode, no gradients; the mode is restored) over a
    batch, returning the points' statistics (`ReluMonitor.stats`). `rows` [B*T] bool: real rows."""
    was = net.training
    net.eval()
    try:
        o = dict(obs)
        if rows is not None:
            o[ROWS_KEY] = rows.to(obs["entity_mask"].device).bool()
        with ReluMonitor(net, inner) as mon:
            net.forward_seq(o, labels, priv, s0, t)
        return mon.stats()
    finally:
        net.train(was)


def stats_to_host(stats: dict) -> dict:
    """The statistics as Python numbers (one host sync per tensor; for reports)."""
    out = {}
    for k, v in stats.items():
        out[k] = {"pos_frac": float(v["pos_frac"]), "never_active": int(v["never_active"]), "channels": int(v["channels"]), "elements": int(v["elements"])}
    return out


@contextlib.contextmanager
def monitoring(net, inner: bool = True):
    """`ReluMonitor` as a plain context manager yielding the monitor."""
    with ReluMonitor(net, inner) as mon:
        yield mon
