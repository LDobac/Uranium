"""The model thread of one inference server (032-R7, R14, R16; AMB-000-4). Sessions run on Rust
threads and queue observations in a `uranium._core.Broker`; this thread pulls a batch, gathers
the batch rows of the `[capacity, …]` Core state kept on the device, runs one `act` (sampling in
the forward, 032-R9), writes the rows back and replies. Opened slots start from zero (032-R5).
Batching changes throughput only: every row is computed as it would be alone."""

import contextlib
import json
import os
import sys
import threading
import time

import torch

from uranium import ckpt, kernels
from uranium.device import DEVICE_LOCK, autocast, compiled
from uranium.model import to_obs
from uranium.model.heads import HEAD_NAMES
from uranium.model.layers import env_choice

# AMB-000-4 step 2: `URANIUM_INFER_GRAPH` - `act` replayed from a CUDA / HIP graph captured per
# input shape and temperature (`on`) or run eagerly (`off`). The graph runs the same kernels, so
# the outputs are the eager ones (CUDA host: 22-36 ms -> 2.4-9.3 ms per batch; Linux ROCm host:
# batch 1 20 -> 12 ms, from batch 16 the GPU work dominates). Batches are padded to a power of two
# with copies of row 0, whose outputs are dropped, so few graphs are captured.
INFER_GRAPH_MODES = ("off", "on")
# `URANIUM_INFER_STREAM`: the server's GPU work on the thread's current stream (`shared`) or on a
# stream of its own (`own`), so its input copies and act do not queue behind an RL learner in the
# same process. `own` stalled learner and inference together for 20-310 s on the Linux ROCm host
# (2 of 2 runs, cause unknown), so `shared` is the default until a backend is shown safe.
INFER_STREAM_MODES = ("shared", "own")
# `URANIUM_INFER_PROFILE=<path prefix>`: one JSON line per batch in `<prefix>-<checkpoint>.jsonl`
# with the latency parts (broker wait, GIL, locks, input copy, GPU time and wait, reply), for
# AMB-042-10. Off when unset.


def leaves(s) -> list[torch.Tensor]:
    """The tensors of a Core state (LSTM `(c, h)`, GTrXL `([mem per layer], mask)`) in order."""
    if isinstance(s, torch.Tensor):
        return [s]
    return [t for x in s for t in leaves(x)]


def rebuild(s, it):
    """`s`'s structure with its tensors taken from `it` in `leaves` order."""
    if isinstance(s, torch.Tensor):
        return next(it)
    out = [rebuild(x, it) for x in s]
    return tuple(out) if isinstance(s, tuple) else out


def select_rows(s, rows: torch.Tensor):
    return rebuild(s, iter([t.index_select(0, rows) for t in leaves(s)]))


def scatter_rows(s, rows: torch.Tensor, sub):
    """`s[rows] = sub`; other rows are kept bit for bit (032-R14)."""
    return rebuild(s, iter([t.index_copy(0, rows, u.to(t.dtype)) for t, u in zip(leaves(s), leaves(sub))]))


def zero_rows(s, rows: torch.Tensor):
    return rebuild(s, iter([t.index_fill(0, rows, 0) for t in leaves(s)]))


def _tree(x, f):
    """`x` (nested dicts / lists / tuples of tensors, None) with `f` applied to every tensor."""
    if isinstance(x, torch.Tensor):
        return f(x)
    if isinstance(x, dict):
        return {k: _tree(v, f) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        out = [_tree(v, f) for v in x]
        return tuple(out) if isinstance(x, tuple) else out
    return x


class GraphedAct:
    """`net.act` replayed from graphs captured per (padded batch, input shapes, temperature), all
    in one memory pool. The returned tensors live in the graph's memory: read or copy them before
    the next call (the server copies the state rows and reads the replies at once)."""

    def __init__(self, act, device: torch.device, precision: str, max_batch: int = 0):
        self.act, self.device, self.precision = act, device, precision
        # Batches larger than this (0 = no limit) run eagerly: every captured shape keeps its
        # memory, and where the GPU work dominates a graph gains nothing (Linux ROCm host:
        # batch 1 20 -> 12 ms, batch 16 and up unchanged).
        self.max_batch = max_batch
        self.graphs: dict = {}
        self.pool = torch.cuda.graph_pool_handle()

    def _act(self, obs, state, temp: float):
        with autocast(self.device, self.precision):
            return self.act(obs, state, temp)

    def __call__(self, obs: dict, state, temp: float) -> dict:
        b = obs["entity_mask"].shape[0]
        if self.max_batch and b > self.max_batch:
            return self._act(obs, state, temp)
        bp = 1 << (b - 1).bit_length()
        if bp != b:
            idx = torch.cat([torch.arange(b, device=self.device), torch.zeros(bp - b, dtype=torch.long, device=self.device)])
            obs = {k: v.index_select(0, idx) for k, v in obs.items()}
            state = select_rows(state, idx)
        key = (float(temp), tuple((k, tuple(v.shape), v.dtype) for k, v in sorted(obs.items())))
        g = self.graphs.get(key)
        if g is None:
            with DEVICE_LOCK.exclusive():
                g = self.graphs[key] = self._capture(obs, state, temp)
        for k, v in obs.items():
            g["obs"][k].copy_(v)
        for dst, src in zip(leaves(g["state"]), leaves(state)):
            dst.copy_(src)
        g["graph"].replay()
        return g["out"] if bp == b else _tree(g["out"], lambda t: t[:b])

    def _capture(self, obs: dict, state, temp: float) -> dict:
        g = {"obs": {k: v.clone() for k, v in obs.items()}, "state": _tree(state, torch.clone), "graph": torch.cuda.CUDAGraph()}
        # Warm-up on a side stream first (library kernel selection, MIOpen's first call).
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._act(g["obs"], g["state"], temp)
        torch.cuda.current_stream().wait_stream(side)
        # Thread-local capture: in RL the learner (another thread of this process) keeps training
        # on the device while a new shape is captured; global mode would fail its calls.
        with torch.cuda.graph(g["graph"], pool=self.pool, capture_error_mode="thread_local"):
            g["out"] = self._act(g["obs"], g["state"], temp)
        return g


def dummy_obs(b: int, n: int, device, dtype=torch.float32) -> dict:
    """An observation batch of the live schema with every mask open (warm-up only)."""
    from uranium import _core

    d = json.loads(_core.feature_dims())
    m = d["map"]
    z = lambda *s, t=dtype: torch.zeros(*s, dtype=t, device=device)  # noqa: E731
    o = lambda *s: torch.ones(*s, dtype=torch.bool, device=device)  # noqa: E731
    return {
        "entity_cat": z(b, n, d["entity_cat"], t=torch.long), "entity_cont": z(b, n, d["entity_cont"]), "entity_mask": o(b, n),
        "entity_cell": z(b, n, t=torch.long), "entity_pos": z(b, n, 2), "spatial": z(b, d["spatial"], m, m), "scalar": z(b, d["scalar"]),
        "scalar_cat": z(b, 3, t=torch.long), "z_build": z(b, d["z_build"], t=torch.long), "z_units": z(b, d["z_units"]),
        "z_present": z(b, t=torch.bool), "type_mask": o(b, d["actions"]), "unit_cap": o(b, n, d["caps"]),
        "loc_class_mask": o(b, d["loc_classes"], m, m), "is_building": z(b, n, t=torch.bool),
    }


class InferenceServer(threading.Thread):
    """Serves `broker` with the checkpoint in `broker.dir` (or `net`) on `device`."""

    def __init__(self, broker, device: torch.device, precision: str, net=None, poll_ms: int = 50):
        super().__init__(name=f"infer-{broker.checkpoint_id}", daemon=True)
        self.broker = broker
        self.device = device
        self.precision = precision
        self.poll_ms = poll_ms
        if net is None:
            net, _ = ckpt.load_model(broker.dir, device)
        # 032 §2: no dropout at inference.
        self.net = net.to(device).eval()
        self.temp = broker.temp
        kernels.warm(device)  # outside the compiled act (its loader cannot be traced)
        # `URANIUM_INFER_COMPILE`: `act` through torch.compile first (CUDA host, inside the graph:
        # batch 1 2.4 -> 1.3 ms, batch 32 9.3 -> 6.2 ms).
        self.act_fn = compiled(self.net.act, "URANIUM_INFER_COMPILE")
        graph_max = int(os.environ.get("URANIUM_INFER_GRAPH_MAX_BATCH", "") or 0)
        self.graphed = GraphedAct(self.act_fn, device, precision, graph_max) if device.type == "cuda" and env_choice("URANIUM_INFER_GRAPH", INFER_GRAPH_MODES, "off") == "on" else None
        self.state = self.net.core.init_state(broker.capacity, device)
        self.version = 0
        self.warmed = False
        prof = os.environ.get("URANIUM_INFER_PROFILE", "")
        # `URANIUM_INFER_STREAM=own`: a stream of its own (in RL the learner queues seconds of kernels
        # on the default stream; Linux ROCm host, shared: input copy median 1.5 -> 8.8 ms, p95
        # 124 ms during updates). Ordered after the setup work above (the constructing thread's stream).
        own = device.type == "cuda" and env_choice("URANIUM_INFER_STREAM", INFER_STREAM_MODES, "shared") == "own"
        self.stream = torch.cuda.Stream(device) if own else None
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(device))
        self.prof = open(f"{prof}-{broker.checkpoint_id.replace('@', '_').replace(':', '_')}.jsonl", "a", encoding="utf-8") if prof and device.type == "cuda" else None
        self.stop = threading.Event()
        self.error: BaseException | None = None
        # Held while a work item runs; a weight swap waits for the batch in flight.
        self.lock = threading.Lock()

    def on_stream(self):
        """The server's stream as the current one (CPU: nothing)."""
        return torch.cuda.stream(self.stream) if self.stream is not None else contextlib.nullcontext()

    def swap(self, state_dict: dict, version: int) -> None:
        """032-R16: new weights from the next batch on (any thread)."""
        caller = torch.cuda.current_stream(self.device) if self.stream is not None else None
        with self.lock, DEVICE_LOCK.shared(), torch.no_grad(), self.on_stream():
            if caller is not None:
                self.stream.wait_stream(caller)  # the caller's writes to `state_dict` first
            # Copied into the existing parameters: captured graphs keep reading them.
            self.net.load_state_dict(state_dict)
            self.version = version
            self.broker.set_version(version)
        if caller is not None:
            caller.wait_stream(self.stream)  # the caller does not overwrite the sources mid-copy

    def act(self, obs: dict, state) -> dict:
        """One `act` of the served net, from a captured graph when `URANIUM_INFER_GRAPH` is on."""
        if self.graphed is not None:
            return self.graphed(obs, state, self.temp)
        with autocast(self.device, self.precision):
            return self.act_fn(obs, state, self.temp)

    @torch.no_grad()
    def warm_up(self) -> None:
        """012-R19: one batch-1 forward per padded entity count up to `warmup_entities`, so
        kernel selection (and the graph capture) does not stall a live game."""
        pad = max(self.broker.pad_entities, 1)
        top = self.broker.warmup_entities
        # A server of several games also runs batch 2 once: a compiled act otherwise treats the
        # batch size 1 of the warm-up as fixed and recompiles at the first larger batch.
        sizes = [(1, n) for n in range(pad, -(-top // pad) * pad + 1, pad)] + ([(2, pad)] if self.broker.capacity > 1 else [])
        self.warmed = True
        for b, n in sizes:
            self._warm_one(b, n)

    @torch.no_grad()
    def _warm_one(self, b: int, n: int) -> None:
        with self.on_stream():
            t = time.perf_counter()
            out = self.act(dummy_obs(b, n, self.device), self.net.core.init_state(b, self.device))
            out["action"]["action"].cpu()
            print(f"[infer] warm-up B={b} N={n}: {time.perf_counter() - t:.1f}s", file=sys.stderr, flush=True)

    def run(self) -> None:
        try:
            # A server not warmed up by its starter (RL learners and league snapshots) does it here,
            # before its first request: compiling or capturing inside a game stalls it.
            if not self.warmed and self.broker.warmup_entities > 0:
                with DEVICE_LOCK.shared():
                    self.warm_up()
            while not self.stop.is_set() and not self.broker.closed():
                w = self.broker.next(self.poll_ms)
                if w is None:
                    continue
                t_py = time.time_ns()
                try:
                    with self.lock, DEVICE_LOCK.shared(), self.on_stream():
                        self.handle(w, t_py)
                except Exception as e:  # noqa: BLE001 - the sessions get the error, the server goes on
                    print(f"[infer] {self.broker.checkpoint_id}: {e!r}", file=sys.stderr, flush=True)
                    self.broker.fail(repr(e))
        except BaseException as e:  # noqa: BLE001
            self.error = e
            raise
        finally:
            self.broker.shutdown("model thread stopped")

    @torch.no_grad()
    def handle(self, w: dict, t_py: int | None = None) -> None:
        dev = self.device
        t_lock = time.time_ns()
        if w["resets"]:
            self.state = zero_rows(self.state, torch.tensor(w["resets"], device=dev))
        if w["states"]:
            parts, shapes = [], []
            for slot in w["states"]:
                ts = [t[slot:slot + 1].float().cpu() for t in leaves(self.state)]
                parts.append([t.reshape(-1).numpy() for t in ts])
                shapes.append([list(t.shape) for t in ts])
            self.broker.reply_states(parts, shapes)
        b = w["batch"]
        if b is None:
            return
        rows = torch.tensor(b["slots"], device=dev)
        obs = to_obs(b, dev)
        st = select_rows(self.state, rows)
        if self.prof:
            graphs = len(self.graphed.graphs) if self.graphed else 0
            t_prep = time.time_ns()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
        out = self.act(obs, st)
        if self.prof:
            e1.record()
            t_submit = time.time_ns()
        self.state = scatter_rows(self.state, rows, out["state"])
        a = out["action"]
        i64 = lambda t: t.reshape(-1).to(torch.int64).cpu().numpy()  # noqa: E731
        f32 = lambda t: t.reshape(-1).float().cpu().numpy()  # noqa: E731
        self.broker.reply(
            i64(a["action"]), i64(a["delay"]), i64(a["queued"]), i64(a["selected"]), i64(a["target_unit"]), i64(a["target_loc"]),
            f32(out["log_prob"]), f32(out["head_log_prob"]), f32(out["value"]),
            None if out["win_open"] is None else f32(out["win_open"]),
        )
        if self.prof:
            t_end = time.time_ns()
            e1.synchronize()
            ms = lambda a, z: round((z - a) / 1e6, 3)  # noqa: E731
            rec = {
                "t": t_py, "rows": len(b["slots"]), "n": b["n"], "captured": (len(self.graphed.graphs) if self.graphed else 0) > graphs,
                "queue_mean": round(b["wait_mean_ns"] / 1e6, 3), "queue_max": round(b["wait_max_ns"] / 1e6, 3),
                "gil": ms(b["returned_unix_ns"], b["gil_unix_ns"]), "to_py": ms(b["gil_unix_ns"], t_py), "locks": ms(t_py, t_lock),
                "prep": ms(t_lock, t_prep), "submit": ms(t_prep, t_submit), "gpu": round(e0.elapsed_time(e1), 3),
                "sync_reply": ms(t_submit, t_end), "total": ms(b["formed_unix_ns"], t_end),
            }
            self.prof.write(json.dumps(rec) + "\n")
            self.prof.flush()


assert len(HEAD_NAMES) == 6, "032 §2.5: six heads in the broker's order"
