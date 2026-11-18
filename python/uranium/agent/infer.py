import contextlib
import json
import os
import sys
import threading
import time

import torch

from uranium import ckpt, kernels
from uranium.device import DEVICE_LOCK, autocast, compiled, exit_if_device_fatal
from uranium.env import env_choice
from uranium.model import to_obs
from uranium.model.heads import HEAD_NAMES
from uranium.model.state import leaves, rebuild, scatter_rows, select_rows, zero_rows

INFER_GRAPH_MODES = ("off", "on")
INFER_STREAM_MODES = ("shared", "own")


def state_to_host(s, slot: int) -> tuple[list, list]:
    ts = [t[slot:slot + 1].float().cpu() for t in leaves(s)]
    return [t.reshape(-1).numpy() for t in ts], [list(t.shape) for t in ts]


def state_from_host(s, parts: list, device: torch.device):
    ls = leaves(s)
    if len(parts) != len(ls):
        raise ValueError(f"{len(parts)} state parts for a Core state of {len(ls)} (051-R4)")
    out = []
    for i, (t, (shape, v)) in enumerate(zip(ls, parts)):
        want = [1, *t.shape[1:]]
        if list(shape) != want:
            raise ValueError(f"state part {i}: shape {list(shape)}, the slot's is {want} (051-R4)")
        x = torch.as_tensor(v, dtype=torch.float32).reshape(want)
        if t.dtype == torch.bool:
            if not ((x == 0) | (x == 1)).all():
                raise ValueError(f"state part {i}: a mask part with values other than 0 and 1 (051-R4)")
        elif t.dtype != torch.float32:
            raise ValueError(f"state part {i}: the slot's state is {t.dtype}, host states are float32 (051-R4)")
        out.append(x.to(device=device, dtype=t.dtype))
    return rebuild(s, iter(out))


def type_logp_rows(type_logp, type_mask, rows: list):
    if not rows:
        return None
    idx = torch.tensor(rows, device=type_logp.device)
    lp = type_logp.index_select(0, idx).float()
    open_ = type_mask.index_select(0, idx).to(torch.bool)
    return torch.where(open_, lp, torch.full_like(lp, float("-inf"))).reshape(-1).cpu().numpy()


def _tree(x, f):
    if isinstance(x, torch.Tensor):
        return f(x)
    if isinstance(x, dict):
        return {k: _tree(v, f) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        out = [_tree(v, f) for v in x]
        return tuple(out) if isinstance(x, tuple) else out
    return x


class GraphedAct:
    def __init__(self, act, device: torch.device, precision: str, max_batch: int = 0):
        self.act, self.device, self.precision = act, device, precision
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
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._act(g["obs"], g["state"], temp)
        torch.cuda.current_stream().wait_stream(side)
        with torch.cuda.graph(g["graph"], pool=self.pool, capture_error_mode="thread_local"):
            g["out"] = self._act(g["obs"], g["state"], temp)
        return g


def dummy_obs(b: int, n: int, device, dtype=torch.float32) -> dict:
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
    def __init__(self, broker, device: torch.device, precision: str, net=None, poll_ms: int = 50):
        super().__init__(name=f"infer-{broker.checkpoint_id}", daemon=True)
        self.broker = broker
        self.device = device
        self.precision = precision
        self.poll_ms = poll_ms
        if net is None:
            net, _ = ckpt.load_model(broker.dir, device)
        self.net = net.to(device).eval()
        self.temp = broker.temp
        kernels.warm(device)
        self.act_fn = compiled(self.net.act, "URANIUM_INFER_COMPILE")
        graph_max = int(os.environ.get("URANIUM_INFER_GRAPH_MAX_BATCH", "") or 0)
        self.graphed = GraphedAct(self.act_fn, device, precision, graph_max) if device.type == "cuda" and env_choice("URANIUM_INFER_GRAPH", INFER_GRAPH_MODES, "off") == "on" else None
        self.state = self.net.init_state(broker.capacity, device)
        self.version = 0
        self.warmed = False
        prof = os.environ.get("URANIUM_INFER_PROFILE", "")
        own = device.type == "cuda" and env_choice("URANIUM_INFER_STREAM", INFER_STREAM_MODES, "shared") == "own"
        self.stream = torch.cuda.Stream(device) if own else None
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(device))
        self.prof = open(f"{prof}-{broker.checkpoint_id.replace('@', '_').replace(':', '_')}.jsonl", "a", encoding="utf-8") if prof and device.type == "cuda" else None
        self.stop = threading.Event()
        self.release = False
        self.error: BaseException | None = None
        self.lock = threading.Lock()

    def close(self) -> None:
        self.release = True
        self.stop.set()
        if self.ident is None:
            self._free()

    def _free(self) -> None:
        with DEVICE_LOCK.exclusive():
            try:
                cuda = self.device.type == "cuda"
                if cuda:
                    torch.cuda.synchronize(self.device)
                if self.graphed is not None:
                    for g in self.graphed.graphs.values():
                        g["graph"].reset()
                    self.graphed.graphs.clear()
                self.net = self.act_fn = self.graphed = self.state = None
                if cuda:
                    torch.cuda.empty_cache()
            except Exception as e:
                exit_if_device_fatal(e, f"inference server {self.broker.checkpoint_id} close")
                raise

    def on_stream(self):
        return torch.cuda.stream(self.stream) if self.stream is not None else contextlib.nullcontext()

    def swap(self, state_dict: dict, version: int) -> None:
        caller = torch.cuda.current_stream(self.device) if self.stream is not None else None
        with self.lock, DEVICE_LOCK.shared(), torch.no_grad(), self.on_stream():
            if caller is not None:
                self.stream.wait_stream(caller)
            self.net.load_state_dict(state_dict)
            self.version = version
            self.broker.set_version(version)
        if caller is not None:
            caller.wait_stream(self.stream)

    def act(self, obs: dict, state) -> dict:
        if self.graphed is not None:
            return self.graphed(obs, state, self.temp)
        with autocast(self.device, self.precision):
            return self.act_fn(obs, state, self.temp)

    @torch.no_grad()
    def warm_up(self, batches: list[int] | None = None) -> None:
        pad = max(self.broker.pad_entities, 1)
        top = self.broker.warmup_entities
        counts = list(range(pad, -(-top // pad) * pad + 1, pad))
        sizes = [(1, n) for n in counts] + ([(2, pad)] if self.broker.capacity > 1 else [])
        if batches:
            sizes = [(b, n) for b in batches for n in counts]
        self.warmed = True
        for b, n in sizes:
            self._warm_one(b, n)

    @torch.no_grad()
    def _warm_one(self, b: int, n: int) -> None:
        with self.on_stream():
            t = time.perf_counter()
            out = self.act(dummy_obs(b, n, self.device), self.net.init_state(b, self.device))
            out["action"]["action"].cpu()
            print(f"[infer] warm-up B={b} N={n}: {time.perf_counter() - t:.1f}s", file=sys.stderr, flush=True)

    def run(self) -> None:
        try:
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
                except Exception as e:  # noqa: BLE001
                    exit_if_device_fatal(e, f"inference server {self.broker.checkpoint_id}")
                    print(f"[infer] {self.broker.checkpoint_id}: {e!r}", file=sys.stderr, flush=True)
                    self.broker.fail(repr(e))
        except BaseException as e:  # noqa: BLE001
            exit_if_device_fatal(e, f"inference server {self.broker.checkpoint_id} warm-up")
            self.error = e
            raise
        finally:
            self.broker.shutdown("model thread stopped")
            if self.release:
                self._free()

    @torch.no_grad()
    def restore(self, restores: list) -> None:
        errors = []
        st = self.state
        for slot, parts in restores:
            try:
                sub = state_from_host(st, parts, self.device)
            except ValueError as e:
                errors.append(str(e))
                continue
            st = scatter_rows(st, torch.tensor([slot], device=self.device), sub)
            errors.append(None)
        self.state = st
        self.broker.reply_restores(errors)

    @torch.no_grad()
    def handle(self, w: dict, t_py: int | None = None) -> None:
        dev = self.device
        t_lock = time.time_ns()
        if w["resets"]:
            self.state = zero_rows(self.state, torch.tensor(w["resets"], device=dev))
        if w.get("restores"):
            self.restore(w["restores"])
        if w["states"]:
            parts, shapes = [], []
            for slot in w["states"]:
                p, s = state_to_host(self.state, slot)
                parts.append(p)
                shapes.append(s)
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
            type_logp_rows(out["type_logp"], obs["type_mask"], b.get("logp_rows") or []),
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
