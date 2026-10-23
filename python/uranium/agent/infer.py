"""The model thread of one inference server (032-R7, R14, R16; AMB-000-4). Sessions run on Rust
threads and queue observations in a `uranium._core.Broker`; this thread pulls a batch, gathers
the batch rows of the `[capacity, …]` Core state kept on the device, runs one `act` (sampling in
the forward, 032-R9), writes the rows back and replies. Opened slots start from zero (032-R5).
Batching changes throughput only: every row is computed as it would be alone."""

import json
import sys
import threading
import time

import torch

from uranium import ckpt
from uranium.device import autocast
from uranium.model import to_obs
from uranium.model.heads import HEAD_NAMES


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
        self.state = self.net.core.init_state(broker.capacity, device)
        self.version = 0
        self.stop = threading.Event()
        self.error: BaseException | None = None

    def swap(self, state_dict: dict, version: int) -> None:
        """032-R16: new weights from the next batch on (call between batches)."""
        with torch.no_grad():
            self.net.load_state_dict(state_dict)
        self.version = version
        self.broker.set_version(version)

    @torch.no_grad()
    def warm_up(self) -> None:
        """012-R19: one batch-1 forward per padded entity count up to `warmup_entities`, so
        kernel selection does not stall a live game."""
        pad = max(self.broker.pad_entities, 1)
        top = self.broker.warmup_entities
        for n in range(pad, -(-top // pad) * pad + 1, pad):
            t = time.perf_counter()
            with autocast(self.device, self.precision):
                out = self.net.act(dummy_obs(1, n, self.device), self.net.core.init_state(1, self.device), self.temp)
            out["action"]["action"].cpu()
            print(f"[infer] warm-up N={n}: {time.perf_counter() - t:.1f}s", file=sys.stderr, flush=True)

    def run(self) -> None:
        try:
            while not self.stop.is_set() and not self.broker.closed():
                w = self.broker.next(self.poll_ms)
                if w is None:
                    continue
                try:
                    self.handle(w)
                except Exception as e:  # noqa: BLE001 - the sessions get the error, the server goes on
                    print(f"[infer] {self.broker.checkpoint_id}: {e!r}", file=sys.stderr, flush=True)
                    self.broker.fail(repr(e))
        except BaseException as e:  # noqa: BLE001
            self.error = e
            raise
        finally:
            self.broker.shutdown("model thread stopped")

    @torch.no_grad()
    def handle(self, w: dict) -> None:
        dev = self.device
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
        with autocast(dev, self.precision):
            out = self.net.act(obs, select_rows(self.state, rows), self.temp)
        self.state = scatter_rows(self.state, rows, out["state"])
        a = out["action"]
        i64 = lambda t: t.reshape(-1).to(torch.int64).cpu().numpy()  # noqa: E731
        f32 = lambda t: t.reshape(-1).float().cpu().numpy()  # noqa: E731
        self.broker.reply(
            i64(a["action"]), i64(a["delay"]), i64(a["queued"]), i64(a["selected"]), i64(a["target_unit"]), i64(a["target_loc"]),
            f32(out["log_prob"]), f32(out["head_log_prob"]), f32(out["value"]),
            None if out["win_open"] is None else f32(out["win_open"]),
        )


assert len(HEAD_NAMES) == 6, "032 §2.5: six heads in the broker's order"
