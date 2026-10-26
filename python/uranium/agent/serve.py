"""`python -m uranium serve | scr | run | random-checkpoint` (001-R6, 012-R19, 020): the Rust agent
server, SC:R server and scheduler run their sessions on Rust threads; each model they use gets
a `Broker` and an `InferenceServer` thread here (AMB-000-4). Ctrl+C ends the process at once, as
the Rust binaries did (finished sessions are already written)."""

import argparse
import json
import signal
import subprocess
import sys
from pathlib import Path

SURRENDER = ["off", "shadow", "on"]


def _model_args(p, surrender_default: str) -> None:
    p.add_argument("--checkpoint", help="checkpoint folder; relative paths are under URANIUM_SHARED_DIR (061)")
    p.add_argument("--backend", choices=["cuda", "rocm", "cpu"], help="default: this host's GPU, else cpu")
    p.add_argument("--sample", choices=["argmax", "stochastic"], default="argmax")
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--max-wait-ms", type=int, default=5)
    p.add_argument("--zero-last-failed", action="store_true", help="030-R10 / E-5: feed last_failed = 0 (BC checkpoint evaluation)")
    p.add_argument("--surrender", choices=SURRENDER, default=surrender_default, help="031-R12 / 020-R14")
    p.add_argument("--surrender-chat", default="gg", help="020-R14: chat line before leaving (empty = none, at most 79 bytes)")


def add_parsers(sub) -> None:
    s = sub.add_parser("serve", help="agent server: bridges connect, one record per session (001-R6)")
    s.add_argument("--listen", default="127.0.0.1:7600")
    s.add_argument("--out", required=True, help="JSONL file receiving one record per finished session")
    s.add_argument("--policy", choices=["scripted-t0", "model"], default="scripted-t0")
    s.add_argument("--respond-delay-ms", type=int, default=0, help="011-AC7 lag test")
    s.add_argument("--sessions", type=int, default=0, help="exit after this many sessions (0 = forever)")
    _model_args(s, "shadow")
    s.set_defaults(fn=serve)
    c = sub.add_parser("scr", help="SC:R live play (012-R19): realtime sessions, surrender on, warm-up first")
    c.add_argument("--listen", default="127.0.0.1:7700")
    c.add_argument("--out", default="runs/scr/sessions.jsonl")
    _model_args(c, "on")
    c.set_defaults(fn=scr)
    r = sub.add_parser("run", help="play every game of a matchup plan (020)")
    r.add_argument("--plan", required=True)
    r.add_argument("--out", help="default runs/<plan name>")
    r.add_argument("--backend", choices=["cuda", "rocm", "cpu"])
    r.add_argument("--max-batch", type=int, default=32)
    r.add_argument("--max-wait-ms", type=int, default=5)
    r.set_defaults(fn=run)
    e = sub.add_parser("eval-worker", help="020-R15: play an evaluation job's games while holding its lease")
    e.add_argument("--job", required=True, help="job folder with job.toml; relative paths are under URANIUM_SHARED_DIR")
    e.add_argument("--priority", type=int, required=True, help="the faster inference host gets the larger number (061-R16)")
    e.add_argument("--backend", choices=["cuda", "rocm", "cpu"])
    e.add_argument("--local-hostd", help="this PC's hostd address in the hosts file: no new game goes there once this worker stops")
    e.add_argument("--hostd", help="uranium-hostd binary to run for this worker (own console and process group, so Ctrl+C here leaves it to finish the games; stopped after them)")
    e.add_argument("--log", help="also append everything written to stderr (Rust and Python) to this file")
    e.add_argument("--no-warmup", action="store_true", help="skip the warm-up of every (batch, entity count) shape before the first game of a lease period")
    e.add_argument("--max-batch", type=int, default=32)
    e.add_argument("--max-wait-ms", type=int, default=5)
    e.set_defaults(fn=eval_worker)
    t = sub.add_parser("eval-status", help="020-R15: progress of an evaluation job, merged records and summaries per stage")
    t.add_argument("--job", required=True)
    t.add_argument("--no-write", action="store_true", help="only print the progress")
    t.set_defaults(fn=eval_status)
    k = sub.add_parser("random-checkpoint", help="a randomly initialised checkpoint (smoke tests before a BC checkpoint)")
    k.add_argument("--out", required=True)
    k.add_argument("--core", choices=["lstm", "gtrxl"], default="gtrxl")
    k.add_argument("--win-open", action="store_true", help="032-R18: include the win_open head")
    k.add_argument("--seed", type=int, default=0)
    k.set_defaults(fn=random_checkpoint)


def start_server(checkpoint: str, sample: str, backend: str | None, max_batch: int, max_wait_ms: int, zero_last_failed: bool = False, warmup: bool = False, warm_batches: list[int] | None = None):
    """A broker for `checkpoint` and its model thread (started). `warm_batches`: the warm-up also
    runs these batch sizes (see `InferenceServer.warm_up`)."""
    from uranium import _core
    from uranium.agent.infer import InferenceServer
    from uranium.device import configure, resolve

    backend, device, precision = resolve(backend)
    configure(backend)
    n_max = json.loads(_core.feature_dims())["n_max"] if warmup else 0
    broker = _core.Broker(checkpoint, sample, max_batch=max_batch, max_wait_ms=max_wait_ms, zero_last_failed=zero_last_failed, warmup_entities=n_max)
    server = InferenceServer(broker, device, precision)
    if warmup:
        server.warm_up(warm_batches)
    server.start()
    print(f"[infer] {broker.checkpoint_id} ready ({backend}, {precision}, {sample}, max_batch {max_batch})", file=sys.stderr, flush=True)
    return broker, server


def _sigint_kills() -> None:
    # The session loops run in Rust with the GIL released; Python would only see Ctrl+C after they
    # return. Keep the Rust binaries' behaviour: Ctrl+C ends the process.
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def serve(a) -> int:
    from uranium import _core

    _sigint_kills()
    broker = None
    if a.policy == "model":
        if not a.checkpoint:
            raise SystemExit("a model policy needs --checkpoint")
        broker, _ = start_server(a.checkpoint, a.sample, a.backend, a.max_batch, a.max_wait_ms, a.zero_last_failed)
    _core.serve(a.listen, a.out, broker, surrender=a.surrender, surrender_chat=a.surrender_chat, respond_delay_ms=a.respond_delay_ms, sessions=a.sessions)
    return 0


def scr(a) -> int:
    """012-R19: realtime sessions get the Act deadline (012-R7) and last_failed = 0 (012-R14);
    every padded entity count is warmed up before the first game."""
    from uranium import _core

    _sigint_kills()
    if not a.checkpoint:
        raise SystemExit("scr needs --checkpoint")
    broker, _ = start_server(a.checkpoint, a.sample, a.backend, a.max_batch, a.max_wait_ms, a.zero_last_failed, warmup=True)
    _core.serve(a.listen, a.out, broker, surrender=a.surrender, surrender_chat=a.surrender_chat)
    return 0


def run(a) -> int:
    from uranium import _core

    _sigint_kills()
    models = []
    for ckpt, sample in _core.plan_models(a.plan):
        broker, _ = start_server(ckpt, sample, a.backend, a.max_batch, a.max_wait_ms)
        models.append((ckpt, sample, broker))
    _core.run_plan(a.plan, models, out=a.out)
    return 0


def eval_worker(a) -> int:
    """020-R15: wait for the job's lease, load the models of the stages with games left, play
    until the lease goes (yield to a faster host: the games in flight finish here first) or
    Ctrl+C, unload, repeat. The models stay off the device while another worker holds the lease
    (a Linux ROCm host shares its memory with a val scoring run)."""
    import gc

    import torch

    from uranium import _core
    from uranium.device import resolve

    if a.log:
        _tee_stderr(a.log)
    backend = resolve(a.backend)[0]
    w = _core.EvalWorker(a.job, a.priority, backend, a.local_hostd)
    hostd = _start_hostd(a.hostd, a.local_hostd, a.log) if a.hostd else None
    reload = False
    try:
        while True:
            # "reload": games are left whose models were not loaded; the lease is still ours.
            if not reload:
                end = w.wait_lease()
                if end != "lease":
                    print(f"[eval] {end}", file=sys.stderr, flush=True)
                    break
            reload = False
            models, servers = [], []
            try:
                # Every power-of-two batch up to max_batch (the graph pads to one) at every entity
                # count: a shape first met in a game stalls every game of the server for its
                # capture (seconds to minutes with several models).
                batches = [1 << i for i in range(max(a.max_batch, 1).bit_length())]
                for ckpt, sample in w.needed_models():
                    if w.stop_level() > 0:
                        break
                    broker, server = start_server(ckpt, sample, a.backend, a.max_batch, a.max_wait_ms, warmup=not a.no_warmup, warm_batches=batches)
                    models.append((ckpt, sample, broker))
                    servers.append(server)
                # Ctrl+C while loading: no game was handed out, `close` lets the lease go.
                end = "stopped" if w.stop_level() > 0 else w.dispatch(models)
            finally:
                for s in servers:
                    s.stop.set()
                for s in servers:
                    s.join(timeout=30)
                del models, servers
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"[eval] lease period ended: {end}", file=sys.stderr, flush=True)
            if end in ("done", "stopped"):
                break
            reload = end == "reload"
    finally:
        w.close()
        if hostd is not None:
            # Another worker may be playing there (020-R15 ⑥): no new game goes there any more,
            # the ones running finish first (Ctrl+C again: stop now).
            w.wait_local_hostd_idle()
            hostd.terminate()
            try:
                hostd.wait(timeout=15)
            except subprocess.TimeoutExpired:
                hostd.kill()
            print("[eval] hostd stopped", file=sys.stderr, flush=True)
    return 0


def _tee_stderr(path: str) -> None:
    """File descriptor 2 becomes a pipe; a thread copies it to the old stderr and to `path`, so the
    Rust log lines (written to fd 2) land in the file as well."""
    import os
    import threading

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    log = open(path, "ab", buffering=0)
    console = os.dup(2)
    r, w = os.pipe()
    os.dup2(w, 2)
    os.close(w)
    sys.stderr = open(2, "w", encoding="utf-8", errors="replace", buffering=1, closefd=False)

    def pump() -> None:
        # A failed write (full log disk, closed console) must not stop the pump: a full pipe would
        # block every stderr writer, the Rust log lines included.
        while chunk := os.read(r, 65536):
            for out in (lambda c: os.write(console, c), log.write):
                try:
                    out(chunk)
                except OSError:
                    pass

    t = threading.Thread(target=pump, name="stderr-tee", daemon=True)
    t.start()

    def drain() -> None:
        # The last lines are still in the pipe at exit: close its only writer and let them out.
        sys.stderr.flush()
        os.close(2)
        t.join(timeout=5)

    import atexit

    atexit.register(drain)


def _start_hostd(exe: str, listen: str | None, log: str | None):
    """020-R15 ⑥: this PC's hostd for an eval worker, out of reach of the worker's Ctrl+C (own
    console window on Windows, own session elsewhere) so the games in flight can finish. It listens
    on `listen` (the address the job's hosts file gives this PC; every worker reads the same file,
    so it cannot be 127.0.0.1). Elsewhere its output goes to `<log>.hostd` (not the worker's
    stderr pipe, which ends with the worker)."""
    import os

    env = dict(os.environ)
    if listen:
        env["URANIUM_HOSTD_LISTEN"] = listen
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
        p = subprocess.Popen([exe, "serve"], creationflags=flags, env=env)
    else:
        out = open(f"{log}.hostd" if log else os.devnull, "ab")
        p = subprocess.Popen([exe, "serve"], start_new_session=True, env=env, stdout=out, stderr=out)
    print(f"[eval] hostd pid {p.pid}", file=sys.stderr, flush=True)
    return p


def eval_status(a) -> int:
    from uranium import _core

    print(_core.eval_status(a.job, write=not a.no_write))
    return 0


def random_checkpoint(a) -> int:
    import torch

    from uranium import ckpt
    from uranium.bc.trainer import new_net_config
    from uranium.model import UraniumNet

    torch.manual_seed(a.seed)
    cfg = new_net_config(a.core, a.win_open)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    net = UraniumNet(cfg)
    ckpt.save_model(out, net)
    ckpt.save_meta(out, cfg, {"random_init": True, "seed": a.seed}, "f32")
    print(f"{out.name}@{ckpt.policy_version(out)[:12]}")
    return 0
