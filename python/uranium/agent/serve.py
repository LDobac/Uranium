import argparse
import json
import signal
import subprocess
import sys
from pathlib import Path

SURRENDER = ["off", "shadow", "on"]


def _model_args(p, surrender_default: str) -> None:
    p.add_argument("--checkpoint", help="checkpoint folder; relative paths are under URANIUM_SHARED_DIR")
    p.add_argument("--backend", choices=["cuda", "rocm", "cpu"], help="default: this host's GPU, else cpu")
    p.add_argument("--sample", choices=["argmax", "stochastic"], default="argmax")
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--max-wait-ms", type=int, default=5)
    p.add_argument("--zero-last-failed", action="store_true", help="feed last_failed = 0 (BC checkpoint evaluation)")
    p.add_argument("--surrender", choices=SURRENDER, default=surrender_default, help="surrender when lost: off, shadow (record the frame only) or on")
    p.add_argument("--surrender-chat", default="gg", help="chat line before leaving (empty = none, at most 79 bytes)")


def add_parsers(sub) -> None:
    s = sub.add_parser("serve", help="agent server: bridges connect, one record per session")
    s.add_argument("--listen", default="127.0.0.1:7600")
    s.add_argument("--out", required=True, help="JSONL file receiving one record per finished session")
    s.add_argument("--policy", choices=["scripted-t0", "model"], default="scripted-t0")
    s.add_argument("--respond-delay-ms", type=int, default=0, help="delay every reply by this many ms (lag test)")
    s.add_argument("--sessions", type=int, default=0, help="exit after this many sessions (0 = forever)")
    _model_args(s, "shadow")
    s.set_defaults(fn=serve)
    c = sub.add_parser("scr", help="SC:R live play: realtime sessions, surrender on, warm-up first")
    c.add_argument("--listen", default="127.0.0.1:7700")
    c.add_argument("--out", default="runs/scr/sessions.jsonl")
    _model_args(c, "on")
    c.set_defaults(fn=scr)
    r = sub.add_parser("run", help="play every game of a matchup plan")
    r.add_argument("--plan", required=True)
    r.add_argument("--out", help="default runs/<plan name>")
    r.add_argument("--backend", choices=["cuda", "rocm", "cpu"])
    r.add_argument("--max-batch", type=int, default=32)
    r.add_argument("--max-wait-ms", type=int, default=5)
    r.set_defaults(fn=run)
    e = sub.add_parser("eval-worker", help="play an evaluation job's games while holding its lease")
    e.add_argument("--job", required=True, help="job folder with job.toml; relative paths are under URANIUM_SHARED_DIR")
    e.add_argument("--priority", type=int, required=True, help="the faster inference host gets the larger number")
    e.add_argument("--backend", choices=["cuda", "rocm", "cpu"])
    e.add_argument("--local-hostd", help="this PC's hostd address in the hosts file: no new game goes there once this worker stops")
    e.add_argument("--hostd", help="uranium-hostd binary to start for this worker in its own console; stopped after its games")
    e.add_argument("--log", help="also append everything written to stderr (Rust and Python) to this file")
    e.add_argument("--no-warmup", action="store_true", help="skip the warm-up of every (batch, entity count) shape before the first game of a lease period")
    e.add_argument("--max-batch", type=int, default=32)
    e.add_argument("--max-wait-ms", type=int, default=5)
    e.set_defaults(fn=eval_worker)
    t = sub.add_parser("eval-status", help="progress of an evaluation job, merged records and summaries per stage")
    t.add_argument("--job", required=True)
    t.add_argument("--no-write", action="store_true", help="only print the progress")
    t.set_defaults(fn=eval_status)
    k = sub.add_parser("random-checkpoint", help="a randomly initialised checkpoint (smoke tests before a BC checkpoint)")
    k.add_argument("--out", required=True)
    k.add_argument("--core", choices=["lstm", "gtrxl"], default="gtrxl")
    k.add_argument("--win-open", action="store_true", help="include the win_open head")
    k.add_argument("--seed", type=int, default=0)
    k.set_defaults(fn=random_checkpoint)


def new_infer_seed() -> int:
    import secrets

    return secrets.randbits(63)


def seed_inference(seed: int) -> None:
    import torch

    torch.manual_seed(seed)


def start_server(checkpoint: str, sample: str, backend: str | None, max_batch: int, max_wait_ms: int, seed: int, zero_last_failed: bool = False, warmup: bool = False, warm_batches: list[int] | None = None):
    from uranium import _core
    from uranium.agent.infer import InferenceServer
    from uranium.device import configure, resolve

    backend, device, precision = resolve(backend)
    configure(backend)
    n_max = json.loads(_core.feature_dims())["n_max"] if warmup else 0
    broker = _core.Broker(checkpoint, sample, max_batch=max_batch, max_wait_ms=max_wait_ms, zero_last_failed=zero_last_failed, warmup_entities=n_max, seed=seed)
    server = InferenceServer(broker, device, precision)
    if warmup:
        server.warm_up(warm_batches)
    server.start()
    print(f"[infer] {broker.checkpoint_id} ready ({backend}, {precision}, {sample}, max_batch {max_batch}, seed {seed})", file=sys.stderr, flush=True)
    return broker, server


def _sigint_kills() -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def serve(a) -> int:
    from uranium import _core

    _sigint_kills()
    broker = None
    if a.policy == "model":
        if not a.checkpoint:
            raise SystemExit("a model policy needs --checkpoint")
        seed = new_infer_seed()
        broker, _ = start_server(a.checkpoint, a.sample, a.backend, a.max_batch, a.max_wait_ms, seed, a.zero_last_failed)
        seed_inference(seed)
    _core.serve(a.listen, a.out, broker, surrender=a.surrender, surrender_chat=a.surrender_chat, respond_delay_ms=a.respond_delay_ms, sessions=a.sessions)
    return 0


def scr(a) -> int:
    from uranium import _core

    _sigint_kills()
    if not a.checkpoint:
        raise SystemExit("scr needs --checkpoint")
    seed = new_infer_seed()
    broker, _ = start_server(a.checkpoint, a.sample, a.backend, a.max_batch, a.max_wait_ms, seed, a.zero_last_failed, warmup=True)
    seed_inference(seed)
    _core.serve(a.listen, a.out, broker, surrender=a.surrender, surrender_chat=a.surrender_chat)
    return 0


def run(a) -> int:
    from uranium import _core

    _sigint_kills()
    models = []
    seed = new_infer_seed()
    for ckpt, sample in _core.plan_models(a.plan):
        broker, _ = start_server(ckpt, sample, a.backend, a.max_batch, a.max_wait_ms, seed)
        models.append((ckpt, sample, broker))
    seed_inference(seed)
    _core.run_plan(a.plan, models, out=a.out)
    return 0


def eval_worker(a) -> int:
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
            if not reload:
                end = w.wait_lease()
                if end != "lease":
                    print(f"[eval] {end}", file=sys.stderr, flush=True)
                    break
            reload = False
            models, servers = [], []
            try:
                batches = [1 << i for i in range(max(a.max_batch, 1).bit_length())]
                seed = new_infer_seed()
                for ckpt, sample in _needed_models(w):
                    if w.stop_level() > 0:
                        break
                    broker, server = start_server(ckpt, sample, a.backend, a.max_batch, a.max_wait_ms, seed, warmup=not a.no_warmup, warm_batches=batches)
                    models.append((ckpt, sample, broker))
                    servers.append(server)
                if w.stop_level() > 0:
                    end = "stopped"
                else:
                    seed_inference(seed)
                    end = w.dispatch(models)
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
            w.wait_local_hostd_idle()
            hostd.terminate()
            try:
                hostd.wait(timeout=15)
            except subprocess.TimeoutExpired:
                hostd.kill()
            print("[eval] hostd stopped", file=sys.stderr, flush=True)
    return 0


def _needed_models(w) -> list:
    import time

    while True:
        try:
            return w.needed_models()
        except RuntimeError as e:
            if w.stop_level() > 0:
                return []
            print(f"[eval] cannot list the models to load ({e}); retrying in 10 s", file=sys.stderr, flush=True)
            time.sleep(10)


def _tee_stderr(path: str) -> None:
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
        while chunk := os.read(r, 65536):
            for out in (lambda c: os.write(console, c), log.write):
                try:
                    out(chunk)
                except OSError:
                    pass

    t = threading.Thread(target=pump, name="stderr-tee", daemon=True)
    t.start()

    def drain() -> None:
        sys.stderr.flush()
        os.close(2)
        t.join(timeout=5)

    import atexit

    atexit.register(drain)


def _start_hostd(exe: str, listen: str | None, log: str | None):
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
    from uranium.model import UraniumNet
    from uranium.model.config import new_net_config

    torch.manual_seed(a.seed)
    cfg = new_net_config(a.core, a.win_open)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    net = UraniumNet(cfg)
    ckpt.save_model(out, net)
    ckpt.save_meta(out, cfg, {"random_init": True, "seed": a.seed}, "f32")
    print(f"{out.name}@{ckpt.policy_version(out)[:12]}")
    return 0
