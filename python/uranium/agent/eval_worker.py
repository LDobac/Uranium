import subprocess
import sys
from pathlib import Path

from uranium.agent.serve import new_infer_seed, seed_inference, start_server


def add_parsers(sub) -> None:
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
