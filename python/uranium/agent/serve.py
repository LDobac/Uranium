"""`python -m uranium serve | scr | run | random-checkpoint` (001-R6, 012-R19, 020): the Rust agent
server, SC:R server and scheduler run their sessions on Rust threads; each model they use gets
a `Broker` and an `InferenceServer` thread here (AMB-000-4). Ctrl+C ends the process at once, as
the Rust binaries did (finished sessions are already written)."""

import argparse
import json
import signal
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
    k = sub.add_parser("random-checkpoint", help="a randomly initialised checkpoint (smoke tests before a BC checkpoint)")
    k.add_argument("--out", required=True)
    k.add_argument("--core", choices=["lstm", "gtrxl"], default="gtrxl")
    k.add_argument("--win-open", action="store_true", help="032-R18: include the win_open head")
    k.add_argument("--seed", type=int, default=0)
    k.set_defaults(fn=random_checkpoint)


def start_server(checkpoint: str, sample: str, backend: str | None, max_batch: int, max_wait_ms: int, zero_last_failed: bool = False, warmup: bool = False):
    """A broker for `checkpoint` and its model thread (started)."""
    from uranium import _core
    from uranium.agent.infer import InferenceServer
    from uranium.device import configure, resolve

    backend, device, precision = resolve(backend)
    configure(backend)
    n_max = json.loads(_core.feature_dims())["n_max"] if warmup else 0
    broker = _core.Broker(checkpoint, sample, max_batch=max_batch, max_wait_ms=max_wait_ms, zero_last_failed=zero_last_failed, warmup_entities=n_max)
    server = InferenceServer(broker, device, precision)
    if warmup:
        server.warm_up()
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
