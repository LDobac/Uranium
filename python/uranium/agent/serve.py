import json
import signal
import sys

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
