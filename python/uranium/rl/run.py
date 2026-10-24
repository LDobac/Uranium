"""`python -m uranium rl` (042): the Rust run loop (league, games, segment queues, snapshots,
checkpoints, 061 lock) with the PyTorch learners and the agents' inference servers in this
process (AMB-000-4). Exit codes as before: 75 when the run's lock is another host's (061-R5)."""

import argparse
import functools
import json
import sys
from pathlib import Path

AGENTS = ("main", "main-exploiter", "league-exploiter")
EXIT_BUSY = 75


def add_parser(sub) -> None:
    p = sub.add_parser("rl", help="RL + league: learners, inference servers, agent server and game dispatch (042)")
    p.add_argument("--run", required=True, help="run id (061 checkpoints/<run>/ and the lock)")
    p.add_argument("--bc", required=True, help="BC checkpoint: starting policy and the KL reference")
    p.add_argument("--config", default="config/rl.toml")
    p.add_argument("--algo", choices=["vtrace", "ppo"], default="vtrace")
    p.add_argument("--backend", choices=["cuda", "rocm", "cpu"], help="default: this host's GPU, else cpu")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--take-over", action="store_true", help="take the run lock from another host now (061-R5)")
    p.add_argument("--logs", default="runs/rl", help="train.csv, games.csv, games.jsonl, sessions.jsonl (042-R2)")
    p.add_argument("--z-table", help="z table from `uranium-train z-table` (033-R8)")
    p.add_argument("--max-updates", type=int, default=0, help="stop after this many updates (smoke runs)")
    p.add_argument("--max-minutes", type=int, default=0)
    p.add_argument("--max-agent-steps", type=int, default=0, help="042-R12: clean stop at this many agent steps (042-AC1: 10,000,000)")
    p.add_argument("--checkpoint-minutes", type=int, default=30, help="061-R4")
    p.add_argument("--max-batch", type=int, default=32, help="inference batch cap (020-R11 TBD(M6))")
    p.add_argument("--max-wait-ms", type=int, default=5)
    p.add_argument("--ac1-race", choices=["terran", "protoss", "zerg"], help="042-AC1 small setting: Main vs this T0 race on Fighting Spirit only")
    p.set_defaults(fn=main)


class Learners:
    """The three learning agents and their inference servers, for the Rust run loop
    (`uranium_league::learner::RlLearners`)."""

    def __init__(self, cfg: dict, bc_dir: str, resume: str | None, device, precision: str, algo: str, logs: Path, max_batch: int, max_wait_ms: int):
        from uranium import _core, ckpt
        from uranium.agent.infer import InferenceServer
        from uranium.device import impl_settings
        from uranium.rl.learner import Learner

        self.device, self.precision = device, precision
        self.max_batch, self.max_wait_ms = max_batch, max_wait_ms
        no_dropout = lambda c: _no_dropout(c)  # noqa: E731
        bc, _ = ckpt.load_model(Path(bc_dir), device, adjust=no_dropout)
        self.learners: list = []
        self.servers: list = []
        self.brokers_: list = []
        for name in AGENTS:
            net, _ = ckpt.load_model(Path(bc_dir), device, adjust=no_dropout)
            update = 0
            if resume is not None:
                d = Path(resume) / ckpt.AGENTS_DIR / name
                net, meta = ckpt.load_model(d, device, adjust=no_dropout)
                update = int(meta["train"].get("update", 0))
            l = Learner(cfg, net, _copy(bc), device, precision, algo, update)
            if resume is not None and (Path(resume) / ckpt.AGENTS_DIR / name / ckpt.optim_file(ckpt.OPTIM_MAIN)).is_file():
                ckpt.load_optimizer(Path(resume) / ckpt.AGENTS_DIR / name, ckpt.OPTIM_MAIN, l.net, l.opt)
            self.learners.append(l)
            # The server starts from the learner's current weights (folder for the 020-R5 id).
            d = logs / "infer" / name
            d.mkdir(parents=True, exist_ok=True)
            ckpt.save_model(d, l.net)
            ckpt.save_meta(d, l.net.cfg, {"agent": name, "update": update}, precision, impl_settings())
            b = _core.Broker(str(d), "stochastic", max_batch=max_batch, max_wait_ms=max_wait_ms, warmup_entities=_n_max())
            s = InferenceServer(b, device, precision, net=_copy(l.net))
            s.swap(l.net.state_dict(), update)
            s.start()
            self.brokers_.append(b)
            self.servers.append(s)

    def brokers(self) -> list:
        return self.brokers_

    def version(self, i: int) -> int:
        return self.learners[i].update_count

    # Device work of the run loop's callbacks is `DEVICE_LOCK.shared()`: an inference server may
    # be capturing a graph meanwhile (uranium.device.DeviceLock).
    def update(self, i: int, batch, weights) -> str:
        from uranium.device import DEVICE_LOCK

        with DEVICE_LOCK.shared():
            return json.dumps(self.learners[i].update(batch, weights))

    def push(self, i: int) -> None:
        l = self.learners[i]
        self.servers[i].swap(l.net.state_dict(), l.update_count)

    def reset_to_bc(self, i: int) -> None:
        from uranium.device import DEVICE_LOCK

        with DEVICE_LOCK.shared():
            self.learners[i].reset_to_bc()

    def save(self, i: int, dir: str, train_json: str) -> None:
        from uranium import ckpt
        from uranium.device import DEVICE_LOCK, impl_settings

        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        l = self.learners[i]
        with DEVICE_LOCK.shared():
            ckpt.save_model(d, l.net)
            ckpt.save_optimizer(d, ckpt.OPTIM_MAIN, l.net, l.opt)
        ckpt.save_meta(d, l.net.cfg, json.loads(train_json), self.precision, impl_settings())

    def snapshot_server(self, dir: str):
        """A league snapshot's agent as an opponent (stochastic, 042)."""
        from uranium import _core
        from uranium.agent.infer import InferenceServer

        from uranium.device import DEVICE_LOCK

        b = _core.Broker(str(dir), "stochastic", max_batch=self.max_batch, max_wait_ms=self.max_wait_ms, warmup_entities=_n_max())
        with DEVICE_LOCK.shared():
            s = InferenceServer(b, self.device, self.precision)
        s.start()
        return b


def _n_max() -> int:
    """Entity count the servers warm up to (012-R19, the server thread does it before serving)."""
    from uranium import _core

    return json.loads(_core.feature_dims())["n_max"]


def _no_dropout(c):
    # 032 §2: dropout 0 in RL (0.1 is the BC setting). The actors run without dropout, so a
    # learner with dropout would see π ≠ μ even on-policy (V-trace ρ, KL to π_BC).
    c.entity.dropout = 0.0
    return c


def _copy(net):
    import copy

    return copy.deepcopy(net)


def _make(a, device, precision, cfg_json, resume):
    return Learners(json.loads(cfg_json), a.bc, None if resume is None else str(resume), device, precision, a.algo, Path(a.logs), a.max_batch, a.max_wait_ms)


def main(a) -> int:
    from uranium import _core

    # 042-R10 / 060-R6: first, so a SIGINT during the pull or the model load is kept.
    _core.install_stop_handler("rl")
    from uranium.device import configure, resolve

    backend, device, precision = resolve(a.backend)
    configure(backend)
    print(f"[rl] {backend} ({precision}), algo {a.algo}", file=sys.stderr, flush=True)
    end = _core.run_rl(
        functools.partial(_make, a, device, precision), a.run, a.config, algo=a.algo, seed=a.seed, take_over=a.take_over, logs=a.logs,
        z_table=a.z_table, max_updates=a.max_updates, max_minutes=a.max_minutes, max_agent_steps=a.max_agent_steps,
        checkpoint_minutes=a.checkpoint_minutes, fixed_t0=a.ac1_race,
    )
    return EXIT_BUSY if end == "busy" else 0
