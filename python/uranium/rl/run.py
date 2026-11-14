"""`python -m uranium rl` (042): the Rust run loop (league, games, segment queues, snapshots,
checkpoints, 061 lock) with the PyTorch learners and the agents' inference servers in this
process (AMB-000-4). Exit codes as before: 75 when the run's lock is another host's (061-R5);
70 (`os._exit`, no save) on a fatal device error in any of the learners' or servers' calls
(042-R20), so the 060-R6 watchdog restarts the run from LATEST."""

import argparse
import functools
import json
import os
import sys
from pathlib import Path

AGENTS = ("main", "main-exploiter", "league-exploiter")
EXIT_BUSY = 75
# Key of the start record in each agent's train JSON (`meta.json` "train"): where its weights
# started, {"from": "init" | "bc", "path" (as given), "abs_path", and for init "update" /
# "step"} (042, AMB-042-13).
START_KEY = "start"
# 044-R9: key of the KL reference record in each agent's train JSON (only with `[kl.reference]`):
# per lobby race {"path" (as configured, "--bc" for the --bc model), "abs_path", "sha256" (weights,
# 020-R5)}.
KL_REFERENCE_KEY = "kl_reference"
# Written by the Rust run loop at a checkpoint's root (uranium_store::ckpt::STATE_FILE, 061).
STATE_FILE = "train_state.json"


def add_parser(sub) -> None:
    p = sub.add_parser("rl", help="RL + league: learners, inference servers, agent server and game dispatch (042)")
    p.add_argument("--run", required=True, help="run id (061 checkpoints/<run>/ and the lock)")
    p.add_argument("--bc", required=True, help="BC checkpoint: the KL reference (per lobby race the config's [kl.reference] when given, 044-R9), the exploiters' start and reset policy, and Main's start without --init")
    p.add_argument("--init", help="042 (AMB-042-13): Main's start weights (only) from an RL checkpoint folder `ckpt-<step>` (its agents/main) or that agents/main folder; new runs only, a run with a LATEST resumes from it")
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

    def __init__(self, cfg: dict, bc_dir: str, init: str | None, resume: str | None, device, precision: str, algo: str, logs: Path, max_batch: int, max_wait_ms: int):
        from uranium import _core, ckpt
        from uranium.agent.infer import InferenceServer
        from uranium.device import impl_settings
        from uranium.rl.learner import Learner

        self.device, self.precision = device, precision
        self.max_batch, self.max_wait_ms = max_batch, max_wait_ms
        # KL reference (AMB-042-9) and exploiter reset policy: always --bc, also with --init (042).
        bc, _ = ckpt.load_model(Path(bc_dir), device, adjust=_no_dropout)
        # 044-R9: the other [kl.reference] models, loaded once and shared by the three learners
        # (frozen; the run loop runs one learner update at a time).
        refs, race_ref, self.kl_reference = kl_references(cfg, bc_dir, bc, device)
        self.learners: list = []
        self.servers: list = []
        self.brokers_: list = []
        # League snapshot / anchor servers by instance token (042, AMB-042-14).
        self.snapshot_servers: dict = {}
        self.snapshot_tokens = 0
        # Per agent: where its weights started (042, AMB-042-13), written into every checkpoint.
        self.starts: list = []
        for name, (net, update, optim_dir, start) in zip(AGENTS, start_policies(bc_dir, bc, init, resume, device), strict=True):
            l = Learner(cfg, net, _copy(bc), device, precision, algo, update, refs=refs, race_ref=race_ref)
            if optim_dir is not None:
                ckpt.load_optimizer(optim_dir, ckpt.OPTIM_MAIN, l.net, l.opt)
            self.learners.append(l)
            self.starts.append(start)
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
        from uranium.device import DEVICE_LOCK, device_fatal_guard

        import time

        t0 = time.time_ns()
        with device_fatal_guard(f"learner {AGENTS[i]} update"), DEVICE_LOCK.shared():
            t1 = time.time_ns()
            out = json.dumps(self.learners[i].update(batch, weights))
        prof = os.environ.get("URANIUM_INFER_PROFILE", "")
        if prof:  # learner update spans next to the servers' batches (AMB-042-10)
            with open(f"{prof}-learner.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"agent": i, "t0": t0, "t_lock": t1, "t1": time.time_ns()}) + "\n")
        return out

    def push(self, i: int) -> None:
        from uranium.device import device_fatal_guard

        l = self.learners[i]
        with device_fatal_guard(f"learner {AGENTS[i]} weight push"):
            self.servers[i].swap(l.net.state_dict(), l.update_count)

    def reset_to_bc(self, i: int) -> None:
        from uranium.device import DEVICE_LOCK, device_fatal_guard

        with device_fatal_guard(f"learner {AGENTS[i]} reset"), DEVICE_LOCK.shared():
            self.learners[i].reset_to_bc()

    def save(self, i: int, dir: str, train_json: str) -> None:
        from uranium import ckpt
        from uranium.device import DEVICE_LOCK, device_fatal_guard, impl_settings

        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        l = self.learners[i]
        # 042-R20: a checkpoint after a fatal device error is never written (exit 70).
        with device_fatal_guard(f"learner {AGENTS[i]} save"), DEVICE_LOCK.shared():
            ckpt.save_model(d, l.net)
            ckpt.save_optimizer(d, ckpt.OPTIM_MAIN, l.net, l.opt)
        train = json.loads(train_json)
        if self.starts[i] is not None:  # None: resumed from a checkpoint written before the record
            train[START_KEY] = self.starts[i]
        if self.kl_reference is not None:  # 044-R9: which checkpoint each race's KL used
            train[KL_REFERENCE_KEY] = self.kl_reference
        ckpt.save_meta(d, l.net.cfg, train, self.precision, impl_settings())

    def snapshot_server(self, dir: str) -> tuple:
        """A league snapshot's or the anchor's agent as an opponent (stochastic, 042): (broker,
        token). The token names this server instance for `snapshot_server_alive` and
        `close_snapshot_server` (another server of the same folder is a different instance)."""
        from uranium import _core
        from uranium.agent.infer import InferenceServer

        from uranium.device import DEVICE_LOCK, device_fatal_guard

        b = _core.Broker(str(dir), "stochastic", max_batch=self.max_batch, max_wait_ms=self.max_wait_ms, warmup_entities=_n_max())
        with device_fatal_guard(f"snapshot server {dir}"), DEVICE_LOCK.shared():
            s = InferenceServer(b, self.device, self.precision)
        s.start()
        self.snapshot_tokens += 1
        self.snapshot_servers[self.snapshot_tokens] = s
        return b, self.snapshot_tokens

    def snapshot_server_alive(self, token: int) -> bool:
        """The server still serves: its model thread runs without an error (a failed warm-up
        ends it after `snapshot_server` returned), and it was not closed."""
        s = self.snapshot_servers.get(token)
        return s is not None and s.is_alive() and s.error is None and not s.stop.is_set()

    def close_snapshot_server(self, token: int) -> None:
        """042 스냅샷 추론 서버 (AMB-042-14): the run loop calls this once no game uses the server
        and no opponent choice can pick it (or its thread died); its thread stops and frees the
        model's device memory (the snapshot's files stay). Only this instance: a newer server of
        the same folder has another token."""
        s = self.snapshot_servers.pop(token, None)
        if s is not None:
            s.close()


def start_policies(bc_dir: str, bc, init: str | None, resume: str | None, device) -> list[tuple]:
    """042 출발 정책 `--init` (AMB-042-13): per agent of `AGENTS`, (net, update, optimizer folder
    or None, start record or None). `bc` is the net loaded from `bc_dir` (copied, not re-read).

    - `resume` (the run's LATEST, 042-R1 / 061) wins over `init`: every agent continues its own
      folder (weights, optimizer, update) and keeps the start record written there.
    - otherwise Main starts from `init` when given (weights only: new optimizer, update 0) and
      the other agents from `bc_dir`."""
    from uranium import ckpt

    out = []
    if resume is not None:
        if init is not None:
            print(f"[rl] resuming {resume}: --init ignored (042, AMB-042-13)", file=sys.stderr, flush=True)
        for name in AGENTS:
            d = Path(resume) / ckpt.AGENTS_DIR / name
            net, meta = ckpt.load_model(d, device, adjust=_no_dropout)
            optim = d if (d / ckpt.optim_file(ckpt.OPTIM_MAIN)).is_file() else None
            out.append((net, int(meta["train"].get("update", 0)), optim, meta["train"].get(START_KEY)))
        return out
    for name in AGENTS:
        if name == AGENTS[0] and init is not None:
            net, start = load_init(init, bc.cfg, device)
            print(f"[rl] {name} starts from --init {start['abs_path']} (update {start['update']}, step {start.get('step')}); KL reference and exploiters from --bc {bc_dir} (042)", file=sys.stderr, flush=True)
        else:
            net = _copy(bc)
            start = {"from": "bc", "path": str(bc_dir), "abs_path": str(Path(bc_dir).resolve())}
        out.append((net, 0, None, start))
    return out


def reference_dir(path: str) -> Path:
    """A `[kl.reference]` value as a folder (044-R9): absolute as given, relative under
    `URANIUM_SHARED_DIR` (as `[league] anchor`, 020-R11; `--bc` itself stays relative to the
    working folder)."""
    p = Path(path.strip())
    if p.is_absolute():
        return p
    shared = os.environ.get("URANIUM_SHARED_DIR", "").strip()
    if not shared:
        raise ValueError(f"[kl.reference] {path}: a relative path needs URANIUM_SHARED_DIR (044-R9)")
    return Path(shared) / p


def model_key(d: Path, cfg) -> tuple:
    """What makes two checkpoints the same KL reference (044-R9): the weights file's SHA-256
    (the 020-R5 policy version) and the model config."""
    from uranium import ckpt

    return ckpt.policy_version(d), json.dumps(_flat(cfg.to_dict()), sort_keys=True, default=str)


def kl_references(cfg: dict, bc_dir: str, bc, device) -> tuple[list, list | None, dict | None]:
    """044-R9 `[kl.reference]`: (the reference models other than `--bc`, each loaded once; per
    lobby race in `REF_RACES` order the index of its reference in `[bc, *models]`; the record
    each checkpoint keeps, `KL_REFERENCE_KEY`), or `([], None, None)` without the table (one
    reference, `--bc`, as before). A race without a key uses `--bc`;
    keys naming the same checkpoint (same weights and config, `model_key`) share one model, and
    one naming `--bc`'s uses `--bc` itself. A model config other than `--bc`'s is a start error
    (the KL compares the same heads' distributions, 042 출발 정책, 032-R10)."""
    from uranium import ckpt
    from uranium.rl.learner import REF_RACES

    table = (cfg.get("kl") or {}).get("reference") or {}
    if not any(table.get(r) for r in REF_RACES):
        return [], None, None
    bc_key = model_key(Path(bc_dir), bc.cfg)
    keys = {bc_key: 0}
    names = {0: f"--bc {bc_dir}"}
    record: dict = {}
    models: list = []
    race_ref: list = []
    for race in REF_RACES:
        path = table.get(race)
        if not path:
            race_ref.append(0)
            record[race] = {"path": "--bc", "abs_path": str(Path(bc_dir).resolve()), "sha256": bc_key[0]}
            continue
        d = reference_dir(path)
        meta = ckpt.read_meta(d)
        key = model_key(d, _no_dropout(type(bc.cfg).from_dict(meta["config"])))
        if key not in keys:
            net, _ = ckpt.load_model(d, device, adjust=_no_dropout)
            same_model(bc.cfg, net.cfg, f"[kl.reference] {race} = {path}")
            keys[key] = len(models) + 1
            names[keys[key]] = f"{path} (sha256 {key[0][:12]})"
            models.append(net)
        race_ref.append(keys[key])
        record[race] = {"path": path, "abs_path": str(d.resolve()), "sha256": key[0]}
    print("[rl] KL reference per lobby race (044-R9): " + ", ".join(f"{r} {names[i]}" for r, i in zip(REF_RACES, race_ref)) + f"; {len(models)} model(s) besides --bc", file=sys.stderr, flush=True)
    return models, race_ref, record


def same_model(bc_cfg, cfg, what: str) -> None:
    """A model config other than the BC model's (`bc_cfg`) is a start error: the KL compares the
    same heads' distributions (042, 032-R10). Both pass the schema check of `ckpt.read_meta`, so
    a feature version on the load-compatible list (030-R5) is not a model difference."""
    want, got = _flat(bc_cfg.to_dict()), _flat(cfg.to_dict())
    if got != want:
        diff = sorted(k for k in want.keys() | got.keys() if want.get(k) != got.get(k))
        raise ValueError(f"{what}: model config differs from the --bc model ({', '.join(diff)}); the KL to the BC policy needs the same model (042, 032-R10)")


def init_agent_dir(init: str) -> Path:
    """`--init` as Main's model folder: an RL checkpoint folder (`ckpt-<step>`, its
    `agents/main`) or that `agents/main` folder itself (042, AMB-042-13)."""
    from uranium import ckpt

    p = Path(init)
    main = p / ckpt.AGENTS_DIR / AGENTS[0]
    if (main / ckpt.WEIGHTS_FILE).is_file():
        return main
    if p.name == AGENTS[0] and p.parent.name == ckpt.AGENTS_DIR and (p / ckpt.WEIGHTS_FILE).is_file():
        return p
    raise FileNotFoundError(f"--init {init}: no {ckpt.AGENTS_DIR}/{AGENTS[0]}/{ckpt.WEIGHTS_FILE} (expected an RL checkpoint folder `ckpt-<step>` or its `{ckpt.AGENTS_DIR}/{AGENTS[0]}`, 042)")


def load_init(init: str, bc_cfg, device) -> tuple:
    """Main's start net from `--init` (weights only) and its start record: the path as given and
    resolved to an absolute path, the update in its train JSON and its checkpoint's step (`train_state.json`) when present.
    A model config other than the BC model's (`bc_cfg`) is a start error: the KL compares the same
    heads' distributions (042, 032-R10). Both pass the schema check of `ckpt.read_meta`, so a
    feature version on the load-compatible list (030-R5) is not a model difference (same inputs)."""
    from uranium import ckpt

    d = init_agent_dir(init)
    net, meta = ckpt.load_model(d, device, adjust=_no_dropout)
    same_model(bc_cfg, net.cfg, f"--init {init}")
    start = {"from": "init", "path": str(init), "abs_path": str(Path(init).resolve()), "update": meta["train"].get("update")}
    state = d.parent.parent / STATE_FILE
    if state.is_file():
        start["step"] = json.loads(state.read_text(encoding="utf-8")).get("step")
    return net, start


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def logs_dir(logs: str) -> Path:
    """`--logs` as an absolute path (042 `--logs` path, 2026-10-31 fix): the inference servers'
    model folders `<logs>/infer/<agent>` go to the Rust Broker, which resolves a relative
    checkpoint path against `URANIUM_SHARED_DIR` (020-R11), so a relative `--logs` is made
    absolute against the working directory first."""
    return Path(logs).absolute()


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
    return Learners(json.loads(cfg_json), a.bc, a.init, None if resume is None else str(resume), device, precision, a.algo, Path(a.logs), a.max_batch, a.max_wait_ms)


def main(a) -> int:
    from uranium import _core

    # 042-R10 / 060-R6: first, so a SIGINT during the pull or the model load is kept.
    _core.install_stop_handler("rl")
    from uranium.device import configure, resolve

    from uranium.rl.learner import MICRO_LANES_ENV, host_micro_lanes

    backend, device, precision = resolve(a.backend)
    configure(backend)
    a.logs = str(logs_dir(a.logs))
    # The learners read the same value (an invalid override fails here, before the lock is taken).
    m = host_micro_lanes(json.loads(_core.rl_config(a.config)))
    src = MICRO_LANES_ENV if os.environ.get(MICRO_LANES_ENV, "").strip() else "config micro_lanes"
    print(f"[rl] {backend} ({precision}), algo {a.algo}, learner lane group {m if m else 'all'} lanes ({src}), logs {a.logs}", file=sys.stderr, flush=True)
    from uranium.device import device_fatal_guard

    # 042-R20: a fatal device error that reached the run loop as an error (its message carries the
    # Python traceback) exits 70 too, like one caught in the learners' calls.
    with device_fatal_guard("rl run loop"):
        end = _core.run_rl(
            functools.partial(_make, a, device, precision), a.run, a.config, algo=a.algo, seed=a.seed, take_over=a.take_over, logs=a.logs,
            z_table=a.z_table, max_updates=a.max_updates, max_minutes=a.max_minutes, max_agent_steps=a.max_agent_steps,
            checkpoint_minutes=a.checkpoint_minutes, fixed_t0=a.ac1_race,
        )
    return EXIT_BUSY if end == "busy" else 0
