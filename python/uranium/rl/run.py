import argparse
import functools
import json
import os
import sys
from pathlib import Path

AGENTS = ("main", "main-exploiter", "league-exploiter")
EXIT_BUSY = 75
START_KEY = "start"
KL_REFERENCE_KEY = "kl_reference"
STATE_FILE = "train_state.json"


def add_parser(sub) -> None:
    p = sub.add_parser("rl", help="RL + league: learners, inference servers, agent server and game dispatch")
    p.add_argument("--run", required=True, help="run id (checkpoints/<run>/ on the shared drive and its lock)")
    p.add_argument("--bc", required=True, help="BC checkpoint: KL reference, exploiters' start and reset policy, Main's start without --init")
    p.add_argument("--init", help="Main's start weights from an RL checkpoint `ckpt-<step>` or its agents/main; new runs only")
    p.add_argument("--config", default="config/rl.toml")
    p.add_argument("--algo", choices=["vtrace", "ppo"], default="vtrace")
    p.add_argument("--backend", choices=["cuda", "rocm", "cpu"], help="default: this host's GPU, else cpu")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--take-over", action="store_true", help="take the run lock from another host now")
    p.add_argument("--logs", default="runs/rl", help="train.csv, games.csv, games.jsonl, sessions.jsonl")
    p.add_argument("--z-table", help="z table from `uranium-train z-table`")
    p.add_argument("--max-updates", type=int, default=0, help="stop after this many updates (smoke runs)")
    p.add_argument("--max-minutes", type=int, default=0)
    p.add_argument("--max-agent-steps", type=int, default=0, help="clean stop at this many agent steps")
    p.add_argument("--checkpoint-minutes", type=int, default=30, help="minutes between checkpoints")
    p.add_argument("--max-batch", type=int, default=32, help="inference batch cap")
    p.add_argument("--max-wait-ms", type=int, default=5)
    p.add_argument("--ac1-race", choices=["terran", "protoss", "zerg"], help="small check setting: Main vs this T0 race on Fighting Spirit only")
    p.set_defaults(fn=main)


class Learners:
    def __init__(self, cfg: dict, bc_dir: str, init: str | None, resume: str | None, device, precision: str, algo: str, logs: Path, max_batch: int, max_wait_ms: int):
        from uranium import _core, ckpt, kernels
        from uranium.agent.infer import InferenceServer
        from uranium.rl.learner import Learner

        self.device, self.precision = device, precision
        self.max_batch, self.max_wait_ms = max_batch, max_wait_ms
        bc, _ = ckpt.load_model(Path(bc_dir), device, adjust=ckpt.no_dropout)
        refs, race_ref, self.kl_reference = kl_references(cfg, bc_dir, bc, device)
        self.learners: list = []
        self.servers: list = []
        self.brokers_: list = []
        self.snapshot_servers: dict = {}
        self.snapshot_tokens = 0
        self.starts: list = []
        for name, (net, update, optim_dir, start) in zip(AGENTS, start_policies(bc_dir, bc, init, resume, device), strict=True):
            served = _copy(net)
            l = Learner(cfg, net, _copy(bc), device, precision, algo, update, refs=refs, race_ref=race_ref)
            if optim_dir is not None:
                ckpt.load_optimizer(optim_dir, ckpt.OPTIM_MAIN, l.net, l.opt)
            self.learners.append(l)
            self.starts.append(start)
            d = logs / "infer" / name
            d.mkdir(parents=True, exist_ok=True)
            ckpt.save_model(d, l.net)
            ckpt.save_meta(d, l.net.cfg, {"agent": name, "update": update}, precision, kernels.meta(l.net, device))
            b = _core.Broker(str(d), "stochastic", max_batch=max_batch, max_wait_ms=max_wait_ms, warmup_entities=_n_max())
            s = InferenceServer(b, device, precision, net=served)
            s.swap(l.net.state_dict(), update)
            s.start()
            self.brokers_.append(b)
            self.servers.append(s)

    def brokers(self) -> list:
        return self.brokers_

    def version(self, i: int) -> int:
        return self.learners[i].update_count

    def update(self, i: int, batch, weights) -> str:
        from uranium.device import DEVICE_LOCK, device_fatal_guard

        import time

        t0 = time.time_ns()
        with device_fatal_guard(f"learner {AGENTS[i]} update"), DEVICE_LOCK.shared():
            t1 = time.time_ns()
            out = json.dumps(self.learners[i].update(batch, weights))
        prof = os.environ.get("URANIUM_INFER_PROFILE", "")
        if prof:
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
        from uranium import ckpt, kernels
        from uranium.device import DEVICE_LOCK, device_fatal_guard

        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        l = self.learners[i]
        with device_fatal_guard(f"learner {AGENTS[i]} save"), DEVICE_LOCK.shared():
            ckpt.save_model(d, l.net)
            ckpt.save_optimizer(d, ckpt.OPTIM_MAIN, l.net, l.opt)
        train = json.loads(train_json)
        if self.starts[i] is not None:
            train[START_KEY] = self.starts[i]
        if self.kl_reference is not None:
            train[KL_REFERENCE_KEY] = self.kl_reference
        ckpt.save_meta(d, l.net.cfg, train, self.precision, kernels.meta(l.net, self.device))

    def snapshot_server(self, dir: str) -> tuple:
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
        s = self.snapshot_servers.get(token)
        return s is not None and s.is_alive() and s.error is None and not s.stop.is_set()

    def close_snapshot_server(self, token: int) -> None:
        s = self.snapshot_servers.pop(token, None)
        if s is not None:
            s.close()


def start_policies(bc_dir: str, bc, init: str | None, resume: str | None, device) -> list[tuple]:
    from uranium import ckpt

    out = []
    if resume is not None:
        if init is not None:
            print(f"[rl] resuming {resume}: --init ignored (042, AMB-042-13)", file=sys.stderr, flush=True)
        for name in AGENTS:
            d = Path(resume) / ckpt.AGENTS_DIR / name
            net, meta = ckpt.load_model(d, device, adjust=ckpt.no_dropout)
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
    p = Path(path.strip())
    if p.is_absolute():
        return p
    shared = os.environ.get("URANIUM_SHARED_DIR", "").strip()
    if not shared:
        raise ValueError(f"[kl.reference] {path}: a relative path needs URANIUM_SHARED_DIR (044-R9)")
    return Path(shared) / p


def model_key(d: Path, cfg) -> tuple:
    from uranium import ckpt

    return ckpt.policy_version(d), json.dumps(_flat(cfg.to_dict()), sort_keys=True, default=str)


def kl_references(cfg: dict, bc_dir: str, bc, device) -> tuple[list, list | None, dict | None]:
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
        key = model_key(d, ckpt.no_dropout(type(bc.cfg).from_dict(meta["config"])))
        if key not in keys:
            net, _ = ckpt.load_model(d, device, adjust=ckpt.no_dropout)
            same_model(bc.cfg, net.cfg, f"[kl.reference] {race} = {path}")
            keys[key] = len(models) + 1
            names[keys[key]] = f"{path} (sha256 {key[0][:12]})"
            models.append(net)
        race_ref.append(keys[key])
        record[race] = {"path": path, "abs_path": str(d.resolve()), "sha256": key[0]}
    print("[rl] KL reference per lobby race (044-R9): " + ", ".join(f"{r} {names[i]}" for r, i in zip(REF_RACES, race_ref)) + f"; {len(models)} model(s) besides --bc", file=sys.stderr, flush=True)
    return models, race_ref, record


def same_model(bc_cfg, cfg, what: str) -> None:
    want, got = _flat(bc_cfg.to_dict()), _flat(cfg.to_dict())
    if got != want:
        diff = sorted(k for k in want.keys() | got.keys() if want.get(k) != got.get(k))
        raise ValueError(f"{what}: model config differs from the --bc model ({', '.join(diff)}); the KL to the BC policy needs the same model (042, 032-R10)")


def init_agent_dir(init: str) -> Path:
    from uranium import ckpt

    p = Path(init)
    main = p / ckpt.AGENTS_DIR / AGENTS[0]
    if (main / ckpt.WEIGHTS_FILE).is_file():
        return main
    if p.name == AGENTS[0] and p.parent.name == ckpt.AGENTS_DIR and (p / ckpt.WEIGHTS_FILE).is_file():
        return p
    raise FileNotFoundError(f"--init {init}: no {ckpt.AGENTS_DIR}/{AGENTS[0]}/{ckpt.WEIGHTS_FILE} (expected an RL checkpoint folder `ckpt-<step>` or its `{ckpt.AGENTS_DIR}/{AGENTS[0]}`, 042)")


def load_init(init: str, bc_cfg, device) -> tuple:
    from uranium import ckpt

    d = init_agent_dir(init)
    net, meta = ckpt.load_model(d, device, adjust=ckpt.no_dropout)
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
    return Path(logs).absolute()


def _n_max() -> int:
    from uranium import _core

    return json.loads(_core.feature_dims())["n_max"]


def _copy(net):
    import copy

    return copy.deepcopy(net)


def _make(a, device, precision, cfg_json, resume):
    return Learners(json.loads(cfg_json), a.bc, a.init, None if resume is None else str(resume), device, precision, a.algo, Path(a.logs), a.max_batch, a.max_wait_ms)


def main(a) -> int:
    from uranium import _core

    _core.install_stop_handler("rl")
    from uranium.device import configure, resolve

    from uranium.rl.learner import MICRO_LANES_ENV, host_micro_lanes

    backend, device, precision = resolve(a.backend)
    configure(backend, rl=True)
    a.logs = str(logs_dir(a.logs))
    m = host_micro_lanes(json.loads(_core.rl_config(a.config)))
    src = MICRO_LANES_ENV if os.environ.get(MICRO_LANES_ENV, "").strip() else "config micro_lanes"
    print(f"[rl] {backend} ({precision}), algo {a.algo}, learner lane group {m if m else 'all'} lanes ({src}), logs {a.logs}", file=sys.stderr, flush=True)
    from uranium.device import device_fatal_guard

    with device_fatal_guard("rl run loop"):
        end = _core.run_rl(
            functools.partial(_make, a, device, precision), a.run, a.config, algo=a.algo, seed=a.seed, take_over=a.take_over, logs=a.logs,
            z_table=a.z_table, max_updates=a.max_updates, max_minutes=a.max_minutes, max_agent_steps=a.max_agent_steps,
            checkpoint_minutes=a.checkpoint_minutes, fixed_t0=a.ac1_race,
        )
    return EXIT_BUSY if end == "busy" else 0
