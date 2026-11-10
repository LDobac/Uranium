"""Lineage measurement protocol (050-R9, 050-AC5; plan "측정 프로토콜", docs/measurement_protocol.md):
config/protocol/lineage.toml pins, per lineage point, the metric menu and the conditions Checkpoint 1 was
measured with; this tool turns a (point, checkpoint) into eval jobs and refuses anything that differs (plan
conditions, sampling, zero_last_failed, family, run prefix, T1 platform, OSS bot builds). Every job gets a
`protocol.json` record: checkpoint sha256, git commit and uncommitted protocol files (paths, diff hash), plan
content hashes, host platforms and bot hashes; `verify` checks a job against it.

The Checkpoint 1 lineage (`--lineage ck1`) is measured already: cells it has are not measured again but
re-tabulated from its existing outputs with both held / bunker definitions (`tables`, also games of a
protocol plan inside a larger job). It takes the cue diagnostics and, per point, the 050-R9 point-table cells
it never measured (`ck1_supplement`), with the same plans; its dead_channels cell is not measured (no
training-log statistics).

The diagnostics (`diag`, `check-diag`): cue_diag runs `python -m uranium diag` on the pinned hold-outs (the Zerg
teacher hold-out and human TvT / TvP games) with the pinned settings; dead_channels summarises the run's
041-R25 training-log records (lineage.toml [metric.cue_diag], [metric.dead_channels]); tables:
tools/eval/diag_table.py.
The s1e2 per-checkpoint loop (041-R20 ③): tools/eval/s1e2_loop.py.

    python tools/eval/protocol.py list
    python tools/eval/protocol.py job --lineage m2 --point s2e2 --checkpoint checkpoints/m2-s2e2/ckpt-N \\
        --metrics rush,macro,tvz --job m2-s2e2-20261112
    python tools/eval/protocol.py tables --lineage ck1 --point r1 [--json out.json]
    python tools/eval/protocol.py render --plan rush_check --checkpoint checkpoints/<run>/ckpt-N --name <stage>
    python tools/eval/protocol.py verify <job dir> --lineage m2 --point s2e2
    python tools/eval/protocol.py bc-eval --metric val500 --lineage m2 --point s2e2 --checkpoint <dir> [--out <json>]
    python tools/eval/protocol.py check-bc-eval <output json> --metric val500
    python tools/eval/protocol.py diag --lineage m2 --point s1e1 --checkpoint checkpoints/m2-s1/ckpt-N --train-log <log> [--metrics cue_diag,dead_channels] [--run]
    python tools/eval/protocol.py check-diag <shared>/artifacts/diag/<run>__<ckpt> [--metrics ...] [--checkpoint <dir>]
    python tools/eval/protocol.py bots [--hosts tools/eval/hosts.toml]
    python tools/eval/protocol.py compare <tables json A> <tables json B>

Host lists (addresses, platforms, ssh aliases, bot folders) live in the git-ignored tools/eval/hosts.toml
(template tools/eval/hosts.example.toml, 001-R10). Relative checkpoints are under URANIUM_SHARED_DIR (061).
DAgger generation plans are written by tools/eval/rush_check_plan.py --dagger (docs/commands.md).
"""

from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_common as ec  # noqa: E402

ROOT = ec.ROOT
LINEAGE = ROOT / "config" / "protocol" / "lineage.toml"
BOTS = ROOT / "config" / "protocol" / "t1_bots.toml"
HOSTS = Path(__file__).resolve().parent / "hosts.toml"
WEIGHTS = "model.safetensors"
RECORD = "protocol.json"
PROTOCOL_PATHS = ("tools/eval", "config/plans/protocol", "config/protocol")
PLATFORMS = ("windows", "linux")
# Plan defaults of crates/uranium-runner/src/plan.rs (Plan, Matchup, Side) for the normalised form.
PLAN_DEFAULTS = {"seed_base": 1000, "same_seed": False, "max_frames": 86_400, "wall_timeout_s": 1800, "privileged": False,
                 "save_replay": False, "zero_last_failed": False, "surrender": "shadow", "surrender_chat": "gg"}
CKPT, NAME = "@CHECKPOINT@", "@NAME@"
# 032-R8: CUDA runs bf16 mixed, ROCm f32 (recorded per inference backend).
PRECISION = {"cuda": "bf16-mixed", "rocm": "f32", "cpu": "f32"}


class Refused(Exception):
    """A request that differs from the pinned protocol (050-R9)."""


# ---------------------------------------------------------------- lineage and templates


def load_lineage(path: Path = LINEAGE) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def model_sides(plan: dict):
    for m in plan["matchup"]:
        for s in ("p0", "p1"):
            if m[s]["kind"] == "model":
                yield m[s]


def normalize(plan: dict) -> dict:
    """The plan's conditions: defaults filled in, the name dropped, every model checkpoint (p0 or p1)
    replaced by @CHECKPOINT@ (sampling mode kept), seed_base / same_seed dropped when an explicit seed
    list is used."""
    p = {**PLAN_DEFAULTS, **{k: v for k, v in plan.items() if k not in ("name", "matchup")}}
    if "seeds" in p:
        p.pop("seed_base")
        p.pop("same_seed")
    ms = []
    for m in plan["matchup"]:
        m = {"p1_random_ratio": 0.0, **m}
        sides = {}
        for s in ("p0", "p1"):
            side = dict(m[s])
            if side["kind"] == "model":
                side = {"sample": "argmax", **side, "checkpoint": CKPT}
            elif side["kind"] == "t0":
                side = {"build": "standard", **side}
            sides[s] = side
        ms.append({**m, **sides})
    p["matchup"] = ms
    return p


def conditions_sha256(plan: dict) -> str:
    return hashlib.sha256(json.dumps(normalize(plan), sort_keys=True).encode("utf-8")).hexdigest()


def template(lin: dict, plan_id: str) -> tuple[str, dict]:
    spec = lin["plan"][plan_id]
    text = (ROOT / spec["template"]).read_text(encoding="utf-8")
    return text, tomllib.loads(text)


def check_template(lin: dict, plan_id: str) -> dict:
    """The template still has the pinned conditions (lineage.toml) and placeholders; returns the plan."""
    spec = lin["plan"][plan_id]
    text, plan = template(lin, plan_id)
    sides = list(model_sides(plan))
    if plan.get("name") != NAME or not sides or any(s.get("checkpoint") != CKPT for s in sides):
        raise Refused(f"template {spec['template']}: name / every model checkpoint (p0 and p1) must be the placeholders")
    h = conditions_sha256(plan)
    if h != spec["conditions_sha256"]:
        raise Refused(f"template {spec['template']} conditions {h[:12]} differ from lineage.toml {spec['conditions_sha256'][:12]}")
    n = normalize(plan)
    pinned = {"zero_last_failed": n["zero_last_failed"], "max_frames": n["max_frames"], "games": sum(m["games"] for m in n["matchup"]),
              "sample": sorted({s["sample"] for s in model_sides(n)})}
    want = {"zero_last_failed": spec["zero_last_failed"], "max_frames": spec["max_frames"], "games": spec["games"], "sample": sorted(spec["sample"])}
    if pinned != want:
        raise Refused(f"template {spec['template']}: {pinned} differs from lineage.toml {want}")
    return plan


def render(lin: dict, plan_id: str, checkpoint: str, name: str) -> str:
    check_template(lin, plan_id)
    text, _ = template(lin, plan_id)
    out = text.replace(f'name = "{NAME}"', f'name = "{name}"', 1).replace(f'"{CKPT}"', f'"{checkpoint}"')
    plan = tomllib.loads(out)
    body = "\n".join(l for l in out.splitlines() if not l.lstrip().startswith("#"))
    if plan["name"] != name or any(s["checkpoint"] != checkpoint for s in model_sides(plan)) or CKPT in body or NAME in body:
        raise Refused(f"template {plan_id}: substitution of the name / checkpoint failed")
    if conditions_sha256(plan) != lin["plan"][plan_id]["conditions_sha256"]:
        raise Refused(f"template {plan_id}: rendered plan changed its conditions")
    return out


def identify(lin: dict, plan: dict) -> str | None:
    """Protocol plan id whose template has exactly this plan's conditions, else None."""
    h = conditions_sha256(plan)
    for pid, spec in lin["plan"].items():
        if spec["conditions_sha256"] == h:
            return pid
    return None


def point(lin: dict, pid: str) -> dict:
    for p in lin["point"]:
        if p["id"] == pid:
            return p
    raise Refused(f"no lineage point {pid!r} (config/protocol/lineage.toml: {', '.join(p['id'] for p in lin['point'])})")


def metric_plans(lin: dict, pt: dict, metric: str) -> list[str]:
    """Plan ids of a game metric for this point's family (`tvz` -> tvz_bc / tvz_rl)."""
    m = lin["metric"][metric]
    plans = m.get("plans", {})
    if isinstance(plans, dict):
        plans = plans.get(pt["family"])
        if plans is None:
            raise Refused(f"metric {metric} has no plan for family {pt['family']} (lineage.toml)")
    for p in plans:
        if pt["family"] not in lin["plan"][p]["families"]:
            raise Refused(f"plan {p} is not pinned for family {pt['family']}")
    return list(plans)


def check_metric(lin: dict, pt: dict, metric: str) -> dict:
    if metric not in lin["metric"]:
        raise Refused(f"no metric {metric!r} (lineage.toml)")
    if metric not in pt["metrics"] + pt.get("optional", []):
        raise Refused(f"metric {metric} is not in the menu of point {pt['id']}")
    m = lin["metric"][metric]
    if m["kind"] == "placeholder":
        raise Refused(f"metric {metric} is a placeholder ({m['note']})")
    return m


def check_new_measurement(lin: dict, lineage: str, metric: str, pt: dict) -> None:
    """050-R9: a lineage that is already measured (Checkpoint 1) is not measured again. It takes its new
    metrics (lineage `new_metrics`: the diagnostics) and, per point, the cells of the 050-R9 point table it
    never measured (`ck1_supplement`, "비교가 필요하면 Checkpoint 1 쪽을 같은 plan으로 보충 측정한다");
    cells it has are re-tabulated from its existing outputs (`protocol.py tables`)."""
    if lineage not in lin["lineage"]:
        raise Refused(f"no lineage {lineage!r}")
    why = lin["metric"].get(metric, {}).get("not_measured", {}).get(lineage)
    if why:
        raise Refused(f"{metric} of lineage {lineage}: {why}")
    allowed = lin["lineage"][lineage].get("new_metrics")
    if allowed is None:
        return
    allowed = list(allowed) + pt.get(f"{lineage}_supplement", [])
    if metric not in allowed:
        have = f" (it has {metric}: `protocol.py tables`)" if metric in pt.get(f"{lineage}_outputs", {}) else ""
        raise Refused(f"lineage {lineage} is measured already: no new {metric} measurement at point {pt['id']}{have}; "
                      f"allowed there: {', '.join(allowed)}")


def run_of(checkpoint: str) -> str:
    parts = [s for s in checkpoint.replace("\\", "/").split("/") if s]
    i = next((k for k, s in enumerate(parts) if s.startswith("ckpt-")), len(parts) - 1)
    return parts[i - 1] if i > 0 else parts[-1]


def check_checkpoint(lin: dict, lineage: str, pt: dict, checkpoint: str) -> None:
    """Checkpoint 1 points take exactly their recorded checkpoint (or a copy with the same weights); other
    lineages a run with their prefix."""
    if lineage not in lin["lineage"]:
        raise Refused(f"no lineage {lineage!r}")
    if lineage == "ck1":
        if checkpoint.replace("\\", "/").rstrip("/") != pt["ck1_checkpoint"] and checkpoint_sha256(checkpoint) != checkpoint_sha256(pt["ck1_checkpoint"]):
            raise Refused(f"point {pt['id']} of the Checkpoint 1 lineage is {pt['ck1_checkpoint']}, not {checkpoint}")
        return
    prefix = lin["lineage"][lineage]["run_prefix"]
    if not run_of(checkpoint).startswith(prefix):
        raise Refused(f"lineage {lineage} runs start with {prefix!r}; {checkpoint} is run {run_of(checkpoint)!r}")


# ---------------------------------------------------------------- records


def checkpoint_dir(checkpoint: str) -> Path:
    p = Path(checkpoint)
    if p.is_absolute():
        return p
    ec.load_env()
    shared = os.environ.get("URANIUM_SHARED_DIR")
    if not shared:
        raise Refused("URANIUM_SHARED_DIR is not set (.env, 001-R9)")
    return Path(shared) / p


@functools.lru_cache(maxsize=None)
def checkpoint_sha256(checkpoint: str) -> str:
    """sha256 of the weights file; its first 12 hex digits are the results' `model.checkpoint` id
    (crates/uranium-agent/src/infer.rs checkpoint_id)."""
    f = checkpoint_dir(checkpoint) / WEIGHTS
    if not f.is_file():
        raise Refused(f"no {f}")
    h = hashlib.sha256()
    with f.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*a: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True)


def git_state() -> dict:
    """Commit of this checkout and the uncommitted changes of the protocol files (paths and a sha256 of
    `git diff HEAD` plus the untracked files' contents). Refused outside a git checkout."""
    try:
        head = _git("rev-parse", "HEAD")
    except OSError as e:
        raise Refused(f"git is not available: {e}")
    if head.returncode != 0:
        raise Refused(f"{ROOT} is not a git checkout (git rev-parse HEAD: {head.stderr.decode(errors='replace').strip()})")
    st = _git("status", "--porcelain", "--untracked-files=all", "--", *PROTOCOL_PATHS)
    diff = _git("diff", "HEAD", "--", *PROTOCOL_PATHS)
    if st.returncode != 0 or diff.returncode != 0:
        raise Refused("git status / diff of the protocol files failed")
    lines = [l for l in st.stdout.decode().splitlines() if l.strip()]
    paths = sorted(l[3:] for l in lines)
    h = hashlib.sha256(diff.stdout)
    for l in sorted(lines):
        if l.startswith("??"):
            f = ROOT / l[3:]
            h.update(l[3:].encode() + b"\0" + (f.read_bytes() if f.is_file() else b""))
    return {"commit": head.stdout.decode().strip(), "protocol_files_dirty": bool(paths), "dirty_paths": paths,
            "dirty_diff_sha256": h.hexdigest() if paths else None}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_record(job: Path) -> dict | None:
    f = job / RECORD
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


# ---------------------------------------------------------------- hosts, T1 platform, OSS bot builds


def load_hosts(path: Path) -> list[dict]:
    if not path.is_file():
        raise Refused(f"no host list {path} (copy tools/eval/hosts.example.toml, git-ignored, 001-R10)")
    # utf-8-sig: Windows PowerShell 5.1 writes a BOM.
    hosts = tomllib.loads(path.read_text(encoding="utf-8-sig")).get("host", [])
    if not hosts:
        raise Refused(f"{path}: no [[host]]")
    return hosts


def job_hosts_toml(hosts: list[dict]) -> str:
    """The job's hosts.toml (020: addr, max_games only)."""
    out = []
    for h in hosts:
        out.append(f'[[host]]\naddr = "{h["addr"]}"\n' + (f"max_games = {h['max_games']}\n" if "max_games" in h else ""))
    return "\n".join(out)


def check_t1_platform(pt: dict, hosts: list[dict]) -> list[str]:
    """User decision 2026-11-09 (050-R9): T1 games run on the platform of the point's Checkpoint 1
    counterpart (BC baseline on the Linux ROCm hosts, Checkpoint 1 on the Windows host), because the
    OSS bot binaries differ between platforms. Every host of a T1 job needs `platform` in the host list."""
    allowed = pt.get("t1_platforms")
    if not allowed:
        raise Refused(f"point {pt['id']} has no Checkpoint 1 T1 counterpart platform (lineage.toml t1_platforms)")
    bad = [f"{h['addr']} ({h.get('platform', 'no platform')})" for h in hosts if h.get("platform") not in allowed]
    if bad:
        raise Refused(f"T1 of point {pt['id']} runs on {', '.join(allowed)} hosts only; host list has " + ", ".join(bad))
    return sorted({h["platform"] for h in hosts})


def local_bot_hashes(bots: list[str]) -> dict[str, str]:
    ec.load_env()
    d = Path(os.environ.get("URANIUM_OSS_BOTS_DIR") or ROOT / "build" / "oss_bots")
    out = {}
    for b in bots:
        for name in ("module.dll", "module.so"):
            f = d / b / name
            if f.is_file():
                out[b] = file_sha256(f)
                break
    return out


def remote_bot_hashes(ssh: str, bots_dir: str, bots: list[str]) -> dict[str, str]:
    cmd = "; ".join(f'f="{bots_dir}/{b}/module.so"; [ -f "$f" ] && echo "{b} $(sha256sum < "$f" | cut -c1-64)"' for b in bots)
    p = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", ssh, cmd], capture_output=True, text=True, timeout=120)
    out = {}
    for line in p.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def check_bots(hosts: list[dict], bots: list[str], given: Path | None = None) -> dict:
    """Every game host's OSS bot binaries of this job are accepted NDEBUG builds (t1_bots.toml). Hashes
    come from a file of `<addr> <bot> <sha256>` lines, `local = true` (this PC's URANIUM_OSS_BOTS_DIR) or
    `ssh = "<alias>"` with `oss_bots_dir` (the host's URANIUM_OSS_BOTS_DIR, module.so)."""
    accepted = tomllib.loads(BOTS.read_text(encoding="utf-8"))["bot"]
    seen: dict[str, dict[str, str]] = {}
    problems = []
    if given is not None:
        for line in given.read_text(encoding="utf-8-sig").splitlines():
            parts = line.split()
            if len(parts) == 3 and not line.startswith("#"):
                seen.setdefault(parts[0], {})[parts[1]] = parts[2]
    for h in hosts:
        if h["addr"] in seen:
            continue
        if h.get("local"):
            seen[h["addr"]] = local_bot_hashes(bots)
        elif h.get("ssh"):
            if not h.get("oss_bots_dir"):
                problems.append(f"{h['addr']}: ssh host without oss_bots_dir (its URANIUM_OSS_BOTS_DIR) in the host list")
                continue
            seen[h["addr"]] = remote_bot_hashes(h["ssh"], h["oss_bots_dir"], bots)
    for h in hosts:
        got = seen.get(h["addr"])
        if got is None:
            if not any(p.startswith(h["addr"] + ":") for p in problems):
                problems.append(f"{h['addr']}: no way to read its bot hashes (local / ssh + oss_bots_dir in the host list, or --bot-hashes)")
            continue
        for b in bots:
            if b not in got:
                problems.append(f"{h['addr']}: {b} missing")
            elif got[b] not in accepted.get(b, {}).get("sha256", []):
                problems.append(f"{h['addr']}: {b} {got[b][:12]} is not an accepted NDEBUG build (config/protocol/t1_bots.toml)")
    if problems:
        raise Refused("bot builds:\n  " + "\n  ".join(problems))
    return {h["addr"]: seen[h["addr"]] for h in hosts}


def oss_bots(plans: list[dict]) -> list[str]:
    return sorted({m["p1"]["name"] for p in plans for m in p["matchup"] if m["p1"]["kind"] == "oss"})


# ---------------------------------------------------------------- diagnostics (cue_diag, dead_channels)
# 050-R9 "추가 지표" (user decisions 2026-11-10, AMB-050-4 / -5 / -6): cue_diag = `python -m uranium diag` on
# the pinned hold-outs (lineage.toml [metric.cue_diag.<set>]); dead_channels = the run's 041-R25 training-log
# records (tools/eval/spatial_logs.py). `diag` prints (or with --run, runs) the commands; `check-diag`
# refuses outputs made with other settings, data or checkpoint, or without the finished record.

DIAG_KINDS = ("diag_cue", "diag_train_log")
DIAG_RECORD = "protocol.json"
RUNNING = "running.json"  # the attempt marker of `diag --run` (becomes protocol.json when it finishes)
COPY = "copy"  # the step that copies a local dump to <out>/<set>/dump (.part, then rename)
HOLDOUT_DIR = "diag_holdout"  # under ec.cache_dir(): the hold-out link trees
LOCAL_DUMP_DIR = "diag_dump"  # under ec.cache_dir(): the dumps before their copy to the shared drive
TRAIN_LOG = "spatial_log.json"


def diag_metric(lin: dict, metric: str) -> dict:
    m = lin["metric"].get(metric)
    if m is None or m["kind"] not in DIAG_KINDS:
        raise Refused(f"{metric} is not a diagnostic metric ({', '.join(k for k, v in lin['metric'].items() if v['kind'] in DIAG_KINDS)})")
    return m


def cue_set(lin: dict, name: str) -> dict:
    return lin["metric"]["cue_diag"][name]


def files_fingerprint(src: Path, parts: list[str]) -> tuple[int, str]:
    """(file count, sha256 of the sorted "<part>/<file> <bytes>" lines) of the shard files under src/<part>/
    (a source on the shared drive or a link tree of it)."""
    lines = sorted(f"{p}/{f.name} {f.stat().st_size}" for p in parts for f in (src / p).glob("shard_*.bin.zst"))
    return len(lines), hashlib.sha256("".join(l + "\n" for l in lines).encode()).hexdigest()


def manifest_fingerprint(shared: Path, datasets: list[str]) -> tuple[int, str]:
    """(file count, sha256 of the sorted "<source>/<file> <sha256> <bytes>" lines) of datasets' manifest.json
    (061-R7; <shared>/datasets/<dataset>/manifest.json, source = the dataset's folder name)."""
    lines = []
    for ds in datasets:
        m = json.loads((shared / "datasets" / ds / "manifest.json").read_text(encoding="utf-8"))
        lines += [f"{Path(ds).name}/{k} {v['sha256']} {v['bytes']}" for k, v in m["files"].items()]
    lines.sort()
    return len(lines), hashlib.sha256("".join(l + "\n" for l in lines).encode()).hexdigest()


def _check_files(name: str, got: tuple[int, str], want: tuple[int, str], where) -> None:
    if got != want:
        raise Refused(f"hold-out {name} ({where}): {got[0]} files, fingerprint {got[1][:12]}; pinned {want[0]} files {want[1][:12]}")


def set_data(lin: dict, name: str, shared: Path | None = None, work: Path | None = None, create: bool = True) -> list[str]:
    """The data arguments of a cue hold-out set on this host, after its fingerprint check:
    - `holdout` (link trees): `--shards <work>/<name>/<source name>` per entry (per_action.link_tree of the
      pinned parts; work = <cache>/diag_holdout); the source and, once linked, the link tree are checked.
      create=False: only the paths (and the source check), nothing written.
    - `datasets`: `--dataset <name>` per dataset, the manifests checked."""
    import per_action

    s = cue_set(lin, name)
    shared = shared or ec.shared_dir()
    if "datasets" in s:
        _check_files(name, manifest_fingerprint(shared, s["datasets"]), (s["files"], s["files_sha256"]), "dataset manifests")
        return [x for ds in s["datasets"] for x in ("--dataset", ds)]
    work = work or ec.cache_dir() / HOLDOUT_DIR
    out = []
    for h in s["holdout"]:
        src = shared / h["source"]
        _check_files(h["name"], files_fingerprint(src, h["parts"]), (h["files"], h["files_sha256"]), src)
        dst = work / h["name"] / Path(h["source"]).name
        if create:
            per_action.link_tree(src, h["parts"], dst)
            _check_files(h["name"], files_fingerprint(dst, h["parts"]), (h["files"], h["files_sha256"]), f"link tree {dst}")
        out += ["--shards", str(dst)]
    return out


def data_tails(s: dict) -> list[str]:
    """The ends of the data folders a dump of the set records (`diag dump` manifest `data`, in its order)."""
    if "datasets" in s:
        return sorted(s["datasets"])  # storage::dataset_dirs sorts the dataset names
    return [f"{h['name']}/{Path(h['source']).name}" for h in s["holdout"]]


def output_name(checkpoint: str) -> str:
    """`<run>__<ckpt>` of a checkpoint path (an RL agent folder adds its tail: ckpt-N-agents-main)."""
    parts = [s for s in checkpoint.replace("\\", "/").split("/") if s]
    i = next((k for k, s in enumerate(parts) if s.startswith("ckpt-")), len(parts) - 1)
    return f"{run_of(checkpoint)}__{'-'.join(parts[i:])}"


def diag_out(lin: dict, metric: str, checkpoint: str) -> str:
    """The output folder of a checkpoint's diagnostics, relative to URANIUM_SHARED_DIR."""
    run, ck = output_name(checkpoint).split("__", 1)
    return diag_metric(lin, metric)["output"].replace("<run>", run).replace("<ckpt>", ck)


def cue_commands(lin: dict, name: str, ckpt: str, out: Path, data: list[str], py: str = "python", dump: Path | None = None) -> list[list[str]]:
    """The pinned `python -m uranium diag` commands of a cue hold-out set for the checkpoint folder `ckpt` (a
    path on the running host) writing into `out` (<out>/<set>); the dump goes to `dump` (default <out>/dump;
    `diag --run` dumps into this host's cache and copies it to <out>/dump afterwards)."""
    m = diag_metric(lin, "cue_diag")
    s = cue_set(lin, name)
    d, p = s["dump"], s["probes"]
    dump = dump or out / "dump"
    cmd = [py, "-m", "uranium", "diag", "dump", "--checkpoint", ckpt, *data, "--splits", ",".join(s["splits"])]
    if s.get("games"):
        cmd += ["--games", str(s["games"]), "--seed", str(s["seed"])]
    cmd += ["--config", m["config"], "--backend", m["backend"], "--lanes-per-group", str(d["lanes_per_group"]), "--row-until", f"{d['row_until']:g}",
            "--act-until", f"{d['act_until']:g}", "--act-every", str(d["act_every"]), "--core-points", d["core_points"]]
    if d["cf"]:
        cmd += ["--cf", "--cf-starts", ";".join(f"{r}:{a:g},{b:g}" for r, (a, b) in d["cf_starts"].items())]
    cmd += ["--out", str(dump)]
    cmds = [cmd, [py, "-m", "uranium", "diag", "probes", str(dump), "--races", ",".join(p["races"]), "--stages", ",".join(p["stages"]), "--gain", p["gain"],
                  "--boot", str(p["boot"]), "--cap", str(p["cap"]), "--seed", str(p["seed"]), "--device", "cpu", "--out", str(out / "probes.json")]]
    if "intervene" in s:
        iv = s["intervene"]
        cmds.append([py, "-m", "uranium", "diag", "intervene", str(dump), "--races", ",".join(iv["races"]), "--window", ",".join(f"{x:g}" for x in iv["window"]),
                     "--boot", str(iv["boot"]), "--seed", str(iv["seed"]), "--out", str(out / "intervene.json")])
    return cmds


def summary_command(lin: dict, pt: dict, name: str, out: Path, py: str = "python") -> list[str]:
    """`diag summary` of one cue set (<out>/<set>/summary.json)."""
    cmd = [py, "-m", "uranium", "diag", "summary", "--point", pt["id"], "--probes", str(out / "probes.json")]
    if "intervene" in cue_set(lin, name):
        cmd += ["--intervene", str(out / "intervene.json")]
    return cmd + ["--out", str(out / "summary.json")]


def train_log_command(lin: dict, pt: dict, checkpoint: str, log: str, ckpt: str, out: Path, py: str = "python") -> list[str]:
    """tools/eval/spatial_logs.py on the run's training log: records from the pinned update to the checkpoint's
    (BC family: ckpt-N = update N; RL: the whole log of agent `main`)."""
    m = diag_metric(lin, "dead_channels")
    cmd = [py, "tools/eval/spatial_logs.py", log, "--checkpoint", ckpt, "--from-update", str(m["from_update"])]
    n = re.match(r"ckpt-(\d+)$", Path(checkpoint).name)
    if pt["family"] == "bc" and n:
        cmd += ["--until-update", n.group(1)]
    return cmd + ["--out", str(out / TRAIN_LOG)]


def step_name(c: list[str]) -> str:
    return COPY if c[0] == COPY else c[4] if c[2:4] == ["uranium", "diag"] else Path(c[1]).stem


def _ends_with(path: str, tail: str) -> bool:
    return path.replace("\\", "/").rstrip("/").endswith(tail)


def _load(f: Path, problems: list[str]) -> dict | None:
    if not f.is_file():
        problems.append(f"no {f}")
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def check_cue_set(lin: dict, name: str, d: Path, sha: str | None) -> list[str]:
    """Problems of one cue set's outputs (<out>/<set>) against its pinned settings."""
    m = diag_metric(lin, "cue_diag")
    s = cue_set(lin, name)
    dm, p, rep = s["dump"], s["probes"], s["report"]
    tails = data_tails(s)
    names = [h["name"] for h in s.get("holdout", [])] or tails
    problems: list[str] = []

    def same(what: str, got, want) -> None:
        if got != want:
            problems.append(f"{name} {what}: {got!r}, pinned {want!r}")

    def data(what: str, got) -> None:
        # the folders in the pinned order: this tool's link trees (<name>/<source>) or a folder named <name>
        # holding the same parts (the lambda runs); the row / game counts fingerprint the content
        got = list(got or [])
        if len(got) != len(tails) or not all(_ends_with(g, t) or _ends_with(g, n) for g, t, n in zip(got, tails, names)):
            problems.append(f"{name} {what} data {got}: not the pinned hold-out (.../{', .../'.join(tails)})")

    man = _load(d / "dump" / "manifest.json", problems)
    if man is not None:
        data("dump", man.get("data"))
        for k, want in (("splits", s["splits"]), ("games_per_split", s.get("games")), ("config", m["config"]), ("backend", m["backend"]),
                        ("precision", PRECISION[m["backend"]]), ("lanes_per_group", dm["lanes_per_group"]), ("row_until", dm["row_until"]),
                        ("act_until", dm["act_until"]), ("act_every", dm["act_every"])):
            same(f"dump {k}", man.get(k), want)
        if s.get("games"):
            same("dump seed", man.get("seed"), s["seed"])
        st = man.get("stats", {})
        for k, got in (("expect_games", man.get("games")), ("expect_rows", st.get("rows")), ("expect_act_rows", st.get("act_rows"))):
            if k in dm:
                same(f"dump {k.removeprefix('expect_')}", got, dm[k])
        if dm["cf"]:
            same("dump cf_starts", man.get("cf_starts"), {r: list(v) for r, v in dm["cf_starts"].items()})
            if len(man.get("passes") or []) < 2:
                problems.append(f"{name} dump passes {man.get('passes')}: made without --cf (no counterfactual passes)")
        elif len(man.get("passes") or ["real"]) != 1:
            problems.append(f"{name} dump passes {man.get('passes')}: counterfactual passes not pinned for this set")
        missing = [x for x in rep["stages"] if x not in (man.get("stages") or [])]
        if missing:
            problems.append(f"{name} dump has no activations at {missing}")
        if sha is not None:
            same("dump checkpoint_sha256", man.get("checkpoint_sha256"), sha)
        sha = sha or man.get("checkpoint_sha256")
    pr = _load(d / "probes.json", problems)
    if pr is not None:
        hd = pr.get("dump", {})
        same("probes dump checkpoint_sha256", hd.get("checkpoint_sha256"), sha)
        data("probes", hd.get("data"))
        for k, want in (("gain_method", p["gain"]), ("boot", p["boot"]), ("cap", p["cap"]), ("mlp", True)):
            same(f"probes {k}", pr.get(k), want)
        same("probes races", sorted(pr.get("races", {})), sorted(p["races"]))
        for race in p["races"]:
            tasks = pr.get("races", {}).get(race, {}).get("tasks", {})
            for t, (rows, games) in p["expect_tasks"].items():
                x = tasks.get(t, {})
                same(f"probes {race} {t} rows / games", [x.get("rows"), x.get("games")], [rows, games])
            for t in set(p["expect_tasks"]) | set(rep.get("required", [])):
                x = tasks.get(t, {})
                if "metrics" not in x:
                    problems.append(f"{name} probes {race} {t}: no values ({x.get('skipped', 'not run')})")
                    continue
                lack = [st for st in rep["stages"] if f"{st}|ridge" not in x["metrics"]]
                if lack:
                    problems.append(f"{name} probes {race} {t}: no probe at {lack} (stages other than pinned)")
            for t, why in p["expect_skipped"].items():
                if why not in str(tasks.get(t, {}).get("skipped")):
                    problems.append(f"{name} probes {race} {t}: expected skipped ({why}), got {tasks.get(t, {}).get('skipped')!r}")
    if "intervene" in s:
        iv = s["intervene"]
        ivr = _load(d / "intervene.json", problems)
        if ivr is not None:
            hd = ivr.get("dump", {})
            same("intervene dump checkpoint_sha256", hd.get("checkpoint_sha256"), sha)
            data("intervene", hd.get("data"))
            same("intervene window", ivr.get("window"), [float(x) for x in iv["window"]])
            same("intervene races", sorted(ivr.get("races", {})), sorted(iv["races"]))
    return problems + data_consistency(lin, name, d)


def data_fingerprint(lin: dict, name: str, d: Path) -> dict | None:
    """What a cue set's outputs say about its data only (the same for every checkpoint): the dump's games /
    rows / activation rows and the reported probe tasks' rows / games per race."""
    f = d / "dump" / "manifest.json"
    if not f.is_file():
        return None
    man = json.loads(f.read_text(encoding="utf-8"))
    fp = {"games": man.get("games"), "rows": man.get("stats", {}).get("rows"), "act_rows": man.get("stats", {}).get("act_rows")}
    pf = d / "probes.json"
    if pf.is_file():
        pr = json.loads(pf.read_text(encoding="utf-8"))
        for race, rr in sorted(pr.get("races", {}).items()):
            for t in cue_set(lin, name)["report"]["tasks"]:
                x = rr.get("tasks", {}).get(t, {})
                fp[f"{race} {t}"] = [x.get("rows"), x.get("games")]
    return fp


def data_consistency(lin: dict, name: str, d: Path) -> list[str]:
    """The set's data fingerprint against every other finished output folder beside this one
    (<shared>/artifacts/diag/*/<set>): the same hold-out gives the same numbers for every checkpoint, so a
    difference means other data (guards the sets whose fingerprint is not pinned yet)."""
    mine = data_fingerprint(lin, name, d)
    out = d.parent
    if mine is None or not out.parent.is_dir():
        return []
    for sib in sorted(out.parent.iterdir()):
        if sib == out or ".failed-" in sib.name or not (sib / DIAG_RECORD).is_file():
            continue
        try:
            if not json.loads((sib / DIAG_RECORD).read_text(encoding="utf-8")).get("finished"):
                continue
        except (OSError, json.JSONDecodeError):
            continue
        other = data_fingerprint(lin, name, sib / name)
        if other is None:
            continue
        diff = {k: (v, other.get(k)) for k, v in mine.items() if k in other and other[k] != v}
        if diff:
            return [f"{name} data differs from the finished {sib.name}: " + ", ".join(f"{k} {a!r} vs {b!r}" for k, (a, b) in diff.items())]
    return []


def unpinned(lin: dict, out: Path) -> list[str]:
    """The data fingerprints a set does not pin yet (TBD, measured on the first run): the values found."""
    notes = []
    for name in diag_metric(lin, "cue_diag")["sets"]:
        dm = cue_set(lin, name)["dump"]
        f = out / name / "dump" / "manifest.json"
        if f.is_file():
            man = json.loads(f.read_text(encoding="utf-8"))
            got = {"expect_games": man.get("games"), "expect_rows": man.get("stats", {}).get("rows"), "expect_act_rows": man.get("stats", {}).get("act_rows")}
            notes += [f"{name} {k} = {v} (not pinned yet)" for k, v in got.items() if k not in dm]
    return notes


def check_train_log(lin: dict, out: Path) -> list[str]:
    m = diag_metric(lin, "dead_channels")
    problems: list[str] = []
    r = _load(out / TRAIN_LOG, problems)
    if r is not None:
        if r.get("from_update") != m["from_update"]:
            problems.append(f"{TRAIN_LOG} from_update {r.get('from_update')!r}, pinned {m['from_update']}")
        if not r.get("sp_records_from"):
            problems.append(f"{TRAIN_LOG}: no sp_* record from update {m['from_update']} (a run without [monitor] spatial_every?)")
        lack = [k for k in m["layers"] if k not in (r.get("dead") or {})]
        if r.get("sp_records_from") and lack:
            problems.append(f"{TRAIN_LOG}: no dead count for {lack}")
    return problems


def check_diag(lin: dict, metric: str, out: Path, sha: str | None = None) -> list[str]:
    """Problems of a diagnostic output folder against the pinned settings (empty = conforms). `sha` = the
    checkpoint's weights sha256 the outputs must be of. The folder needs the finished protocol.json of
    `diag --run` with the checkpoint sha256, the metric and the git commit (050-R9 record)."""
    m = diag_metric(lin, metric)
    problems: list[str] = []
    rec = _load(out / DIAG_RECORD, problems)
    if rec is not None:
        if not rec.get("finished"):
            problems.append(f"{DIAG_RECORD}: not finished (an attempt in progress or failed)")
        if metric not in (rec.get("metrics") or []):
            problems.append(f"{DIAG_RECORD}: {metric} not among the measured metrics {rec.get('metrics')}")
        if not (rec.get("git") or {}).get("commit"):
            problems.append(f"{DIAG_RECORD}: no git commit")
        if not rec.get("checkpoint_sha256"):
            problems.append(f"{DIAG_RECORD}: no checkpoint sha256")
        elif sha is not None and rec["checkpoint_sha256"] != sha:
            problems.append(f"{DIAG_RECORD} checkpoint_sha256: {rec['checkpoint_sha256']!r}, pinned {sha!r}")
        sha = sha or rec.get("checkpoint_sha256")
    if m["kind"] == "diag_train_log":
        return problems + check_train_log(lin, out)
    for name in m["sets"]:
        problems += check_cue_set(lin, name, out / name, sha)
    return problems


def next_failed(p: Path) -> Path:
    """<p>.failed-<k> with the first free k (a failed attempt is renamed, never deleted)."""
    k = 1
    while p.with_name(f"{p.name}.failed-{k}").exists():
        k += 1
    return p.with_name(f"{p.name}.failed-{k}")


def copy_dump(src: Path, dst: Path) -> None:
    """A local dump to the shared drive: copy into <dst>.part, then rename (as the lambda-run scripts)."""
    import shutil

    part = dst.with_name(dst.name + ".part")
    if part.exists():
        part.rename(next_failed(part))
    shutil.copytree(src, part)
    os.replace(part, dst)


def diag_record_cmd(c: list[str], roots: dict[str, Path]) -> list[str]:
    """A command as protocol.json keeps it: `python` for the interpreter, host paths as <shared> / <cache>."""
    return redact_cmd(c if c[0] == COPY else ["python", *c[1:]], roots)


def check_final_only(pt: dict, checkpoint: str) -> None:
    """A point with `diag_final_only` (s1e2, AMB-050-6): the cue diagnostics at the run's final checkpoint only
    (the run folder has DONE and no later ckpt-N)."""
    if not pt.get("diag_final_only"):
        return
    d = checkpoint_dir(checkpoint)
    run = d.parent
    steps = sorted(int(x.name[5:]) for x in run.glob("ckpt-*") if x.name[5:].isdigit())
    if not (run / "DONE").is_file() or not steps or d.name != f"ckpt-{steps[-1]}":
        raise Refused(f"point {pt['id']}: cue diagnostics at the run's final checkpoint only (AMB-050-6); {checkpoint} is not (DONE {'present' if (run / 'DONE').is_file() else 'missing'}, last ckpt-{steps[-1] if steps else '?'})")


def cmd_diag(a) -> None:
    """050-R9 diagnostics of a lineage point's checkpoint. Without --run: the pinned commands for this host
    (paths of this host; nothing is written). With --run: builds the hold-out link trees, marks the attempt
    (<out>/running.json), runs the commands with the dumps in this host's cache, copies each dump to
    <out>/<set>/dump, writes <out>/protocol.json and runs `check-diag`. A folder without a finished
    protocol.json (an earlier attempt that failed) is renamed to <out>.failed-<k> first. Run it on a Linux
    ROCm host (the pinned backend)."""
    lin = load_lineage()
    pt = point(lin, a.point)
    metrics = [x.strip() for x in a.metrics.split(",") if x.strip()]
    for metric in metrics:
        check_new_measurement(lin, a.lineage, metric, pt)
        if check_metric(lin, pt, metric)["kind"] not in DIAG_KINDS:
            raise Refused(f"metric {metric} is not a diagnostic")
    check_checkpoint(lin, a.lineage, pt, a.checkpoint)
    if "cue_diag" in metrics:  # AMB-050-6 restricts the cue diagnostics (단서 진단) only
        check_final_only(pt, a.checkpoint)
    if "dead_channels" in metrics and not a.train_log:
        raise Refused(f"dead_channels reads the run's training log: --train-log <BC stderr log | RL train.csv> of {run_of(a.checkpoint)}")
    outs = {diag_out(lin, metric, a.checkpoint) for metric in metrics}
    if len(outs) != 1:
        raise Refused(f"the metrics write to different folders {sorted(outs)}")
    shared = ec.shared_dir()
    out = Path(a.out) if a.out else shared / outs.pop()
    sha = checkpoint_sha256(a.checkpoint)
    ck = str(checkpoint_dir(a.checkpoint))
    py = sys.executable if a.run else "python"
    cache = ec.cache_dir()
    local = cache / LOCAL_DUMP_DIR / out.name
    cmds = []
    if "cue_diag" in metrics:
        for name in diag_metric(lin, "cue_diag")["sets"]:
            cmds += cue_commands(lin, name, ck, out / name, set_data(lin, name, shared, create=a.run), py, dump=local / name / "dump")
            cmds.append([COPY, str(local / name / "dump"), str(out / name / "dump")])
            cmds.append(summary_command(lin, pt, name, out / name, py))
    log_sha = None
    if "dead_channels" in metrics:
        log = Path(a.train_log)
        if not log.is_file():
            raise Refused(f"no training log {log}")
        log_sha = file_sha256(log)
        cmds.append(train_log_command(lin, pt, a.checkpoint, str(log), ck, out, py))
    if not a.run:
        for c in cmds:
            if c[0] == COPY:
                part = c[2] + ".part"
                print(f"cp -r {shlex.quote(c[1])} {shlex.quote(part)} && mv {shlex.quote(part)} {shlex.quote(c[2])}")
            else:
                print(show(c))
        print(f"python tools/eval/protocol.py check-diag {out} --metrics {','.join(metrics)} --checkpoint {a.checkpoint}")
        print(f"# pinned: backend {diag_metric(lin, 'cue_diag')['backend']}; checkpoint sha256 {sha}; git {git_state()['commit']}")
        print(f"# --run also makes the hold-out link trees under {cache / HOLDOUT_DIR}, writes {out / RUNNING} at the start and {out / DIAG_RECORD} at the end")
        return
    git = git_state()
    if git["protocol_files_dirty"] and not a.allow_dirty:
        raise Refused("uncommitted protocol files (" + ", ".join(git["dirty_paths"]) + "); --allow-dirty records them in protocol.json")
    if (out / DIAG_RECORD).exists():
        raise Refused(f"{out / DIAG_RECORD} exists (one finished measurement per folder; --out for another)")
    if (out / RUNNING).is_file() and not a.takeover:
        try:
            prev = json.loads((out / RUNNING).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prev = {}
        if not any("error" in s for s in prev.get("steps", [])):
            raise Refused(f"{out / RUNNING}: an attempt in progress (host {prev.get('host', '?')}, started {prev.get('started', '?')}, "
                          f"no failed step); --takeover when its host is gone")
    for p in (out, local):
        if p.exists():  # an earlier attempt without a finished record: kept under another name
            moved = next_failed(p)
            p.rename(moved)
            print(f"[diag] earlier attempt kept as {moved}", flush=True)
    out.mkdir(parents=True)
    if "cue_diag" in metrics:
        for name in diag_metric(lin, "cue_diag")["sets"]:
            (out / name).mkdir()
    roots = {"<shared>": shared, "<cache>": cache}
    rec = {"protocol": "config/protocol/lineage.toml (050-R9)", "lineage": a.lineage, "point": pt["id"], "family": pt["family"],
           "metrics": metrics, "checkpoint": a.checkpoint, "checkpoint_sha256": sha, "checkpoint_id": f"{Path(a.checkpoint).name}@{sha[:12]}",
           "backend": diag_metric(lin, "cue_diag")["backend"] if "cue_diag" in metrics else None,
           "precision": PRECISION[diag_metric(lin, "cue_diag")["backend"]] if "cue_diag" in metrics else None,
           "sets": {n: {k: v for k, v in cue_set(lin, n).items() if k in ("holdout", "datasets", "files", "files_sha256", "splits", "games", "seed")}
                    for n in diag_metric(lin, "cue_diag")["sets"]} if "cue_diag" in metrics else None,
           "train_log": {"file": Path(a.train_log).name, "sha256": log_sha} if log_sha else None,
           "git": git, "host": socket.gethostname(), "platform": sys.platform, "commands": [diag_record_cmd(c, roots) for c in cmds],
           "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "finished": None, "steps": []}

    def save(final: bool = False) -> None:
        (out / RUNNING).write_text(json.dumps(rec, indent=1), encoding="utf-8")
        if final:
            os.replace(out / RUNNING, out / DIAG_RECORD)

    save()
    for c in cmds:
        print("[diag]", show(diag_record_cmd(c, roots)), flush=True)
        t0 = datetime.datetime.now()
        try:
            if c[0] == COPY:
                copy_dump(Path(c[1]), Path(c[2]))
            else:
                subprocess.run(c, cwd=ROOT, check=True)
        except (subprocess.CalledProcessError, OSError) as e:
            rec["steps"].append({"step": step_name(c), "error": str(e)})
            save()
            raise
        rec["steps"].append({"step": step_name(c), "secs": round((datetime.datetime.now() - t0).total_seconds())})
        save()
    rec["finished"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    save(final=True)
    problems = [p for metric in metrics for p in check_diag(lin, metric, out, sha)]
    rec["check"] = problems
    rec["unpinned"] = unpinned(lin, out) if "cue_diag" in metrics else []
    (out / DIAG_RECORD).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    for n in rec["unpinned"]:
        print("NOTE", n)
    for p in problems:
        print("MISMATCH", p)
    if problems:
        raise SystemExit(1)
    print(f"{out}: {', '.join(metrics)} conform to the protocol")


def cmd_check_diag(a) -> None:
    lin = load_lineage()
    sha = checkpoint_sha256(a.checkpoint) if a.checkpoint else None
    metrics = [x.strip() for x in a.metrics.split(",") if x.strip()]
    problems = [p for metric in metrics for p in check_diag(lin, metric, Path(a.out), sha)]
    if "cue_diag" in metrics:
        for n in unpinned(lin, Path(a.out)):
            print("NOTE", n)
    for p in problems:
        print("MISMATCH", p)
    if problems:
        raise SystemExit(1)
    print(f"{a.out}: {', '.join(metrics)} conform to the protocol")


# ---------------------------------------------------------------- commands


def cmd_job(a) -> None:
    lin = load_lineage()
    pt = point(lin, a.point)
    metrics = [m.strip() for m in a.metrics.split(",") if m.strip()]
    stages: list[tuple[str, str]] = []  # (metric, plan id)
    for m in metrics:
        check_new_measurement(lin, a.lineage, m, pt)
        spec = check_metric(lin, pt, m)
        if spec["kind"] != "games":
            raise Refused(f"metric {m} is not a game metric (use the bc-eval command)")
        stages += [(m, p) for p in metric_plans(lin, pt, m)]
    check_checkpoint(lin, a.lineage, pt, a.checkpoint)
    git = git_state()
    if git["protocol_files_dirty"] and not a.allow_dirty:
        raise Refused("uncommitted protocol files (" + ", ".join(git["dirty_paths"]) + "); --allow-dirty records them in protocol.json")
    sha = checkpoint_sha256(a.checkpoint)
    label = a.label or run_of(a.checkpoint)
    # A job's stages that share the model must agree on zero_last_failed (crates/uranium-runner eval.rs):
    # one job per value, as the Checkpoint 1 jobs did (rush + macro in one job, TvZ apart).
    groups: dict[bool, list[tuple[str, str, str]]] = {}
    for m, p in stages:
        text = render(lin, p, a.checkpoint.replace("\\", "/"), f"{label}-{p.replace('_', '-')}")
        groups.setdefault(lin["plan"][p]["zero_last_failed"], []).append((m, p, text))
    hosts = load_hosts(Path(a.hosts))
    has_t1 = any(lin["plan"][p]["kind"] == "t1" for _, p in stages)
    platforms = check_t1_platform(pt, hosts) if has_t1 else None
    plans = [tomllib.loads(t) for g in groups.values() for _, _, t in g]
    bots = oss_bots(plans)
    bot_hashes = check_bots(hosts, bots, Path(a.bot_hashes) if a.bot_hashes else None) if bots else {}
    out_root = Path(a.out_root) if a.out_root else ec.shared_dir() / "eval"
    many = len(groups) > 1
    for zlf, items in groups.items():
        name = a.job + ("-" + "-".join(dict.fromkeys(m for m, _, _ in items)) if many else "")
        d = out_root / name
        if a.dry_run:
            print(f"[dry run] {d}: " + ", ".join(p for _, p, _ in items))
            continue
        (d / "plans").mkdir(parents=True, exist_ok=False)
        gbots = oss_bots([tomllib.loads(t) for _, _, t in items])
        g_t1 = any(lin["plan"][p]["kind"] == "t1" for _, p, _ in items)
        rec_plans = []
        stage_files = []
        for i, (m, p, text) in enumerate(items, start=1):
            f = d / "plans" / f"{i:02d}-{p}.toml"
            f.write_bytes(text.encode("utf-8"))
            stage_files.append(f"plans/{f.name}")
            rec_plans.append({"metric": m, "plan": p, "file": f"plans/{f.name}", "content_sha256": file_sha256(f),
                              "conditions_sha256": lin["plan"][p]["conditions_sha256"], "template": lin["plan"][p]["template"]})
        (d / "job.toml").write_text("stages = [" + ", ".join(f'"{s}"' for s in stage_files) + ']\nhosts = "hosts.toml"\n', encoding="utf-8")
        (d / "hosts.toml").write_text(job_hosts_toml(hosts), encoding="utf-8")
        rec = {
            "protocol": "config/protocol/lineage.toml (050-R9)", "lineage": a.lineage, "point": pt["id"], "family": pt["family"],
            "checkpoint": a.checkpoint, "checkpoint_sha256": sha, "checkpoint_id": f"{Path(a.checkpoint).name}@{sha[:12]}",
            "zero_last_failed": zlf, "git": git, "plans": rec_plans,
            "hosts": [{"addr": h["addr"], "platform": h.get("platform")} for h in hosts],
            "t1_platforms": platforms if g_t1 else None,
            "bot_hashes": {h: {b: v for b, v in hs.items() if b in gbots} for h, hs in bot_hashes.items()} if gbots else {},
            "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        (d / RECORD).write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print("wrote", d)


def cmd_render(a) -> None:
    lin = load_lineage()
    text = render(lin, a.plan, a.checkpoint.replace("\\", "/"), a.name)
    if a.out:
        Path(a.out).write_bytes(text.encode("utf-8"))
        print("wrote", a.out)
    else:
        sys.stdout.write(text)


def verify_job(lin: dict, job: Path, lineage: str, pt: dict) -> list[str]:
    """Problems of an eval job against the protocol (050-AC5; empty = conforms). Jobs of a lineage measured
    after the protocol must carry protocol.json; Checkpoint 1 jobs predate it."""
    problems = []
    rec = read_record(job)
    if rec is None:
        if lineage != "ck1":
            problems.append(f"no {RECORD} (jobs of lineage {lineage} are made by protocol.py job)")
    else:
        if (rec.get("lineage"), rec.get("point")) != (lineage, pt["id"]):
            problems.append(f"{RECORD} is for {rec.get('lineage')} / {rec.get('point')}")
        for p in rec.get("plans", []):
            f = job / p["file"]
            if not f.is_file() or file_sha256(f) != p["content_sha256"]:
                problems.append(f"{p['file']} differs from its recorded content_sha256")
        git = rec.get("git", {})
        if git.get("protocol_files_dirty"):
            problems.append(f"made with uncommitted protocol files {git.get('dirty_paths')} (diff {str(git.get('dirty_diff_sha256'))[:12]})")
    allowed = {}
    for m in pt["metrics"] + pt.get("optional", []):
        if lin["metric"][m]["kind"] == "games":
            for p in metric_plans(lin, pt, m):
                allowed[p] = m
    rows = ec.read_results(job)
    for k, f, plan in ec.job_stages(job):
        pid = identify(lin, plan)
        if pid is None:
            problems.append(f"stage {k} {f.name}: conditions match no protocol plan")
            continue
        if pid not in allowed:
            problems.append(f"stage {k} {f.name}: plan {pid} is not in the menu of point {pt['id']} ({pt['family']})")
        if lin["plan"][pid]["kind"] == "t1" and rec is not None:
            got = rec.get("t1_platforms") or []
            if not got or set(got) - set(pt.get("t1_platforms") or []):
                problems.append(f"stage {k}: T1 ran on {got}, point {pt['id']} pins {pt.get('t1_platforms')}")
        cks = {s["checkpoint"] for s in model_sides(plan)}
        for ck in cks:
            try:
                check_checkpoint(lin, lineage, pt, ck)
            except Refused as e:
                problems.append(f"stage {k}: {e}")
        mr = ec.model_rows(rows, k)
        zlf = {r["model"].get("zero_last_failed") for r in mr.values()}
        if zlf - {lin["plan"][pid]["zero_last_failed"]}:
            problems.append(f"stage {k}: rows ran with zero_last_failed {sorted(zlf)}, pinned {lin['plan'][pid]['zero_last_failed']}")
        ids = {r["model"].get("checkpoint") for r in mr.values()}
        for ck in cks:
            try:
                want = f"{Path(ck).name}@{checkpoint_sha256(ck)[:12]}"
            except Refused:
                want = None
            if want and ids - {want}:
                problems.append(f"stage {k}: rows ran {sorted(ids)}, checkpoint is {want}")
    return problems


def cmd_verify(a) -> None:
    lin = load_lineage()
    pt = point(lin, a.point)
    problems = verify_job(lin, Path(a.job), a.lineage, pt)
    for p in problems:
        print("MISMATCH", p)
    if problems:
        raise SystemExit(1)
    print(f"{a.job}: conforms to point {pt['id']} ({a.lineage})")


def cmd_tables(a) -> None:
    """050-R9: the point's existing Checkpoint 1 eval jobs (ck1_outputs) re-tabulated with both held and
    both bunker definitions (no new games)."""
    import tables

    lin = load_lineage()
    pt = point(lin, a.point)
    if a.lineage != "ck1":
        raise Refused("tables of other lineages: run tools/eval/tables.py on their jobs")
    ec.load_env()
    cache = Path(a.cache) if a.cache else ec.cache_dir() / "eval_cmddump"
    res = ck1_game_tables(lin, pt, ec.shared_dir(), cache)
    tables.print_tables(res)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("wrote", a.json)


def output_entries(out) -> list[dict]:
    """A ck1_outputs value as entries {job, stage?, as_plan?}: a job folder, a list of them, or an inline
    table naming one stage of a job (`stage`) or a protocol plan whose games lie inside the job (`as_plan`)."""
    items = out if isinstance(out, list) else [out]
    return [{"job": o} if isinstance(o, str) else dict(o) for o in items]


def ck1_game_tables(lin: dict, pt: dict, shared: Path, cache: Path) -> dict:
    """050-R9: the tables of a Checkpoint 1 point's existing game outputs (both held / bunker definitions)."""
    import tables

    res: dict = {}
    seen = set()
    for m, out in pt.get("ck1_outputs", {}).items():
        if lin["metric"][m]["kind"] != "games":
            continue
        for e in output_entries(out):
            key = (e["job"], e.get("stage"), e.get("as_plan"))
            if key in seen:
                continue
            seen.add(key)
            job = shared / e["job"]
            if e.get("as_plan"):
                res.update(tables.subset_tables(job, e["as_plan"], lin))
            else:
                res.update(tables.job_tables(job, cache, lin=lin, stages={e["stage"]} if e.get("stage") else None))
    if not res:
        raise Refused(f"point {pt['id']} has no Checkpoint 1 game outputs")
    return res


def cmd_bc_eval(a) -> None:
    """Prints the pinned bc-eval / per_action command of a teacher-forced metric (run it on a host of the
    pinned backend)."""
    lin = load_lineage()
    pt = point(lin, a.point)
    check_new_measurement(lin, a.lineage, a.metric, pt)
    m = check_metric(lin, pt, a.metric)
    check_checkpoint(lin, a.lineage, pt, a.checkpoint)
    if m["kind"] == "bc_eval":
        args = bc_eval_args(m, a.checkpoint, a.out)
    elif m["kind"] == "per_action":
        args = ["python", "tools/eval/per_action.py", "--checkpoint", a.checkpoint, "--holdout", m["holdout"],
                "--backend", m["backend"], "--config", m["config"], "--out", a.out or f"<{a.metric}>.json"]
    elif m["kind"] in DIAG_KINDS:
        raise Refused(f"metric {a.metric} is a diagnostic: `protocol.py diag`")
    else:
        raise Refused(f"metric {a.metric} is not teacher-forced")
    print(" ".join(args))
    print(f"# pinned: backend {m['backend']} ({PRECISION[m['backend']]}); checkpoint sha256 {checkpoint_sha256(a.checkpoint)}; git {git_state()['commit']}")


def bc_eval_args(m: dict, checkpoint: str, out: str | None = None, py: str = "python") -> list[str]:
    """The pinned `bc-eval` command of a bc_eval metric (`--out`: also write the JSON report there)."""
    args = [py, "-m", "uranium", "bc-eval"]
    for d in m["datasets"]:
        args += ["--dataset", d]
    args += ["--checkpoint", checkpoint, "--split", m["split"]]
    if m.get("games"):
        args += ["--games", str(m["games"])]
    if m.get("seed") is not None:
        args += ["--seed", str(m["seed"])]
    args += ["--backend", m["backend"], "--config", m["config"]]
    # 050-R10 ③: val500 / val500_race's --bunker-before 5760 (adds a field, the rest of the output is the same).
    args += [str(x) for x in m.get("extra_args", [])]
    if out:
        args += ["--out", out]
    return args


def show(cmd: list[str]) -> str:
    """A command line to paste into a POSIX shell (an argument with shell syntax such as `;` quoted)."""
    return " ".join(shlex.quote(x) if re.search(r"[\s;&|<>()$`'\"*?]", x) else x for x in cmd)


def redact_cmd(cmd: list[str], roots: dict[str, Path]) -> list[str]:
    """A command with the host's shared drive / cache paths written as `<shared>` / `<cache>` (001-R10)."""
    out = []
    for x in cmd:
        for tag, r in roots.items():
            rs = str(r)
            if rs and x.startswith(rs):
                x = tag + x[len(rs):].replace("\\", "/")
                break
        out.append(x)
    return out


def cmd_check_bc_eval(a) -> None:
    lin = load_lineage()
    m = lin["metric"].get(a.metric)
    if m is None or m["kind"] != "bc_eval":
        raise Refused(f"{a.metric} is not a bc-eval metric ({', '.join(k for k, v in lin['metric'].items() if v['kind'] == 'bc_eval')})")
    d = json.loads(Path(a.output).read_text(encoding="utf-8"))
    want = {"split": m["split"], "games": m.get("expect_games", m.get("games")), "selection_seed": m.get("seed")}
    got = {k: d.get(k) for k in want}
    if got != want:
        raise SystemExit(f"MISMATCH {a.output}: {got} != pinned {want}")
    field = m.get("requires_field")
    if field:
        # val500_race (050-R9 / 050-R10 ②): per-enemy-race scores in the bc-eval output.
        per = d.get(field)
        if not isinstance(per, dict) or not set(m["races"]) <= set(per):
            raise SystemExit(f"MISMATCH {a.output}: no `{field}` with {', '.join(m['races'])} ({m['note']})")
        for r in m["races"]:
            x = per[r]
            if not isinstance(x, dict) or not x.get("steps") or x.get("action_top1") is None:
                raise SystemExit(f"MISMATCH {a.output}: `{field}.{r}` has no scored step ({m['note']})")
            print(f"{a.output}: {a.metric} vs {r}: action_top1 {x['action_top1'] * 100:.2f}%, action NLL {x['nll']['action']:.4f}, steps {x['steps']}")
        return
    print(f"{a.output}: {a.metric} conditions ok, action_top1 {d['action_top1'] * 100:.2f}%")


def cmd_bots(a) -> None:
    hosts = load_hosts(Path(a.hosts))
    bots = sorted(tomllib.loads(BOTS.read_text(encoding="utf-8"))["bot"])
    seen = check_bots(hosts, a.bots.split(",") if a.bots else bots, Path(a.bot_hashes) if a.bot_hashes else None)
    for h, hs in seen.items():
        print(h, "ok", len(hs), "bots")


def cmd_list(a) -> None:
    lin = load_lineage()
    for p in lin["point"]:
        t1 = f"  T1 on {'/'.join(p['t1_platforms'])}" if p.get("t1_platforms") else ""
        t1 += f"  ck1 supplement {','.join(p['ck1_supplement'])}" if p.get("ck1_supplement") else ""
        print(f"{p['id']:8} {p['family']:3} {p['ck1_checkpoint']:48} metrics {','.join(p['metrics'])}  optional {','.join(p.get('optional', []))}{t1}")


def cmd_compare(a) -> None:
    """Count metrics of two tables.py JSON outputs with Fisher p (050-R2: sampling noise is reported)."""
    A = json.loads(Path(a.a).read_text(encoding="utf-8"))
    B = json.loads(Path(a.b).read_text(encoding="utf-8"))
    for key in sorted(set(A) & set(B)):
        ca, cb = A[key].get("counts", {}), B[key].get("counts", {})
        for row in sorted(set(ca) & set(cb)):
            for metric, (x, n) in ca[row].items():
                if metric in cb[row]:
                    y, m = cb[row][metric]
                    print(f"{key} {row} {metric}: {x}/{n} vs {y}/{m}  Fisher p {ec.fisher_p(x, n, y, m):.3f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list")
    s.set_defaults(f=cmd_list)
    s = sub.add_parser("job", help="eval jobs of a lineage point")
    s.add_argument("--lineage", required=True)
    s.add_argument("--point", required=True)
    s.add_argument("--checkpoint", required=True, help="relative to URANIUM_SHARED_DIR")
    s.add_argument("--metrics", required=True, help="comma separated game metrics of the point's menu")
    s.add_argument("--job", required=True, help="job folder name under <shared>/eval (per zero_last_failed group: -<metrics>)")
    s.add_argument("--label", default=None, help="stage name prefix (default the run name)")
    s.add_argument("--hosts", default=str(HOSTS))
    s.add_argument("--bot-hashes", default=None, help="file of `<addr> <bot> <sha256>` lines for hosts without local / ssh")
    s.add_argument("--out-root", default=None, help="default <shared>/eval")
    s.add_argument("--allow-dirty", action="store_true", help="record uncommitted protocol files (paths, diff hash) instead of refusing")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(f=cmd_job)
    s = sub.add_parser("tables", help="re-tabulate a Checkpoint 1 point's existing eval jobs")
    s.add_argument("--lineage", required=True)
    s.add_argument("--point", required=True)
    s.add_argument("--json", default=None)
    s.add_argument("--cache", default=None)
    s.set_defaults(f=cmd_tables)
    s = sub.add_parser("render", help="one measurement plan from its template")
    s.add_argument("--plan", required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--out", default=None)
    s.set_defaults(f=cmd_render)
    s = sub.add_parser("verify", help="check an eval job against the protocol (050-AC5)")
    s.add_argument("job")
    s.add_argument("--lineage", required=True)
    s.add_argument("--point", required=True)
    s.set_defaults(f=cmd_verify)
    s = sub.add_parser("bc-eval", help="the pinned command of a teacher-forced metric")
    s.add_argument("--metric", required=True)
    s.add_argument("--lineage", required=True)
    s.add_argument("--point", required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--out", default=None, help="the JSON report file (bc-eval --out)")
    s.set_defaults(f=cmd_bc_eval)
    s = sub.add_parser("check-bc-eval")
    s.add_argument("output")
    s.add_argument("--metric", required=True)
    s.set_defaults(f=cmd_check_bc_eval)
    s = sub.add_parser("diag", help="the pinned diagnostics (cue_diag, dead_channels) of a checkpoint")
    s.add_argument("--lineage", required=True)
    s.add_argument("--point", required=True)
    s.add_argument("--checkpoint", required=True, help="relative to URANIUM_SHARED_DIR")
    s.add_argument("--metrics", default="cue_diag,dead_channels")
    s.add_argument("--train-log", default=None, help="dead_channels: the run's training log (BC stderr log or RL train.csv)")
    s.add_argument("--out", default=None, help="default <shared>/<metric output>")
    s.add_argument("--run", action="store_true", help="run the commands here (else print them; nothing is written, the hold-out link trees are made by --run)")
    s.add_argument("--allow-dirty", action="store_true")
    s.add_argument("--takeover", action="store_true", help="redo a folder whose running.json shows an attempt in progress (its host is gone); kept as .failed-<k>")
    s.set_defaults(f=cmd_diag)
    s = sub.add_parser("check-diag", help="refuse diagnostic outputs made with other settings or data")
    s.add_argument("out", help="the diagnostics output folder")
    s.add_argument("--metrics", default="cue_diag,dead_channels")
    s.add_argument("--checkpoint", default=None, help="the outputs must be of this checkpoint's weights")
    s.set_defaults(f=cmd_check_diag)
    s = sub.add_parser("bots", help="check the game hosts' OSS bot builds")
    s.add_argument("--hosts", default=str(HOSTS))
    s.add_argument("--bots", default=None)
    s.add_argument("--bot-hashes", default=None)
    s.set_defaults(f=cmd_bots)
    s = sub.add_parser("compare")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(f=cmd_compare)
    a = ap.parse_args(argv)
    try:
        a.f(a)
    except Refused as e:
        raise SystemExit(f"REFUSED: {e}")


if __name__ == "__main__":
    main()
