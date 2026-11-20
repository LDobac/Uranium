"""Lineage measurement protocol (docs/guide/measurement_protocol.md)"""

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

from uranium.measure import eval_common as ec
from uranium.measure import per_action, tables
from uranium.measure.lineage import RECORD, Refused, identify, load_lineage, model_sides, read_record, render

ROOT = ec.ROOT
BOTS = ROOT / "config" / "protocol" / "t1_bots.toml"
HOSTS = ROOT / "tools" / "eval" / "hosts.toml"
WEIGHTS = "model.safetensors"
PROTOCOL_PATHS = ("python/uranium/measure", "tools/eval", "config/plans/protocol", "config/protocol")
PLATFORMS = ("windows", "linux")


def point(lin: dict, pid: str) -> dict:
    for p in lin["point"]:
        if p["id"] == pid:
            return p
    raise Refused(f"no lineage point {pid!r} (config/protocol/lineage.toml: {', '.join(p['id'] for p in lin['point'])})")


def metric_plans(lin: dict, pt: dict, metric: str) -> list[str]:
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


def check_lineage_point(lin: dict, lineage: str, pt: dict) -> None:
    only = pt.get("lineages")
    if only and lineage not in only:
        raise Refused(f"point {pt['id']} is a point of lineage {', '.join(only)} only (lineage.toml lineages)")
    pts = lin["lineage"][lineage].get("points")
    if pts is not None and pt["id"] not in pts:
        raise Refused(f"lineage {lineage} has the points {', '.join(pts)} only (lineage.toml points)")


def check_new_measurement(lin: dict, lineage: str, metric: str, pt: dict) -> None:
    if lineage not in lin["lineage"]:
        raise Refused(f"no lineage {lineage!r}")
    check_lineage_point(lin, lineage, pt)
    no_counterpart(lineage, pt)
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


def no_counterpart(lineage: str, pt: dict) -> None:
    if lineage == "ck1" and not pt.get("ck1_checkpoint"):
        raise Refused(f"point {pt['id']} has no Checkpoint 1 counterpart (050-R9 point table): not a point of lineage ck1")


def check_checkpoint(lin: dict, lineage: str, pt: dict, checkpoint: str) -> None:
    if lineage not in lin["lineage"]:
        raise Refused(f"no lineage {lineage!r}")
    check_lineage_point(lin, lineage, pt)
    if lineage == "ck1":
        no_counterpart(lineage, pt)
        if checkpoint.replace("\\", "/").rstrip("/") != pt["ck1_checkpoint"] and checkpoint_sha256(checkpoint) != checkpoint_sha256(pt["ck1_checkpoint"]):
            raise Refused(f"point {pt['id']} of the Checkpoint 1 lineage is {pt['ck1_checkpoint']}, not {checkpoint}")
        return
    prefix = lin["lineage"][lineage]["run_prefix"]
    run = run_of(checkpoint)
    if not run.startswith(prefix):
        raise Refused(f"lineage {lineage} runs start with {prefix!r}; {checkpoint} is run {run!r}")
    pat = pt.get("run_pattern")
    if pat and not re.fullmatch(pat, run[len(prefix):]):
        raise Refused(f"point {pt['id']} takes runs {prefix}<{pat}> (lineage.toml run_pattern); {checkpoint} is run {run!r}")


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


def load_hosts(path: Path) -> list[dict]:
    if not path.is_file():
        raise Refused(f"no host list {path} (copy tools/eval/hosts.example.toml, git-ignored, 001-R10)")
    hosts = tomllib.loads(path.read_text(encoding="utf-8-sig")).get("host", [])
    if not hosts:
        raise Refused(f"{path}: no [[host]]")
    return hosts


def job_hosts_toml(hosts: list[dict]) -> str:
    out = []
    for h in hosts:
        out.append(f'[[host]]\naddr = "{h["addr"]}"\n' + (f"max_games = {h['max_games']}\n" if "max_games" in h else ""))
    return "\n".join(out)


def check_t1_platform(pt: dict, hosts: list[dict]) -> list[str]:
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
                out[b] = ec.file_sha256(f)
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


DIAG_KINDS = ("diag_cue", "diag_train_log")
DIAG_RECORD = "protocol.json"
RUNNING = "running.json"
COPY = "copy"
FETCH = "fetch"
GPU_STEPS = ("dump", COPY)
HOLDOUT_DIR = "diag_holdout"
LOCAL_DUMP_DIR = "diag_dump"
TRAIN_LOG = "spatial_log.json"


def diag_metric(lin: dict, metric: str) -> dict:
    m = lin["metric"].get(metric)
    if m is None or m["kind"] not in DIAG_KINDS:
        raise Refused(f"{metric} is not a diagnostic metric ({', '.join(k for k, v in lin['metric'].items() if v['kind'] in DIAG_KINDS)})")
    return m


def cue_set(lin: dict, name: str) -> dict:
    return lin["metric"]["cue_diag"][name]


def files_fingerprint(src: Path, parts: list[str]) -> tuple[int, str]:
    lines = sorted(f"{p}/{f.name} {f.stat().st_size}" for p in parts for f in (src / p).glob("shard_*.bin.zst"))
    return len(lines), hashlib.sha256("".join(l + "\n" for l in lines).encode()).hexdigest()


def manifest_fingerprint(shared: Path, datasets: list[str]) -> tuple[int, str]:
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
        dst = work / "by_name" / h["name"]
        if create:
            per_action.link_tree(src, h["parts"], dst)
            _check_files(h["name"], files_fingerprint(dst, h["parts"]), (h["files"], h["files_sha256"]), f"link tree {dst}")
        out += ["--shards", str(dst)]
    return out


def data_tails(s: dict) -> list[str]:
    if "datasets" in s:
        return sorted(s["datasets"])
    return [h["name"] for h in s["holdout"]]


def output_name(checkpoint: str) -> str:
    parts = [s for s in checkpoint.replace("\\", "/").split("/") if s]
    i = next((k for k, s in enumerate(parts) if s.startswith("ckpt-")), len(parts) - 1)
    return f"{run_of(checkpoint)}__{'-'.join(parts[i:])}"


def diag_out(lin: dict, metric: str, checkpoint: str) -> str:
    run, ck = output_name(checkpoint).split("__", 1)
    return diag_metric(lin, metric)["output"].replace("<run>", run).replace("<ckpt>", ck)


def probe_stages(lin: dict, name: str, lineage: str | None = None) -> list[str]:
    drop = set(lin["lineage"].get(lineage, {}).get("cue_drop_stages", [])) if lineage else set()
    return [x for x in cue_set(lin, name)["probes"]["stages"] if x not in drop]


def cue_commands(lin: dict, name: str, ckpt: str, out: Path, data: list[str], py: str = "python", dump: Path | None = None,
                 lineage: str | None = None) -> list[list[str]]:
    m = diag_metric(lin, "cue_diag")
    s = cue_set(lin, name)
    d, p = s["dump"], s["probes"]
    dump = dump or out / "dump"
    cmd = [py, "-m", "uranium", "diag", "dump", "--checkpoint", ckpt, *data, "--splits", ",".join(s["splits"])]
    if s.get("games"):
        cmd += ["--games", str(s["games"]), "--seed", str(s["seed"])]
    cmd += ["--config", m["config"], "--backend", m["backend"], "--lanes-per-group", str(d["lanes_per_group"]), "--row-until", f"{d['row_until']:g}",
            "--act-until", f"{d['act_until']:g}", "--act-every", str(d["act_every"]), "--core-points", d["core_points"]]
    if d.get("read_until"):
        cmd += ["--read-until", f"{d['read_until']:g}"]
    if d["cf"]:
        cmd += ["--cf", "--cf-starts", ";".join(f"{r}:{a:g},{b:g}" for r, (a, b) in d["cf_starts"].items())]
    cmd += ["--out", str(dump)]
    cmds = [cmd, [py, "-m", "uranium", "diag", "probes", str(dump), "--races", ",".join(p["races"]), "--stages", ",".join(probe_stages(lin, name, lineage)), "--gain", p["gain"],
                  "--boot", str(p["boot"]), "--cap", str(p["cap"]), "--seed", str(p["seed"]), "--device", "cpu", "--out", str(out / "probes.json")]]
    if "intervene" in s:
        iv = s["intervene"]
        cmds.append([py, "-m", "uranium", "diag", "intervene", str(dump), "--races", ",".join(iv["races"]), "--window", ",".join(f"{x:g}" for x in iv["window"]),
                     "--boot", str(iv["boot"]), "--seed", str(iv["seed"]), "--out", str(out / "intervene.json")])
    return cmds


def summary_command(lin: dict, pt: dict, name: str, out: Path, py: str = "python") -> list[str]:
    cmd = [py, "-m", "uranium", "diag", "summary", "--point", pt["id"], "--probes", str(out / "probes.json")]
    if "intervene" in cue_set(lin, name):
        cmd += ["--intervene", str(out / "intervene.json")]
    return cmd + ["--out", str(out / "summary.json")]


def train_log_command(lin: dict, pt: dict, checkpoint: str, log: str, ckpt: str, out: Path, py: str = "python") -> list[str]:
    m = diag_metric(lin, "dead_channels")
    cmd = [py, "tools/eval/spatial_logs.py", log, "--checkpoint", ckpt, "--from-update", str(m["from_update"])]
    n = re.match(r"ckpt-(\d+)$", Path(checkpoint).name)
    if pt["family"] == "bc" and n:
        cmd += ["--until-update", n.group(1)]
    return cmd + ["--out", str(out / TRAIN_LOG)]


def step_name(c: list[str]) -> str:
    return c[0] if c[0] in (COPY, FETCH) else c[4] if c[2:4] == ["uranium", "diag"] else Path(c[1]).stem


def _ends_with(path: str, tail: str) -> bool:
    return path.replace("\\", "/").rstrip("/").endswith(tail)


def _load(f: Path, problems: list[str]) -> dict | None:
    if not f.is_file():
        problems.append(f"no {f}")
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def check_cue_set(lin: dict, name: str, d: Path, sha: str | None) -> list[str]:
    m = diag_metric(lin, "cue_diag")
    s = cue_set(lin, name)
    dm, p, rep = s["dump"], s["probes"], s["report"]
    tails = data_tails(s)
    problems: list[str] = []

    def same(what: str, got, want) -> None:
        if got != want:
            problems.append(f"{name} {what}: {got!r}, pinned {want!r}")

    def data(what: str, got) -> None:
        got = list(got or [])
        if len(got) != len(tails) or not all(_ends_with(g, t) for g, t in zip(got, tails)):
            problems.append(f"{name} {what} data {got}: not the pinned hold-out (.../{', .../'.join(tails)}; a folder named otherwise orders the games otherwise)")

    man = _load(d / "dump" / "manifest.json", problems)
    if man is not None:
        data("dump", man.get("data"))
        for k, want in (("splits", s["splits"]), ("games_per_split", s.get("games")), ("config", m["config"]), ("backend", m["backend"]),
                        ("precision", ec.PRECISION[m["backend"]]), ("lanes_per_group", dm["lanes_per_group"]), ("row_until", dm["row_until"]),
                        ("act_until", dm["act_until"]), ("act_every", dm["act_every"]), ("read_until", dm.get("read_until"))):
            same(f"dump {k}", man.get(k), want)
        if s.get("games"):
            same("dump seed", man.get("seed"), s["seed"])
        st = man.get("stats", {})
        for k, got in (("expect_games", man.get("games")), ("expect_rows", st.get("rows")), ("expect_act_rows", st.get("act_rows"))):
            if k in dm:
                same(f"dump {k.removeprefix('expect_')}", got, dm[k])
        if "expect_game_rows_sha256" in dm:
            same("dump game order (game_rows_sha256)", game_rows_sha256(d / "dump"), dm["expect_game_rows_sha256"])
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
            for t in set(p["expect_tasks"]) | (set(rep.get("required", [])) - set(not_measurable(rep, race))):
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


def not_measurable(rep: dict, race: str) -> dict[str, str]:
    return dict((rep.get("not_measurable") or {}).get(race, {}))


def game_rows_sha256(dump: Path) -> str | None:
    import numpy as np

    parts = sorted(Path(dump).glob("rows_p*.npz"))
    if not parts:
        return None
    games = np.concatenate([np.load(p)["game"] for p in parts])
    return hashlib.sha256(",".join(str(int(n)) for n in np.bincount(games)).encode()).hexdigest()


def data_fingerprint(lin: dict, name: str, d: Path) -> dict | None:
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
    notes = []
    for name in diag_metric(lin, "cue_diag")["sets"]:
        dm = cue_set(lin, name)["dump"]
        f = out / name / "dump" / "manifest.json"
        if f.is_file():
            man = json.loads(f.read_text(encoding="utf-8"))
            got = {"expect_games": man.get("games"), "expect_rows": man.get("stats", {}).get("rows"), "expect_act_rows": man.get("stats", {}).get("act_rows")}
            notes += [f"{name} {k} = {v} (not pinned yet)" for k, v in got.items() if k not in dm]
    return notes


def no_targets(lin: dict, metric: str, lineage: str | None) -> str | None:
    m = lin["metric"].get(metric, {})
    return (m.get("not_measured") or {}).get(lineage) if lineage in m.get("no_targets", []) else None


def check_train_log(lin: dict, out: Path, lineage: str | None = None) -> list[str]:
    m = diag_metric(lin, "dead_channels")
    problems: list[str] = []
    r = _load(out / TRAIN_LOG, problems)
    if r is not None:
        if r.get("from_update") != m["from_update"]:
            problems.append(f"{TRAIN_LOG} from_update {r.get('from_update')!r}, pinned {m['from_update']}")
        if r.get("not_measurable"):
            if not r.get("spatial_patch"):
                problems.append(f"{TRAIN_LOG}: not_measurable without a patch spatial path (spatial_patch {r.get('spatial_patch')!r})")
            if not no_targets(lin, "dead_channels", lineage):
                problems.append(f"{TRAIN_LOG}: not_measurable ({r['not_measurable']}) in a folder of lineage {lineage!r}, which measures dead channels")
            return problems
        if not r.get("sp_records_from"):
            problems.append(f"{TRAIN_LOG}: no sp_* record from update {m['from_update']} (a run without [monitor] spatial_every?)")
        lack = [k for k in m["layers"] if k not in (r.get("dead") or {})]
        if r.get("sp_records_from") and lack:
            problems.append(f"{TRAIN_LOG}: no dead count for {lack}")
    return problems


def check_diag(lin: dict, metric: str, out: Path, sha: str | None = None) -> list[str]:
    m = diag_metric(lin, metric)
    problems: list[str] = []
    rec = _load(out / DIAG_RECORD, problems)
    skip = no_targets(lin, metric, (rec or {}).get("lineage"))
    if rec is not None:
        if not rec.get("finished"):
            problems.append(f"{DIAG_RECORD}: not finished (an attempt in progress or failed)")
        if metric not in (rec.get("metrics") or []) and not skip:
            problems.append(f"{DIAG_RECORD}: {metric} not among the measured metrics {rec.get('metrics')}")
        if not (rec.get("git") or {}).get("commit"):
            problems.append(f"{DIAG_RECORD}: no git commit")
        if not rec.get("checkpoint_sha256"):
            problems.append(f"{DIAG_RECORD}: no checkpoint sha256")
        elif sha is not None and rec["checkpoint_sha256"] != sha:
            problems.append(f"{DIAG_RECORD} checkpoint_sha256: {rec['checkpoint_sha256']!r}, pinned {sha!r}")
        sha = sha or rec.get("checkpoint_sha256")
    if m["kind"] == "diag_train_log":
        if skip and not (out / TRAIN_LOG).is_file():
            return problems
        return problems + check_train_log(lin, out, (rec or {}).get("lineage"))
    for name in m["sets"]:
        problems += check_cue_set(lin, name, out / name, sha)
    return problems


def next_failed(p: Path) -> Path:
    k = 1
    while p.with_name(f"{p.name}.failed-{k}").exists():
        k += 1
    return p.with_name(f"{p.name}.failed-{k}")


def copy_dump(src: Path, dst: Path) -> None:
    import shutil

    part = dst.with_name(dst.name + ".part")
    if part.exists():
        part.rename(next_failed(part))
    shutil.copytree(src, part)
    os.replace(part, dst)


def diag_record_cmd(c: list[str], roots: dict[str, Path]) -> list[str]:
    return redact_cmd(c if c[0] in (COPY, FETCH) else ["python", *c[1:]], roots)


def final_only_metrics(pt: dict) -> list[str]:
    f = pt.get("diag_final_only")
    return list(f) if isinstance(f, list) else (["cue_diag"] if f else [])


def check_final_only(pt: dict, checkpoint: str) -> None:
    if not pt.get("diag_final_only"):
        return
    d = checkpoint_dir(checkpoint)
    d = next((p for p in (d, d.parent, d.parent.parent) if p.name.startswith("ckpt-")), d)
    run = d.parent
    steps = sorted(int(x.name[5:]) for x in run.glob("ckpt-*") if x.name[5:].isdigit())
    if not (run / "DONE").is_file() or not steps or d.name != f"ckpt-{steps[-1]}":
        what = "cue diagnostics" if final_only_metrics(pt) == ["cue_diag"] else f"diagnostics ({', '.join(final_only_metrics(pt))})"
        why = "AMB-050-6" if pt["id"] == "s1e2" else "user decision 2026-11-14"
        raise Refused(f"point {pt['id']}: {what} at the run's final checkpoint only ({why}); {checkpoint} is not (DONE {'present' if (run / 'DONE').is_file() else 'missing'}, last ckpt-{steps[-1] if steps else '?'})")


def step_part_name(step: str) -> str:
    return "gpu" if step in GPU_STEPS else "cpu"


def step_part(c: list[str]) -> str:
    return step_part_name(step_name(c))


def cpu_order(cmds: list[list[str]], out: Path, local: Path) -> list[list[str]]:
    todo, fetched = [], set()
    for c in cmds:
        if step_part(c) != "cpu":
            continue
        if c[2:4] == ["uranium", "diag"] and c[4] in ("probes", "intervene"):
            name = Path(c[5]).parent.name
            if name not in fetched:
                fetched.add(name)
                todo.append([FETCH, str(out / name / "dump"), str(local / name / "dump")])
        todo.append(c)
    return todo


def cmd_diag(a) -> None:
    lin = load_lineage()
    pt = point(lin, a.point)
    metrics = [x.strip() for x in a.metrics.split(",") if x.strip()]
    for metric in metrics:
        check_new_measurement(lin, a.lineage, metric, pt)
        if check_metric(lin, pt, metric)["kind"] not in DIAG_KINDS:
            raise Refused(f"metric {metric} is not a diagnostic")
    check_checkpoint(lin, a.lineage, pt, a.checkpoint)
    if set(final_only_metrics(pt)) & set(metrics):
        check_final_only(pt, a.checkpoint)
    if "dead_channels" in metrics and not a.train_log and a.part != "gpu":
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
            cmds += cue_commands(lin, name, ck, out / name, set_data(lin, name, shared, create=a.run and a.part != "cpu"), py, dump=local / name / "dump",
                                 lineage=a.lineage)
            cmds.append([COPY, str(local / name / "dump"), str(out / name / "dump")])
            cmds.append(summary_command(lin, pt, name, out / name, py))
    log_sha = None
    if "dead_channels" in metrics and a.part != "gpu":
        log = Path(a.train_log)
        if not log.is_file():
            raise Refused(f"no training log {log}")
        log_sha = ec.file_sha256(log)
        cmds.append(train_log_command(lin, pt, a.checkpoint, str(log), ck, out, py))
    if a.part == "cpu":
        cmds = cpu_order(cmds, out, local)
    elif a.part == "gpu":
        cmds = [c for c in cmds if step_part(c) == "gpu"]
    if not a.run:
        for c in cmds:
            if c[0] in (COPY, FETCH):
                part = c[2] + ".part"
                print(f"cp -r {shlex.quote(c[1])} {shlex.quote(part)} && mv {shlex.quote(part)} {shlex.quote(c[2])}")
            else:
                print(show(c))
        if a.part != "gpu":
            print(f"python tools/eval/protocol.py check-diag {out} --metrics {','.join(metrics)} --checkpoint {a.checkpoint}")
        print(f"# pinned: backend {diag_metric(lin, 'cue_diag')['backend']}; checkpoint sha256 {sha}; git {git_state()['commit']}")
        print(f"# --run also makes the hold-out link trees under {cache / HOLDOUT_DIR}, writes {out / RUNNING} at the start and {out / DIAG_RECORD} at the end")
        return
    git = git_state()
    if git["protocol_files_dirty"] and not a.allow_dirty:
        raise Refused("uncommitted protocol files (" + ", ".join(git["dirty_paths"]) + "); --allow-dirty records them in protocol.json")
    if (out / DIAG_RECORD).exists():
        raise Refused(f"{out / DIAG_RECORD} exists (one finished measurement per folder; --out for another)")
    roots = {"<shared>": shared, "<cache>": cache}
    host = socket.gethostname()
    now = lambda: datetime.datetime.now().astimezone().isoformat(timespec="seconds")  # noqa: E731
    if a.part == "cpu":
        rec = cpu_record(out, sha, metrics, git, a.takeover)
        if local.exists():
            moved = next_failed(local)
            local.rename(moved)
            print(f"[diag] earlier local copy kept as {moved}", flush=True)
        rec["parts"]["cpu"] = {"host": host, "platform": sys.platform, "git": git, "started": now(), "finished": None}
        rec["train_log"] = {"file": Path(a.train_log).name, "sha256": log_sha} if log_sha else None
        rec["commands"] = [c for c in rec["commands"] if step_part(c) == "gpu"] + [diag_record_cmd(c, roots) for c in cmds]
    else:
        if (out / RUNNING).is_file() and not a.takeover:
            try:
                prev = json.loads((out / RUNNING).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                prev = {}
            if not any("error" in s for s in prev.get("steps", [])):
                raise Refused(f"{out / RUNNING}: an attempt in progress (host {prev.get('host', '?')}, started {prev.get('started', '?')}, "
                              f"no failed step); --takeover when its host is gone")
        for p in (out, local):
            if p.exists():
                moved = next_failed(p)
                p.rename(moved)
                print(f"[diag] earlier attempt kept as {moved}", flush=True)
        out.mkdir(parents=True)
        if "cue_diag" in metrics:
            for name in diag_metric(lin, "cue_diag")["sets"]:
                (out / name).mkdir()
        rec = {"protocol": "config/protocol/lineage.toml (050-R9)", "lineage": a.lineage, "point": pt["id"], "family": pt["family"],
               "metrics": metrics, "checkpoint": a.checkpoint, "checkpoint_sha256": sha, "checkpoint_id": f"{Path(a.checkpoint).name}@{sha[:12]}",
               "backend": diag_metric(lin, "cue_diag")["backend"] if "cue_diag" in metrics else None,
               "precision": ec.PRECISION[diag_metric(lin, "cue_diag")["backend"]] if "cue_diag" in metrics else None,
               "sets": {n: {k: v for k, v in cue_set(lin, n).items() if k in ("holdout", "datasets", "files", "files_sha256", "splits", "games", "seed")}
                        for n in diag_metric(lin, "cue_diag")["sets"]} if "cue_diag" in metrics else None,
               "train_log": {"file": Path(a.train_log).name, "sha256": log_sha} if log_sha else None,
               "git": git, "host": host, "platform": sys.platform, "commands": [diag_record_cmd(c, roots) for c in cmds],
               "started": now(), "finished": None, "steps": []}
        if a.part == "gpu":
            rec["parts"] = {"gpu": {"host": host, "platform": sys.platform, "git": git, "started": rec["started"], "finished": None}}

    def save(final: bool = False) -> None:
        (out / RUNNING).write_text(json.dumps(rec, indent=1), encoding="utf-8")
        if final:
            os.replace(out / RUNNING, out / DIAG_RECORD)

    save()
    for c in cmds:
        print("[diag]", show(diag_record_cmd(c, roots)), flush=True)
        t0 = datetime.datetime.now()
        try:
            if c[0] in (COPY, FETCH):
                copy_dump(Path(c[1]), Path(c[2]))
            else:
                subprocess.run(c, cwd=ROOT, check=True)
        except (subprocess.CalledProcessError, OSError) as e:
            rec["steps"].append({"step": step_name(c), "host": host, "error": str(e)})
            save()
            raise
        rec["steps"].append({"step": step_name(c), "host": host, "secs": round((datetime.datetime.now() - t0).total_seconds())})
        save()
    if a.part == "gpu":
        rec["parts"]["gpu"]["finished"] = now()
        save()
        print(f"[diag] dumps on the shared drive; finish on a CPU host at git {git['commit']}: python tools/eval/protocol.py diag "
              f"--lineage {a.lineage} --point {a.point} --checkpoint {a.checkpoint} --metrics {','.join(metrics)} --run --part cpu"
              + (f" --out {a.out}" if a.out else "") + (" --train-log <log>" if "dead_channels" in metrics else ""), flush=True)
        return
    if a.part == "cpu":
        rec["parts"]["cpu"]["finished"] = now()
    rec["finished"] = now()
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


def cpu_record(out: Path, sha: str, metrics: list[str], git: dict, takeover: bool) -> dict:
    f = out / RUNNING
    if not f.is_file():
        raise Refused(f"no {f}: run `diag --run --part gpu` first (on the Linux ROCm host)")
    rec = json.loads(f.read_text(encoding="utf-8"))
    gpu = (rec.get("parts") or {}).get("gpu") or {}
    if not gpu.get("finished"):
        raise Refused(f"{f}: the gpu part has not finished (host {gpu.get('host', rec.get('host', '?'))}, steps {[s['step'] for s in rec.get('steps', [])]})")
    if rec.get("checkpoint_sha256") != sha:
        raise Refused(f"{f}: dumps of checkpoint sha256 {str(rec.get('checkpoint_sha256'))[:12]}, this checkpoint {sha[:12]}")
    if rec.get("metrics") != metrics:
        raise Refused(f"{f}: metrics {rec.get('metrics')}, this attempt {metrics}")
    if (gpu.get("git") or {}).get("commit") != git.get("commit"):
        raise Refused(f"{f}: the dumps were made at git {(gpu.get('git') or {}).get('commit')}, this host is at {git.get('commit')} (same commit on both parts)")
    g = gpu.get("git") or {}
    if (g.get("dirty_paths") or [], g.get("dirty_diff_sha256")) != (git.get("dirty_paths") or [], git.get("dirty_diff_sha256")):
        raise Refused(f"{f}: uncommitted protocol files differ between the parts (gpu {g.get('dirty_paths') or []}, this host {git.get('dirty_paths') or []}); "
                      f"the same protocol files on both parts")
    cpu = rec["parts"].get("cpu")
    if cpu and not cpu.get("finished") and not takeover:
        failed = any("error" in s for s in rec.get("steps", []) if s.get("host") == cpu.get("host") and step_part_name(s["step"]) == "cpu")
        if not failed:
            raise Refused(f"{f}: a cpu part in progress (host {cpu.get('host', '?')}, started {cpu.get('started', '?')}, no failed step); --takeover when its host is gone")
    return rec


def cmd_check_diag(a) -> None:
    lin = load_lineage()
    sha = checkpoint_sha256(a.checkpoint) if a.checkpoint else None
    metrics = [x.strip() for x in a.metrics.split(",") if x.strip()]
    problems = [p for metric in metrics for p in check_diag(lin, metric, Path(a.out), sha)]
    rec_f = Path(a.out) / DIAG_RECORD
    if rec_f.is_file():
        rec = json.loads(rec_f.read_text(encoding="utf-8"))
        parts = rec.get("parts") or {"all": {"host": rec.get("host")}}
        print("hosts:", ", ".join(f"{k} {v.get('host')}" for k, v in parts.items()))
    if "cue_diag" in metrics:
        for n in unpinned(lin, Path(a.out)):
            print("NOTE", n)
    for p in problems:
        print("MISMATCH", p)
    if problems:
        raise SystemExit(1)
    print(f"{a.out}: {', '.join(metrics)} conform to the protocol")


def job_names(job: str, groups: dict[bool, list]) -> dict[bool, str]:
    if len(groups) == 1:
        return {zlf: job for zlf in groups}
    base = {zlf: job + "-" + "-".join(dict.fromkeys(m for m, _, _ in items)) for zlf, items in groups.items()}
    out = {zlf: n + ("-lf0" if zlf else "-real") if list(base.values()).count(n) > 1 else n for zlf, n in base.items()}
    assert len(set(out.values())) == len(out), out
    return out


def cmd_job(a) -> None:
    lin = load_lineage()
    pt = point(lin, a.point)
    metrics = [m.strip() for m in a.metrics.split(",") if m.strip()]
    stages: list[tuple[str, str]] = []
    for m in metrics:
        check_new_measurement(lin, a.lineage, m, pt)
        spec = check_metric(lin, pt, m)
        if spec["kind"] == "games_table":
            raise Refused(f"metric {m} is a table of the point's game jobs (052-R8: tools/eval/tables.py / `protocol.py tables`), not a job")
        if spec["kind"] != "games":
            raise Refused(f"metric {m} is not a game metric (use the bc-eval command)")
        stages += [(m, p) for p in metric_plans(lin, pt, m)]
    check_checkpoint(lin, a.lineage, pt, a.checkpoint)
    git = git_state()
    if git["protocol_files_dirty"] and not a.allow_dirty:
        raise Refused("uncommitted protocol files (" + ", ".join(git["dirty_paths"]) + "); --allow-dirty records them in protocol.json")
    sha = checkpoint_sha256(a.checkpoint)
    label = a.label or run_of(a.checkpoint)
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
    names = job_names(a.job, groups)
    for zlf, items in groups.items():
        d = out_root / names[zlf]
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
            rec_plans.append({"metric": m, "plan": p, "file": f"plans/{f.name}", "content_sha256": ec.file_sha256(f),
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
            if not f.is_file() or ec.file_sha256(f) != p["content_sha256"]:
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
            why = "a branch-game plan (051) is never a lineage measurement" if ec.is_branch_plan(plan) else "conditions match no protocol plan"
            problems.append(f"stage {k} {f.name}: {why}")
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
    lin = load_lineage()
    pt = point(lin, a.point)
    if a.lineage != "ck1":
        raise Refused("tables of other lineages: run tools/eval/tables.py on their jobs")
    no_counterpart(a.lineage, pt)
    ec.load_env()
    cache = Path(a.cache) if a.cache else ec.cache_dir() / "eval_cmddump"
    res = ck1_game_tables(lin, pt, ec.shared_dir(), cache, a.ops, a.ops_jobs)
    tables.print_tables(res)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("wrote", a.json)


def output_entries(out) -> list[dict]:
    items = out if isinstance(out, list) else [out]
    return [{"job": o} if isinstance(o, str) else dict(o) for o in items]


def ck1_game_tables(lin: dict, pt: dict, shared: Path, cache: Path, ops: str = "replay", ops_jobs: int = 1) -> dict:
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
                res.update(tables.subset_tables(job, e["as_plan"], lin, cache, ops, ops_jobs))
            else:
                res.update(tables.job_tables(job, cache, lin=lin, stages={e["stage"]} if e.get("stage") else None, ops=ops, ops_jobs=ops_jobs))
    if not res:
        raise Refused(f"point {pt['id']} has no Checkpoint 1 game outputs")
    return res


def cmd_bc_eval(a) -> None:
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
    print(f"# pinned: backend {m['backend']} ({m.get('precision') or ec.PRECISION[m['backend']]}); checkpoint sha256 {checkpoint_sha256(a.checkpoint)}; git {git_state()['commit']}")


def bc_eval_args(m: dict, checkpoint: str, out: str | None = None, py: str = "python") -> list[str]:
    args = [py, "-m", "uranium", "bc-eval"]
    for d in m["datasets"]:
        args += ["--dataset", d]
    args += ["--checkpoint", checkpoint, "--split", m["split"]]
    if m.get("games"):
        args += ["--games", str(m["games"])]
    if m.get("seed") is not None:
        args += ["--seed", str(m["seed"])]
    args += ["--backend", m["backend"], "--config", m["config"]]
    if m.get("precision"):
        args += ["--precision", m["precision"]]
    args += [str(x) for x in m.get("extra_args", [])]
    if out:
        args += ["--out", out]
    return args


def show(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) if re.search(r"[\s;&|<>()$`'\"*?]", x) else x for x in cmd)


def redact_cmd(cmd: list[str], roots: dict[str, Path]) -> list[str]:
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
    if "chain_start_sources" in d:
        raise SystemExit(f"MISMATCH {a.output}: scored with --chain-start-sources {d['chain_start_sources']} (041-R30: pinned metrics use the current observation)")
    want = {"split": m["split"], "games": m.get("expect_games", m.get("games")), "selection_seed": m.get("seed")}
    got = {k: d.get(k) for k in want}
    if got != want:
        raise SystemExit(f"MISMATCH {a.output}: {got} != pinned {want}")
    if "--bunker-before" in [str(x) for x in m.get("extra_args", [])] and "bunker_before" not in d:
        print(f"WARNING {a.output}: no `bunker_before` (run without --bunker-before 5760): regression gate check 3a will be MISSING")
    if m.get("precision") and (d.get("backend"), d.get("precision")) != (m["backend"], m["precision"]):
        raise SystemExit(f"MISMATCH {a.output}: backend / precision {d.get('backend')!r} / {d.get('precision')!r} != pinned {m['backend']!r} / {m['precision']!r}")
    if m.get("loc_groups"):
        g = d.get("target_loc_groups")
        if not isinstance(g, dict) or not set(m["loc_groups"]) <= set(g):
            raise SystemExit(f"MISMATCH {a.output}: no `target_loc_groups` with {', '.join(m['loc_groups'])} (bc-eval before 045-R7)")
        for k in m["loc_groups"]:
            if not (g[k] or {}).get("steps"):
                raise SystemExit(f"MISMATCH {a.output}: `target_loc_groups.{k}` has no scored step")
        print(f"{a.output}: {a.metric} location NLL {g['all']['nll']:.4f}, within 4 cells {g['all']['within4']:.4f}, "
              f"build exact {g['build']['exact']:.4f}, rally distance {g['rally']['err_mean']:.3f} (steps {g['all']['steps']})")
        return
    field = m.get("requires_field")
    if field:
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
        t1 += f"  lineage {','.join(p['lineages'])} only, compared with {','.join(p.get('compare', []))}" if p.get("lineages") else ""
        print(f"{p['id']:8} {p['family']:3} {p.get('ck1_checkpoint', '- (no Checkpoint 1 counterpart)'):48} metrics {','.join(p['metrics'])}  optional {','.join(p.get('optional', []))}{t1}")


def stage_backends(s: dict) -> list[str]:
    return sorted({f"{b.split('@')[0]} ({x.get('precision', '?')})" for b, x in (s.get("inference") or {}).items()})


def backend_notes(a: dict, b: dict) -> list[str]:
    out = []
    ba, bb = stage_backends(a), stage_backends(b)
    if not ba or not bb:
        if ba or bb:
            out.append(f"inference backend not recorded on one side: {', '.join(ba) or 'not recorded'} vs {', '.join(bb) or 'not recorded'}")
    elif ba != bb:
        out.append(f"inference backend differs: {', '.join(ba)} vs {', '.join(bb)}")
    pa, pb = ((s.get("protocol") or {}).get("t1_platforms") for s in (a, b))
    if (a.get("kind") == "t1" or b.get("kind") == "t1") and sorted(pa or []) != sorted(pb or []):
        out.append(f"T1 platform differs: {pa or '?'} vs {pb or '?'}")
    return out


def cmd_compare(a) -> None:
    A = json.loads(Path(a.a).read_text(encoding="utf-8"))
    B = json.loads(Path(a.b).read_text(encoding="utf-8"))
    for key in sorted(set(A) & set(B)):
        for note in backend_notes(A[key], B[key]):
            print(f"NOTE {key}: {note}")
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
    s.add_argument("--job", required=True, help="job folder name under <shared>/eval (one folder per zero_last_failed group, suffixed)")
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
    s.add_argument("--ops", choices=("replay", "live", "off"), default="replay", help="ops tables: opsdump replay values (the path comparison's), live row values, or none")
    s.add_argument("--ops-jobs", type=int, default=1, help="parallel opsdump processes")
    s.set_defaults(f=cmd_tables)
    s = sub.add_parser("render", help="one measurement plan from its template")
    s.add_argument("--plan", required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--out", default=None)
    s.set_defaults(f=cmd_render)
    s = sub.add_parser("verify", help="check an eval job against the protocol")
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
    s.add_argument("--run", action="store_true", help="run the commands here (else only print them; nothing is written)")
    s.add_argument("--allow-dirty", action="store_true")
    s.add_argument("--takeover", action="store_true", help="redo a folder whose running.json shows an attempt of a gone host; kept as .failed-<k>")
    s.add_argument("--part", choices=["all", "gpu", "cpu"], default="all",
                   help="gpu: the dumps and their copies, leaving running.json; cpu: the rest, on a host at the same commit")
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
