"""Lineage measurement protocol (050-R9, 050-AC5; plan "측정 프로토콜", docs/measurement_protocol.md):
config/protocol/lineage.toml pins, per lineage point, the metric menu and the conditions Checkpoint 1 was
measured with; this tool turns a (point, checkpoint) into eval jobs and refuses anything that differs (plan
conditions, sampling, zero_last_failed, family, run prefix, T1 platform, OSS bot builds). Every job gets a
`protocol.json` record: checkpoint sha256, git commit and uncommitted protocol files (paths, diff hash), plan
content hashes, host platforms and bot hashes; `verify` checks a job against it.

The Checkpoint 1 lineage (`--lineage ck1`) is measured already: cells it has are not measured again but
re-tabulated from its existing outputs with both held / bunker definitions (`tables`, also games of a
protocol plan inside a larger job). It takes the diagnostics (cue_diag, dead_channels, other work units) and,
per point, the 050-R9 point-table cells it never measured (`ck1_supplement`), with the same plans.

    python tools/eval/protocol.py list
    python tools/eval/protocol.py job --lineage m2 --point s2e2 --checkpoint checkpoints/m2-s2e2/ckpt-N \\
        --metrics rush,macro,tvz --job m2-s2e2-20261112
    python tools/eval/protocol.py tables --lineage ck1 --point r1 [--json out.json]
    python tools/eval/protocol.py render --plan rush_check --checkpoint checkpoints/<run>/ckpt-N --name <stage>
    python tools/eval/protocol.py verify <job dir> --lineage m2 --point s2e2
    python tools/eval/protocol.py bc-eval --metric val500 --lineage m2 --point s2e2 --checkpoint <dir>
    python tools/eval/protocol.py check-bc-eval <output json> --metric val500
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
    cache = Path(a.cache) if a.cache else Path(os.environ.get("URANIUM_CACHE_DIR") or ROOT / "runs") / "eval_cmddump"
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
        args = ["python", "-m", "uranium", "bc-eval"]
        for d in m["datasets"]:
            args += ["--dataset", d]
        args += ["--checkpoint", a.checkpoint, "--split", m["split"]]
        if m.get("games"):
            args += ["--games", str(m["games"])]
        if m.get("seed") is not None:
            args += ["--seed", str(m["seed"])]
        args += ["--backend", m["backend"], "--config", m["config"]]
    elif m["kind"] == "per_action":
        args = ["python", "tools/eval/per_action.py", "--checkpoint", a.checkpoint, "--holdout", m["holdout"],
                "--backend", m["backend"], "--config", m["config"], "--out", f"<{a.metric}>.json"]
    else:
        raise Refused(f"metric {a.metric} is not teacher-forced")
    print(" ".join(args))
    print(f"# pinned: backend {m['backend']} ({PRECISION[m['backend']]}); checkpoint sha256 {checkpoint_sha256(a.checkpoint)}; git {git_state()['commit']}")


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
    s.set_defaults(f=cmd_bc_eval)
    s = sub.add_parser("check-bc-eval")
    s.add_argument("output")
    s.add_argument("--metric", required=True)
    s.set_defaults(f=cmd_check_bc_eval)
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
