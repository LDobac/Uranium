"""Measurement protocol of the Checkpoint 1 lineage and its successors (plan "측정 프로토콜",
docs/measurement_protocol.md): config/protocol/lineage.toml pins, per lineage point, the metric menu and
the conditions Checkpoint 1 was measured with; this tool turns a (point, checkpoint) into eval jobs and
refuses anything that differs (plan conditions, sampling, zero_last_failed, family, run prefix, T1 bot
builds). Every job gets a `protocol.json` record: checkpoint sha256, git commit, plan content hashes.

    python tools/eval/protocol.py list
    python tools/eval/protocol.py job --lineage m2 --point s2e2 --checkpoint checkpoints/m2-s2e2/ckpt-N \\
        --metrics rush,macro,tvz --job m2-s2e2-20261112
    python tools/eval/protocol.py render --plan dagger_r1 --checkpoint checkpoints/<run>/ckpt-N --name dagger-r1 --out <file>
    python tools/eval/protocol.py verify <job dir> --lineage ck1 --point r1
    python tools/eval/protocol.py bc-eval --metric val500 --lineage m2 --point s2e2 --checkpoint <dir>
    python tools/eval/protocol.py check-bc-eval <output json> --metric val500
    python tools/eval/protocol.py bots [--hosts tools/eval/hosts.toml]
    python tools/eval/protocol.py compare <tables json A> <tables json B>

Host lists (addresses, ssh aliases) live in the git-ignored tools/eval/hosts.toml (template
tools/eval/hosts.example.toml, 001-R10). Relative checkpoints are under URANIUM_SHARED_DIR (061).
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
# Plan defaults of crates/uranium-runner/src/plan.rs (Plan, Matchup, Side) for the normalised form.
PLAN_DEFAULTS = {"seed_base": 1000, "same_seed": False, "max_frames": 86_400, "wall_timeout_s": 1800, "privileged": False,
                 "save_replay": False, "zero_last_failed": False, "surrender": "shadow", "surrender_chat": "gg"}
CKPT, NAME = "@CHECKPOINT@", "@NAME@"
# 032-R8: CUDA runs bf16 mixed, ROCm f32 (recorded per inference backend).
PRECISION = {"cuda": "bf16-mixed", "rocm": "f32", "cpu": "f32"}


class Refused(Exception):
    """A request that differs from the pinned protocol."""


# ---------------------------------------------------------------- lineage and templates


def load_lineage(path: Path = LINEAGE) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def normalize(plan: dict) -> dict:
    """The plan's conditions: defaults filled in, the name dropped, every model checkpoint replaced by
    @CHECKPOINT@ (sampling mode kept), seed_base / same_seed dropped when an explicit seed list is used."""
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
    if plan.get("name") != NAME or any(m["p0"].get("checkpoint") != CKPT for m in plan["matchup"]):
        raise Refused(f"template {spec['template']}: name / every p0 checkpoint must be the placeholders")
    h = conditions_sha256(plan)
    if h != spec["conditions_sha256"]:
        raise Refused(f"template {spec['template']} conditions {h[:12]} differ from lineage.toml {spec['conditions_sha256'][:12]}")
    n = normalize(plan)
    pinned = {"zero_last_failed": n["zero_last_failed"], "max_frames": n["max_frames"], "games": sum(m["games"] for m in n["matchup"]),
              "sample": sorted({m["p0"]["sample"] for m in n["matchup"]})}
    want = {"zero_last_failed": spec["zero_last_failed"], "max_frames": spec["max_frames"], "games": spec["games"], "sample": sorted(spec["sample"])}
    if pinned != want:
        raise Refused(f"template {spec['template']}: {pinned} differs from lineage.toml {want}")
    return plan


def render(lin: dict, plan_id: str, checkpoint: str, name: str) -> str:
    check_template(lin, plan_id)
    text, _ = template(lin, plan_id)
    out = text.replace(f'name = "{NAME}"', f'name = "{name}"', 1).replace(f'"{CKPT}"', f'"{checkpoint}"')
    tomllib.loads(out)
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


def run_of(checkpoint: str) -> str:
    parts = [s for s in checkpoint.replace("\\", "/").split("/") if s]
    i = next((k for k, s in enumerate(parts) if s.startswith("ckpt-")), len(parts) - 1)
    return parts[i - 1] if i > 0 else parts[-1]


def check_checkpoint(lin: dict, lineage: str, pt: dict, checkpoint: str) -> None:
    """Checkpoint 1 points take exactly their recorded checkpoint; other lineages a run with their prefix."""
    if lineage not in lin["lineage"]:
        raise Refused(f"no lineage {lineage!r}")
    if lineage == "ck1":
        # A job-folder copy of the checkpoint (eval/bc-e2-games/<run>__<ckpt>/ckpt) counts when its weights are the same.
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


def git_state() -> dict:
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    dirty = git("status", "--porcelain", "--", "tools/eval", "config/plans/protocol", "config/protocol")
    return {"commit": git("rev-parse", "HEAD"), "protocol_files_dirty": bool(dirty)}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- T1 / OSS bot builds


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
    cmd = "; ".join(f'f={bots_dir}/{b}/module.so; [ -f "$f" ] && echo "{b} $(sha256sum < "$f" | cut -c1-64)"' for b in bots)
    p = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", ssh, cmd], capture_output=True, text=True, timeout=120)
    out = {}
    for line in p.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def check_bots(hosts: list[dict], bots: list[str], given: Path | None = None) -> dict:
    """Every game host's OSS bot binaries of this job are accepted NDEBUG builds (t1_bots.toml). Hashes
    come from `local = true` (this PC's URANIUM_OSS_BOTS_DIR), `ssh = "<alias>"` (module.so under
    `oss_bots_dir`, default ~/Workspace/Uranium/build/oss_bots) or a file of `<addr> <bot> <sha256>` lines."""
    accepted = tomllib.loads(BOTS.read_text(encoding="utf-8"))["bot"]
    seen: dict[str, dict[str, str]] = {}
    if given is not None:
        for line in given.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 3 and not line.startswith("#"):
                seen.setdefault(parts[0], {})[parts[1]] = parts[2]
    for h in hosts:
        if h["addr"] in seen:
            continue
        if h.get("local"):
            seen[h["addr"]] = local_bot_hashes(bots)
        elif h.get("ssh"):
            seen[h["addr"]] = remote_bot_hashes(h["ssh"], h.get("oss_bots_dir", "~/Workspace/Uranium/build/oss_bots"), bots)
    problems = []
    for h in hosts:
        got = seen.get(h["addr"])
        if got is None:
            problems.append(f"{h['addr']}: no way to read its bot hashes (set local / ssh in the host list or pass --bot-hashes)")
            continue
        for b in bots:
            if b not in got:
                problems.append(f"{h['addr']}: {b} missing")
            elif got[b] not in accepted.get(b, {}).get("sha256", []):
                problems.append(f"{h['addr']}: {b} {got[b][:12]} is not an accepted NDEBUG build (config/protocol/t1_bots.toml)")
    if problems:
        raise Refused("bot builds:\n  " + "\n  ".join(problems))
    return seen


def oss_bots(plans: list[dict]) -> list[str]:
    return sorted({m["p1"]["name"] for p in plans for m in p["matchup"] if m["p1"]["kind"] == "oss"})


# ---------------------------------------------------------------- commands


def cmd_job(a) -> None:
    lin = load_lineage()
    pt = point(lin, a.point)
    check_checkpoint(lin, a.lineage, pt, a.checkpoint)
    metrics = [m.strip() for m in a.metrics.split(",") if m.strip()]
    stages: list[tuple[str, str]] = []  # (metric, plan id)
    for m in metrics:
        spec = check_metric(lin, pt, m)
        if spec["kind"] != "games":
            raise Refused(f"metric {m} is not a game metric (use the bc-eval command)")
        stages += [(m, p) for p in metric_plans(lin, pt, m)]
    git = git_state()
    if git["protocol_files_dirty"] and not a.allow_dirty:
        raise Refused("tools/eval, config/plans/protocol or config/protocol has uncommitted changes (--allow-dirty to record anyway)")
    sha = checkpoint_sha256(a.checkpoint)
    label = a.label or run_of(a.checkpoint)
    # A job's stages that share the model must agree on zero_last_failed (crates/uranium-runner eval.rs):
    # one job per value, as the Checkpoint 1 jobs did (rush + macro in one job, TvZ apart).
    groups: dict[bool, list[tuple[str, str, str]]] = {}
    for m, p in stages:
        text = render(lin, p, a.checkpoint.replace("\\", "/"), f"{label}-{p.replace('_', '-')}")
        groups.setdefault(lin["plan"][p]["zero_last_failed"], []).append((m, p, text))
    hosts = load_hosts(Path(a.hosts))
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
            "protocol": "config/protocol/lineage.toml", "lineage": a.lineage, "point": pt["id"], "family": pt["family"],
            "checkpoint": a.checkpoint, "checkpoint_sha256": sha, "checkpoint_id": f"{Path(a.checkpoint).name}@{sha[:12]}",
            "zero_last_failed": zlf, "git": git, "plans": rec_plans,
            "bot_hashes": {h: {b: v for b, v in hs.items() if b in gbots} for h, hs in bot_hashes.items()} if gbots else {},
            "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        (d / "protocol.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
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
    """Problems of an eval job against the protocol (empty = conforms)."""
    problems = []
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
        cks = {m["p0"]["checkpoint"] for m in plan["matchup"]}
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


def cmd_bc_eval(a) -> None:
    """Prints the pinned bc-eval / per_action command of a teacher-forced metric (run it on a host of the
    pinned backend)."""
    lin = load_lineage()
    pt = point(lin, a.point)
    check_checkpoint(lin, a.lineage, pt, a.checkpoint)
    m = check_metric(lin, pt, a.metric)
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
    m = lin["metric"][a.metric]
    d = json.loads(Path(a.output).read_text(encoding="utf-8"))
    want = {"split": m["split"], "games": m.get("expect_games", m.get("games")), "selection_seed": m.get("seed")}
    got = {k: d.get(k) for k in want}
    if got != want:
        raise SystemExit(f"MISMATCH {a.output}: {got} != pinned {want}")
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
        print(f"{p['id']:8} {p['family']:3} {p['ck1_checkpoint']:48} metrics {','.join(p['metrics'])}  optional {','.join(p.get('optional', []))}")


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
    s.add_argument("--allow-dirty", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(f=cmd_job)
    s = sub.add_parser("render", help="one plan from its template (e.g. a DAgger generation plan)")
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
