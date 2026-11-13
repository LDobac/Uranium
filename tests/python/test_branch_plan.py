"""051 branch-game plan tools (tools/eval/branch_plan.py, branch_collect.py) on synthetic trunk and branch jobs:
the trunk plan (051-R11 ①), points / branch list / branch plan from trunk results with every 051-R10 refusal,
determinism and repeat-outermost order, replacement trunks, the branch records (051-R12) and the exclusion
of branch-game stages from the evaluation tables (050). CPU, no game data."""

import json
import shutil
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "eval"))

import branch_collect as bcol  # noqa: E402
import branch_common as bc  # noqa: E402
import branch_plan as bp  # noqa: E402
import eval_common as ec  # noqa: E402
import protocol  # noqa: E402
import tables  # noqa: E402

CONFIG = ROOT / "config" / "branch_b1.toml"
CKPT = "checkpoints/m2-s1/ckpt-39096"
SCHEMAS = {"feature": 7, "action": 2}
T0 = "t0@9.9.9"
IDS = bc.action_ids()
BUNKER, MARINE = IDS["Build_Bunker"], IDS["Train_Marine"]
# Field sets of the U1 records (crates/uranium-env/src/branch.rs): PointRecord, TrunkRef, BranchPlanLine,
# BranchRecord, GameRef.
POINT_KEYS = {"point", "trunk", "seed", "map", "opponent", "condition", "checkpoint_sha256", "policy", "t0_policy", "git", "schemas",
              "backend", "precision", "infer_seed", "i_t", "f_t", "digest_t", "type_mask", "scalar_t", "trunk_action", "state",
              "state_sha256", "trunk_outcome", "valid", "skip_reason"}
TRUNK_KEYS = {"job", "stage", "plan_game", "hostd_game", "replay", "acts", "opponent_acts"}
LINE_KEYS = {"trunk", "point", "acts", "opponent_acts", "state", "state_sha256", "i_t", "f_t", "seed", "map", "opponent", "alt", "repeat"}
RECORD_KEYS = {"point", "alt", "rules", "repeat", "game", "checkpoint_sha256", "git", "valid", "invalid", "prefix", "state_check",
               "intervention", "samples", "metrics", "dilution", "end"}
GAME_KEYS = {"job", "stage", "plan_game", "hostd_game", "attempt", "replay", "host", "backend", "precision", "infer_seed"}


def cond(seed: int) -> str:
    """Stand-in for the 020-R21 generator: odd seeds hit."""
    return "hit" if seed % 2 else "miss"


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    monkeypatch.setattr(ec, "load_env", lambda: None)


def jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def act(i: int) -> dict:
    return {"cmds": [], "advance_frames": 8, "valid_until_frame": None, "surrender": None}


def digest(i: int, salt: int = 0) -> str:
    return f"{(i * 7919 + salt) & 0xFFFFFFFFFFFFFFFF:016x}"


class TrunkJob:
    """A played trunk job: the generated trunk plan, merged stage files, act logs, snapshot records, state files."""

    def __init__(self, root: Path, hit: int = 1, miss: int = 1):
        self.root, self.job = root, root / "eval" / "trunk-job"
        cfg = bp.load_config(CONFIG)
        text = bp.trunk_plan(cfg, "b1-trunk", cond, counts={"hit": hit, "miss": miss})
        (self.job / "plans").mkdir(parents=True)
        (self.job / "plans" / "trunk.toml").write_text(text, encoding="utf-8")
        (self.job / "job.toml").write_text("stages = ['plans/trunk.toml']\nhosts = 'hosts.toml'\n", encoding="utf-8")
        self.plan = tomllib.loads(text)
        self.games = bc.plan_games(self.plan)
        ck = root / CKPT
        ck.mkdir(parents=True)
        (ck / "model.safetensors").write_bytes(b"weights")
        self.ckpt_sha = bc.file_sha256(ck / "model.safetensors")
        self.sd = self.job / "b1-trunk"
        for n in range(1, len(self.games) + 1):
            self.play(n)
        self.merge()

    def play(self, n: int, marine: bool = True, frames=(3300, 4000), same=False, fire=(True, True)):
        """Game n: 12 observations, b1-p1 at i 5 (frame 3300), b1-p2 at i 8 (frame 4000)."""
        frame_of = {5: frames[0], 8: frames[1]}
        lines, snaps, opp = [], [], []
        f = 100
        for i in range(12):
            f = frame_of.get(i, f + 100)
            lines.append({"i": i, "frame": f, "digest": digest(i, n), "act": act(i),
                          "decision": {"action": 0, "delay_bucket": 1, "units": [], "rejected": False}, "state": None, "forced": False})
            opp.append({"i": i, "frame": f, "digest": digest(i, 1000 + n), "act": act(i), "forced": False})
        for rule, i, fired in (("b1-p1", 5, fire[0]), ("b1-p2", 8 if not same else 5, fire[1])):
            if not fired:
                continue
            sid = f"g{n}-o{i}"
            if same and snaps:
                snaps[0]["rules"].append(rule)
                continue
            body = f"state {n} {i}".encode()
            (self.sd / "states").mkdir(parents=True, exist_ok=True)
            (self.sd / "states" / f"{sid}.state").write_bytes(body)
            sha = bc.file_sha256(self.sd / "states" / f"{sid}.state")
            lines[i]["state"], lines[i]["state_sha256"] = sid, sha
            mask = [0, BUNKER] + ([MARINE] if marine else [])
            snaps.append({"rules": [rule], "state": sid, "state_sha256": sha, "i": i, "frame": lines[i]["frame"], "digest": lines[i]["digest"],
                          "type_mask": mask, "scalar": [0.5, float(n)]})
        end = {"obs": 12, "decisions": 12, "end_frame": 11568, "reason": "TimeLimitDraw", "player": 0, "players_index": 0,
               "policy": f"model:ckpt-39096@{self.ckpt_sha[:12]}:stochastic", "git": "abc", "feature_schema": 7, "action_schema": 2}
        jsonl(self.sd / "acts" / f"g{n}-p0.jsonl", [*lines, {"end": end}])
        oend = {**end, "decisions": 0, "player": 1, "players_index": 1, "policy": T0}
        jsonl(self.sd / "acts" / f"g{n}-p1.jsonl", [*opp, {"end": oend}])
        self.snaps = getattr(self, "snaps", {})
        self.snaps[n] = snaps

    def merge(self, outcomes=None):
        outcomes = outcomes or {}
        games, results = [], []
        for n in range(1, len(self.games) + 1):
            games.append({"game_id": n, "hostd_id": 100_000 + n, "outcome": outcomes.get(n, "Finished"), "host": "h", "backend": "rocm",
                          "opponent": "t0-zerg-rush@9.9.9", "map": "x"})
            results.append({"game_id": n, "slot": 0, "won": None, "end_frame": 11568, "reason": "TimeLimitDraw",
                            "model": {"scv_8m": 9, "scv_min_1m": 4, "bunker_s": 170.0}, "snapshots": self.snaps[n], "infer_seed": 5})
            results.append({"game_id": n, "slot": 1, "won": None, "model": None})
        jsonl(self.sd / "games.jsonl", games)
        jsonl(self.sd / "results.jsonl", results)


def args(tj: TrunkJob, out: Path, **kw):
    a = bp.argparse.Namespace(config=CONFIG, root=str(tj.root), trunk_job=str(tj.job), out=str(out), stage=["b1-trunk"], name=None,
                              alts=None, repeats=None, hit=1, miss=1, verify_state=False, hosts=None, replacements=None,
                              replacement_name=None, t0_policy=T0, schemas=None)
    for k, v in kw.items():
        setattr(a, k, v)
    a.condition_fn, a.schemas_fn = cond, lambda: dict(SCHEMAS)
    return a


def test_trunk_plan_051_r11():
    """051-R11 ① / 051-R15: per map the first hit / miss seeds of the rush5 T0 zerg_rush block in list order, the
    051-R1 settings, act_log and the config's snapshot rules; deterministic; excluded seeds skipped."""
    cfg = bp.load_config(CONFIG)
    text = bp.trunk_plan(cfg, "b1-trunk", cond)
    assert text == bp.trunk_plan(cfg, "b1-trunk", cond)
    p = tomllib.loads(text)
    assert (p["save_replay"], p["zero_last_failed"], p["surrender"], p["max_frames"], p["act_log"]) == (True, True, "off", 11568, True)
    assert [s["name"] for s in p["snapshots"]] == ["b1-p1", "b1-p2"]
    assert p["snapshots"][0] == {"name": "b1-p1", "from_frame": 3240, "to_frame": 5760, "any_open": ["Build_Bunker"]}
    rush = bp.rush_blocks(cfg)
    games = bc.plan_games(p)
    assert len(games) == 40
    for m in ("fighting_spirit", "circuit_breaker"):
        seeds = [g["seed"] for g in games if g["map"] == m]
        want = [s for s in rush[m] if cond(s) == "hit"][:10] + [s for s in rush[m] if cond(s) == "miss"][:10]
        assert seeds == sorted(want, key=rush[m].index), "list order"
    for g in games:
        assert g["p0"] == {"kind": "model", "checkpoint": CKPT, "sample": "stochastic"}
        assert g["p1"] == {"kind": "t0", "race": "zerg", "build": "zerg_rush"}
    first = games[0]["seed"]
    again = tomllib.loads(bp.trunk_plan(cfg, "x", cond, exclude={first}))
    assert first not in ec.plan_seeds(again)


def test_branch_plan_points_and_list_051_r10(tmp_path):
    """051-R10 / 051-R12: points.jsonl and the branch list carry the U1 fields, paths relative to the shared drive;
    repeats outermost, then points, then alternatives; a force closed at t drops that alternative there; the plan's
    game n is line n's seed / map / opponent; same inputs give the same bytes."""
    tj = TrunkJob(tmp_path)
    tj.play(2, marine=False)
    tj.merge()
    out = tmp_path / "eval" / "branch-job"
    s = bp.build(args(tj, out, hosts="hosts.toml"))
    points = bc.read_jsonl(out / "points.jsonl")
    assert len(points) == 8 and all(set(p) == POINT_KEYS and set(p["trunk"]) == TRUNK_KEYS for p in points)
    p = points[0]
    assert p["point"] == "trunk-job/b1-trunk/g1#b1-p1"
    assert p["trunk"]["acts"] == "eval/trunk-job/b1-trunk/acts/g1-p0.jsonl"
    assert p["state"] == "eval/trunk-job/b1-trunk/states/g1-o5.state"
    assert (p["i_t"], p["f_t"], p["opponent"], p["condition"], p["valid"]) == (5, 3300, "t0-zerg-rush", cond(p["seed"]), True)
    assert (p["checkpoint_sha256"], p["backend"], p["precision"], p["infer_seed"]) == (tj.ckpt_sha, "rocm", "f32", 5)
    assert p["trunk_outcome"]["held_8m"] is True and p["trunk_outcome"]["held_s"] is False
    assert p["trunk_outcome"]["end"] == {"result": "draw", "end_frame": 11568, "reason": "TimeLimitDraw", "before_horizon": False}
    lines = bc.read_jsonl(out / "plans" / "branches_plan.jsonl")
    assert all(set(x) == LINE_KEYS for x in lines)
    # 4 trunks x 2 points x 4 alternatives x 4 repeats, minus force_marine at game 2's two points.
    assert len(lines) == s["games"] == (4 * 2 * 4 - 2) * 4
    assert s["dropped"] == [{"point": f"trunk-job/b1-trunk/g2#{r}", "alt": "force_marine", "closed": ["Train_Marine"]} for r in ("b1-p1", "b1-p2")]
    assert [x["repeat"] for x in lines] == sorted(x["repeat"] for x in lines), "repeats outermost"
    per = len(lines) // 4
    assert [(x["point"], x["alt"]["name"]) for x in lines[:per]] == [(x["point"], x["alt"]["name"]) for x in lines[per:2 * per]]
    assert [x["alt"]["name"] for x in lines[:4]] == ["none", "force_bunker", "ban_bunker_90s", "force_marine"]
    assert lines[1]["alt"]["rules"] == [{"op": "force", "action": "Build_Bunker", "from_s": 0, "to_s": 0}]
    assert lines[0]["alt"] == {"name": "none", "rules": []}
    plan = tomllib.loads((out / "plans" / "branch-b1.toml").read_text(encoding="utf-8"))
    assert plan["branches"] == "branches_plan.jsonl" and plan["surrender"] == "off" and plan["max_frames"] == 11568 and plan["act_log"]
    assert "verify_state" not in plan and "snapshots" not in plan
    games = bc.plan_games(plan)
    assert [(g["seed"], g["map"]) for g in games] == [(x["seed"], x["map"]) for x in lines]
    assert all(g["p1"] == {"kind": "t0", "race": "zerg", "build": "zerg_rush"} and g["p0"]["checkpoint"] == CKPT for g in games)
    assert (out / "job.toml").read_text(encoding="utf-8") == "stages = [\"plans/branch-b1.toml\"]\nhosts = \"hosts.toml\"\n"
    # Determinism: the same inputs again, byte for byte.
    out2 = tmp_path / "eval" / "branch-job2"
    bp.build(args(tj, out2, hosts="hosts.toml"))
    for f in ("points.jsonl", "plans/branches_plan.jsonl", "plans/branch-b1.toml", "job.toml"):
        assert (out / f).read_bytes() == (out2 / f).read_bytes(), f
    # 051-AC6 options: two alternatives, one repeat, verify_state.
    out3 = tmp_path / "eval" / "ac6"
    s = bp.build(args(tj, out3, alts="none,force_bunker", repeats=1, verify_state=True))
    assert s["games"] == 4 * 2 * 2
    assert tomllib.loads((out3 / "plans" / "branch-b1.toml").read_text(encoding="utf-8"))["verify_state"] is True


@pytest.mark.parametrize("case", ["incomplete", "state_sha", "checkpoint", "t0", "schemas", "not_fired", "same_decision", "aborted", "line"])
def test_branch_plan_refusals_051_r10(tmp_path, case):
    """051-R10: each refusal leaves the trunk unused with its reason; the cell is short and --replacements writes the
    next unused seed of that map and condition (051-R11 ①)."""
    tj = TrunkJob(tmp_path)
    n = 1
    acts = tj.sd / "acts" / f"g{n}-p0.jsonl"
    if case == "incomplete":
        acts.write_text("".join(acts.read_text(encoding="utf-8").splitlines(keepends=True)[:-1]), encoding="utf-8")
    elif case == "state_sha":
        (tj.sd / "states" / f"g{n}-o5.state").write_bytes(b"other")
    elif case == "checkpoint":
        rows = bc.read_jsonl(acts)
        rows[-1]["end"]["policy"] = "model:ckpt-39096@000000000000:stochastic"
        jsonl(acts, rows)
    elif case == "t0":
        rows = bc.read_jsonl(tj.sd / "acts" / f"g{n}-p1.jsonl")
        rows[-1]["end"]["policy"] = "t0@0.0.1"
        jsonl(tj.sd / "acts" / f"g{n}-p1.jsonl", rows)
    elif case == "schemas":
        rows = bc.read_jsonl(acts)
        rows[-1]["end"]["feature_schema"] = 6
        jsonl(acts, rows)
    elif case == "not_fired":
        tj.play(n, fire=(True, False))
        tj.merge()
    elif case == "same_decision":
        tj.play(n, same=True)
        tj.merge()
    elif case == "aborted":
        tj.merge({n: "Aborted"})
    elif case == "line":
        rows = bc.read_jsonl(acts)
        rows[5]["digest"] = digest(99)
        jsonl(acts, rows)
    out = tmp_path / "eval" / "branch-job"
    rep = tmp_path / "eval" / "trunk-job" / "plans" / "trunk-r.toml"
    s = bp.build(args(tj, out, replacements=str(rep)))
    t = s["trunks"][0]
    assert not t["used"] and s["replaced"] == 1
    why = json.dumps(t)
    want = {"incomplete": "end line", "state_sha": "sha256 differs", "checkpoint": "checkpoint sha256", "t0": "T0 policy",
            "schemas": "schemas", "not_fired": "did not fire", "same_decision": "same decision", "aborted": "game Aborted",
            "line": "not the snapshot's"}[case]
    assert want in why, why
    assert all(not x["point"].startswith("trunk-job/b1-trunk/g1#") for x in bc.read_jsonl(out / "plans" / "branches_plan.jsonl"))
    mine = [p for p in bc.read_jsonl(out / "points.jsonl") if p["point"].startswith("trunk-job/b1-trunk/g1#")]
    assert all(not p["valid"] and p["skip_reason"] for p in mine), "a replaced trunk's points are not valid"
    if case in ("state_sha", "not_fired", "same_decision"):
        assert any(p["skip_reason"] == "trunk replaced (051-R11 1)" for p in mine)
    cell = f"{tj.games[0]['map']}/{cond(tj.games[0]['seed'])}"
    assert s["cells"][cell]["short"] == 1 and s["replacements"]["short"] == {cell: 1}
    r = tomllib.loads(rep.read_text(encoding="utf-8"))
    used = set(ec.plan_seeds(tj.plan))
    (seed,) = ec.plan_seeds(r)
    rush = bp.rush_blocks(bp.load_config(CONFIG))[tj.games[0]["map"]]
    assert seed == next(x for x in rush if x not in used and cond(x) == cond(tj.games[0]["seed"]))
    assert r["snapshots"] == tj.plan["snapshots"] and r["max_frames"] == 11568
    assert r["name"] == "branch-b1-trunk-r"
    if case == "incomplete":
        # A second replacement round: the first replacement stage (`rep`, in the trunk job's plans) is a
        # stage of the trunk job, so the new one gets -r2.
        assert rep.parent == tj.job / "plans"
        (tj.job / "job.toml").write_text("stages = ['plans/trunk.toml', 'plans/trunk-r.toml']\nhosts = 'hosts.toml'\n", encoding="utf-8")
        rep2 = tmp_path / "trunk-r2.toml"
        bp.build(args(tj, out, replacements=str(rep2)))
        assert tomllib.loads(rep2.read_text(encoding="utf-8"))["name"] == "branch-b1-trunk-r2"
        with pytest.raises(bp.Refused, match="already has that stage"):
            bp.build(args(tj, out, replacements=str(rep2), replacement_name="branch-b1-trunk-r"))


def branch_job(tmp_path: Path, tj: TrunkJob) -> Path:
    out = tmp_path / "eval" / "branch-job"
    bp.build(args(tj, out, alts="none,force_bunker", repeats=1, hosts="h.toml"))
    return out


def test_branch_collect_records_051_r12(tmp_path):
    """051-R12: one U1 BranchRecord per game with a branch outcome (fields as the session wrote them), the opponent
    prefix against the trunk's log, a mismatch before f_t marked invalid; games without an outcome counted."""
    tj = TrunkJob(tmp_path)
    out = branch_job(tmp_path, tj)
    lines = bc.read_jsonl(out / "plans" / "branches_plan.jsonl")
    sd = out / "branch-b1"
    games, results = [], []
    outcome = {"valid": True, "invalid": None, "prefix_obs": 6, "prefix_matched": 6, "state_check": None,
               "intervention": [{"op": "force", "action": "Build_Bunker", "from_s": 0, "to_s": 0, "in_window": 1, "applied": 1,
                                 "force_frame": 3300, "missed": False, "rejected": False}],
               "samples": {"t60": {"frame": 4800, "scalar": [1.0]}, "t120": None, "t240": None, "horizon": None, "d8": None, "d32": None},
               "metrics": {"held_8m": True, "held_s": True}, "dilution": [{"action": "Build_Bunker", "first_frame": None, "picks_30s": 0}],
               "end": {"result": "draw", "end_frame": 11568, "reason": "TimeLimitDraw", "before_horizon": False}}
    for n, ln in enumerate(lines, start=1):
        if n == 3:
            continue  # not played
        games.append({"game_id": n, "hostd_id": 10_000_000 + 100_000 + n, "outcome": "Finished", "host": "h", "backend": "cuda"})
        if n == 4:
            results.append({"game_id": n, "model": {"x": 1}})  # no branch outcome
            continue
        if n == 7:  # aborted: the session never saw GameEnd (valid, no end) -> no outcome
            results.append({"game_id": n, "model": {"x": 1}, "branch": {**outcome, "end": None}})
            continue
        results.append({"game_id": n, "model": {"x": 1}, "branch": outcome, "infer_seed": 3})
        trunk = bc.read_jsonl(bc.on_shared(tmp_path, ln["opponent_acts"]))[:-1]
        mine = [dict(x) for x in trunk]
        # The opponent's observations differ from the first one after f_t (game 2: before f_t).
        cut = 2 if n == 2 else next(x["i"] for x in trunk if x["frame"] > ln["f_t"])
        for x in mine[cut:]:
            x["digest"] = digest(x["i"], 555)
        if n != 6:  # game 6: the opponent's act log is missing
            jsonl(sd / "acts" / f"g{n}-p1.jsonl", [*mine, {"end": {"policy": T0}}])
        # game 5 ran another checkpoint (051-R1 ④)
        policy = f"model:ckpt-39096@{'0' * 12 if n == 5 else tj.ckpt_sha[:12]}:stochastic"
        jsonl(sd / "acts" / f"g{n}-p0.jsonl", [{"end": {"git": "def", "policy": policy}}])
    (sd / "replays").mkdir(parents=True)
    (sd / "replays" / "g1.rep").write_bytes(b"rep")
    jsonl(sd / "games.jsonl", games)
    jsonl(sd / "results.jsonl", results)
    s = bcol.collect(out, None, tmp_path)
    recs = bc.read_jsonl(out / "branches.jsonl")
    assert len(recs) == len(lines) - 3 == s["records"]
    assert 7 not in {r["game"]["plan_game"] for r in recs}
    assert all(set(r) == RECORD_KEYS and set(r["game"]) == GAME_KEYS for r in recs)
    r = recs[0]
    assert (r["point"], r["alt"], r["repeat"], r["valid"]) == (lines[0]["point"], "none", 1, True)
    assert r["game"] == {"job": "branch-job", "stage": "branch-b1", "plan_game": 1, "hostd_game": 10_100_001, "attempt": 1,
                         "replay": "eval/branch-job/branch-b1/replays/g1.rep", "host": "h", "backend": "cuda", "precision": "bf16-mixed", "infer_seed": 3}
    assert (r["checkpoint_sha256"], r["git"]) == (tj.ckpt_sha, "def")
    assert r["prefix"]["opponent"] == {"obs": 5, "matched": 5, "first_mismatch_frame": bc.read_jsonl(bc.on_shared(tmp_path, lines[0]["opponent_acts"]))[6]["frame"]}
    assert r["prefix"]["opponent"]["first_mismatch_frame"] > lines[0]["f_t"]
    for k in ("intervention", "samples", "metrics", "dilution", "end"):
        assert r[k] == outcome[k], k
    bad = recs[1]
    assert not bad["valid"] and bad["invalid"]["reason"] == "digest" and bad["prefix"]["opponent"]["matched"] == 2
    assert bad["invalid"]["side"] == "opponent"
    by_game = {r["game"]["plan_game"]: r for r in recs}
    assert by_game[5]["invalid"]["reason"] == "missing_input" and "policy" in by_game[5]["invalid"]["policy_mismatch"]
    assert by_game[6]["valid"] and by_game[6]["prefix"]["opponent"] is None
    assert s["invalid_by_reason"] == {"digest": 1, "missing_input": 1}
    assert s["opponent_prefix_unchecked"] == [6] and s["act_log_incomplete"] == {"ours": [], "opponent": [6]}
    assert [m["game"] for m in s["missing"]] == [3, 4, 7]
    first = (out / "branches.jsonl").read_bytes()
    assert bcol.collect(out, None, tmp_path) == s and (out / "branches.jsonl").read_bytes() == first, "deterministic"


def test_branch_stages_left_out_of_tables_050(tmp_path):
    """050 "분기 게임(051)": a stage with a 020-R22 key is never a protocol plan nor a table, --kind included."""
    plan = {"name": "b", "act_log": True, "matchup": [{"map": "both", "games": 1, "p0": {"kind": "t0", "race": "terran"}, "p1": {"kind": "t0", "race": "zerg"}}]}
    lin = protocol.load_lineage()
    assert ec.is_branch_plan(plan) and not ec.is_branch_plan({**plan, "act_log": False}) and not ec.is_branch_plan({**plan, "act_log": False, "snapshots": []})
    assert protocol.identify(lin, plan) is None
    with pytest.raises(tables.UnknownPlan, match="branch-game"):
        tables.stage_kind(lin, plan, {"b": "winrate"})
    job = tmp_path / "job"
    (job / "plans").mkdir(parents=True)
    (job / "plans" / "b.toml").write_text("name = 'b'\nact_log = true\n[[matchup]]\nmap = 'both'\ngames = 1\np0 = {kind = 't0', race = 'terran'}\np1 = {kind = 't0', race = 'zerg'}\n", encoding="utf-8")
    (job / "job.toml").write_text("stages = ['plans/b.toml']\nhosts = 'h.toml'\n", encoding="utf-8")
    assert tables.job_tables(job, tmp_path / "cache", override={"b": "winrate"}, lin=lin) == {}


def test_act_log_cut_short_is_incomplete_051_r2(tmp_path):
    """051-R2: a log whose last line was cut mid-write is incomplete: lenient reads keep the lines before it
    (no end line), strict reads refuse it (no JSONDecodeError escapes)."""
    p = tmp_path / "g1-p1.jsonl"
    p.write_text('{"i":0,"frame":1,"digest":"a"}\n{"i":1,"frame":9,"digest":"b"}\n{"i":2,"fra', encoding="utf-8")
    log = bc.ActLog.read(p, lenient=True)
    assert log.end is None and [x["i"] for x in log.lines] == [0, 1]
    with pytest.raises(bc.Refused, match="not JSON"):
        bc.ActLog.read(p)


def test_compare_prefix_matches_rust_051_r12():
    """bc.compare_prefix is uranium_env::branch::compare_prefix (branch_compare_prefix_051_r12 cases)."""
    L = lambda fr, d: [{"frame": f, "digest": x} for f, x in zip(fr, d)]  # noqa: E731
    t = L([1, 2, 3, 4], "abcd")
    assert bc.compare_prefix(t, t, 3) == {"obs": 2, "matched": 2, "first_mismatch_frame": None}
    assert bc.compare_prefix(t, L([1, 2, 3, 4], "abxd"), 4) == {"obs": 3, "matched": 2, "first_mismatch_frame": 3}
    assert bc.compare_prefix(t, L([1, 2], "ab"), 4) == {"obs": 3, "matched": 2, "first_mismatch_frame": 3}
    assert bc.compare_prefix(L([1, 2], "ab"), t, 9)["first_mismatch_frame"] == 3


def test_config_checks_051_r7(tmp_path):
    """The alternative config: 051-R7 rules with known actions and non-empty windows, no NoOp ban."""
    base = CONFIG.read_text(encoding="utf-8")
    p = tmp_path / "c.toml"
    for old, new in (('"Train_Marine"', '"Train_Marne"'), ("from_s = 0, to_s = 90", "from_s = 90, to_s = 0"),
                     ('op = "ban", action = "Build_Bunker"', 'op = "ban", action = "NoOp"'), ('name = "force_marine"', 'name = "none"')):
        assert old in base
        p.write_text(base.replace(old, new), encoding="utf-8")
        with pytest.raises(bc.Refused):
            bp.load_config(p)
    shutil.copy(CONFIG, p)
    assert [a["name"] for a in bp.load_config(p)["alternative"]] == ["none", "force_bunker", "ban_bunker_90s", "force_marine"]
