import json
import shutil
import tomllib
from pathlib import Path

import pytest

from eval_helpers import ROOT

from uranium.measure import branch_collect as bcol
from uranium.measure import branch_common as bc
from uranium.measure import branch_plan as bp
from uranium.measure import eval_common as ec
from uranium.measure import protocol
from uranium.measure import tables

CONFIG = ROOT / "config" / "branch_b1.toml"
CKPT = "checkpoints/m2-gn-t1e2/ckpt-2844"
SCHEMAS = {"feature": 7, "action": 2}
T0 = "t0@9.9.9"
IDS = bc.action_ids()
BUNKER, MARINE = IDS["Build_Bunker"], IDS["Train_Marine"]
POINT_KEYS = {"point", "trunk", "seed", "map", "opponent", "condition", "checkpoint_sha256", "policy", "t0_policy", "git", "schemas",
              "backend", "precision", "infer_seed", "i_t", "f_t", "digest_t", "type_mask", "scalar_t", "trunk_action", "state",
              "state_sha256", "trunk_outcome", "valid", "skip_reason", "policy_top"}
TRUNK_KEYS = {"job", "stage", "plan_game", "hostd_game", "replay", "acts", "opponent_acts"}
LINE_KEYS = {"trunk", "point", "acts", "opponent_acts", "state", "state_sha256", "i_t", "f_t", "seed", "map", "opponent", "alt", "repeat"}
RECORD_KEYS = {"point", "alt", "rules", "repeat", "game", "checkpoint_sha256", "git", "valid", "invalid", "prefix", "state_check",
               "intervention", "samples", "metrics", "dilution", "end", "labels", "ops", "own_horizon", "score"}
GAME_KEYS = {"job", "stage", "plan_game", "hostd_game", "attempt", "replay", "host", "backend", "precision", "infer_seed"}


def cond(seed: int) -> str:
    return "hit" if seed % 2 else "miss"


def jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def act(i: int) -> dict:
    return {"cmds": [], "advance_frames": 8, "valid_until_frame": None, "surrender": None}


def digest(i: int, salt: int = 0) -> str:
    return f"{(i * 7919 + salt) & 0xFFFFFFFFFFFFFFFF:016x}"


class TrunkJob:
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
        self.ckpt_sha = ec.file_sha256(ck / "model.safetensors")
        self.sd = self.job / "b1-trunk"
        for n in range(1, len(self.games) + 1):
            self.play(n)
        self.merge()

    def play(self, n: int, marine: bool = True, frames=(3300, 4000), same=False, fire=(True, True)):
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
            sha = ec.file_sha256(self.sd / "states" / f"{sid}.state")
            lines[i]["state"], lines[i]["state_sha256"] = sid, sha
            mask = [0, BUNKER] + ([MARINE] if marine else [])
            snaps.append({"rules": [rule], "state": sid, "state_sha256": sha, "i": i, "frame": lines[i]["frame"], "digest": lines[i]["digest"],
                          "type_mask": mask, "scalar": [0.5, float(n)]})
        end = {"obs": 12, "decisions": 12, "end_frame": 11568, "reason": "TimeLimitDraw", "player": 0, "players_index": 0,
               "policy": f"model:ckpt-2844@{self.ckpt_sha[:12]}:stochastic", "git": "abc", "feature_schema": 7, "action_schema": 2}
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
    cfg = bp.load_config(CONFIG)
    text = bp.trunk_plan(cfg, "b1-trunk", cond)
    assert text == bp.trunk_plan(cfg, "b1-trunk", cond)
    p = tomllib.loads(text)
    assert (p["save_replay"], p["zero_last_failed"], p["surrender"], p["max_frames"], p["act_log"]) == (True, True, "off", 11568, True)
    assert [s["name"] for s in p["snapshots"]] == ["b1-p1", "b1-p2"]
    assert p["snapshots"][0] == {"name": "b1-p1", "from_frame": 3240, "to_frame": 5760, "any_open": ["Build_Bunker"]}
    assert p["snapshots"][1] == {"name": "b1-p2", "from_frame": 3960, "to_frame": 5760, "any_open": ["Build_Bunker"], "after_rule": "b1-p1", "after_s": 30}
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


def test_exclude_job_resolves_under_shared_or_refuses_051_r11(tmp_path, monkeypatch):
    tj = TrunkJob(tmp_path)
    rel = tj.job.relative_to(tmp_path)
    assert bp.job_seeds(tj.job)
    assert bp.excluded_seeds([rel], tmp_path) == bp.job_seeds(tj.job)
    monkeypatch.delenv("URANIUM_SHARED_DIR", raising=False)
    assert bp.excluded_seeds([tj.job], None) == bp.job_seeds(tj.job)
    with pytest.raises(bc.Refused, match="no job.toml"):
        bp.excluded_seeds([Path("eval/missing")], tmp_path)
    with pytest.raises(SystemExit, match="no job.toml"):
        bp.main(["trunk", "--out", str(tmp_path / "trunk.toml"), "--root", str(tmp_path), "--exclude-job", "eval/missing"])


def test_branch_plan_points_and_list_051_r10(tmp_path):
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
    out2 = tmp_path / "eval" / "branch-job2"
    bp.build(args(tj, out2, hosts="hosts.toml"))
    for f in ("points.jsonl", "plans/branches_plan.jsonl", "plans/branch-b1.toml", "job.toml"):
        assert (out / f).read_bytes() == (out2 / f).read_bytes(), f
    out3 = tmp_path / "eval" / "ac6"
    s = bp.build(args(tj, out3, alts="none,force_bunker", repeats=1, verify_state=True))
    assert s["games"] == 4 * 2 * 2
    assert tomllib.loads((out3 / "plans" / "branch-b1.toml").read_text(encoding="utf-8"))["verify_state"] is True


@pytest.mark.parametrize("case", ["incomplete", "state_sha", "checkpoint", "t0", "schemas", "not_fired", "same_decision", "aborted", "line"])
def test_branch_plan_refusals_051_r10(tmp_path, case):
    tj = TrunkJob(tmp_path)
    n = 1
    acts = tj.sd / "acts" / f"g{n}-p0.jsonl"
    if case == "incomplete":
        acts.write_text("".join(acts.read_text(encoding="utf-8").splitlines(keepends=True)[:-1]), encoding="utf-8")
    elif case == "state_sha":
        (tj.sd / "states" / f"g{n}-o5.state").write_bytes(b"other")
    elif case == "checkpoint":
        rows = bc.read_jsonl(acts)
        rows[-1]["end"]["policy"] = "model:ckpt-2844@000000000000:stochastic"
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
            continue
        games.append({"game_id": n, "hostd_id": 10_000_000 + 100_000 + n, "outcome": "Finished", "host": "h", "backend": "cuda"})
        if n == 4:
            results.append({"game_id": n, "model": {"x": 1}})
            continue
        if n == 7:
            results.append({"game_id": n, "model": {"x": 1}, "branch": {**outcome, "end": None}})
            continue
        results.append({"game_id": n, "model": {"x": 1}, "branch": outcome, "infer_seed": 3})
        trunk = bc.read_jsonl(bc.on_shared(tmp_path, ln["opponent_acts"]))[:-1]
        mine = [dict(x) for x in trunk]
        cut = 2 if n == 2 else next(x["i"] for x in trunk if x["frame"] > ln["f_t"])
        for x in mine[cut:]:
            x["digest"] = digest(x["i"], 555)
        if n != 6:
            jsonl(sd / "acts" / f"g{n}-p1.jsonl", [*mine, {"end": {"policy": T0}}])
        policy = f"model:ckpt-2844@{'0' * 12 if n == 5 else tj.ckpt_sha[:12]}:stochastic"
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
    p = tmp_path / "g1-p1.jsonl"
    p.write_text('{"i":0,"frame":1,"digest":"a"}\n{"i":1,"frame":9,"digest":"b"}\n{"i":2,"fra', encoding="utf-8")
    log = bc.ActLog.read(p, lenient=True)
    assert log.end is None and [x["i"] for x in log.lines] == [0, 1]
    with pytest.raises(bc.Refused, match="not JSON"):
        bc.ActLog.read(p)


def test_compare_prefix_matches_rust_051_r12():
    L = lambda fr, d: [{"frame": f, "digest": x} for f, x in zip(fr, d)]  # noqa: E731
    t = L([1, 2, 3, 4], "abcd")
    assert bc.compare_prefix(t, t, 3) == {"obs": 2, "matched": 2, "first_mismatch_frame": None}
    assert bc.compare_prefix(t, L([1, 2, 3, 4], "abxd"), 4) == {"obs": 3, "matched": 2, "first_mismatch_frame": 3}
    assert bc.compare_prefix(t, L([1, 2], "ab"), 4) == {"obs": 3, "matched": 2, "first_mismatch_frame": 3}
    assert bc.compare_prefix(L([1, 2], "ab"), t, 9)["first_mismatch_frame"] == 3


def test_config_checks_051_r7(tmp_path):
    base = CONFIG.read_text(encoding="utf-8")
    p = tmp_path / "c.toml"
    for old, new in (('"Train_Marine"', '"Train_Marne"'), ("from_s = 0, to_s = 90", "from_s = 90, to_s = 0"),
                     ('op = "ban", action = "Build_Bunker"', 'op = "ban", action = "NoOp"'), ('name = "force_marine"', 'name = "none"'),
                     ('after_rule = "b1-p1"', 'after_rule = "b1-p2"'), ('after_s = 30\n', '\n'), ('after_s = 30\n', 'after_s = 30.5\n'),
                     ('after_s = 30\n', 'after_s = "30"\n')):
        assert old in base
        p.write_text(base.replace(old, new), encoding="utf-8")
        with pytest.raises(bc.Refused):
            bp.load_config(p)
    shutil.copy(CONFIG, p)
    assert [a["name"] for a in bp.load_config(p)["alternative"]] == ["none", "force_bunker", "ban_bunker_90s", "force_marine"]


def test_053_R2_trunk_held_8m_counts_a_win_before_8m():
    base = {"model": {"scv_8m": None, "scv_min_1m": 2}, "end_frame": 9000, "reason": "Elimination"}
    assert bp.trunk_outcome({**base, "won": True}, 11520)["held_8m"] is True
    assert bp.trunk_outcome({**base, "won": False}, 11520)["held_8m"] is False
    assert bp.trunk_outcome({**base, "won": True, "end_frame": 12000}, 11520)["held_8m"] is False
    assert bp.trunk_outcome({**base, "won": None, "model": {"scv_8m": 8}}, 11520)["held_8m"] is True


B2 = ROOT / "config" / "branch_b2.toml"
F = "checkpoints/m2-teacher-t2/ckpt-3258"
OPS = {a: IDS[a] for a in ("Train_SCV", "Build_SupplyDepot", "Build_Barracks", "Build_Refinery")}
B2_RULES = ("b2-p1", "b2-p2", "b2-p3")


def test_b2_trunk_plan_051_r16():
    cfg = bp.load_config(B2)
    text = bp.mix_trunk_plan(cfg, "branch-b2-trunk")
    assert text == bp.mix_trunk_plan(cfg, "branch-b2-trunk"), "deterministic"
    p = tomllib.loads(text)
    assert (p["save_replay"], p["zero_last_failed"], p["surrender"], p["max_frames"], p["act_log"]) == (True, True, "off", 17280 + 48, True)
    assert [s["name"] for s in p["snapshots"]] == list(B2_RULES)
    assert all((s["from_frame"], s["to_frame"], s["min_open_ops"]) == (4320, 11520, 3) for s in p["snapshots"])
    assert [(s.get("after_rule"), s.get("after_s"), s.get("after_gap", False)) for s in p["snapshots"]] == [(None, None, False), ("b2-p1", 60, True), ("b2-p2", 60, True)]
    games = bc.plan_games(p)
    assert len(games) == 100 and ec.plan_seeds(p) == [bp.rcp.branch_b2_seed(k) for k in range(1, 101)]
    by = lambda key: {k: sum(1 for g in games if key(g) == k) for k in {key(g) for g in games}}  # noqa: E731
    assert by(lambda g: g["map"]) == {"fighting_spirit": 50, "circuit_breaker": 50}
    assert by(lambda g: g["p1"]["race"]) == {"zerg": 50, "protoss": 30, "terran": 20}
    assert by(lambda g: g["p1"].get("build")) == {None: 75, "zerg_rush": 25}
    assert all(g["p1"]["kind"] == "t0" and g["p0"] == {"kind": "model", "checkpoint": F, "sample": "stochastic"} for g in games)
    assert not set(ec.plan_seeds(p)) & (bp.rcp.taken_seeds(ROOT / "config" / "plans", ()) | bp.rcp.reserved_seeds(skip_branch_b2=True))
    assert bp.rcp.branch_b2_block() <= bp.rcp.reserved_seeds(), "other generators keep out of the B2 block"
    small = bc.plan_games(tomllib.loads(bp.mix_trunk_plan(cfg, "s", first=991, games=2)))
    assert [(g["map"], g["p1"]["race"], g["seed"]) for g in small] == [("fighting_spirit", "zerg", bp.rcp.branch_b2_seed(991)),
                                                                       ("circuit_breaker", "protoss", bp.rcp.branch_b2_seed(992))]
    with pytest.raises(bp.Refused, match="used up"):
        bp.mix_trunk_plan(cfg, "s", first=999, games=4)


class MixTrunkJob(TrunkJob):
    def __init__(self, root: Path, games: int = 4):
        self.root, self.job = root, root / "eval" / "b2-trunks"
        self.cfg = bp.load_config(B2)
        text = bp.mix_trunk_plan(self.cfg, "b2-trunk", first=101, games=games)
        (self.job / "plans").mkdir(parents=True)
        (self.job / "plans" / "trunk.toml").write_text(text, encoding="utf-8")
        (self.job / "job.toml").write_text("stages = ['plans/trunk.toml']\nhosts = 'hosts.toml'\n", encoding="utf-8")
        self.plan = tomllib.loads(text)
        self.games = bc.plan_games(self.plan)
        ck = root / F
        ck.mkdir(parents=True)
        (ck / "model.safetensors").write_bytes(b"weights")
        self.ckpt_sha = ec.file_sha256(ck / "model.safetensors")
        self.sd = self.job / "b2-trunk"
        self.snaps = {}
        for n in range(1, len(self.games) + 1):
            self.play(n)
        self.merge()

    def play(self, n: int, top=None, fire: int = 3, no_top: bool = False):
        top = top or [[0, 0.4], [IDS["Move_Pos"], 0.2], [OPS["Train_SCV"], 0.15], [OPS["Build_SupplyDepot"], 0.1],
                      [OPS["Build_Barracks"], 0.08], [OPS["Build_Refinery"], 0.02]]
        at = {3: 4400, 6: 5900, 9: 7400}
        lines, opp, snaps = [], [], []
        f = 100
        for i in range(12):
            f = at.get(i, f + 100)
            lines.append({"i": i, "frame": f, "digest": digest(i, n), "act": act(i),
                          "decision": {"action": 0, "delay_bucket": 1, "units": [], "rejected": False}, "state": None, "forced": False})
            opp.append({"i": i, "frame": f, "digest": digest(i, 1000 + n), "act": act(i), "forced": False})
        for rule, i in list(zip(B2_RULES, (3, 6, 9)))[:fire]:
            sid = f"g{n}-o{i}"
            (self.sd / "states").mkdir(parents=True, exist_ok=True)
            (self.sd / "states" / f"{sid}.state").write_bytes(f"state {n} {i}".encode())
            sha = ec.file_sha256(self.sd / "states" / f"{sid}.state")
            lines[i].update({"state": sid, "state_sha256": sha})
            if not no_top:
                lines[i]["policy_top"] = top
            snaps.append({"rules": [rule], "state": sid, "state_sha256": sha, "i": i, "frame": lines[i]["frame"], "digest": lines[i]["digest"],
                          "type_mask": sorted(int(a) for a, _ in top), "scalar": [0.5, float(n)]})
        end = {"obs": 12, "decisions": 12, "end_frame": 17328, "reason": "TimeLimitDraw", "player": 0, "players_index": 0,
               "policy": f"model:ckpt-3258@{self.ckpt_sha[:12]}:stochastic", "git": "abc", "feature_schema": 7, "action_schema": 2}
        jsonl(self.sd / "acts" / f"g{n}-p0.jsonl", [*lines, {"end": end}])
        jsonl(self.sd / "acts" / f"g{n}-p1.jsonl", [*opp, {"end": {**end, "decisions": 0, "player": 1, "players_index": 1, "policy": T0}}])
        self.snaps[n] = snaps

    def merge(self, outcomes=None):
        super().merge(outcomes)
        rows = bc.read_jsonl(self.sd / "games.jsonl")
        for r, g in zip(rows, self.games):
            r["opponent"] = f"t0-{g['p1']['race']}" + (f"-{g['p1']['build'].replace('_', '-')}" if g["p1"].get("build") else "") + "@9.9.9"
        jsonl(self.sd / "games.jsonl", rows)


def b2_args(tj: MixTrunkJob, out: Path, **kw):
    return args(tj, out, config=B2, stage=["b2-trunk"], hit=None, miss=None, **{"games": None, **kw})


def test_b2_points_and_policy_rank_alternatives_051_r16_r10(tmp_path):
    tj = MixTrunkJob(tmp_path)
    tj.play(2, top=[[0, 0.5], [OPS["Train_SCV"], 0.3], [IDS["Move_Pos"], 0.1], [OPS["Build_SupplyDepot"], 0.1]])
    tj.play(3, no_top=True)
    tj.play(4, fire=2)
    tj.merge()
    out = tmp_path / "eval" / "b2"
    rep = tj.job / "plans" / "trunk-r.toml"
    s = bp.build(b2_args(tj, out, hosts="hosts.toml", replacements=str(rep)))
    points = bc.read_jsonl(out / "points.jsonl")
    assert all(set(p) == POINT_KEYS for p in points)
    first = [p for p in points if p["point"].startswith("b2-trunks/b2-trunk/g1#")]
    assert [p["point"].split("#")[1] for p in first] == list(B2_RULES) and all(p["valid"] for p in first)
    assert first[0]["policy_top"][1] == [IDS["Move_Pos"], 0.2]
    assert all(not p["valid"] and "policy_top" in p["skip_reason"] for p in points if "/g3#" in p["point"])
    assert {t["trunk"].split("/")[-1]: t["used"] for t in s["trunks"]} == {"g1": True, "g2": True, "g3": False, "g4": False}
    lines = bc.read_jsonl(out / "plans" / "branches_plan.jsonl")
    g1 = [x for x in lines if x["point"] == first[0]["point"] and x["repeat"] == 1]
    assert [(x["alt"]["name"], [r["action"] for r in x["alt"]["rules"]]) for x in g1] == [
        ("none", []), ("top1", ["Train_SCV"]), ("top2", ["Build_SupplyDepot"]), ("top3", ["Build_Barracks"])]
    assert g1[1]["alt"]["rules"] == [{"op": "force", "action": "Train_SCV", "from_s": 0, "to_s": 0}]
    assert [(d["point"].split("/")[-1], d["alt"]) for d in s["dropped"]] == [(f"g2#{r}", "top3") for r in B2_RULES]
    assert all(not (x["point"].startswith("b2-trunks/b2-trunk/g2#") and x["alt"]["name"] == "top3") for x in lines), "a missing rank drops out"
    assert len(lines) == s["games"] == (3 * 4 + 3 * 3) * 2
    assert [x["repeat"] for x in lines] == sorted(x["repeat"] for x in lines), "repeats outermost"
    plan = tomllib.loads((out / "plans" / "branch-b2.toml").read_text(encoding="utf-8"))
    assert (plan["max_frames"], plan["surrender"], plan["zero_last_failed"]) == (17328, "off", True) and "snapshots" not in plan
    games = bc.plan_games(plan)
    assert [(g["seed"], g["map"]) for g in games] == [(x["seed"], x["map"]) for x in lines]
    p1 = {tg["seed"]: tg["p1"] for tg in tj.games}
    assert all(g["p1"] == p1[g["seed"]] and g["p0"]["checkpoint"] == F for g in games), "each branch plays its trunk's opponent"
    assert [(m["map"], m["p1"].get("build")) for m in plan["matchup"]] == [("fighting_spirit", None), ("circuit_breaker", "zerg_rush")] * 2
    assert s["cells"]["fighting_spirit/zerg"] == {"quota": 13, "used": 1, "short": 12}
    assert s["cells"]["circuit_breaker/zerg-zerg_rush"] == {"quota": 13, "used": 1, "short": 12}
    assert s["cells"]["fighting_spirit/protoss"]["used"] == 0 and sum(v["short"] for v in s["cells"].values()) == 98
    r = tomllib.loads(rep.read_text(encoding="utf-8"))
    assert len(ec.plan_seeds(r)) == 98 and not set(ec.plan_seeds(r)) & set(ec.plan_seeds(tj.plan))
    assert ec.plan_seeds(r)[0] == bp.rcp.branch_b2_seed(1), "replacements take the first unused game numbers of the block"
    assert r["snapshots"] == tj.plan["snapshots"] and r["max_frames"] == 17328
    assert all(p["condition"] is None for p in points)
    small = bp.build(b2_args(tj, tmp_path / "eval" / "b2-small", games=4, replacements=str(tmp_path / "r4.toml")))
    assert small["cells"]["circuit_breaker/terran"] == {"quota": 1, "used": 0, "short": 1}
    assert sum(v["short"] for v in small["cells"].values()) == 2 and len(ec.plan_seeds(tomllib.loads((tmp_path / "r4.toml").read_text(encoding="utf-8")))) == 2
    with pytest.raises(SystemExit, match="opponent-mix"):
        bp.main(["branches", "--config", str(B2), "--trunk-job", str(tj.job), "--stage", "b2-trunk", "--out", str(out), "--hit", "1"])
    out2 = tmp_path / "eval" / "b2-again"
    bp.build(b2_args(tj, out2, hosts="hosts.toml"))
    for f in ("points.jsonl", "plans/branches_plan.jsonl", "plans/branch-b2.toml"):
        assert (out / f).read_bytes() == (out2 / f).read_bytes(), f


def test_b2_config_checks_051_r16(tmp_path):
    base = B2.read_text(encoding="utf-8")
    p = tmp_path / "c.toml"
    for old, new in (("policy_rank = 1", "policy_rank = 0"), ("policy_rank = 2", 'policy_rank = 2\nrules = [{ op = "force", action = "Train_SCV", from_s = 0, to_s = 0 }]'),
                     ("min_open_ops = 3\n\n", "min_open_ops = 0\n\n"), ("after_s = 60\nafter_gap = true\n\n[[snapshots]]", "after_s = 0\nafter_gap = true\n\n[[snapshots]]"),
                     ('race = "terran" }', 'race = "terran", name = "x" }'), ('{ kind = "t0", race = "protoss" }', '{ kind = "t0" }'), ('{ share = 20, p1 = { kind = "t0"', '{ share = 20, p1 = { kind = "oss"')):
        assert old in base, old
        p.write_text(base.replace(old, new, 1), encoding="utf-8")
        with pytest.raises(bc.Refused):
            bp.load_config(p)


def test_b2_collect_outcome_extras_051_r16_053_r2(tmp_path):
    tj = MixTrunkJob(tmp_path, games=2)
    out = tmp_path / "eval" / "b2"
    bp.build(b2_args(tj, out, alts="none,top1", repeats=1, hosts="h.toml"))
    lines = bc.read_jsonl(out / "plans" / "branches_plan.jsonl")
    sd = out / "branch-b2"
    scalar = [0.0] * 567
    outcome = {"valid": True, "invalid": None, "prefix_obs": 3, "prefix_matched": 3, "state_check": None, "intervention": [],
               "samples": {"horizon": {"frame": 17280, "scalar": scalar}}, "metrics": {"held_8m": True}, "dilution": [],
               "end": {"result": "draw", "end_frame": 17328, "reason": "TimeLimitDraw", "before_horizon": False}}
    games, results = [], []
    for n, ln in enumerate(lines, start=1):
        games.append({"game_id": n, "hostd_id": 100_000 + n, "outcome": "Aborted" if n == 2 else "Finished", "host": "h", "backend": "rocm"})
        results.append({"game_id": n, "won": None, "end_frame": 17328, "reason": "TimeLimitDraw", "branch": outcome,
                        "model": {"scv_8m": 12, "scv_min_1m": 5, "ops": {"inc_10m": 812.0, "inc_scv_10m": 30.5}}})
        jsonl(sd / "acts" / f"g{n}-p1.jsonl", [*bc.read_jsonl(bc.on_shared(tmp_path, ln["opponent_acts"]))[:-1], {"end": {"policy": T0}}])
    jsonl(sd / "games.jsonl", games)
    jsonl(sd / "results.jsonl", results)
    bcol.collect(out, None, tmp_path)
    recs = {r["game"]["plan_game"]: r for r in bc.read_jsonl(out / "branches.jsonl")}
    assert all(set(r) == RECORD_KEYS for r in recs.values())
    assert recs[1]["labels"] == {"held_8m": 1, "held_s": 1, "alive_10m": 1, "win": None}
    assert recs[2]["labels"] == {"held_8m": None, "held_s": None, "alive_10m": None, "win": None}, "Aborted: no outcome label (053-R2 8)"
    assert recs[1]["ops"]["inc_10m"] == 812.0
    from uranium.diag.planning.dump import own_columns
    assert list(recs[1]["own_horizon"]) == own_columns()[0] and recs[1]["own_horizon"]["supply_used"] == 0.0


B3 = ROOT / "config" / "branch_b3.toml"
B3_PLANS = {
    "none": [],
    "cc_now": [("force", "Build_CommandCenter", 0, 90)],
    "cc_late": [("ban", "Build_CommandCenter", 0, 180)],
    "factory_now": [("force", "Build_Refinery", 0, 30), ("force", "Build_Factory", 60, 180), ("ban", "Build_Barracks", 0, 120)],
    "rax_more": [("force", "Build_Barracks", 0, 30), ("force", "Build_Barracks", 30, 75), ("ban", "Build_Factory", 0, 180)],
    "scv_cut": [("ban", "Train_SCV", 0, 90)],
    "bunker_now": [("force", "Build_Bunker", 0, 30)],
}


def b3_args(tj: MixTrunkJob, out: Path, **kw):
    return args(tj, out, config=B3, stage=["b2-trunk"], hit=None, miss=None, **{"games": None, **kw})


def test_b3_config_reuses_b2_trunks_055_r2_r3(tmp_path):
    b2, b3 = bp.load_config(B2), bp.load_config(B3)
    assert {k: b3[k] for k in ("checkpoint", "horizon", "trunks", "snapshots")} == {k: b2[k] for k in ("checkpoint", "horizon", "trunks", "snapshots")}
    assert (b3["name"], b3["repeats"]) == ("branch-b3", 1)
    assert {a["name"]: [(r["op"], r["action"], r["from_s"], r["to_s"]) for r in a["rules"]] for a in b3["alternative"]} == B3_PLANS
    tj = MixTrunkJob(tmp_path)
    out = tmp_path / "eval" / "b3"
    s = bp.build(b3_args(tj, out, hosts="hosts.toml"))
    assert s["dropped"] == [] and s["repeats"] == 1 and s["alternatives"] == list(B3_PLANS)
    lines = bc.read_jsonl(out / "plans" / "branches_plan.jsonl")
    assert len(lines) == s["games"] == 4 * 3 * 7, "every point gets all seven plans (no t-only force to drop)"
    for p in {x["point"] for x in lines}:
        got = [(x["alt"]["name"], [(r["op"], r["action"], r["from_s"], r["to_s"]) for r in x["alt"]["rules"]]) for x in lines if x["point"] == p]
        assert got == list(B3_PLANS.items())
    union = list(dict.fromkeys(r["action"] for x in lines for r in x["alt"]["rules"]))
    assert union == ["Build_CommandCenter", "Build_Refinery", "Build_Factory", "Build_Barracks", "Train_SCV", "Build_Bunker"], "none dilution list (055-R1)"
    plan = tomllib.loads((out / "plans" / "branch-b3.toml").read_text(encoding="utf-8"))
    assert (plan["name"], plan["max_frames"], plan["surrender"], plan["zero_last_failed"]) == ("branch-b3", 17328, "off", True)
    b2out = tmp_path / "eval" / "b2"
    bp.build(b2_args(tj, b2out, hosts="hosts.toml"))
    assert (out / "points.jsonl").read_bytes() == (b2out / "points.jsonl").read_bytes(), "B3 points are the B2 points"
    again = tmp_path / "eval" / "b3-again"
    bp.build(b3_args(tj, again, hosts="hosts.toml"))
    for f in ("points.jsonl", "plans/branches_plan.jsonl", "plans/branch-b3.toml"):
        assert (out / f).read_bytes() == (again / f).read_bytes(), f


def test_b3_collect_score_055_r6(tmp_path):
    tj = MixTrunkJob(tmp_path, games=2)
    out = tmp_path / "eval" / "b3"
    bp.build(b3_args(tj, out, alts="none,cc_now", hosts="h.toml"))
    lines = bc.read_jsonl(out / "plans" / "branches_plan.jsonl")
    sd = out / "branch-b3"
    ends = {1: ("win", True), 2: ("draw", None), 3: ("loss", False), 4: ("win", True), 5: ("win", True), 6: ("loss", False)}
    games, results = [], []
    for n, ln in enumerate(lines, start=1):
        result, won = ends.get(n, ("draw", None))
        outcome = {"valid": True, "invalid": None, "prefix_obs": 3, "prefix_matched": 3, "state_check": None, "intervention": [],
                   "samples": {}, "metrics": {}, "dilution": [], "end": {"result": result, "end_frame": 9000, "reason": "x", "before_horizon": n != 5}}
        games.append({"game_id": n, "hostd_id": 100_000 + n, "outcome": "Aborted" if n == 4 else "Finished", "host": "h", "backend": "rocm"})
        results.append({"game_id": n, "won": won, "end_frame": 9000, "reason": "x", "branch": outcome, "model": {"scv_8m": 12, "scv_min_1m": 5},
                        **({"error": "x"} if n == 6 else {})})
        jsonl(sd / "acts" / f"g{n}-p1.jsonl", [*bc.read_jsonl(bc.on_shared(tmp_path, ln["opponent_acts"]))[:-1], {"end": {"policy": T0}}])
    jsonl(sd / "games.jsonl", games)
    jsonl(sd / "results.jsonl", results)
    bcol.collect(out, None, tmp_path)
    recs = {r["game"]["plan_game"]: r for r in bc.read_jsonl(out / "branches.jsonl")}
    assert all(set(r) == RECORD_KEYS for r in recs.values())
    assert [recs[n]["score"] for n in (1, 2, 3, 4, 5, 6)] == [1.0, 0.5, 0.0, None, 0.5, None],         "win 1, draw 0.5, loss 0, Aborted none, end after H 0.5, session error none (055-R6, 053-R2 8)"
