"""041-R27 (AMB-041-16): the teacher hold-out scoring split by opponent class (tools/eval/per_action.py): each game's
class from the generating plan's games.jsonl and its replay file hash; per-class tallies that add up to the whole
and match scoring only that class's games; the macro games' mean p(Build_Bunker) before 4:00. CPU, synthetic
shards (uranium._core with Batch.lane_games)."""

import functools
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "eval"))

import per_action as pa  # noqa: E402
import per_action_table as pat  # noqa: E402


def test_opponent_class_041_r27():
    for o in ("t0-zerg-rush@0.1.0", "ualbertabot12poollings@558899d-12poollings", "ualbertabot9pool10lings@x", "ualbertabot9poolhydra@x"):
        assert pa.opponent_class(o) == "rush"
    for o in ("ualbertabot2hatchhydra@x", "ualbertabot3hatchmuta@x", "steamhammer2hatchmuta@5.3.6-2hatchmuta", "t0-zerg-muta@0.1.0", "t0-zerg@0.1.0"):
        assert pa.opponent_class(o) == "macro"
    with pytest.raises(SystemExit, match="neither a rush"):
        pa.opponent_class("iron@1")


def fake_plan(root: Path, folder: str, games: list[tuple[int, str]]) -> dict[str, str]:
    """A teacher plan folder: games.jsonl rows (game_id, opponent) and replays/g<id>.rep; returns {sha: opponent}."""
    d = root / folder
    (d / "replays").mkdir(parents=True)
    out = {}
    with open(d / "games.jsonl", "w", encoding="utf-8") as f:
        for gid, opp in games:
            f.write(json.dumps({"game_id": gid, "opponent": opp, "outcome": "Finished"}) + "\n")
            data = f"{folder}/{gid}".encode()
            (d / "replays" / f"g{gid}.rep").write_bytes(data)
            out[hashlib.sha256(data).hexdigest()] = opp
    return out


def test_teacher_classes_from_plans_041_r27(tmp_path):
    raw = tmp_path / "raw"
    rush = fake_plan(raw, "teacher-1/teacher_botgen_rush", [(1, "t0-zerg-rush@0.1.0"), (2, "ualbertabot9poolhydra@x")])
    macro = fake_plan(raw, "teacher-1/teacher_botgen_macro", [(1, "t0-zerg@0.1.0")])
    macro3 = fake_plan(raw, "teacher-3-macro3/teacher_botgen_macro3", [(7, "ualbertabot3hatchmuta@x")])
    # a games.jsonl row without its replay is skipped
    with open(raw / "teacher-1/teacher_botgen_macro/games.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"game_id": 9, "opponent": "t0-zerg@0.1.0"}) + "\n")
    cache = tmp_path / "cache" / "classes.json"
    got = pa.teacher_classes(raw, cache)
    assert {s: v["class"] for s, v in got.items()} == {**{s: "rush" for s in rush}, **{s: "macro" for s in {**macro, **macro3}}}
    assert got[next(iter(macro3))] == {"class": "macro", "opponent": "ualbertabot3hatchmuta", "plan": "teacher-3-macro3/teacher_botgen_macro3"}
    # the cache is reused while games.jsonl is unchanged (a replaced replay is not re-hashed); it keeps the
    # opponent only, the class is derived on every call
    (raw / "teacher-1/teacher_botgen_rush/replays/g1.rep").write_bytes(b"changed")
    assert pa.teacher_classes(raw, cache) == got
    cached = json.loads(cache.read_text(encoding="utf-8"))
    assert all(set(v) == {"opponent"} for e in cached.values() for v in e["games"].values())
    # the same replay under two plans with different classes is an error
    dup = raw / "teacher-9/teacher_botgen_macro9"
    (dup / "replays").mkdir(parents=True)
    (dup / "games.jsonl").write_text(json.dumps({"game_id": 2, "opponent": "t0-zerg@0.1.0"}) + "\n", encoding="utf-8")
    (dup / "replays" / "g2.rep").write_bytes(b"teacher-1/teacher_botgen_rush/2")
    with pytest.raises(SystemExit, match="is rush in teacher-1/teacher_botgen_rush and macro in teacher-9/teacher_botgen_macro9"):
        pa.teacher_classes(raw, None)
    # load_classes: a JSON file of either form
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"a": "rush", "b": {"class": "macro"}}), encoding="utf-8")
    assert pa.load_classes(str(f), None) == {"a": "rush", "b": "macro"}
    assert pa.load_classes(None, None) is None and pa.load_classes("none", "orig") is None
    f.write_text(json.dumps({"a": "other"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="rush / macro"):
        pa.load_classes(str(f), None)


@pytest.fixture
def scored(tiny_config, tmp_path):
    """A tiny model and the synthetic `races` shards (15 games of 1..16 steps, enemies Terran / Protoss / Zerg),
    with the race of each game id read from the stream."""
    from uranium import _core

    if not hasattr(_core.Batch, "lane_games"):
        pytest.skip("uranium._core without Batch.lane_games (rebuild the extension)")
    from test_bc_eval_race import _tiny_make

    shards = tmp_path / "races"
    _core.write_test_shards(str(shards), "races", 4)
    out = tmp_path / "out"
    assert _core.run_bc(functools.partial(_tiny_make, None, None), str(tiny_config), out=str(out), shards=[str(shards)], steps=1, synthetic_maps=True) == "done"
    race = {}
    stream = _core.EvalStream(str(tiny_config), [str(shards)], split="train", synthetic_maps=True)
    while (b := stream.next()) is not None:
        for gid, r, n in zip(b.lane_games(), b.lane_enemy_races(), b.lane_steps()):
            if n:
                race[gid] = r
    return tiny_config, shards, out / "last", race


def run(cfg, shards, ckpt, classes):
    return pa.score(str(ckpt), [], [str(shards)], "cpu", str(cfg), classes, synthetic_maps=True, split="train")


def test_rush_macro_split_041_r27(scored, tmp_path):
    """Zerg-enemy games as `rush`, the rest `macro`: per-class steps and per-action counts add up to the whole,
    the overall `per_action` is the same as without the split, and the rush class equals scoring only the Zerg
    games (games, steps, per-action counts)."""
    from uranium import _core

    cfg, shards, ckpt, race = scored
    assert len(race) == 15 and set(race.values()) == {"terran", "protoss", "zerg"}
    classes = {g: "rush" if r == "zerg" else "macro" for g, r in race.items()}
    res = run(cfg, shards, ckpt, classes)
    plain = run(cfg, shards, ckpt, None)
    assert "by_class" not in plain and "r27" not in plain
    assert res["per_action"] == plain["per_action"] and res["steps"] == plain["steps"]
    by = res["by_class"]
    assert by["rush"]["games"] == sum(1 for c in classes.values() if c == "rush") and by["macro"]["games"] == 15 - by["rush"]["games"]
    assert by["rush"]["steps"] + by["macro"]["steps"] == res["steps"]
    whole = {r["action"]: r["n"] for r in res["per_action"]}
    for a, n in whole.items():
        assert sum({r["action"]: r["n"] for r in by[c]["per_action"]}.get(a, 0) for c in ("rush", "macro")) == n, a
    # synthetic frames (0..120) are all before 4:00: every scored step of a class counts
    assert by["macro"]["bunker_p_before_4m"]["steps"] == by["macro"]["steps"]
    assert 0.0 <= res["r27"]["macro_bunker_p_before_4m"] <= 1.0
    assert res["r27"]["macro_games"] == by["macro"]["games"] and res["r27"]["rush_games"] == by["rush"]["games"]
    # Build_Bunker is not a label of the synthetic games: no rush bunker top-1
    assert res["r27"]["rush_bunker_top1"] is None and res["r27"]["rush_bunker_steps"] == 0

    only = tmp_path / "only_zerg"
    _core.write_test_shards(str(only), "races", 4, only_race="zerg")
    z = run(cfg, only, ckpt, None)
    assert z["games"] == by["rush"]["games"] and z["steps"] == by["rush"]["steps"]
    assert {r["action"]: r["n"] for r in z["per_action"]} == {r["action"]: r["n"] for r in by["rush"]["per_action"]}

    # every game must have a class
    missing = dict(classes)
    missing.pop(next(iter(missing)))
    with pytest.raises(SystemExit, match="has no rush / macro class"):
        run(cfg, shards, ckpt, missing)


def test_bunker_before_4m_boundary_041_r27(scored, monkeypatch):
    """Only steps observed before the boundary count (frame < F; synthetic steps are 8 frames apart from 0)."""
    cfg, shards, ckpt, race = scored
    classes = {g: "macro" for g in race}
    full = run(cfg, shards, ckpt, classes)["by_class"]["macro"]
    monkeypatch.setattr(pa, "BUNKER_BEFORE", 64)
    part = run(cfg, shards, ckpt, classes)["by_class"]["macro"]
    assert part["steps"] == full["steps"] and 0 < part["bunker_p_before_4m"]["steps"] < full["bunker_p_before_4m"]["steps"]
    monkeypatch.setattr(pa, "BUNKER_BEFORE", 0)
    none = run(cfg, shards, ckpt, classes)["by_class"]["macro"]["bunker_p_before_4m"]
    assert none["steps"] == 0 and none["mean_p"] is None


def test_per_action_table_r27_rows_041_r27(tmp_path):
    """per_action_table shows the rush-only bunker top-1 (extension rule input) and its change, and the macro
    p(bunker) before 4:00; an old column without the split shows '-'."""
    def col(name, top1, r27):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({"steps": 10, "games": 2, "per_action": [{"action": "Build_Bunker", "n": 10, "top1": top1, "loc_n": 0, "tu_n": 0}],
                                 **({"r27": r27} if r27 else {})}), encoding="utf-8")
        return f"{name}={p}"

    def r27(t, p):
        return {"rush_bunker_top1": t, "rush_bunker_steps": 8, "rush_games": 129, "macro_bunker_p_before_4m": p, "macro_steps_before_4m": 900, "macro_games": 117}

    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pat.main([col("old", 0.5, None), col("e1", 0.6, r27(0.62, 0.0123)), col("e2", 0.7, r27(0.65, 0.0101))])
    text = buf.getvalue()
    assert "Build_Bunker top-1, rush-opponent games (041-R27) | - | 62.0 | 65.0" in text
    assert "macro mean p(Build_Bunker) < 4:00 (record) | - | 0.0123 | 0.0101" in text
    assert "041-R27 rush bunker top-1 per column [62.0, 65.0], last change +3.0 %p" in text


def test_macro_value_over_the_full_hold_out_041_r27():
    """The recorded macro p(bunker) before 4:00 is the step-weighted value over the orig pass's macro games and
    the added parts' macro games (main-session interim decision 2026-11-10); the rest of r27 is unchanged."""
    r27 = {"rush_bunker_top1": 0.6, "rush_bunker_steps": 50, "rush_games": 129,
           "macro_bunker_p_before_4m": 0.02, "macro_steps_before_4m": 100, "macro_games": 36, "macro_holdout": "orig"}
    added = {"games": 81, "steps": 900, "bunker_p_before_4m": {"before_frame": 5760, "steps": 300, "mean_p": 0.06}}
    got = pa.macro_full(r27, added)
    assert got["macro_bunker_p_before_4m"] == pytest.approx((0.02 * 100 + 0.06 * 300) / 400)
    assert (got["macro_steps_before_4m"], got["macro_games"], got["macro_added_games"], got["macro_holdout"]) == (400, 117, 81, "full")
    assert {k: got[k] for k in ("rush_bunker_top1", "rush_bunker_steps", "rush_games")} == {"rush_bunker_top1": 0.6, "rush_bunker_steps": 50, "rush_games": 129}
    assert r27["macro_games"] == 36, "the input is not changed"
    assert pa.macro_full(r27, None) == r27


def test_tp_teacher_folders_left_out_044_r8(tmp_path, monkeypatch):
    """044-R8 / 022-R12: botgen folders of the 022 TvT / TvP teacher (teacher-tp-*) are neither classified (their
    T0 Protoss / Terran opponents have no 041-R27 class) nor part of the `full` macro pass."""
    raw = tmp_path / "raw"
    rush = fake_plan(raw, "teacher-1/teacher_botgen_rush", [(1, "t0-zerg-rush@0.1.0")])
    fake_plan(raw, "teacher-tp-20261114/teacher_botgen_tp", [(1, "t0-protoss@0.1.0"), (2, "iron@aiide2017")])
    got = pa.teacher_classes(raw, None)
    assert {s: v["class"] for s, v in got.items()} == {s: "rush" for s in rush}
    shared = tmp_path / "shared"
    src = shared / "datasets" / "replays" / "shards" / "botgen"
    for part in (*pa.ORIG, "teacher-20261107-macro2", "teacher-tp-20261114", "teacher-tp-20261114-v2"):
        (src / part).mkdir(parents=True)
        (src / part / "shard_00000.bin.zst").write_bytes(part.encode())
    monkeypatch.setattr(pa.ec, "shared_dir", lambda: shared)
    monkeypatch.setattr(pa.ec, "cache_dir", lambda: tmp_path / "cache")
    (added,) = pa.added_dirs()
    assert sorted(p.name for p in Path(added).iterdir()) == ["teacher-20261107-macro2"]


def tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def botgen(tmp_path, monkeypatch, parts):
    shared = tmp_path / "shared"
    src = shared / "datasets" / "replays" / "shards" / "botgen"
    for part in parts:
        (src / part).mkdir(parents=True, exist_ok=True)
        (src / part / "shard_00000.bin.zst").write_bytes(part.encode())
    monkeypatch.setattr(pa.ec, "shared_dir", lambda: shared)
    monkeypatch.setattr(pa.ec, "cache_dir", lambda: tmp_path / "cache")
    return src


def test_full_holdout_excludes_tp_teacher_041_r27_044_r8(tmp_path, monkeypatch):
    """041-R27 / 044-R8: `--holdout full` is a link tree of every botgen part but teacher-tp-* (also the -v2 shards
    added before teacher round t1) = ORIG + added_dirs() parts; `orig` stays the ORIG parts only."""
    added = ["teacher-20261107-macro2", "teacher-20261109-macro3"]
    src = botgen(tmp_path, monkeypatch, (*pa.ORIG, *added, "teacher-tp-20261114", "teacher-tp-20261114-v2"))
    datasets, (full,) = pa.holdout_dirs("full")
    assert datasets == []
    assert sorted(p.name for p in Path(full).iterdir()) == sorted([*pa.ORIG, *added])
    assert tree(Path(full)) == {f"{p}/shard_00000.bin.zst": p.encode() for p in [*pa.ORIG, *added]}
    (ad,) = pa.added_dirs()
    assert sorted({*pa.ORIG, *(p.name for p in Path(ad).iterdir())}) == sorted(p.name for p in Path(full).iterdir())
    datasets, (orig,) = pa.holdout_dirs("orig")
    assert datasets == [] and tree(Path(orig)) == {f"{p}/shard_00000.bin.zst": p.encode() for p in pa.ORIG}
    assert Path(orig) != Path(full)
    assert pa.holdout_parts(src) == sorted([*pa.ORIG, *added])
    with pytest.raises(SystemExit, match="unknown --holdout"):
        pa.holdout_dirs("tp")


def test_link_tree_drops_stale_entries_041_r27_044_r8(tmp_path, monkeypatch):
    """041-R27 / 044-R8: a rebuilt link tree holds exactly the current parts: a part dropped from the list (or a
    stray teacher-tp-* folder from an older tree) and a shard file gone from its source part are removed; the
    source files stay."""
    src = botgen(tmp_path, monkeypatch, (*pa.ORIG, "teacher-20261107-macro2", "teacher-20261109-macro3"))
    (src / "teacher-20261109-macro3" / "shard_00001.bin.zst").write_bytes(b"second")
    (ad,) = pa.added_dirs()
    work = Path(ad)
    assert sorted(tree(work)) == ["teacher-20261107-macro2/shard_00000.bin.zst", "teacher-20261109-macro3/shard_00000.bin.zst",
                                  "teacher-20261109-macro3/shard_00001.bin.zst"]
    # an older tree's teacher-tp folder, a part no longer in the source, a shard removed from its source part
    (work / "teacher-tp-20261114-v2").mkdir()
    (work / "teacher-tp-20261114-v2" / "shard_00000.bin.zst").write_bytes(b"tp")
    import shutil
    shutil.rmtree(src / "teacher-20261107-macro2")
    (src / "teacher-20261109-macro3" / "shard_00001.bin.zst").unlink()
    (ad2,) = pa.added_dirs()
    assert Path(ad2) == work
    assert tree(work) == {"teacher-20261109-macro3/shard_00000.bin.zst": b"teacher-20261109-macro3"}
    assert (src / "teacher-20261109-macro3" / "shard_00000.bin.zst").read_bytes() == b"teacher-20261109-macro3"
    assert all((src / p / "shard_00000.bin.zst").exists() for p in pa.ORIG)
    # no added part left: [] (the stale tree is not read)
    shutil.rmtree(src / "teacher-20261109-macro3")
    assert pa.added_dirs() == []
