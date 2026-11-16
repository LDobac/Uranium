import functools
import hashlib
import json
import os
from pathlib import Path

import pytest

import eval_helpers  # noqa: F401

import per_action as pa
import per_action_table as pat


def test_opponent_class_041_r27():
    for o in ("t0-zerg-rush@0.1.0", "ualbertabot12poollings@558899d-12poollings", "ualbertabot9pool10lings@x", "ualbertabot9poolhydra@x"):
        assert pa.opponent_class(o) == "rush"
    for o in ("ualbertabot2hatchhydra@x", "ualbertabot3hatchmuta@x", "steamhammer2hatchmuta@5.3.6-2hatchmuta", "t0-zerg-muta@0.1.0", "t0-zerg@0.1.0"):
        assert pa.opponent_class(o) == "macro"
    with pytest.raises(SystemExit, match="neither a rush"):
        pa.opponent_class("iron@1")


def fake_plan(root: Path, folder: str, games: list[tuple[int, str]]) -> dict[str, str]:
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
    with open(raw / "teacher-1/teacher_botgen_macro/games.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"game_id": 9, "opponent": "t0-zerg@0.1.0"}) + "\n")
    cache = tmp_path / "cache" / "classes.json"
    got = pa.teacher_classes(raw, cache)
    assert {s: v["class"] for s, v in got.items()} == {**{s: "rush" for s in rush}, **{s: "macro" for s in {**macro, **macro3}}}
    assert got[next(iter(macro3))] == {"class": "macro", "opponent": "ualbertabot3hatchmuta", "plan": "teacher-3-macro3/teacher_botgen_macro3"}
    (raw / "teacher-1/teacher_botgen_rush/replays/g1.rep").write_bytes(b"changed")
    assert pa.teacher_classes(raw, cache) == got
    cached = json.loads(cache.read_text(encoding="utf-8"))
    assert all(set(v) == {"opponent"} for e in cached.values() for v in e["games"].values())
    dup = raw / "teacher-9/teacher_botgen_macro9"
    (dup / "replays").mkdir(parents=True)
    (dup / "games.jsonl").write_text(json.dumps({"game_id": 2, "opponent": "t0-zerg@0.1.0"}) + "\n", encoding="utf-8")
    (dup / "replays" / "g2.rep").write_bytes(b"teacher-1/teacher_botgen_rush/2")
    with pytest.raises(SystemExit, match="is rush in teacher-1/teacher_botgen_rush and macro in teacher-9/teacher_botgen_macro9"):
        pa.teacher_classes(raw, None)
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"a": "rush", "b": {"class": "macro"}}), encoding="utf-8")
    assert pa.load_classes(str(f), None) == {"a": "rush", "b": "macro"}
    assert pa.load_classes(None, None) is None and pa.load_classes("none", "orig") is None
    f.write_text(json.dumps({"a": "other"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="rush / macro"):
        pa.load_classes(str(f), None)


@pytest.fixture
def scored(tiny_config, tmp_path):
    from uranium import _core

    if not hasattr(_core.Batch, "lane_games"):
        pytest.skip("uranium._core without Batch.lane_games (rebuild the extension)")
    from testkit import tiny_make as _tiny_make

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
    assert by["macro"]["bunker_p_before_4m"]["steps"] == by["macro"]["steps"]
    assert 0.0 <= res["r27"]["macro_bunker_p_before_4m"] <= 1.0
    assert res["r27"]["macro_games"] == by["macro"]["games"] and res["r27"]["rush_games"] == by["rush"]["games"]
    assert res["r27"]["rush_bunker_top1"] is None and res["r27"]["rush_bunker_steps"] == 0

    only = tmp_path / "only_zerg"
    _core.write_test_shards(str(only), "races", 4, only_race="zerg")
    z = run(cfg, only, ckpt, None)
    assert z["games"] == by["rush"]["games"] and z["steps"] == by["rush"]["steps"]
    assert {r["action"]: r["n"] for r in z["per_action"]} == {r["action"]: r["n"] for r in by["rush"]["per_action"]}

    missing = dict(classes)
    missing.pop(next(iter(missing)))
    with pytest.raises(SystemExit, match="has no rush / macro class"):
        run(cfg, shards, ckpt, missing)


def test_bunker_before_4m_boundary_041_r27(scored, monkeypatch):
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
    r27 = {"rush_bunker_top1": 0.6, "rush_bunker_steps": 50, "rush_games": 129,
           "macro_bunker_p_before_4m": 0.02, "macro_steps_before_4m": 100, "macro_games": 36, "macro_holdout": "orig"}
    added = {"games": 81, "steps": 900, "bunker_p_before_4m": {"before_frame": 5760, "steps": 300, "mean_p": 0.06}}
    got = pa.macro_full(r27, added)
    assert got["macro_bunker_p_before_4m"] == pytest.approx((0.02 * 100 + 0.06 * 300) / 400)
    assert (got["macro_steps_before_4m"], got["macro_games"], got["macro_added_games"], got["macro_holdout"]) == (400, 117, 81, "full")
    assert {k: got[k] for k in ("rush_bunker_top1", "rush_bunker_steps", "rush_games")} == {"rush_bunker_top1": 0.6, "rush_bunker_steps": 50, "rush_games": 129}
    assert r27["macro_games"] == 36, "the input is not changed"
    assert pa.macro_full(r27, None) == r27




def tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def write_manifest(src: Path, parts) -> None:
    files = {f"{p}/{f.name}": {"sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "bytes": f.stat().st_size}
             for p in parts for f in sorted((src / p).glob("shard_*.bin.zst"))}
    (src / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")


def botgen(tmp_path, monkeypatch, parts, listed=None):
    shared = tmp_path / "shared"
    src = shared / "datasets" / "replays" / "shards" / "botgen"
    for part in parts:
        (src / part).mkdir(parents=True, exist_ok=True)
        (src / part / "shard_00000.bin.zst").write_bytes(part.encode())
    write_manifest(src, parts if listed is None else listed)
    monkeypatch.setattr(pa.ec, "shared_dir", lambda: shared)
    monkeypatch.setattr(pa.ec, "cache_dir", lambda: tmp_path / "cache")
    return src


ADDED = ["teacher-20261107-macro2", "teacher-20261109-macro3"]
TP_V1, TP_V2 = "teacher-tp-20261114", "teacher-tp-20261114-v2"


def test_tp_teacher_folders_left_out_044_r8(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    rush = fake_plan(raw, "teacher-1/teacher_botgen_rush", [(1, "t0-zerg-rush@0.1.0")])
    fake_plan(raw, "teacher-tp-20261114/teacher_botgen_tp", [(1, "t0-protoss@0.1.0"), (2, "iron@aiide2017")])
    got = pa.teacher_classes(raw, None)
    assert {s: v["class"] for s, v in got.items()} == {s: "rush" for s in rush}
    botgen(tmp_path, monkeypatch, (*pa.ORIG, ADDED[0], TP_V1, TP_V2), listed=(*pa.ORIG, ADDED[0], TP_V2))
    (added,) = pa.added_dirs()
    assert sorted(p.name for p in Path(added).iterdir()) == [ADDED[0]]


def test_full_holdout_excludes_tp_teacher_041_r27_044_r8(tmp_path, monkeypatch):
    src = botgen(tmp_path, monkeypatch, (*pa.ORIG, *ADDED, TP_V1, TP_V2), listed=(*pa.ORIG, *ADDED, TP_V2))
    (src / ADDED[0] / "shard_00007.bin.zst").write_bytes(b"not in the manifest")
    parts, files = pa.holdout_parts(src)
    assert parts == sorted([*pa.ORIG, *ADDED]) and files == {p: ["shard_00000.bin.zst"] for p in parts}
    datasets, (full,) = pa.holdout_dirs("full")
    assert datasets == []
    assert tree(Path(full)) == {f"{p}/shard_00000.bin.zst": p.encode() for p in [*pa.ORIG, *ADDED]}
    (ad,) = pa.added_dirs()
    assert sorted({*pa.ORIG, *(p.name for p in Path(ad).iterdir())}) == sorted(p.name for p in Path(full).iterdir())
    datasets, (orig,) = pa.holdout_dirs("orig")
    assert datasets == [] and tree(Path(orig)) == {f"{p}/shard_00000.bin.zst": p.encode() for p in pa.ORIG}
    assert len({Path(orig), Path(full), Path(ad)}) == 3
    with pytest.raises(SystemExit, match="unknown --holdout"):
        pa.holdout_dirs("tp")


def test_full_holdout_refuses_a_manifest_mismatch_041_r27_044_r8(tmp_path, monkeypatch):
    src = botgen(tmp_path, monkeypatch, (*pa.ORIG, *ADDED, TP_V1), listed=(*pa.ORIG, ADDED[0]))
    with pytest.raises(SystemExit, match="are not the manifest's parts"):
        pa.holdout_dirs("full")
    with pytest.raises(SystemExit, match="are not the manifest's parts"):
        pa.added_dirs()
    write_manifest(src, (*pa.ORIG, *ADDED))
    assert pa.holdout_parts(src)[0] == sorted([*pa.ORIG, *ADDED])
    (src / ADDED[1] / "shard_00000.bin.zst").write_bytes(b"other size")
    with pytest.raises(SystemExit, match="the manifest says"):
        pa.holdout_dirs("full")
    (src / ADDED[1] / "shard_00000.bin.zst").unlink()
    with pytest.raises(SystemExit, match="is missing"):
        pa.holdout_dirs("full")
    (src / "manifest.json").unlink()
    with pytest.raises(SystemExit, match="no botgen manifest"):
        pa.holdout_dirs("full")


def test_link_tree_drops_stale_entries_041_r27_044_r8(tmp_path, monkeypatch):
    src = botgen(tmp_path, monkeypatch, (*pa.ORIG, *ADDED))
    (src / ADDED[1] / "shard_00001.bin.zst").write_bytes(b"second")
    write_manifest(src, (*pa.ORIG, *ADDED))
    (ad,) = pa.added_dirs()
    work = Path(ad)
    assert sorted(tree(work)) == [f"{ADDED[0]}/shard_00000.bin.zst", f"{ADDED[1]}/shard_00000.bin.zst", f"{ADDED[1]}/shard_00001.bin.zst"]
    (work / TP_V2).mkdir()
    (work / TP_V2 / "shard_00000.bin.zst").write_bytes(b"tp")
    import shutil
    shutil.rmtree(src / ADDED[0])
    (src / ADDED[1] / "shard_00001.bin.zst").unlink()
    write_manifest(src, (*pa.ORIG, ADDED[1]))
    pa.link_tree(src, [ADDED[1]], work)
    assert (work / TP_V2 / "shard_00000.bin.zst").exists() and (work / ADDED[1] / "shard_00001.bin.zst").exists()
    (ad2,) = pa.added_dirs()
    assert Path(ad2) == work
    assert tree(work) == {f"{ADDED[1]}/shard_00000.bin.zst": ADDED[1].encode()}
    assert (src / ADDED[1] / "shard_00000.bin.zst").read_bytes() == ADDED[1].encode()
    assert all((src / p / "shard_00000.bin.zst").exists() for p in pa.ORIG)
    shutil.rmtree(src / ADDED[1])
    write_manifest(src, pa.ORIG)
    assert pa.added_dirs() == []


def dir_link(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    try:
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    except (ImportError, OSError):
        pytest.skip("no directory symbolic link or junction on this host")


def test_link_tree_refuses_links_and_trees_inside_the_source_041_r27_044_r8(tmp_path):
    src = tmp_path / "src"
    for part in ("a", "b"):
        (src / part).mkdir(parents=True)
        (src / part / "shard_00000.bin.zst").write_bytes(part.encode())
    before = tree(src)
    dst = tmp_path / "dst"
    dir_link(dst, src)
    with pytest.raises(SystemExit, match="symbolic link or junction"):
        pa.link_tree(src, ["a"], dst, prune=True)
    assert tree(src) == before
    os.rmdir(dst) if dst.is_junction() else dst.unlink()
    dst.mkdir()
    dir_link(dst / "a", src / "b")
    with pytest.raises(SystemExit, match="symbolic link or junction"):
        pa.link_tree(src, ["a"], dst, prune=True)
    assert tree(src) == before
    with pytest.raises(SystemExit, match="inside its source"):
        pa.link_tree(src, ["a"], src / "b" / "tree", prune=True)
    with pytest.raises(SystemExit, match="inside its source"):
        pa.link_tree(src / "a", ["x"], src, prune=True)
    assert tree(src) == before


def test_orig_macro_parts_checked_before_scoring_041_r27_044_r8(tmp_path, monkeypatch):
    src = botgen(tmp_path, monkeypatch, (*pa.ORIG, *ADDED), listed=(*pa.ORIG, ADDED[0]))
    cls = tmp_path / "classes.json"
    cls.write_text(json.dumps({"x": "rush"}), encoding="utf-8")
    out_json = tmp_path / "out.json"
    args = ["--checkpoint", str(tmp_path / "ck"), "--out", str(out_json), "--holdout", "orig", "--classes", str(cls)]

    def unreachable(*a, **k):
        raise AssertionError("score() ran before the manifest check")

    monkeypatch.setattr(pa, "score", unreachable)
    with pytest.raises(SystemExit, match="are not the manifest's parts"):
        pa.main(args)
    assert not out_json.exists()

    write_manifest(src, (*pa.ORIG, *ADDED))
    calls = []

    def fake(ck, datasets, shards, backend, config, classes=None, chain_start=None, **k):
        calls.append((list(shards), chain_start))
        early = {"before_frame": pa.BUNKER_BEFORE, "steps": 0, "mean_p": None}
        return {"games": 1, "steps": 0, "per_action": [], "by_class": {"macro": {"games": 1, "bunker_p_before_4m": early}},
                "r27": {"rush_bunker_top1": None, "rush_bunker_steps": 0, "rush_games": 0, "macro_bunker_p_before_4m": None,
                        "macro_steps_before_4m": 0, "macro_games": 0}}

    monkeypatch.setattr(pa, "score", fake)
    taken = []
    real_added = pa.added_dirs
    monkeypatch.setattr(pa, "added_dirs", lambda: taken.append(1) or real_added())
    pa.main(args)
    assert len(taken) == 1
    (orig,), (added,) = pa.holdout_dirs("orig")[1], real_added()
    assert calls == [([orig], None), ([added], None), ([orig], ["botgen"]), ([added], ["botgen"])]
    out = json.loads(out_json.read_text(encoding="utf-8"))
    assert out["r27"]["macro_holdout"] == "full" and out["r27"]["macro_added_games"] == 1
