import importlib.util
import json
import os
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "archive" / "archive_shared.py"


@pytest.fixture
def tool():
    spec = importlib.util.spec_from_file_location("archive_shared", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tree(root: Path) -> dict:
    out = {}
    for r, ds, fs in os.walk(root):
        rel = Path(r).relative_to(root).as_posix()
        if not ds and not fs:
            out[rel + "/"] = None
        for f in fs:
            out[Path(rel, f).as_posix()] = Path(r, f).read_bytes()
    return out


@pytest.fixture
def shared(tmp_path: Path) -> Path:
    root = tmp_path / "shared"
    for run in ("old-a", "old-b"):
        for step in (10, 20):
            d = root / "checkpoints" / run / f"ckpt-{step}"
            d.mkdir(parents=True)
            (d / "model.safetensors").write_bytes(os.urandom(5000) + bytes(step))
            (d / "meta.json").write_text(json.dumps({"step": step}), encoding="utf-8")
        (root / "checkpoints" / run / "LATEST").write_text("ckpt-20", encoding="utf-8")
    (root / "eval" / "job-x" / "games" / "empty").mkdir(parents=True)
    (root / "eval" / "job-x" / "result.json").write_text("{}", encoding="utf-8")
    (root / "locks").mkdir()
    return root


def plan(tmp_path: Path, units: dict) -> Path:
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"units": {u: {"items": [{"path": x} for x in items]} for u, items in units.items()}}), encoding="utf-8")
    return p


def index(root: Path) -> dict:
    return json.loads((root / "archive" / "INDEX.json").read_text(encoding="utf-8"))


def pack(tool, root, plan_path, *units):
    args = ["pack", "--shared", str(root), "--plan", str(plan_path), "--workers", "0", "--level", "1"]
    return tool.main(args + (["--units", *units] if units else []))


def test_pack_restore_round_trip_061_ac10_061_r17(tool, shared, tmp_path):
    items = ["checkpoints/old-a/ckpt-10", "eval/job-x"]
    before = {k: v for k, v in tree(shared).items() if k.startswith(tuple(items))}
    assert "eval/job-x/games/empty/" in before
    assert pack(tool, shared, plan(tmp_path, {"u1": items})) == 0
    assert not (shared / "checkpoints/old-a/ckpt-10").exists() and not (shared / "eval/job-x").exists()
    assert (shared / "checkpoints/old-a/ckpt-20/model.safetensors").exists()
    e = index(shared)["units"][0]
    assert e["unit"] == "u1" and e["items"] == items and e["files"] == 3 and e["originals_removed"] is True
    man = json.loads((shared / "archive/u1.manifest.json").read_text(encoding="utf-8"))
    assert man["archive_sha256"] == e["archive_sha256"] and {"path", "size", "mtime", "sha256"} <= set(man["files"][0])
    assert not list((shared / "archive").glob("*.tmp"))
    assert tool.main(["restore", "--shared", str(shared), "--unit", "u1"]) == 0
    after = {k: v for k, v in tree(shared).items() if k.startswith(tuple(items))}
    assert after == before


def test_tampered_archive_keeps_originals_061_ac10_061_r17(tool, shared, tmp_path, monkeypatch):
    real = tool.verify

    def tampered(path, manifest):
        b = bytearray(Path(path).read_bytes())
        b[len(b) // 2] ^= 0xFF
        Path(path).write_bytes(bytes(b))
        return real(path, manifest)

    monkeypatch.setattr(tool, "verify", tampered)
    before = tree(shared)
    assert pack(tool, shared, plan(tmp_path, {"u1": ["checkpoints/old-a"]})) == 1
    assert tree(shared) == before | {"archive/": None}
    assert not (shared / "archive/u1.tar.zst").exists() and not (shared / "archive/u1.tar.zst.tmp").exists()
    assert not (shared / "archive/u1.manifest.json").exists()


def test_verify_rejects_extra_and_missing_members_061_ac10(tool, shared, tmp_path):
    assert pack(tool, shared, plan(tmp_path, {"u2": ["checkpoints/old-a"]})) == 0
    path = shared / "archive/u2.tar.zst"
    man = json.loads((shared / "archive/u2.manifest.json").read_text(encoding="utf-8"))
    assert tool.verify(path, man) is None
    assert "missing" in tool.verify(path, {**man, "files": man["files"] + [{"path": "x", "size": 1, "sha256": "0"}]})
    assert "unexpected" in tool.verify(path, {**man, "files": man["files"][1:]})
    assert "sha256" in tool.verify(path, {**man, "archive_sha256": "0" * 64})


def test_locked_items_are_left_061_ac10_061_r17(tool, shared, tmp_path):
    (shared / "locks/old-a.json").write_text("{}", encoding="utf-8")
    (shared / "locks/eval-job-x.json").write_text("{}", encoding="utf-8")
    assert pack(tool, shared, plan(tmp_path, {"u1": ["checkpoints/old-a/ckpt-10", "eval/job-x", "checkpoints/old-b/ckpt-10"]})) == 0
    assert (shared / "checkpoints/old-a/ckpt-10/meta.json").exists() and (shared / "eval/job-x/result.json").exists()
    assert not (shared / "checkpoints/old-b/ckpt-10").exists()
    assert index(shared)["units"][0]["items"] == ["checkpoints/old-b/ckpt-10"]
    (shared / "locks/old-b__ckpt-20.json").write_text("{}", encoding="utf-8")
    assert pack(tool, shared, plan(tmp_path, {"u2": ["checkpoints/old-b"]})) == 1
    assert (shared / "checkpoints/old-b/ckpt-20").exists() and not (shared / "archive/u2.tar.zst").exists()


def test_changed_original_is_not_removed_061_ac10_061_r17(tool, shared, tmp_path, monkeypatch):
    real = tool.remove_originals
    target = shared / "checkpoints/old-a/ckpt-10/meta.json"

    def fail_once(root, items, files):
        monkeypatch.setattr(tool, "remove_originals", real)
        raise OSError("share dropped")

    monkeypatch.setattr(tool, "remove_originals", fail_once)
    p = plan(tmp_path, {"u1": ["checkpoints/old-a/ckpt-10"]})
    with pytest.raises(OSError):
        pack(tool, shared, p)
    assert index(shared)["units"][0]["originals_removed"] is False
    target.write_text('{"step": 11}', encoding="utf-8")
    (shared / "checkpoints/old-a/ckpt-10/new.txt").write_text("late", encoding="utf-8")
    assert pack(tool, shared, p) == 0
    assert index(shared)["units"][0]["originals_removed"] is True
    assert target.read_text(encoding="utf-8") == '{"step": 11}'
    assert (shared / "checkpoints/old-a/ckpt-10/new.txt").exists()
    assert not (shared / "checkpoints/old-a/ckpt-10/model.safetensors").exists()


def test_existing_unit_name_is_refused_061_ac10_061_r17(tool, shared, tmp_path):
    assert pack(tool, shared, plan(tmp_path, {"u1": ["checkpoints/old-a/ckpt-10"]})) == 0
    data = (shared / "archive/u1.tar.zst").read_bytes()
    assert pack(tool, shared, plan(tmp_path, {"u1": ["checkpoints/old-b/ckpt-10"]})) == 1
    assert (shared / "checkpoints/old-b/ckpt-10").exists() and (shared / "archive/u1.tar.zst").read_bytes() == data
    (shared / "archive/u2.tar.zst").write_bytes(b"x")
    assert pack(tool, shared, plan(tmp_path, {"u2": ["checkpoints/old-b/ckpt-10"]})) == 1
    assert (shared / "checkpoints/old-b/ckpt-10").exists() and (shared / "archive/u2.tar.zst").read_bytes() == b"x"


def test_restore_checks_manifest_and_never_overwrites_061_ac10_061_r17(tool, shared, tmp_path):
    assert pack(tool, shared, plan(tmp_path, {"u1": ["checkpoints/old-a"]})) == 0
    restore = ["restore", "--shared", str(shared), "--unit", "u1"]
    assert tool.main(restore + ["--only", "checkpoints/old-a/ckpt-20"]) == 0
    assert (shared / "checkpoints/old-a/ckpt-20/meta.json").exists() and not (shared / "checkpoints/old-a/ckpt-10").exists()
    meta = shared / "checkpoints/old-a/ckpt-20/meta.json"
    meta.write_text("local edit", encoding="utf-8")
    assert tool.main(restore) == 1
    assert meta.read_text(encoding="utf-8") == "local edit" and not (shared / "checkpoints/old-a/ckpt-10").exists()
    man_path = shared / "archive/u1.manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    for m in man["files"]:
        if m["path"] == "checkpoints/old-a/ckpt-10/meta.json":
            m["sha256"] = "0" * 64
    man_path.write_text(json.dumps(man), encoding="utf-8")
    assert tool.main(restore + ["--only", "checkpoints/old-a/ckpt-10"]) == 1
