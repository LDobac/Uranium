"""Packs finished shared-drive data into archive/<unit>.tar.zst and restores it (061-R17)"""

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path

from compression.zstd import CompressionParameter, ZstdError

CHUNK = 8 << 20
KEPT_TOP = ("archive", "datasets", "locks", "queue", "runs", "workers")


class HashReader:
    def __init__(self, f):
        self.f = f
        self.h = hashlib.sha256()
        self.n = 0

    def read(self, size=-1):
        b = self.f.read(size)
        self.h.update(b)
        self.n += len(b)
        return b


class HashWriter:
    def __init__(self, f):
        self.f = f
        self.h = hashlib.sha256()
        self.n = 0

    def write(self, b):
        self.h.update(b)
        self.n += len(b)
        return self.f.write(b)

    def flush(self):
        self.f.flush()


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def write_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=1, ensure_ascii=False))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def parts(p):
    return tuple(x.casefold() for x in p.strip("/").split("/") if x)


def overlaps(a, b):
    pa, pb = parts(a), parts(b)
    n = min(len(pa), len(pb))
    return pa[:n] == pb[:n]


def under(path, prefix):
    pp, px = parts(path), parts(prefix)
    return pp[: len(px)] == px


def lock_prefixes(root):
    out = []
    for p in sorted((root / "locks").glob("*.json")):
        name = p.stem
        x = name[len("eval-"):] if name.startswith("eval-") else name
        out += [f"checkpoints/{x}", f"eval/{x}"]
        if "__" in x:
            run, ckpt = x.split("__", 1)
            out += [f"checkpoints/{run}/{ckpt}", f"artifacts/diag/{x}"]
    return out


def select_items(root, unit, paths):
    locks = lock_prefixes(root)
    items = []
    for it in paths:
        it = Path(it).as_posix().strip("/")
        if not parts(it) or parts(it)[0] in KEPT_TOP:
            log(f"  {unit}: skip {it}: kept in place (061-R17)")
        elif not (root / it).exists():
            log(f"  {unit}: skip {it}: missing")
        elif hit := [l for l in locks if overlaps(it, l)]:
            log(f"  {unit}: skip {it}: lock present for {hit[0]}")
        elif it in items:
            continue
        else:
            items.append(it)
    return items


def list_entries(root, items):
    files, dirs, seen = [], [], set()
    for it in items:
        p = root / it
        if p.is_file():
            walk = [(str(p.parent), [], [p.name])]
        else:
            walk = []
            for r, ds, fs in os.walk(p):
                ds.sort()
                walk.append((r, ds, sorted(fs)))
        for r, ds, fs in walk:
            if not ds and not fs and Path(r) != root:
                rel = Path(r).relative_to(root).as_posix()
                if rel not in seen:
                    seen.add(rel)
                    dirs.append(rel)
            for f in fs:
                rel = Path(r, f).relative_to(root).as_posix()
                if rel not in seen:
                    seen.add(rel)
                    files.append(rel)
    return files, dirs


def pack(root, unit, items, adir, workers, level):
    files, dirs = list_entries(root, items)
    tmp = adir / f"{unit}.tar.zst.tmp"
    records = []
    opts = {CompressionParameter.nb_workers: workers, CompressionParameter.compression_level: level}
    t, done = time.time(), 0
    with open(tmp, "wb") as raw:
        hw = HashWriter(raw)
        with tarfile.open(fileobj=hw, mode="w:zst", options=opts) as tar:
            for rel in dirs:
                tar.addfile(tar.gettarinfo(str(root / rel), arcname=rel))
            for i, rel in enumerate(files):
                src = root / rel
                st0 = os.stat(src)
                ti = tar.gettarinfo(str(src), arcname=rel)
                if not ti.isfile():
                    raise RuntimeError(f"not a regular file: {rel}")
                with open(src, "rb") as f:
                    hr = HashReader(f)
                    tar.addfile(ti, hr)
                st1 = os.stat(src)
                if hr.n != ti.size or (st1.st_size, st1.st_mtime_ns) != (st0.st_size, st0.st_mtime_ns):
                    raise RuntimeError(f"changed while reading: {rel}")
                records.append({"path": rel, "size": ti.size, "mtime": st0.st_mtime_ns, "sha256": hr.h.hexdigest()})
                done += ti.size
                if i % 200 == 0:
                    log(f"  pack {unit} {i + 1}/{len(files)} {done / 1e9:.1f} GB {done / 1e6 / max(time.time() - t, 1e-3):.0f} MB/s")
        raw.flush()
        os.fsync(raw.fileno())
    return tmp, records, dirs, hw.h.hexdigest()


def open_server(path):
    f = open(path, "rb")
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    elif os.environ.get("URANIUM_SHARED_LOCAL") != "1":
        log("  warning: verifying through the client cache; run on the share-providing PC (061-R12) to read the server copy")
    return f


def drain(f):
    while f.read(CHUNK):
        pass


def verify(path, manifest):
    want = {m["path"]: m for m in manifest["files"]}
    want_dirs = set(manifest.get("dirs", []))
    seen, seen_dirs = set(), set()
    with open_server(path) as raw:
        hr = HashReader(raw)
        try:
            with tarfile.open(fileobj=hr, mode="r|zst") as tar:
                for ti in tar:
                    if ti.isdir():
                        if ti.name not in want_dirs or ti.name in seen_dirs:
                            return f"unexpected or duplicate directory {ti.name}"
                        seen_dirs.add(ti.name)
                        continue
                    m = want.get(ti.name)
                    if not ti.isfile() or m is None or ti.name in seen:
                        return f"unexpected or duplicate member {ti.name}"
                    f = tar.extractfile(ti)
                    h, n = hashlib.sha256(), 0
                    while b := f.read(CHUNK):
                        h.update(b)
                        n += len(b)
                    if n != m["size"] or h.hexdigest() != m["sha256"]:
                        return f"mismatch {ti.name}"
                    seen.add(ti.name)
        except (tarfile.TarError, ZstdError, OSError, EOFError) as e:
            return f"unreadable archive: {e}"
        drain(hr)
    if hr.h.hexdigest() != manifest["archive_sha256"]:
        return "archive sha256 differs"
    missing = (set(want) - seen) | (want_dirs - seen_dirs)
    return f"missing {len(missing)} members, e.g. {sorted(missing)[:3]}" if missing else None


def remove_originals(root, items, files):
    kept = 0
    listed = {m["path"] for m in files}
    for m in files:
        p = root / m["path"]
        try:
            st = os.stat(p)
        except FileNotFoundError:
            continue
        if (st.st_size, st.st_mtime_ns) != (m["size"], m["mtime"]):
            log(f"  warning: {m['path']} changed after packing, kept")
            kept += 1
            continue
        os.remove(p)
    for it in items:
        p = root / it
        if not p.is_dir():
            continue
        for r, ds, fs in os.walk(p, topdown=False):
            for f in fs:
                rel = Path(r, f).relative_to(root).as_posix()
                if rel not in listed:
                    log(f"  warning: {rel} is not in the manifest, kept")
                    kept += 1
            if not os.listdir(r):
                os.rmdir(r)
    return kept


def finish(root, unit, entry, index, idx_path, adir):
    manifest = read_json(adir / f"{unit}.manifest.json", None)
    if manifest is None:
        log(f"FAIL {unit}: manifest missing, originals kept")
        return False
    kept = remove_originals(root, entry["items"], manifest["files"])
    entry["originals_removed"] = True
    write_json(idx_path, index)
    log(f"done {unit}: originals removed" + (f", {kept} changed or unlisted files kept" if kept else ""))
    return True


def pack_unit(root, unit, paths, adir, index, idx_path, workers, level):
    entry = next((u for u in index["units"] if u["unit"] == unit), None)
    if entry is not None:
        if entry.get("originals_removed"):
            log(f"FAIL {unit}: already in INDEX, refused")
            return False
        log(f"resume {unit}: removing originals")
        return finish(root, unit, entry, index, idx_path, adir)
    final = adir / f"{unit}.tar.zst"
    man_path = adir / f"{unit}.manifest.json"
    if final.exists():
        log(f"FAIL {unit}: {final.name} exists, refused")
        return False
    items = select_items(root, unit, paths)
    if not items:
        log(f"FAIL {unit}: nothing to pack")
        return False
    log(f"start {unit}: {len(items)} items")
    tmp = adir / f"{unit}.tar.zst.tmp"
    try:
        tmp, records, dirs, sha = pack(root, unit, items, adir, workers, level)
        manifest = {"unit": unit, "archive_sha256": sha, "dirs": dirs, "files": records}
        write_json(man_path, manifest)
        raw = sum(m["size"] for m in records)
        log(f"  packed {unit}: {len(records)} files {raw / 1e9:.2f} GB -> {tmp.stat().st_size / 1e9:.2f} GB, verifying")
        err = verify(tmp, manifest)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    if err:
        log(f"FAIL {unit}: {err}; originals kept, archive removed")
        tmp.unlink(missing_ok=True)
        man_path.unlink(missing_ok=True)
        return False
    os.replace(tmp, final)
    entry = {"unit": unit, "created": time.strftime("%Y-%m-%d %H:%M"), "items": items, "files": len(records),
             "raw_bytes": raw, "archive_bytes": final.stat().st_size, "archive_sha256": sha, "originals_removed": False}
    index["units"].append(entry)
    write_json(idx_path, index)
    return finish(root, unit, entry, index, idx_path, adir)


def cmd_pack(a):
    root = Path(a.shared)
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    adir = root / "archive"
    adir.mkdir(exist_ok=True)
    idx_path = adir / "INDEX.json"
    index = read_json(idx_path, {"units": []})
    failed = []
    for unit, spec in plan["units"].items():
        if a.units and unit not in a.units:
            continue
        if not pack_unit(root, unit, [i["path"] for i in spec["items"]], adir, index, idx_path, a.workers, a.level):
            failed.append(unit)
    log(f"finished; failed: {failed}")
    return 1 if failed else 0


def cmd_restore(a):
    root = Path(a.shared)
    adir = root / "archive"
    archive = adir / f"{a.unit}.tar.zst"
    manifest = read_json(adir / f"{a.unit}.manifest.json", None)
    if manifest is None or not archive.exists():
        log(f"FAIL {a.unit}: archive or manifest missing")
        return 1
    only = [Path(o).as_posix().strip("/") for o in a.only or []]
    want = {m["path"]: m for m in manifest["files"] if not only or any(under(m["path"], o) for o in only)}
    if not want and not any(any(under(d, o) for o in only) for d in manifest.get("dirs", [])):
        log(f"FAIL {a.unit}: nothing matches {only}")
        return 1
    clash = [p for p in want if (root / p).exists()]
    if clash:
        log(f"FAIL {a.unit}: {len(clash)} files already exist, nothing restored, e.g. {clash[:3]}")
        return 1
    errors, seen = [], set()
    listed = {m["path"] for m in manifest["files"]}
    with open(archive, "rb") as raw:
        hr = HashReader(raw)
        with tarfile.open(fileobj=hr, mode="r|zst") as tar:
            for ti in tar:
                if ti.isdir():
                    if only and not any(under(ti.name, o) for o in only):
                        continue
                    tar.extract(ti, root, filter="data")
                    continue
                if ti.name not in want:
                    if ti.name not in listed:
                        errors.append(f"not in manifest: {ti.name}")
                    continue
                tar.extract(ti, root, filter="data")
                h, n = hashlib.sha256(), 0
                with open(root / ti.name, "rb") as f:
                    while b := f.read(CHUNK):
                        h.update(b)
                        n += len(b)
                m = want[ti.name]
                if n != m["size"] or h.hexdigest() != m["sha256"]:
                    errors.append(f"mismatch {ti.name}")
                seen.add(ti.name)
        drain(hr)
    if hr.h.hexdigest() != manifest.get("archive_sha256"):
        errors.append("archive sha256 differs")
    errors += [f"missing {p}" for p in sorted(set(want) - seen)]
    for e in errors:
        log(f"  {a.unit}: {e}")
    log(f"{'FAIL' if errors else 'done'} {a.unit}: restored {len(seen)} files")
    return 1 if errors else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--shared", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--units", nargs="*")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--level", type=int, default=3)
    r = sub.add_parser("restore")
    r.add_argument("--shared", required=True)
    r.add_argument("--unit", required=True)
    r.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    return cmd_pack(a) if a.cmd == "pack" else cmd_restore(a)


if __name__ == "__main__":
    sys.exit(main())
