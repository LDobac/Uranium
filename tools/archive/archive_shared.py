"""Packs finished shared-drive data into archive/<unit>.tar.zst and restores it (061-R17)"""

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import tarfile
import time
from pathlib import Path

from compression.zstd import CompressionParameter, ZstdError

CHUNK = 8 << 20
KEPT_TOP = ("archive", "datasets", "locks", "queue", "runs", "workers")
BAD_CHARS = set('<>:"|?*')
REPO = Path(__file__).resolve().parents[2]


class ItemError(Exception):
    pass


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
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=1, ensure_ascii=False))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def file_sha256(path):
    h = hashlib.sha256()
    with open_server(path) as f:
        while b := f.read(CHUNK):
            h.update(b)
    return h.hexdigest()


def index_get(idx_path, unit):
    return next((u for u in read_json(idx_path, {"units": []})["units"] if u["unit"] == unit), None)


def index_put(idx_path, entry):
    index = read_json(idx_path, {"units": []})
    index["units"] = [u for u in index["units"] if u["unit"] != entry["unit"]] + [entry]
    write_json(idx_path, index)


def bad_name(n):
    return n in ("", ".", "..") or n[-1] in ". " or any(c in BAD_CHARS or ord(c) < 32 for c in n)


def good_unit(u):
    return isinstance(u, str) and not bad_name(u) and "/" not in u and "\\" not in u


def split_item(s):
    if not isinstance(s, str) or not s or s[0] in "/\\" or re.match(r"[A-Za-z]:", s):
        return None
    pieces = re.split(r"[\\/]", s)
    return None if any(bad_name(p) for p in pieces) else pieces


def parts(p):
    return tuple(x.casefold() for x in p.split("/") if x)


def overlaps(a, b):
    pa, pb = parts(a), parts(b)
    n = min(len(pa), len(pb))
    return pa[:n] == pb[:n]


def under(path, prefix):
    pp, px = parts(path), parts(prefix)
    return pp[: len(px)] == px


def is_link(e):
    return e.is_symlink() or e.is_junction()


def resolve_item(root, pieces):
    cur, real = root, []
    for p in pieces:
        try:
            with os.scandir(cur) as it:
                hits = [e for e in it if e.name.casefold() == p.casefold()]
        except (FileNotFoundError, NotADirectoryError):
            return None
        if not hits:
            return None
        if len(hits) > 1:
            raise ItemError(f"ambiguous case for {p} under {'/'.join(real) or '.'}")
        if is_link(hits[0]):
            raise ItemError(f"link at {'/'.join(real + [hits[0].name])}")
        real.append(hits[0].name)
        cur = Path(hits[0].path)
    rel = "/".join(real)
    rp, rr = (root / rel).resolve(), root.resolve()
    if rp == rr or not rp.is_relative_to(rr):
        raise ItemError(f"outside the shared root: {rel}")
    return rel


def scan(root, rel):
    p = root / rel
    if is_link(p):
        raise ItemError(f"link at {rel}")
    if p.is_file():
        return [rel], [], []
    files, empty, names, stack = [], [], [], [rel]
    while stack:
        d = stack.pop()
        names.append(d.rsplit("/", 1)[-1])
        with os.scandir(root / d) as it:
            entries = sorted(it, key=lambda e: e.name)
        if not entries:
            empty.append(d)
        subdirs = []
        for e in entries:
            r = f"{d}/{e.name}"
            if is_link(e):
                raise ItemError(f"link at {r}")
            if e.is_dir(follow_symlinks=False):
                subdirs.append(r)
            elif e.is_file(follow_symlinks=False):
                files.append(r)
            else:
                raise ItemError(f"not a regular file: {r}")
        stack.extend(reversed(subdirs))
    return files, empty, names


def lock_names(root):
    d = root / "locks"
    return [p.stem for p in sorted(d.glob("*.json"))] if d.is_dir() else []


def lock_hit(rel, names, locks):
    folders = {n.casefold() for n in names} | set(parts(rel))
    for name in locks:
        for x in {name, name[len("eval-"):] if name.startswith("eval-") else name}:
            prefixes = [f"checkpoints/{x}", f"eval/{x}"]
            if "__" in x:
                run, ckpt = x.split("__", 1)
                prefixes += [f"checkpoints/{run}/{ckpt}", f"artifacts/diag/{x}"]
            if hit := next((pre for pre in prefixes if overlaps(rel, pre)), None):
                return f"locks/{name}.json ({hit})"
            if x.casefold() in folders:
                return f"locks/{name}.json (folder {x})"
    return None


def prepare(root, unit, paths):
    chosen = []
    for raw in paths:
        pieces = split_item(raw)
        if pieces is None:
            raise ItemError(f"bad item path {raw!r}")
        if pieces[0].casefold() in KEPT_TOP:
            log(f"  {unit}: skip {raw}: kept in place (061-R17)")
            continue
        rel = resolve_item(root, pieces)
        if rel is None:
            log(f"  {unit}: skip {raw}: missing")
            continue
        if any(under(rel, c) for c in chosen):
            log(f"  {unit}: skip {rel}: inside another item")
            continue
        chosen = [c for c in chosen if not under(c, rel)] + [rel]
    locks = lock_names(root)
    out = []
    for rel in chosen:
        files, empty, names = scan(root, rel)
        if hit := lock_hit(rel, names, locks):
            log(f"  {unit}: skip {rel}: {hit}")
            continue
        out.append((rel, files, empty))
    return out


def recheck(root, items):
    locks = lock_names(root)
    for rel in items:
        real = resolve_item(root, rel.split("/"))
        if real is None:
            continue
        _, _, names = scan(root, real)
        if hit := lock_hit(real, names, locks):
            return f"{rel}: {hit}"
    return None


def pack(root, unit, prepared, adir, workers, level):
    files = [f for _, fs, _ in prepared for f in fs]
    dirs = [d for _, _, ds in prepared for d in ds]
    tmp = adir / f"{unit}.tar.zst.tmp"
    records = []
    opts = {CompressionParameter.nb_workers: workers, CompressionParameter.compression_level: level}
    t, done = time.time(), 0
    with open(tmp, "wb") as raw:
        hw = HashWriter(raw)
        with tarfile.open(fileobj=hw, mode="w:zst", options=opts, copybufsize=CHUNK) as tar:
            for rel in dirs:
                tar.addfile(tar.gettarinfo(str(root / rel), arcname=rel))
            for i, rel in enumerate(files):
                src = root / rel
                st0 = os.stat(src, follow_symlinks=False)
                ti = tar.gettarinfo(str(src), arcname=rel)
                if not ti.isfile():
                    raise ItemError(f"not a regular file: {rel}")
                with open(src, "rb") as f:
                    hr = HashReader(f)
                    tar.addfile(ti, hr)
                st1 = os.stat(src, follow_symlinks=False)
                if hr.n != ti.size or (st1.st_size, st1.st_mtime_ns) != (st0.st_size, st0.st_mtime_ns):
                    raise ItemError(f"changed while reading: {rel}")
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
            with tarfile.open(fileobj=hr, mode="r|zst", copybufsize=CHUNK) as tar:
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
            drain(hr)
        except (tarfile.TarError, ZstdError, OSError, EOFError) as e:
            return f"unreadable archive: {e}"
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
            st = os.stat(p, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as e:
            log(f"  warning: {m['path']}: {e}, kept")
            kept += 1
            continue
        if (st.st_size, st.st_mtime_ns) != (m["size"], m["mtime"]):
            log(f"  warning: {m['path']} changed after packing, kept")
            kept += 1
            continue
        try:
            os.remove(p)
        except OSError as e:
            log(f"  warning: {m['path']}: {e}, kept")
            kept += 1
    for it in items:
        p = root / it
        if not p.is_dir() or is_link(p):
            continue
        for r, ds, fs in os.walk(p, topdown=False, onerror=lambda e: log(f"  warning: {e}")):
            for f in fs:
                rel = Path(r, f).relative_to(root).as_posix()
                if rel not in listed:
                    log(f"  warning: {rel} is not in the manifest, kept")
                    kept += 1
            try:
                if not os.listdir(r):
                    os.rmdir(r)
            except OSError as e:
                log(f"  warning: {e}")
    return kept


def finish(root, unit, entry, idx_path, adir, resume):
    manifest = read_json(adir / f"{unit}.manifest.json", None)
    if manifest is None:
        log(f"FAIL {unit}: manifest missing, originals kept")
        return False
    if resume:
        final = adir / f"{unit}.tar.zst"
        if not final.is_file() or final.stat().st_size != entry["archive_bytes"]:
            log(f"FAIL {unit}: archive missing or size differs from INDEX, originals kept")
            return False
        sha = file_sha256(final)
        if sha != entry["archive_sha256"] or sha != manifest["archive_sha256"]:
            log(f"FAIL {unit}: archive sha256 differs from INDEX or manifest, originals kept")
            return False
    try:
        hit = recheck(root, entry["items"])
    except (ItemError, OSError) as e:
        hit = str(e)
    if hit:
        log(f"FAIL {unit}: {hit}; originals kept, rerun to resume")
        return False
    kept = remove_originals(root, entry["items"], manifest["files"])
    entry["originals_removed"] = True
    index_put(idx_path, entry)
    log(f"done {unit}: originals removed" + (f", {kept} changed or unlisted files kept" if kept else ""))
    return True


def pack_unit(root, unit, paths, adir, idx_path, workers, level):
    if not good_unit(unit):
        log(f"FAIL {unit!r}: bad unit name")
        return False
    entry = index_get(idx_path, unit)
    if entry is not None:
        if entry.get("originals_removed"):
            log(f"FAIL {unit}: already in INDEX, refused")
            return False
        log(f"resume {unit}: removing originals")
        return finish(root, unit, entry, idx_path, adir, resume=True)
    final = adir / f"{unit}.tar.zst"
    man_path = adir / f"{unit}.manifest.json"
    tmp = adir / f"{unit}.tar.zst.tmp"
    if final.exists():
        log(f"FAIL {unit}: {final.name} exists, refused")
        return False
    try:
        prepared = prepare(root, unit, paths)
    except (ItemError, OSError) as e:
        log(f"FAIL {unit}: {e}; nothing packed")
        return False
    if not prepared:
        log(f"FAIL {unit}: nothing to pack")
        return False
    items = [rel for rel, _, _ in prepared]
    log(f"start {unit}: {len(items)} items")
    try:
        tmp, records, dirs, sha = pack(root, unit, prepared, adir, workers, level)
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
    index_put(idx_path, entry)
    return finish(root, unit, entry, idx_path, adir, resume=False)


def shared_local():
    v = os.environ.get("URANIUM_SHARED_LOCAL")
    env = REPO / ".env"
    if v is None and env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            k, sep, val = line.partition("=")
            if sep and k.strip() == "URANIUM_SHARED_LOCAL":
                v = val.strip().strip("\"'")
    return v == "1"


def take_pack_lock(adir):
    p = adir / ".pack.lock"
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        log(f"FAIL: {p} exists; another pack may be running. If none is, remove it by hand")
        return None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}))
    return p


def cmd_pack(a):
    root = Path(a.shared)
    if os.name == "nt" and not shared_local():
        log("FAIL: on Windows pack runs only on the share-providing PC (URANIUM_SHARED_LOCAL=1, 061-R12)")
        return 1
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    units = list(plan["units"])
    if a.units is not None:
        unknown = [u for u in a.units if u not in plan["units"]]
        if not a.units or unknown:
            log(f"FAIL: --units needs names from the plan; unknown {unknown}")
            return 1
        units = [u for u in units if u in a.units]
    adir = root / "archive"
    adir.mkdir(exist_ok=True)
    lock = take_pack_lock(adir)
    if lock is None:
        return 1
    failed = []
    try:
        for unit in units:
            paths = [i["path"] for i in plan["units"][unit]["items"]]
            if not pack_unit(root, unit, paths, adir, adir / "INDEX.json", a.workers, a.level):
                failed.append(unit)
    finally:
        lock.unlink(missing_ok=True)
    log(f"finished; failed: {failed}")
    return 1 if failed else 0


def restore_member(tar, ti, root, m):
    safe = tarfile.data_filter(ti, str(root))
    dst = root / safe.name
    if dst.exists():
        return f"exists, not overwritten: {ti.name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.restore-{os.getpid()}.tmp")
    try:
        h, n = hashlib.sha256(), 0
        src = tar.extractfile(ti)
        with open(tmp, "wb") as f:
            while b := src.read(CHUNK):
                h.update(b)
                n += len(b)
                f.write(b)
        if n != m["size"] or h.hexdigest() != m["sha256"]:
            return f"mismatch {ti.name}"
        if dst.exists():
            return f"exists, not overwritten: {ti.name}"
        os.replace(tmp, dst)
        if "mtime" in m:
            os.utime(dst, ns=(m["mtime"], m["mtime"]))
        return None
    finally:
        tmp.unlink(missing_ok=True)


def cmd_restore(a):
    root = Path(a.shared)
    if not good_unit(a.unit):
        log(f"FAIL {a.unit!r}: bad unit name")
        return 1
    adir = root / "archive"
    archive = adir / f"{a.unit}.tar.zst"
    manifest = read_json(adir / f"{a.unit}.manifest.json", None)
    if manifest is None or not archive.is_file():
        log(f"FAIL {a.unit}: archive or manifest missing")
        return 1
    only = []
    for o in a.only or []:
        pieces = split_item(o)
        if pieces is None:
            log(f"FAIL {a.unit}: bad --only path {o!r}")
            return 1
        only.append("/".join(pieces))
    if file_sha256(archive) != manifest["archive_sha256"]:
        log(f"FAIL {a.unit}: archive sha256 differs from manifest, nothing restored")
        return 1
    picked = lambda p: not only or any(under(p, o) for o in only)
    want = {m["path"]: m for m in manifest["files"] if picked(m["path"])}
    if not want and not any(picked(d) for d in manifest.get("dirs", [])):
        log(f"FAIL {a.unit}: nothing matches {only}")
        return 1
    clash = [p for p in want if (root / p).exists()]
    if clash:
        log(f"FAIL {a.unit}: {len(clash)} files already exist, nothing restored, e.g. {clash[:3]}")
        return 1
    errors, seen = [], set()
    listed = {m["path"] for m in manifest["files"]}
    try:
        with open(archive, "rb") as raw, tarfile.open(fileobj=raw, mode="r|zst", copybufsize=CHUNK) as tar:
            for ti in tar:
                try:
                    if ti.isdir():
                        if picked(ti.name):
                            (root / tarfile.data_filter(ti, str(root)).name).mkdir(parents=True, exist_ok=True)
                    elif ti.name not in want:
                        if ti.name not in listed:
                            errors.append(f"not in manifest: {ti.name}")
                    elif err := restore_member(tar, ti, root, want[ti.name]):
                        errors.append(err)
                    else:
                        seen.add(ti.name)
                except (tarfile.TarError, OSError) as e:
                    errors.append(f"{ti.name}: {e}")
    except (tarfile.TarError, ZstdError, OSError, EOFError) as e:
        errors.append(f"unreadable archive: {e}")
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
