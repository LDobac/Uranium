import hashlib
import json
import os
from pathlib import Path

import numpy as np

LARGE = ("h", "y", "priv", "logp")
BLOCK_ROWS = 65536


def read_schema(dump) -> dict:
    js = json.loads((Path(dump) / "rows" / "schema.json").read_text(encoding="utf-8"))
    if js["rows"] is None:
        raise SystemExit(f"{dump}: unfinished plan-dump (no row count)")
    return js


class FileField:
    def __init__(self, path: Path, n: int, dtype: str, shape: tuple):
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self.shape = (n, *shape)
        self.row_items = int(np.prod(shape, dtype=np.int64))
        self.row_bytes = self.dtype.itemsize * self.row_items

    def __len__(self) -> int:
        return self.shape[0]

    def _span(self, f, a: int, b: int) -> np.ndarray:
        f.seek(a * self.row_bytes)
        out = np.fromfile(f, self.dtype, (b - a) * self.row_items)
        if len(out) != (b - a) * self.row_items:
            raise SystemExit(f"{self.path}: short read at rows {a}..{b}")
        return out.reshape(b - a, *self.shape[1:])

    def read(self, idx) -> np.ndarray:
        idx = np.asarray(idx, np.int64)
        out = np.empty((len(idx), *self.shape[1:]), self.dtype)
        if not len(idx):
            return out
        if idx.min() < 0 or idx.max() >= self.shape[0]:
            raise IndexError(f"{self.path}: row index out of range")
        o = np.argsort(idx, kind="stable")
        s = idx[o]
        with open(self.path, "rb") as f:
            lo = 0
            while lo < len(s):
                a = int(s[lo])
                hi = int(np.searchsorted(s, a + BLOCK_ROWS, side="left"))
                b = int(s[hi - 1]) + 1
                out[o[lo:hi]] = self._span(f, a, b)[s[lo:hi] - a]
                lo = hi
        return out

    def __getitem__(self, key):
        if isinstance(key, slice):
            a, b, step = key.indices(self.shape[0])
            return self.read(np.arange(a, b, step))
        if np.isscalar(key):
            return self.read(np.array([int(key)]))[0]
        return self.read(key)

    def __array__(self, dtype=None, copy=None):
        with open(self.path, "rb") as f:
            x = self._span(f, 0, self.shape[0])
        return x if dtype is None else x.astype(dtype)


def file_rows(dump) -> dict:
    js = read_schema(dump)
    d = Path(dump) / "rows"
    return {k: FileField(d / f"{k}.bin", js["rows"], f["dtype"], tuple(f["shape"])) for k, f in js["fields"].items()}


def load(dump) -> tuple[dict, dict, dict]:
    dump = Path(dump)
    man = json.loads((dump / "manifest.json").read_text(encoding="utf-8"))
    games = {int(k): v for k, v in json.loads((dump / "games.json").read_text(encoding="utf-8")).items()}
    return file_rows(dump), games, man


def within(path: Path, root: Path) -> bool:
    p, r = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(root))
    return p == r or p.startswith(r.rstrip("\\/") + os.sep)


def check_local(root: Path) -> None:
    shared = os.environ.get("URANIUM_SHARED_DIR")
    if shared and within(root, Path(shared)):
        raise SystemExit(f"--mmap {root}: inside the shared drive {shared}; give a local folder (053-R6)")


class Column:
    def __init__(self, path: Path, total: int, shape: tuple, dtype):
        self.path, self.total, self.shape, self.dtype = path, total, tuple(shape), np.dtype(dtype)
        self.done = 0
        self.part = path.with_suffix(".part")
        self.f = open(self.part, "wb")

    def __setitem__(self, key: slice, values) -> None:
        if not isinstance(key, slice) or key.start != self.done:
            raise ValueError(f"{self.path.name}: columns are written in row order")
        v = np.ascontiguousarray(values, self.dtype)
        if v.shape[1:] != self.shape or len(v) != key.stop - key.start:
            raise ValueError(f"{self.path.name}: shape {v.shape}, expected [{key.stop - key.start}, *{self.shape}]")
        self.f.write(v.tobytes())
        self.done += len(v)

    def finish(self) -> np.ndarray:
        self.f.close()
        if self.done != self.total:
            raise ValueError(f"{self.path.name}: {self.done} rows written, {self.total} expected")
        os.replace(self.part, self.path)
        return open_column(self.path, self.total, self.shape, self.dtype)


def open_column(path: Path, n: int, shape: tuple, dtype) -> np.ndarray:
    if not n:
        return np.zeros((0, *shape), dtype)
    return np.memmap(path, dtype=dtype, mode="r", shape=(n, *shape))


def _digest(parts: list) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else json.dumps(p, sort_keys=True).encode())
    return h.hexdigest()


class Store:
    def __init__(self, root):
        self.root = Path(root)
        check_local(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.key = None
        self.reused = []
        self.written = []

    def large(self, field: str) -> bool:
        return field in LARGE

    def bind(self, sources: list, sel: list, overrides: list) -> None:
        parts = []
        for s, (keep, game), ovr in zip(sources, sel, overrides):
            js = read_schema(s.path)
            parts += [os.path.abspath(s.path), js["rows"], js["fields"], keep.astype(np.int64).tobytes(), sorted(ovr)]
            for f in LARGE:
                st = (Path(s.path) / "rows" / f"{f}.bin").stat()
                parts.append([f, st.st_size, st.st_mtime_ns])
        self.key = _digest(parts)

    def cached(self, field: str, total: int, shape: tuple, dtype) -> np.ndarray | None:
        meta = self.root / f"{field}.json"
        path = self.root / f"{field}.bin"
        if not meta.is_file() or not path.is_file():
            return None
        m = json.loads(meta.read_text(encoding="utf-8"))
        want = {"key": self.key, "rows": total, "shape": list(shape), "dtype": np.dtype(dtype).str}
        if m != want or path.stat().st_size != total * np.dtype(dtype).itemsize * int(np.prod(shape, dtype=np.int64)):
            return None
        self.reused.append(field)
        return open_column(path, total, tuple(shape), dtype)

    def column(self, field: str, total: int, shape: tuple, dtype) -> Column:
        (self.root / f"{field}.json").unlink(missing_ok=True)
        return Column(self.root / f"{field}.bin", total, shape, dtype)

    def finish(self, field: str, col: Column) -> np.ndarray:
        arr = col.finish()
        meta = {"key": self.key, "rows": col.total, "shape": list(col.shape), "dtype": col.dtype.str}
        (self.root / f"{field}.json").write_text(json.dumps(meta), encoding="utf-8")
        self.written.append(field)
        return arr

    def describe(self) -> dict:
        return {"key": self.key, "reused": self.reused, "written": self.written}
