import numpy as np
import pytest

from uranium.diag.planning import pred as pp
from uranium.diag.planning import store as dst

from testkit import fake_dump

FIELDS = ("h", "y", "priv", "priv_ok", "own", "action", "delay", "weight", "logp", "mask", "win_open", "frame", "dec", "game")


def _sources(tmp_path):
    a = fake_dump(tmp_path / "a", n_games=10, rows=50, seed=1)
    b = fake_dump(tmp_path / "b", n_games=8, rows=40, seed=2, sha_prefix="b", chain_every=5)
    return [pp.Source("a", "teacher", "split", str(a)), pp.Source("b", "model", "split", str(b))]


def test_mmap_data_equals_ram_data_053_r6(tmp_path):
    srcs = _sources(tmp_path)
    ram = pp.Data(srcs, (2, 4), maps=[0, 1])
    mm = pp.Data(srcs, (2, 4), maps=[0, 1], mmap=tmp_path / "local")
    assert mm.N == ram.N and mm.N > 0
    for f in FIELDS:
        assert np.array_equal(np.asarray(getattr(mm, f)), np.asarray(getattr(ram, f))), f
    for f in dst.LARGE:
        assert isinstance(getattr(mm, f), np.memmap) and not isinstance(getattr(ram, f), np.memmap), f
    for k in (1, 2, 4):
        assert np.array_equal(mm.nxt[k], ram.nxt[k]), k
    for name in ("usable", "chained", "partner", "row_eval", "row_q", "is_op"):
        assert np.array_equal(getattr(mm, name), getattr(ram, name), equal_nan=name == "row_q"), name
    assert mm.split_digest == ram.split_digest
    rows = ram.pair_rows(2, "train")
    assert np.array_equal(rows, mm.pair_rows(2, "train"))
    st = ram.stats(rows, 2)
    st_m = mm.stats(rows, 2)
    for k in st:
        assert np.array_equal(st[k], st_m[k]), k
    idx = np.sort(rows[np.random.default_rng(0).integers(0, len(rows), 64)])
    for v in ("A", "H", "S", "U"):
        b, bm = ram.batch(idx, 2, v, st), mm.batch(idx, 2, v, st)
        assert b.keys() == bm.keys()
        for k in b:
            assert torch_equal(b[k], bm[k]), (v, k)
    assert np.array_equal(ram.gk(8, idx), mm.gk(8, idx))
    assert np.array_equal(ram.probs(idx), mm.probs(idx))
    assert pp.describe(ram) == pp.describe(mm)


def torch_equal(a, b) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and bool((a == b).all())


def test_mmap_cache_reuse_and_refresh_053_r6(tmp_path):
    srcs = _sources(tmp_path)
    local = tmp_path / "local"
    first = pp.Data(srcs, (2,), mmap=local)
    assert sorted(first.store.written) == sorted(dst.LARGE) and not first.store.reused
    again = pp.Data(srcs, (2,), mmap=local)
    assert sorted(again.store.reused) == sorted(dst.LARGE) and not again.store.written
    assert np.array_equal(np.asarray(again.h), np.asarray(first.h))
    assert first.store.dir == again.store.dir == local / first.store.key[:16]
    other = pp.Data(srcs, (2,), maps=[0], mmap=local)
    assert sorted(other.store.written) == sorted(dst.LARGE)
    assert other.store.dir != first.store.dir and np.array_equal(np.asarray(again.h), np.asarray(first.h)), "another selection gets its own folder"
    assert not list(local.rglob("*.part"))
    assert np.array_equal(np.asarray(other.h), np.asarray(pp.Data(srcs, (2,), maps=[0]).h))


def test_file_field_reads_rows_without_mapping_053_r6(tmp_path):
    d = fake_dump(tmp_path / "a", n_games=3, rows=20)
    ff = dst.file_rows(d)
    ref = pp.load_dump(d)[0]
    idx = np.array([45, 3, 3, 59, 0, 21])
    for f in ("h", "logp", "game", "frame"):
        assert np.array_equal(ff[f][idx], np.asarray(ref[f])[idx]), f
        assert np.array_equal(np.asarray(ff[f]), np.asarray(ref[f])), f
        assert np.array_equal(ff[f][5:17], np.asarray(ref[f])[5:17]), f
    with pytest.raises(IndexError):
        ff["h"][np.array([60])]


def test_mmap_folder_must_be_local_053_r6(tmp_path, monkeypatch):
    monkeypatch.setenv("URANIUM_SHARED_DIR", str(tmp_path / "shared"))
    with pytest.raises(SystemExit):
        dst.Store(tmp_path / "shared" / "eval" / "x")
    with pytest.raises(SystemExit):
        dst.Store(tmp_path / "local" / ".." / "shared" / "y")
    dst.Store(tmp_path / "local")


def test_s0_reports_peak_rss_and_store_053_r6(tmp_path):
    from uranium.diag.planning import s0

    srcs = _sources(tmp_path)
    ram = pp.Data(srcs, (2,))
    mm = pp.Data(srcs, (2,), mmap=tmp_path / "local")
    m_ram, m_mm = s0.memory_of(ram), s0.memory_of(mm)
    assert m_ram["peak_rss_gib"] > 0 and m_ram["mmap"] is None
    assert sorted(m_mm["mmap"]["written"]) == sorted(dst.LARGE) and m_mm["mmap"]["folder"] == mm.store.key[:16]
    assert "RSS" in s0.memory_line(m_ram) and m_mm["mmap"]["folder"] in s0.memory_line(m_mm)
    fp = s0.fingerprint(ram)
    assert fp["sources"] == [["a", "teacher", "split"], ["b", "model", "split"]], "old fingerprints unchanged"
