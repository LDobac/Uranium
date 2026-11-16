import json
import os
from pathlib import Path

import pytest

from uranium import env

KEY = "URANIUM_TEST_ENV_FILE_VALUE"


def test_env_file_replaces_the_repo_dotenv_001_r9(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text(f"{KEY}=repo\nURANIUM_TEST_ENV_REPO_ONLY=1\n", encoding="utf-8")
    other = tmp_path / "other.env"
    other.write_text(f"{KEY}=file\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    monkeypatch.setenv(env.ENV_FILE_KEY, str(other))
    assert env.load() == other
    assert os.environ[KEY] == "file" and "URANIUM_TEST_ENV_REPO_ONLY" not in os.environ


def test_env_file_missing_loads_nothing_001_r9(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(f"{KEY}=repo\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(env.ENV_FILE_KEY, str(tmp_path / "missing.env"))
    assert env.load() is None and KEY not in os.environ


def test_process_environment_wins_over_the_env_file_001_r9(tmp_path, monkeypatch):
    other = tmp_path / "other.env"
    other.write_text(f"{KEY}=file\n", encoding="utf-8")
    monkeypatch.setenv(env.ENV_FILE_KEY, str(other))
    monkeypatch.setenv(KEY, "process")
    assert env.load() == other and os.environ[KEY] == "process"


def test_without_env_file_the_repo_dotenv_is_read_001_r9(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(f"{KEY}=repo\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(env.ENV_FILE_KEY, raising=False)
    assert env.load() == tmp_path / ".env" and os.environ[KEY] == "repo"


def test_explicit_missing_path_is_an_error_001_r9(tmp_path):
    with pytest.raises(FileNotFoundError):
        env.load(tmp_path / "missing.env")


def loaded(tmp_path, monkeypatch, lines: list) -> dict:
    keys = [f"URANIUM_TEST_ENV_CASE_{i}" for i in range(len(lines))]
    for k in keys:
        monkeypatch.delenv(k, raising=False)
    f = tmp_path / "cases.env"
    f.write_text("".join(line.replace("K", k, 1) + "\n" for k, line in zip(keys, lines)), encoding="utf-8")
    env.load(f)
    return {line: os.environ.get(k) for k, line in zip(keys, lines)}


def test_quoted_non_ascii_value_is_kept_001_r9(tmp_path, monkeypatch):
    got = loaded(tmp_path, monkeypatch, ['K="C:/데이터/맵 폴더"', "K='D:\\리플레이\\한글'", "K=한글값", 'K="é ü 日本"'])
    assert list(got.values()) == ["C:/데이터/맵 폴더", "D:\\리플레이\\한글", "한글값", "é ü 日本"]


@pytest.mark.parametrize("escape, want", [("\\\\", "\\"), ("\\'", "'"), ('\\"', '"'), ("\\$", "$"), ("\\ ", " "), ("\\n", "\n")])
def test_double_quote_escapes_001_r9(tmp_path, monkeypatch, escape, want):
    line = f'K="a{escape}b"'
    assert loaded(tmp_path, monkeypatch, [line])[line] == f"a{want}b"


def test_windows_path_in_single_quotes_is_literal_001_r9(tmp_path, monkeypatch):
    got = loaded(tmp_path, monkeypatch, [r"K='C:\Users\name\new\table'", r"K='C:\Users\name\dir' # this PC", r"K='\\server\share\$x'"])
    assert list(got.values()) == [r"C:\Users\name\new\table", r"C:\Users\name\dir", r"\\server\share\$x"]


DOTENVY_CASES = [
    ("K=1", "1"),
    ('K="2"', "2"),
    ("K='3'", "3"),
    ("K='fo ur'", "fo ur"),
    ('K="fi ve"', "fi ve"),
    ("K=", ""),
    ("K=     ", ""),
    ("K=   # foo", ""),
    ("K=#foo", ""),
    ('K ="whitespace before ="', "whitespace before ="),
    ('K=    "whitespace after ="', "whitespace after ="),
    ("export   K=1", "1"),
    ("K=v # comment", "v"),
    ("K=v#not-a-comment", "v#not-a-comment"),
    ('K="a # b"', "a # b"),
    ("K='a # b'", "a # b"),
    ('K="a" # comment', "a"),
    ('K="a"\t# comment', "a"),
    ("K='a'   # comment", "a"),
    ('K="it\'s"', "it's"),
    ("K='say \"hi\"'", 'say "hi"'),
    ('K="tab\there"', "tab\there"),
    ('K="a\\\\n"', "a\\n"),
]


def test_values_match_the_rust_loader_001_r9(tmp_path, monkeypatch):
    got = loaded(tmp_path, monkeypatch, [line for line, _ in DOTENVY_CASES])
    assert got == dict(DOTENVY_CASES)


def test_unknown_escape_stops_the_load_like_the_rust_loader_001_r9(tmp_path, monkeypatch, capsys):
    got = loaded(tmp_path, monkeypatch, ["K=before", 'K="C:\\Users\\name"', "K=after"])
    assert list(got.values()) == ["before", None, None]
    assert "line 2" in capsys.readouterr().err


CASES = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "dotenv_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_case_table_shared_with_the_rust_loader_001_r9(case):
    environ = dict(case["env"])
    stop = env.read(case["text"].encode("utf-8"), environ)
    assert environ == {**case["want"], **case["env"]}
    assert (stop[0] if stop else None) == case["error_line"]
    assert set(case["want"]) <= set(case["keys"]) | set(case["env"])


def test_case_table_has_every_line_the_old_parser_read_differently_001_r9():
    texts = {c["text"] for c in CASES}
    for line in ("DK=C:\\data\\sc", "DK=a\\ b", "DK=$DA", "DK=${DA}y", 'DK="${DA} z"', "DK=a b", "DK=a\t# note", 'DK="a"b', "DK='a'b'", 'DK="abc', "DK", "DK-X=1"):
        assert line + "\n" in texts, line


def test_error_warning_names_file_and_line_without_the_value_001_r9(tmp_path, monkeypatch, capsys):
    keys = [f"URANIUM_TEST_ENV_STOP_{i}" for i in range(3)]
    for k in keys:
        monkeypatch.delenv(k, raising=False)
    f = tmp_path / "stop.env"
    f.write_bytes(f"{keys[0]}=before\n{keys[1]}=hiddenvalue\\q\n{keys[2]}=after\n".encode())
    assert env.load(f) == f
    err = capsys.readouterr().err
    assert str(f) in err and "line 2" in err and "hiddenvalue" not in err
    assert [os.environ.get(k) for k in keys] == ["before", None, None]


def test_non_utf8_line_stops_the_load_001_r9(tmp_path, monkeypatch, capsys):
    keys = [f"URANIUM_TEST_ENV_BYTES_{i}" for i in range(3)]
    for k in keys:
        monkeypatch.delenv(k, raising=False)
    f = tmp_path / "bytes.env"
    f.write_bytes(f"{keys[0]}=before\n{keys[1]}=".encode() + b"\xff\n" + f"{keys[2]}=after\n".encode())
    env.load(f)
    assert "line 2" in capsys.readouterr().err
    assert [os.environ.get(k) for k in keys] == ["before", None, None]


def test_substitution_reads_the_process_environment_001_r9(tmp_path, monkeypatch):
    monkeypatch.setenv("URANIUMTESTENVSUB", "process")
    monkeypatch.delenv("URANIUM_TEST_ENV_SUB_OUT", raising=False)
    monkeypatch.delenv("URANIUMTESTENVEARLIER", raising=False)
    f = tmp_path / "sub.env"
    f.write_bytes(b"URANIUMTESTENVEARLIER=file\nURANIUM_TEST_ENV_SUB_OUT=$URANIUMTESTENVSUB/${URANIUMTESTENVEARLIER}\n")
    env.load(f)
    assert os.environ["URANIUM_TEST_ENV_SUB_OUT"] == "process/file"
