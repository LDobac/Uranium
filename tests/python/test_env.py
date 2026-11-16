import os

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
