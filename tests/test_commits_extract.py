from __future__ import annotations

from datetime import UTC, datetime

from config.settings import Settings
from ntech_agent.commits.db import get_connection
from ntech_agent.commits.extract import sync_commits_for_repo


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(_env_file=None, index_dir=tmp_path, github_token="fake-token", **overrides)


class _FakeFile:
    def __init__(self, filename, status="modified", patch=None):
        self.filename = filename
        self.status = status
        self.patch = patch


class _FakeAuthor:
    def __init__(self, login):
        self.login = login


class _FakeGitAuthor:
    def __init__(self, name, date):
        self.name = name
        self.date = date


class _FakeGitCommit:
    def __init__(self, message, author_name, date):
        self.message = message
        self.author = _FakeGitAuthor(author_name, date)


class _FakeCommit:
    def __init__(self, sha, message, author_name, author_login, date, files):
        self.sha = sha
        self.author = _FakeAuthor(author_login) if author_login else None
        self.commit = _FakeGitCommit(message, author_name, date)
        self.files = files


class _FakeRepo:
    def __init__(self, commits):
        self._commits = commits

    def get_commits(self, since=None):
        return [c for c in self._commits if c.commit.author.date >= since]

    def get_commit(self, sha):
        return next(c for c in self._commits if c.sha == sha)


class _FakeGithub:
    def __init__(self, repo):
        self._repo = repo

    def get_repo(self, full_name):
        return self._repo


def test_sync_commits_for_repo_inserts_rows(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    commit = _FakeCommit(
        sha="abc123",
        message="fix: corrige bug\ndetalle en el cuerpo",
        author_name="Roberto",
        author_login="rpribau",
        date=datetime(2026, 7, 10, tzinfo=UTC),
        files=[_FakeFile("a.py"), _FakeFile("b.py")],
    )
    fake_repo = _FakeRepo([commit])
    monkeypatch.setattr("ntech_agent.commits.extract._client", lambda s: _FakeGithub(fake_repo))

    result = sync_commits_for_repo("ws-arg", settings)
    assert result.new_commits == 1
    assert result.new_files == 2

    conn = get_connection(settings)
    rows = conn.execute(
        "SELECT commit_sha, file_path, author, commit_message FROM commit_files "
        "WHERE repo = ? ORDER BY file_path",
        ("ws-arg",),
    ).fetchall()
    assert rows == [
        ("abc123", "a.py", "rpribau", "fix: corrige bug"),
        ("abc123", "b.py", "rpribau", "fix: corrige bug"),
    ]


def test_sync_commits_for_repo_falls_back_to_git_author_name_without_github_login(
    monkeypatch, tmp_path
):
    settings = _settings(tmp_path)
    commit = _FakeCommit(
        sha="def456",
        message="wip",
        author_name="roberto@example.com",
        author_login=None,
        date=datetime(2026, 7, 11, tzinfo=UTC),
        files=[_FakeFile("a.py")],
    )
    fake_repo = _FakeRepo([commit])
    monkeypatch.setattr("ntech_agent.commits.extract._client", lambda s: _FakeGithub(fake_repo))

    sync_commits_for_repo("ws-arg", settings)
    conn = get_connection(settings)
    author = conn.execute(
        "SELECT author FROM commit_files WHERE repo = ? AND commit_sha = ?", ("ws-arg", "def456")
    ).fetchone()[0]
    assert author == "roberto@example.com"


def test_sync_commits_for_repo_is_idempotent(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    commit = _FakeCommit(
        sha="abc123",
        message="fix",
        author_name="Roberto",
        author_login="rpribau",
        date=datetime(2026, 7, 10, tzinfo=UTC),
        files=[_FakeFile("a.py")],
    )
    fake_repo = _FakeRepo([commit])
    monkeypatch.setattr("ntech_agent.commits.extract._client", lambda s: _FakeGithub(fake_repo))

    sync_commits_for_repo("ws-arg", settings)
    result = sync_commits_for_repo("ws-arg", settings)  # segunda corrida: nada nuevo

    conn = get_connection(settings)
    count = conn.execute(
        "SELECT COUNT(*) FROM commit_files WHERE repo = ?", ("ws-arg",)
    ).fetchone()[0]
    assert count == 1  # no se duplicó la fila
    assert result.new_files == 0
    assert result.new_commits == 0


def test_sync_commits_for_repo_excludes_vendored_paths(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    commit = _FakeCommit(
        sha="abc123",
        message="vendor update",
        author_name="Roberto",
        author_login=None,
        date=datetime(2026, 7, 10, tzinfo=UTC),
        files=[_FakeFile("libs/thirdparty.cpp"), _FakeFile("src/main.py")],
    )
    fake_repo = _FakeRepo([commit])
    monkeypatch.setattr("ntech_agent.commits.extract._client", lambda s: _FakeGithub(fake_repo))

    result = sync_commits_for_repo("ws-br", settings)
    assert result.new_files == 1  # libs/ excluido por _is_excluded

    conn = get_connection(settings)
    paths = [
        r[0]
        for r in conn.execute(
            "SELECT file_path FROM commit_files WHERE repo = ?", ("ws-br",)
        ).fetchall()
    ]
    assert paths == ["src/main.py"]
