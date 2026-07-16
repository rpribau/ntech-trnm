from __future__ import annotations

from config.settings import Settings
from ntech_agent.commits.db import get_connection
from ntech_agent.commits.query import org_insights, repo_insights


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(_env_file=None, index_dir=tmp_path, **overrides)


def _insert(conn, repo, sha, file_path, author, committed_at, message, summary=None):
    conn.execute(
        "INSERT INTO commit_files "
        "(repo, commit_sha, file_path, author, committed_at, commit_message, change_type, summary) "
        "VALUES (?, ?, ?, ?, ?, ?, 'modified', ?)",
        (repo, sha, file_path, author, committed_at, message, summary),
    )
    conn.commit()


def test_repo_insights_empty_repo_returns_zeroed_structure(tmp_path):
    settings = _settings(tmp_path)
    facts = repo_insights("ws-arg", settings)
    assert facts["total_commits"] == 0
    assert facts["last_commit"] is None
    assert facts["contributors"] == []
    assert facts["recent"] == []


def test_repo_insights_counts_distinct_commits_and_contributors(tmp_path):
    settings = _settings(tmp_path)
    conn = get_connection(settings)
    _insert(conn, "ws-arg", "sha1", "a.py", "rpribau", "2026-07-01T10:00:00+00:00", "primero")
    _insert(conn, "ws-arg", "sha1", "b.py", "rpribau", "2026-07-01T10:00:00+00:00", "primero")
    _insert(conn, "ws-arg", "sha2", "c.py", "otro-dev", "2026-07-05T10:00:00+00:00", "segundo")

    facts = repo_insights("ws-arg", settings)
    assert facts["total_commits"] == 2  # sha1 cuenta una vez pese a tocar 2 archivos
    assert facts["last_commit"]["commit_sha"] == "sha2"
    assert facts["last_commit"]["author"] == "otro-dev"
    contributors = {c["author"]: c["commits"] for c in facts["contributors"]}
    assert contributors == {"rpribau": 1, "otro-dev": 1}


def test_repo_insights_scoped_by_repo(tmp_path):
    settings = _settings(tmp_path)
    conn = get_connection(settings)
    _insert(conn, "ws-arg", "sha1", "a.py", "dev-a", "2026-07-01T00:00:00+00:00", "m1")
    _insert(conn, "ws-br", "sha2", "b.py", "dev-b", "2026-07-02T00:00:00+00:00", "m2")

    assert repo_insights("ws-arg", settings)["total_commits"] == 1
    assert repo_insights("ws-br", settings)["total_commits"] == 1


def test_org_insights_aggregates_per_repo(tmp_path):
    settings = _settings(tmp_path)
    conn = get_connection(settings)
    _insert(conn, "ws-arg", "sha1", "a.py", "dev-a", "2026-07-01T00:00:00+00:00", "m1")
    _insert(conn, "ws-br", "sha2", "b.py", "dev-b", "2026-07-02T00:00:00+00:00", "m2")

    org = org_insights(["ws-arg", "ws-br"], settings)
    assert set(org.keys()) == {"ws-arg", "ws-br"}
    assert org["ws-arg"]["total_commits"] == 1
    assert org["ws-br"]["total_commits"] == 1


def test_repo_insights_recent_includes_summary(tmp_path):
    settings = _settings(tmp_path)
    conn = get_connection(settings)
    _insert(
        conn,
        "ws-arg",
        "sha1",
        "a.py",
        "dev-a",
        "2026-07-01T00:00:00+00:00",
        "m1",
        summary="Corrige el parseo de fechas.",
    )
    facts = repo_insights("ws-arg", settings)
    assert facts["recent"][0]["summary"] == "Corrige el parseo de fechas."


def test_repo_insights_recent_is_bounded_by_commit_count_not_row_count(tmp_path):
    # Bug real: un límite de FILAS subestimaba cuántos commits distintos entraban
    # en "recent" cuando un commit toca varios archivos — un commit de 3 archivos
    # + un límite de 2 filas dejaba ese commit truncado a mitad y ni siquiera
    # llegaba a incluir el commit más viejo completo.
    settings = _settings(tmp_path, commits_recent_limit=2)
    conn = get_connection(settings)
    # sha1 (más viejo, 3 archivos) debe quedar TOTALMENTE afuera con el límite de 2 commits.
    _insert(conn, "ws-x", "sha1", "a1.py", "dev-a", "2026-06-01T00:00:00+00:00", "m1")
    _insert(conn, "ws-x", "sha1", "a2.py", "dev-a", "2026-06-01T00:00:00+00:00", "m1")
    _insert(conn, "ws-x", "sha1", "a3.py", "dev-a", "2026-06-01T00:00:00+00:00", "m1")
    # sha2 (2 archivos) y sha3 (1 archivo) son los 2 commits más nuevos.
    _insert(conn, "ws-x", "sha2", "b1.py", "dev-b", "2026-06-02T00:00:00+00:00", "m2")
    _insert(conn, "ws-x", "sha2", "b2.py", "dev-b", "2026-06-02T00:00:00+00:00", "m2")
    _insert(conn, "ws-x", "sha3", "c1.py", "dev-c", "2026-06-03T00:00:00+00:00", "m3")

    facts = repo_insights("ws-x", settings)
    shas_in_recent = {r["commit_sha"] for r in facts["recent"]}
    assert shas_in_recent == {"sha2", "sha3"}  # sha1 (el más viejo) queda afuera ENTERO

    sha2_files = {r["file_path"] for r in facts["recent"] if r["commit_sha"] == "sha2"}
    assert sha2_files == {"b1.py", "b2.py"}  # sha2 trae SUS 2 archivos completos, no truncado
