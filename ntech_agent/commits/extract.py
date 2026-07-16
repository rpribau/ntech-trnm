"""Fase 1: extrae metadata de commits (autor, fecha, mensaje, archivos) vía la
API de GitHub y la persiste en ``commit_files``.

Deliberadamente NO usa el clon local (``ntech_agent/github/sync.py`` lo deja
en ``--depth 1`` para el RAG, sin historial). La API de GitHub además resuelve
el autor a su login real de GitHub (mejor para "¿quién participa?" que el
nombre/email crudo de git) y expone ``files`` con su ``status`` sin necesitar
un unshallow del repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from github import Auth, Github

from config.settings import Settings, get_settings
from ntech_agent.commits.db import get_connection
from ntech_agent.repomap.build import _is_excluded, _submodule_prefixes


@dataclass
class ExtractResult:
    repo: str
    new_commits: int
    new_files: int


def _client(settings: Settings) -> Github:
    if not settings.github_token:
        raise RuntimeError("NTECH_GITHUB_TOKEN no está configurado en .env")
    return Github(auth=Auth.Token(settings.github_token), per_page=100)


def _cutoff(repo: str, conn, settings: Settings) -> datetime:
    row = conn.execute(
        "SELECT last_synced_at FROM sync_state WHERE repo = ?", (repo,)
    ).fetchone()
    if row and row[0]:
        return datetime.fromisoformat(row[0])
    return datetime.now(UTC) - timedelta(days=settings.commits_lookback_days)


def sync_commits_for_repo(
    repo: str, settings: Settings | None = None, *, conn=None
) -> ExtractResult:
    """Trae commits nuevos (desde el último sync, o desde ``commits_lookback_days``
    si es la primera vez) y sus archivos modificados, y los guarda en
    ``commit_files``. Idempotente: re-correr no duplica filas."""
    settings = settings or get_settings()
    owned_conn = conn is None
    conn = conn or get_connection(settings)
    try:
        gh = _client(settings)
        gh_repo = gh.get_repo(f"{settings.github_org}/{repo}")
        since = _cutoff(repo, conn, settings)
        submodule_prefixes = _submodule_prefixes(settings.repos_dir / repo)

        new_commits = 0
        new_files = 0
        latest_at = since
        for commit in gh_repo.get_commits(since=since):
            git_commit = commit.commit
            author_login = commit.author.login if commit.author else None
            author = author_login or (git_commit.author.name if git_commit.author else None) or (
                "(desconocido)"
            )
            committed_at = git_commit.author.date if git_commit.author else datetime.now(UTC)
            if committed_at.tzinfo is None:
                committed_at = committed_at.replace(tzinfo=UTC)
            message = (git_commit.message or "").splitlines()[0] if git_commit.message else ""
            latest_at = max(latest_at, committed_at)

            rows = [
                (repo, commit.sha, f.filename, author, committed_at.isoformat(), message, f.status)
                for f in (commit.files or [])
                if not _is_excluded(f.filename, submodule_prefixes)
            ]
            if not rows:
                continue
            cur = conn.executemany(
                "INSERT OR IGNORE INTO commit_files "
                "(repo, commit_sha, file_path, author, committed_at, commit_message, change_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            inserted = max(cur.rowcount, 0)
            new_files += inserted
            if inserted > 0:
                new_commits += 1

        conn.execute(
            "INSERT INTO sync_state (repo, last_synced_at) VALUES (?, ?) "
            "ON CONFLICT(repo) DO UPDATE SET last_synced_at = excluded.last_synced_at",
            (repo, latest_at.isoformat()),
        )
        conn.commit()
        return ExtractResult(repo=repo, new_commits=new_commits, new_files=new_files)
    finally:
        if owned_conn:
            conn.close()
