"""Conexión y esquema SQLite para metadata de commits.

Mismo patrón que el checkpointer de LangGraph (``ntech_agent/graph/builder.py``):
``sqlite3`` crudo, sin ORM. La PK compuesta (repo, commit_sha, file_path) da
idempotencia natural — re-sincronizar no duplica filas (``INSERT OR IGNORE``).
"""

from __future__ import annotations

import sqlite3

from config.settings import Settings, get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS commit_files (
    repo TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    file_path TEXT NOT NULL,
    author TEXT,
    committed_at TEXT NOT NULL,
    commit_message TEXT,
    change_type TEXT,
    summary TEXT,
    PRIMARY KEY (repo, commit_sha, file_path)
);
CREATE INDEX IF NOT EXISTS idx_commit_files_repo_date ON commit_files(repo, committed_at);

CREATE TABLE IF NOT EXISTS sync_state (
    repo TEXT PRIMARY KEY,
    last_synced_at TEXT
);
"""


def get_connection(settings: Settings | None = None) -> sqlite3.Connection:
    """Abre (y crea el esquema si hace falta) la base de metadata de commits."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    conn = sqlite3.connect(str(settings.commits_db_path), check_same_thread=False)
    conn.executescript(_SCHEMA)
    return conn
