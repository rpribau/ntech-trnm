"""Queries de solo lectura sobre ``commit_files`` — compartidas por el tab de
Streamlit y por ``commit_insights_node`` (grafo), para no duplicar SQL en dos
lugares."""

from __future__ import annotations

from config.settings import Settings, get_settings
from ntech_agent.commits.db import get_connection


def repo_insights(repo: str, settings: Settings | None = None) -> dict:
    """Métricas de actividad de un repo: último commit, contribuidores, conteo
    total, actividad semanal y commits recientes (con resumen si ya se generó)."""
    settings = settings or get_settings()
    conn = get_connection(settings)
    try:
        total = conn.execute(
            "SELECT COUNT(DISTINCT commit_sha) FROM commit_files WHERE repo = ?", (repo,)
        ).fetchone()[0]

        last_row = conn.execute(
            "SELECT commit_sha, author, committed_at, commit_message "
            "FROM commit_files WHERE repo = ? ORDER BY committed_at DESC LIMIT 1",
            (repo,),
        ).fetchone()
        last_commit = (
            {
                "commit_sha": last_row[0],
                "author": last_row[1],
                "committed_at": last_row[2],
                "commit_message": last_row[3],
            }
            if last_row
            else None
        )

        contributors = [
            {"author": r[0], "commits": r[1]}
            for r in conn.execute(
                "SELECT author, COUNT(DISTINCT commit_sha) AS n FROM commit_files "
                "WHERE repo = ? GROUP BY author ORDER BY n DESC",
                (repo,),
            ).fetchall()
        ]

        # Bucket semanal (lunes de esa semana). SQLite no soporta '%W' en
        # strftime — 'weekday 0' avanza a el próximo domingo (o se queda si ya
        # lo es), y '-6 days' retrocede al lunes de esa misma semana.
        activity = [
            {"week_start": r[0], "commits": r[1]}
            for r in conn.execute(
                "SELECT date(committed_at, 'weekday 0', '-6 days') AS week_start, "
                "COUNT(DISTINCT commit_sha) AS n FROM commit_files "
                "WHERE repo = ? GROUP BY week_start ORDER BY week_start",
                (repo,),
            ).fetchall()
        ]

        # Acotado por COMMIT (no por fila): buscar primero los N commits más
        # recientes y después TODOS sus archivos, en vez de "las últimas N filas"
        # — con varios archivos por commit, un límite de filas cubría muchos
        # menos commits de lo esperado y perdía autores/commits más viejos
        # (ver settings.commits_recent_limit).
        recent_shas = [
            r[0]
            for r in conn.execute(
                "SELECT commit_sha FROM commit_files WHERE repo = ? "
                "GROUP BY commit_sha ORDER BY MAX(committed_at) DESC LIMIT ?",
                (repo, settings.commits_recent_limit),
            ).fetchall()
        ]
        recent: list[dict] = []
        if recent_shas:
            placeholders = ",".join("?" * len(recent_shas))
            recent = [
                {
                    "commit_sha": r[0],
                    "file_path": r[1],
                    "author": r[2],
                    "committed_at": r[3],
                    "commit_message": r[4],
                    "summary": r[5],
                }
                for r in conn.execute(
                    "SELECT commit_sha, file_path, author, committed_at, commit_message, summary "
                    f"FROM commit_files WHERE repo = ? AND commit_sha IN ({placeholders}) "
                    "ORDER BY committed_at DESC, file_path",
                    (repo, *recent_shas),
                ).fetchall()
            ]

        return {
            "repo": repo,
            "total_commits": total,
            "last_commit": last_commit,
            "contributors": contributors,
            "activity_by_week": activity,
            "recent": recent,
        }
    finally:
        conn.close()


def org_insights(repos: list[str], settings: Settings | None = None) -> dict[str, dict]:
    """Insights por repo, agregados (mismo espíritu que ``generate_org_report``
    en ``ntech_agent/report/generate.py``, pero leyendo la base de commits en
    vez de correr el pipeline de revisión)."""
    settings = settings or get_settings()
    return {repo: repo_insights(repo, settings) for repo in repos}


_ROW_COLUMNS = (
    "repo", "commit_sha", "file_path", "author", "committed_at",
    "commit_message", "change_type", "summary",
)


def repo_commit_rows(repo: str, settings: Settings | None = None) -> list[dict]:
    """Todas las filas (una por archivo) de un repo, sin agregar — a diferencia
    de ``repo_insights``, pensado para que la UI (tab Insights) arme su propio
    DataFrame y filtre/agrupe interactivamente (rango de fechas, autor, texto)."""
    settings = settings or get_settings()
    conn = get_connection(settings)
    try:
        rows = conn.execute(
            f"SELECT {', '.join(_ROW_COLUMNS)} FROM commit_files "
            "WHERE repo = ? ORDER BY committed_at DESC",
            (repo,),
        ).fetchall()
        return [dict(zip(_ROW_COLUMNS, r, strict=True)) for r in rows]
    finally:
        conn.close()
