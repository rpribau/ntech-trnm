"""Transformaciones pandas puras sobre filas de ``commit_files``, usadas por el
tab de Insights (``ui/insights_tab.py``) para graficar y filtrar interactivamente.

Separado de ``query.py`` (que solo lee SQLite) para poder testear la lógica de
agregación con DataFrames sintéticos, sin tocar disco."""

from __future__ import annotations

from datetime import date

import pandas as pd

_ROW_COLUMNS = [
    "repo", "commit_sha", "file_path", "author", "committed_at",
    "commit_message", "change_type", "summary",
]

_CHANGE_TYPE_LABELS = {
    "added": "Agregado",
    "modified": "Modificado",
    "removed": "Eliminado",
    "renamed": "Renombrado",
    "changed": "Cambiado",
    "copied": "Copiado",
}


def to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Arma el DataFrame base a partir de ``query.repo_commit_rows``. Devuelve
    columnas bien tipadas incluso si ``rows`` está vacío (evita KeyError más
    adelante al filtrar/agrupar sobre un repo sin sincronizar)."""
    df = pd.DataFrame(rows, columns=_ROW_COLUMNS)
    df["committed_at"] = pd.to_datetime(df["committed_at"], utc=True, errors="coerce")
    return df


def filter_rows(
    df: pd.DataFrame,
    *,
    since: date | None = None,
    until: date | None = None,
    authors: list[str] | None = None,
    search: str | None = None,
) -> pd.DataFrame:
    """Filtra por rango de fechas, autor(es) y/o texto libre (mensaje, archivo
    o resumen) — la base de las "features de exploración" del dashboard."""
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    if since is not None:
        mask &= df["committed_at"].dt.date >= since
    if until is not None:
        mask &= df["committed_at"].dt.date <= until
    if authors:
        mask &= df["author"].isin(authors)
    if search:
        q = search.lower()
        mask &= (
            df["commit_message"].fillna("").str.lower().str.contains(q, regex=False)
            | df["file_path"].fillna("").str.lower().str.contains(q, regex=False)
            | df["summary"].fillna("").str.lower().str.contains(q, regex=False)
        )
    return df[mask]


def commits_only(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por commit (no por archivo) — para contar/graficar actividad
    sin que un commit de muchos archivos infle el conteo."""
    return df.drop_duplicates(subset="commit_sha")


def daily_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Commits por día, con los días sin actividad rellenados en 0 — necesario
    para que el heatmap tipo calendario no tenga huecos."""
    commits = commits_only(df)
    if commits.empty:
        return pd.DataFrame(columns=["date", "commits"])
    counts = commits.groupby(commits["committed_at"].dt.date).size()
    full_range = pd.date_range(counts.index.min(), counts.index.max(), freq="D").date
    counts = counts.reindex(full_range, fill_value=0)
    return pd.DataFrame({"date": pd.to_datetime(counts.index), "commits": counts.to_numpy()})


def weekly_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Commits por semana (lunes de esa semana como ancla) — para la tendencia."""
    commits = commits_only(df)
    if commits.empty:
        return pd.DataFrame(columns=["week_start", "commits"])
    weekly = (
        commits.set_index("committed_at")
        .resample("W-MON", label="left", closed="left")["commit_sha"]
        .count()
    )
    return pd.DataFrame({"week_start": weekly.index, "commits": weekly.to_numpy()})


def top_contributors(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Contribuidores ordenados por cantidad de commits (no de archivos)."""
    commits = commits_only(df)
    if commits.empty:
        return pd.DataFrame(columns=["author", "commits"])
    counts = commits.groupby("author").size().sort_values(ascending=False).head(limit)
    return pd.DataFrame({"author": counts.index, "commits": counts.to_numpy()})


def top_files(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Archivos que más veces cambiaron (commits distintos que los tocaron) —
    análisis de "hotspots", un indicador clásico de riesgo/complejidad."""
    if df.empty:
        return pd.DataFrame(columns=["file_path", "commits"])
    counts = (
        df.groupby("file_path")["commit_sha"].nunique().sort_values(ascending=False).head(limit)
    )
    return pd.DataFrame({"file_path": counts.index, "commits": counts.to_numpy()})


def change_type_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Archivos por tipo de cambio (agregado/modificado/eliminado/...)."""
    if df.empty:
        return pd.DataFrame(columns=["change_type", "files"])
    counts = df["change_type"].fillna("desconocido").value_counts()
    labels = [_CHANGE_TYPE_LABELS.get(ct, ct) for ct in counts.index]
    return pd.DataFrame({"change_type": labels, "files": counts.to_numpy()})


def metadata_table(df: pd.DataFrame) -> pd.DataFrame:
    """Fase 1 del boceto original: metadata cruda por script-commit, sin resumen."""
    out = df[["commit_sha", "file_path", "author", "committed_at", "commit_message", "repo"]].copy()
    out["commit_sha"] = out["commit_sha"].str[:8]
    return out.rename(columns={"commit_sha": "commit"})


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Fase 2 del boceto original: resumen por script-commit (solo lo ya resumido)."""
    out = df.loc[df["summary"].notna(), ["commit_sha", "file_path", "summary"]].copy()
    out["commit_sha"] = out["commit_sha"].str[:8]
    return out.rename(columns={"commit_sha": "commit"})


def combined_table(df: pd.DataFrame) -> pd.DataFrame:
    """Fase 3 del boceto original: metadata + resumen combinados."""
    cols = [
        "commit_sha", "file_path", "author", "committed_at", "commit_message", "repo", "summary",
    ]
    out = df[cols].copy()
    out["commit_sha"] = out["commit_sha"].str[:8]
    return out.rename(columns={"commit_sha": "commit"})
