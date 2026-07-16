"""Tab "Insights": dashboard de actividad de commits (Governance/DevOps) +
exploración interactiva. Separado de ``streamlit_app.py`` para no inflar ese
archivo — toda la lógica de datos vive en ``ntech_agent/commits/analytics.py``
(testeada por separado); acá solo se arma la UI.

Paleta y specs de los charts siguen la skill de dataviz del proyecto: hues
categóricos en orden fijo (nunca ciclados), un solo hue secuencial para
magnitud (heatmap), tooltip en todo mark interactivo, sin doble eje.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from config.settings import Settings
from ntech_agent.commits import analytics
from ntech_agent.commits.query import repo_commit_rows

# Paleta categórica en orden fijo (referencia: skill de dataviz del proyecto).
# Variantes "dark" porque el tema por defecto observado en esta app es oscuro.
_BLUE = "#3987e5"
_AQUA = "#199e70"
_RED = "#e66767"
_VIOLET = "#9085e9"
_ORANGE = "#d95926"
_MAGENTA = "#d55181"

_SEQUENTIAL_DARK = ["#16233a", "#1c5cab", "#3987e5", "#6da7ec", "#b7d3f6"]
_SEQUENTIAL_LIGHT = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

_CHANGE_TYPE_DOMAIN = ["Agregado", "Modificado", "Eliminado", "Renombrado", "Cambiado", "Copiado"]
_CHANGE_TYPE_RANGE = [_AQUA, _BLUE, _RED, _VIOLET, _ORANGE, _MAGENTA]


def _is_dark_theme() -> bool:
    base = st.get_option("theme.base")
    return base != "light"  # sin config explícita, coincide con lo observado (dark)


def _weekly_trend_chart(weekly: pd.DataFrame, color: str) -> alt.Chart:
    base = alt.Chart(weekly).encode(
        x=alt.X("week_start:T", title=None, axis=alt.Axis(format="%d %b", grid=False)),
        y=alt.Y("commits:Q", title="Commits", axis=alt.Axis(grid=True)),
        tooltip=[
            alt.Tooltip("week_start:T", title="Semana de", format="%d %b %Y"),
            alt.Tooltip("commits:Q", title="Commits"),
        ],
    )
    area = base.mark_area(opacity=0.18, color=color, interpolate="monotone")
    line = base.mark_line(color=color, size=2, interpolate="monotone")
    points = base.mark_circle(color=color, size=40)
    return (
        (area + line + points)
        .properties(height=220, background="transparent")
        .configure_axis(labelColor="#898781", titleColor="#898781", gridColor="#2c2c2a")
        .configure_view(strokeWidth=0)
    )


def _calendar_heatmap_chart(daily: pd.DataFrame, dark: bool) -> alt.Chart | None:
    if daily.empty:
        return None
    df = daily.copy()
    df["weekday"] = df["date"].dt.weekday
    start_monday = df["date"].min() - pd.Timedelta(days=int(df["date"].min().weekday()))
    df["week"] = (df["date"] - start_monday).dt.days // 7
    weekday_labels = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    df["weekday_label"] = df["weekday"].map(dict(enumerate(weekday_labels)))
    scale_range = _SEQUENTIAL_DARK if dark else _SEQUENTIAL_LIGHT

    return (
        alt.Chart(df)
        .mark_rect(cornerRadius=2)
        .encode(
            x=alt.X("week:O", title=None, axis=None),
            y=alt.Y("weekday_label:N", title=None, sort=weekday_labels),
            color=alt.Color(
                "commits:Q",
                title="Commits",
                scale=alt.Scale(range=scale_range),
                legend=alt.Legend(orient="right", gradientLength=100),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Fecha", format="%d %b %Y"),
                alt.Tooltip("commits:Q", title="Commits"),
            ],
        )
        .properties(height=160, background="transparent")
        .configure_view(strokeWidth=0)
    )


def _contributors_chart(top: pd.DataFrame, color: str) -> alt.Chart | None:
    if top.empty:
        return None
    return (
        alt.Chart(top)
        .mark_bar(color=color, cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("commits:Q", title="Commits"),
            y=alt.Y("author:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("author:N", title="Autor"),
                alt.Tooltip("commits:Q", title="Commits"),
            ],
        )
        .properties(height=alt.Step(26), background="transparent")
        .configure_axis(labelColor="#898781", titleColor="#898781", gridColor="#2c2c2a")
        .configure_view(strokeWidth=0)
    )


def _files_chart(top: pd.DataFrame, color: str) -> alt.Chart | None:
    if top.empty:
        return None
    return (
        alt.Chart(top)
        .mark_bar(color=color, cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("commits:Q", title="Commits que lo tocaron"),
            y=alt.Y("file_path:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("file_path:N", title="Archivo"),
                alt.Tooltip("commits:Q", title="Commits"),
            ],
        )
        .properties(height=alt.Step(26), background="transparent")
        .configure_axis(labelColor="#898781", titleColor="#898781", gridColor="#2c2c2a")
        .configure_view(strokeWidth=0)
    )


def _change_type_chart(breakdown: pd.DataFrame) -> alt.Chart | None:
    if breakdown.empty:
        return None
    return (
        alt.Chart(breakdown)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("files:Q", title="Archivos"),
            y=alt.Y("change_type:N", sort="-x", title=None),
            color=alt.Color(
                "change_type:N",
                scale=alt.Scale(domain=_CHANGE_TYPE_DOMAIN, range=_CHANGE_TYPE_RANGE),
                legend=None,  # una sola serie categórica por fila: el eje ya la nombra
            ),
            tooltip=[
                alt.Tooltip("change_type:N", title="Tipo"),
                alt.Tooltip("files:Q", title="Archivos"),
            ],
        )
        .properties(height=alt.Step(26), background="transparent")
        .configure_axis(labelColor="#898781", titleColor="#898781", gridColor="#2c2c2a")
        .configure_view(strokeWidth=0)
    )


def _repo_comparison_chart(summary: pd.DataFrame, color: str) -> alt.Chart | None:
    if summary.empty:
        return None
    return (
        alt.Chart(summary)
        .mark_bar(color=color, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("repo:N", title=None, sort="-y"),
            y=alt.Y("commits:Q", title="Commits"),
            tooltip=[
                alt.Tooltip("repo:N", title="Repo"),
                alt.Tooltip("commits:Q", title="Commits"),
                alt.Tooltip("contribuidores:Q", title="Contribuidores"),
            ],
        )
        .properties(height=260, background="transparent")
        .configure_axis(labelColor="#898781", titleColor="#898781", gridColor="#2c2c2a")
        .configure_view(strokeWidth=0)
    )


_TABLE_COLUMN_CONFIG = {
    "commit": st.column_config.TextColumn("Commit", width="small"),
    "committed_at": st.column_config.DatetimeColumn("Fecha", format="YYYY-MM-DD HH:mm"),
    "author": st.column_config.TextColumn("Autor"),
    "commit_message": st.column_config.TextColumn("Mensaje", width="medium"),
    "file_path": st.column_config.TextColumn("Archivo", width="medium"),
    "repo": st.column_config.TextColumn("Repo", width="small"),
    "summary": st.column_config.TextColumn("Resumen", width="large"),
}


def _render_repo_dashboard(settings: Settings, repo: str) -> None:
    if st.button("🔄 Sincronizar commits", key="sync_commits_btn"):
        from ntech_agent.commits.extract import sync_commits_for_repo
        from ntech_agent.commits.summarize import summarize_new_commits

        with st.spinner(f"Sincronizando commits de {repo}…"):
            extracted = sync_commits_for_repo(repo, settings)
            summarized = summarize_new_commits(repo, settings)
        st.success(
            f"{extracted.new_commits} commits nuevos · {extracted.new_files} "
            f"archivos nuevos · {summarized} resumidos"
        )

    df = analytics.to_dataframe(repo_commit_rows(repo, settings))
    if df.empty:
        st.info("Sin commits sincronizados aún para este repo. Usa el botón de arriba.")
        return

    min_d, max_d = df["committed_at"].min().date(), df["committed_at"].max().date()
    f1, f2, f3 = st.columns([1.2, 1, 1.4])
    date_range = f1.date_input(
        "Rango de fechas", value=(min_d, max_d), min_value=min_d, max_value=max_d
    )
    authors_sel = f2.multiselect("Autores", options=sorted(df["author"].dropna().unique()))
    search = f3.text_input("Buscar", placeholder="mensaje, archivo o resumen…")

    since, until = date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (
        min_d, max_d,
    )
    filtered = analytics.filter_rows(
        df, since=since, until=until, authors=authors_sel or None, search=search or None
    )
    if filtered.empty:
        st.warning("Ningún commit coincide con los filtros aplicados.")
        return

    commits_df = analytics.commits_only(filtered)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Commits", len(commits_df))
    k2.metric("Contribuidores", int(filtered["author"].nunique()))
    k3.metric("Archivos tocados", int(filtered["file_path"].nunique()))
    k4.metric("Último commit", filtered["committed_at"].max().strftime("%Y-%m-%d"))

    dark = _is_dark_theme()
    color = _BLUE

    st.subheader("Actividad diaria")
    heatmap = _calendar_heatmap_chart(analytics.daily_activity(filtered), dark)
    if heatmap is not None:
        st.altair_chart(heatmap, use_container_width=True)

    st.subheader("Tendencia semanal")
    weekly_chart = _weekly_trend_chart(analytics.weekly_activity(filtered), color)
    st.altair_chart(weekly_chart, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top contribuidores")
        chart = _contributors_chart(analytics.top_contributors(filtered), color)
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
    with c2:
        st.subheader("Archivos más modificados")
        chart = _files_chart(analytics.top_files(filtered), _AQUA)
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)

    st.subheader("Tipo de cambio")
    chart = _change_type_chart(analytics.change_type_breakdown(filtered))
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)

    st.subheader("🔍 Explorar datos crudos")
    st.caption(
        "Las 3 tablas del flujo original: metadata cruda → resumen por script → combinado."
    )
    tab1, tab2, tab3 = st.tabs(["Fase 1 · Metadata", "Fase 2 · Resúmenes", "Fase 3 · Combinado"])
    with tab1:
        st.dataframe(
            analytics.metadata_table(filtered), use_container_width=True, hide_index=True,
            column_config=_TABLE_COLUMN_CONFIG,
        )
    with tab2:
        st.dataframe(
            analytics.summary_table(filtered), use_container_width=True, hide_index=True,
            column_config=_TABLE_COLUMN_CONFIG,
        )
    with tab3:
        st.dataframe(
            analytics.combined_table(filtered), use_container_width=True, hide_index=True,
            column_config=_TABLE_COLUMN_CONFIG,
        )


def _render_org_dashboard(settings: Settings, repos: list[str]) -> None:
    per_repo_df = {repo: analytics.to_dataframe(repo_commit_rows(repo, settings)) for repo in repos}
    summary = pd.DataFrame(
        [
            {
                "repo": repo,
                "commits": len(analytics.commits_only(df)),
                "contribuidores": int(df["author"].nunique()) if not df.empty else 0,
                "último_commit": (
                    df["committed_at"].max().date() if not df.empty else None
                ),
            }
            for repo, df in per_repo_df.items()
        ]
    )
    if summary["commits"].sum() == 0:
        st.info("Ningún repo tiene commits sincronizados todavía.")
        return

    k1, k2 = st.columns(2)
    k1.metric("Commits totales (todos los repos)", int(summary["commits"].sum()))
    k2.metric("Repos con actividad", int((summary["commits"] > 0).sum()))

    st.subheader("Commits por repo")
    chart = _repo_comparison_chart(summary, _BLUE)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)

    st.dataframe(
        summary.sort_values("commits", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "repo": st.column_config.TextColumn("Repo"),
            "commits": st.column_config.NumberColumn("Commits"),
            "contribuidores": st.column_config.NumberColumn("Contribuidores"),
            "último_commit": st.column_config.DateColumn("Último commit"),
        },
    )


def render(settings: Settings, repos: list[str]) -> None:
    st.caption(
        "Actividad de commits por repo (autor, fecha, resumen) — visibilidad de flujo de "
        "trabajo para Governance/DevOps."
    )
    if not repos:
        st.info("No hay repos. Corre `python -m scripts.sync_repos`.")
        return

    view = st.radio("Vista", ["Un repo", "Todos los repos"], horizontal=True)
    if view == "Todos los repos":
        _render_org_dashboard(settings, repos)
        return

    insight_repo = st.selectbox("Repo", options=repos, key="insight_repo")
    _render_repo_dashboard(settings, insight_repo)
