"""Generación de reportes ejecutivos (por-repo y org-wide) en Markdown (+ PDF opcional).

Cada reporte tiene 5 secciones fijas (Introducción, Objetivos generales,
Metodología, Principales hallazgos y conclusiones, Recomendaciones) pensadas
para directivos — ver ``ntech_agent/report/executive.py``. La revisión técnica
completa (hallazgos con archivo:línea) sigue disponible en el chat interactivo.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from config.settings import Settings, get_settings
from ntech_agent.commits.query import org_insights, repo_insights
from ntech_agent.graph.nodes import repomap_node, retrieve_node, reviewer_node, static_node
from ntech_agent.graph.supervisor import list_local_repos
from ntech_agent.report.executive import (
    build_org_executive_report,
    build_repo_executive_report,
    render_executive_report,
)

_REVIEW_PROMPT = "Revisa el repositorio {repo}: calidad de código, design patterns y áreas de oportunidad."


def _run_review_pipeline(repo: str) -> dict:
    """Corre retrieve→repomap→static→reviewer directamente con route="review" y
    repo=repo ya fijados, sin pasar por el router ni por synthesize_node — los
    reportes no usan la respuesta de chat (``answer``), así que armarla ahí sería
    una llamada LLM pagada y descartada por cada repo."""
    state: dict = {"question": _REVIEW_PROMPT.format(repo=repo), "route": "review", "repo": repo}
    for node in (retrieve_node, repomap_node, static_node, reviewer_node):
        state.update(node(state))
    return state


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _maybe_pdf(md_path: Path, md_text: str) -> Path | None:
    """Exporta a PDF si ``markdown-pdf`` está instalado (extra opcional)."""
    try:
        from markdown_pdf import MarkdownPdf, Section

        pdf = MarkdownPdf()
        pdf.add_section(Section(md_text))
        out = md_path.with_suffix(".pdf")
        pdf.save(str(out))
        return out
    except Exception:
        return None


def _used_repomap(final: dict) -> bool:
    return any(
        d.metadata.get("source_type") == "repomap_file" for d in final.get("retrieved", [])
    )


def generate_repo_report(
    repo: str, *, pdf: bool = False, settings: Settings | None = None
) -> Path:
    """Genera el reporte ejecutivo de revisión de un repo."""
    settings = settings or get_settings()
    final = _run_review_pipeline(repo)
    report = build_repo_executive_report(
        repo,
        review_data=final.get("review_data") or {},
        review_markdown=final.get("review", ""),
        static_findings=final.get("static_findings") or {},
        used_repomap=_used_repomap(final),
        commit_facts=repo_insights(repo, settings),
        settings=settings,
    )
    md = f"_Generado: {date.today().isoformat()}_\n\n" + render_executive_report(report)
    path = _write(settings.reports_dir / f"{repo}_review.md", md)
    if pdf:
        _maybe_pdf(path, md)
    return path


def generate_org_report(
    *, pdf: bool = False, settings: Settings | None = None
) -> Path:
    """Genera el reporte ejecutivo consolidado de toda la organización."""
    settings = settings or get_settings()
    repos = list_local_repos(settings)

    per_repo_review_data: dict[str, dict] = {}
    languages_by_repo: dict[str, list[str]] = {}
    used_repomap_any = False
    for repo in repos:
        final = _run_review_pipeline(repo)
        per_repo_review_data[repo] = final.get("review_data") or {}
        languages_by_repo[repo] = (final.get("static_findings") or {}).get("languages", [])
        used_repomap_any = used_repomap_any or _used_repomap(final)

    report = build_org_executive_report(
        per_repo_review_data,
        languages_by_repo,
        used_repomap=used_repomap_any,
        commit_facts_by_repo=org_insights(repos, settings),
        settings=settings,
    )
    md = f"_Generado: {date.today().isoformat()}_\n\n" + render_executive_report(report)
    path = _write(settings.reports_dir / "org_report.md", md)
    if pdf:
        _maybe_pdf(path, md)
    return path
