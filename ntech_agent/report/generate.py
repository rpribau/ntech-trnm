"""Generación de reportes (por-repo y org-wide) en Markdown (+ PDF opcional)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from config.settings import Settings, get_settings
from ntech_agent.graph.builder import run_agent
from ntech_agent.graph.supervisor import list_local_repos

_REVIEW_PROMPT = "Revisa el repositorio {repo}: calidad de código, design patterns y áreas de oportunidad."


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


def generate_repo_report(
    repo: str, *, pdf: bool = False, settings: Settings | None = None
) -> Path:
    """Genera el reporte de revisión de un repo."""
    settings = settings or get_settings()
    final = run_agent(_REVIEW_PROMPT.format(repo=repo), thread_id=f"review-{repo}")
    header = f"# Reporte de revisión — {repo}\n\n_Generado: {date.today().isoformat()}_\n\n"
    md = header + final.get("answer", "_(sin contenido)_")
    path = _write(settings.reports_dir / f"{repo}_review.md", md)
    if pdf:
        _maybe_pdf(path, md)
    return path


def generate_org_report(
    *, pdf: bool = False, settings: Settings | None = None
) -> Path:
    """Genera un reporte de toda la organización (panorama + revisión por repo)."""
    settings = settings or get_settings()
    repos = list_local_repos(settings)

    panorama = run_agent(
        "Dame un panorama de toda la organización NTech-TRNM: temas, fortalezas y "
        "áreas de oportunidad comunes.",
        thread_id="org-panorama",
    ).get("answer", "")

    parts = [
        f"# Reporte de la organización NTech-TRNM\n\n_Generado: {date.today().isoformat()}_\n",
        "## Panorama general\n\n" + panorama,
        "\n---\n\n# Revisión por repositorio\n",
    ]
    for repo in repos:
        final = run_agent(_REVIEW_PROMPT.format(repo=repo), thread_id=f"review-{repo}")
        parts.append(f"\n## {repo}\n\n" + final.get("answer", "_(sin contenido)_"))

    md = "\n".join(parts)
    path = _write(settings.reports_dir / "org_report.md", md)
    if pdf:
        _maybe_pdf(path, md)
    return path
