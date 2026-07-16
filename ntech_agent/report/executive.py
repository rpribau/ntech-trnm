"""Reportes ejecutivos: 5 secciones fijas (Introducción, Objetivos generales,
Metodología, Principales hallazgos y conclusiones, Recomendaciones), en prosa
para directivos — sin citas archivo:línea ni jerga de código.

"Objetivos generales" y "Metodología" se generan por template determinístico
(describen el PROCESO fijo, no requieren al LLM y no hay riesgo de que se
invente algo). "Introducción" reutiliza el resumen ya generado por el reviewer
(``resumen_contextualizado``). Solo "Principales hallazgos y conclusiones" y
"Recomendaciones" requieren una llamada LLM adicional (una sola, combinada) que
reelabora los hallazgos técnicos en prosa ejecutiva — mismo patrón
``with_structured_output`` + fallback a texto libre que ``reviewer_node``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from config.settings import Settings, get_settings
from ntech_agent.llm import extract_text, get_chat_model, log_llm_failure


class ExecutiveReport(BaseModel):
    repo: str | None  # None = reporte de organización
    introduccion: str
    objetivos_generales: str
    metodologia: str
    actividad_desarrollo: str
    principales_hallazgos_y_conclusiones: str
    recomendaciones: str


class _NarrativaEjecutiva(BaseModel):
    principales_hallazgos_y_conclusiones: str = Field(
        description="2-4 párrafos en prosa ejecutiva, sin archivo:línea ni jerga técnica"
    )
    recomendaciones: str = Field(
        description="3-6 acciones priorizadas orientadas a decisiones de negocio"
    )


_NARRATIVA_PROMPT = (
    "Eres un consultor que traduce hallazgos técnicos de una revisión de código en un "
    "resumen para directivos SIN conocimiento técnico.\n\n"
    "Reglas:\n"
    "- NO cites archivo:línea, nombres de función/clase/archivo, ni herramientas "
    "(ruff, radon, etc.)\n"
    "- NO uses jerga de programación\n"
    "- Prioriza por impacto de negocio (riesgo, mantenibilidad, velocidad de desarrollo, "
    "seguridad, costo de no actuar)\n"
    "- 'principales_hallazgos_y_conclusiones': 2-4 párrafos narrativos sobre el estado "
    "general y los riesgos/fortalezas más relevantes\n"
    "- 'recomendaciones': 3-6 acciones concretas y priorizadas, orientadas a qué decidir "
    "o invertir primero, no instrucciones de código\n\n"
    "=== EVIDENCIA TÉCNICA (uso interno, no la cites literalmente) ===\n{evidencia}"
)

_ORG_NARRATIVA_PROMPT = (
    "Eres un consultor que traduce hallazgos técnicos de revisiones de código de "
    "varios repositorios en un panorama para directivos SIN conocimiento técnico.\n\n"
    "Reglas:\n"
    "- NO cites archivo:línea, nombres de función/clase/archivo, ni herramientas "
    "(ruff, radon, etc.)\n"
    "- NO uses jerga de programación\n"
    "- Además de resumir el estado, identificá PATRONES QUE SE REPITEN entre los "
    "repositorios: stack tecnológico compartido, anti-patrones o brechas (testing, "
    "documentación, seguridad) que aparecen en más de uno, y diferencias notables "
    "entre proyectos (propósito, madurez, tecnología)\n"
    "- 'principales_hallazgos_y_conclusiones': 2-4 párrafos narrativos que cubran el "
    "estado general de la organización Y los patrones cruzados detectados\n"
    "- 'recomendaciones': 3-6 acciones priorizadas a nivel organización (no repitas "
    "una por cada repo salvo que sea crítico), orientadas a qué decidir o invertir "
    "primero\n\n"
    "=== EVIDENCIA TÉCNICA POR REPOSITORIO (uso interno, no la cites literalmente) ===\n"
    "{evidencia}"
)


def _template_objetivos(repo: str | None) -> str:
    target = (
        f"del repositorio `{repo}`"
        if repo
        else "de los repositorios de la organización NTech-TRNM"
    )
    return (
        f"Evaluar la calidad del código {target} conforme a los lineamientos de "
        "ingeniería de software y ciencia de datos definidos por NTech-TRNM, "
        "identificando fortalezas, riesgos y oportunidades de mejora priorizadas "
        "para apoyar la toma de decisiones."
    )


def _template_metodologia(*, used_repomap: bool, static_languages: list[str]) -> str:
    parts = [
        "La revisión combina recuperación híbrida (búsqueda semántica y por palabras "
        "clave) de código y guidelines internos, con un modelo de lenguaje que aplica "
        "una rúbrica de buenas prácticas de ingeniería de software y ciencia de datos "
        "(diseño, patrones, legibilidad, testing, seguridad y documentación, entre otras "
        "dimensiones)."
    ]
    if used_repomap:
        parts.append(
            "Para priorizar qué revisar en profundidad, se utilizó un mapa estructural "
            "del repositorio basado en el grafo de dependencias entre archivos, lo que "
            "permitió acceder a archivos completos en vez de fragmentos aislados."
        )
    if static_languages:
        langs = ", ".join(static_languages)
        parts.append(
            f"Se complementó con análisis estático automatizado ({langs}) como evidencia "
            "objetiva adicional."
        )
    return " ".join(parts)


def _template_introduccion(repo: str | None, resumen: str) -> str:
    target = (
        f"el repositorio `{repo}`" if repo else "los repositorios de la organización NTech-TRNM"
    )
    intro = (
        "Este documento presenta los resultados de la revisión de calidad de "
        f"código realizada sobre {target}."
    )
    if resumen:
        intro += f" {resumen}"
    return intro


def _template_actividad_desarrollo(repo: str | None, facts: dict | None) -> str:
    """Sección determinística (sin LLM) de actividad de commits — mismo espíritu
    que ``_template_objetivos``/``_template_metodologia``: describe datos ya
    calculados por ``ntech_agent/commits/query.py``, sin riesgo de que el LLM
    invente cifras."""
    if repo:
        facts = facts or {}
        total = facts.get("total_commits", 0)
        if not total:
            return (
                "No hay commits sincronizados todavía para este repositorio "
                "(correr `python -m scripts.sync_commits`)."
            )
        contributors = facts.get("contributors", [])
        last = facts.get("last_commit")
        last_txt = f"{last['committed_at'][:10]} ({last['author']})" if last else "sin registro"
        return (
            f"En la ventana sincronizada se registraron {total} commits de "
            f"{len(contributors)} contribuidor(es). Último commit: {last_txt}."
        )

    # Reporte de organización: facts es dict[repo, repo_insights].
    facts = facts or {}
    total_all = sum(f.get("total_commits", 0) for f in facts.values())
    if not total_all:
        return (
            "No hay commits sincronizados todavía para los repositorios de la organización "
            "(correr `python -m scripts.sync_commits`)."
        )
    contributors_all = {c["author"] for f in facts.values() for c in f.get("contributors", [])}
    lines = [
        f"En conjunto, los {len(facts)} repositorios registraron {total_all} commits de "
        f"{len(contributors_all)} contribuidor(es) distintos en la ventana sincronizada."
    ]
    for repo_name, f in facts.items():
        last = f.get("last_commit")
        last_txt = last["committed_at"][:10] if last else "sin registro"
        lines.append(
            f"- `{repo_name}`: {f.get('total_commits', 0)} commits · "
            f"{len(f.get('contributors', []))} contribuidor(es) · último commit {last_txt}"
        )
    return "\n".join(lines)


def _severity_counts(review_data: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for h in review_data.get("hallazgos", []):
        sev = h.get("severidad", "?")
        counts[sev] = counts.get(sev, 0) + 1
    return counts


def _repo_evidence(review_data: dict, review_markdown: str) -> str:
    if not review_data:
        return review_markdown or "(sin evidencia disponible)"
    lines = [
        f"Resumen: {review_data.get('resumen_contextualizado', '')}",
        f"Fortalezas: {'; '.join(review_data.get('fortalezas', []))}",
        f"Áreas de oportunidad: {'; '.join(review_data.get('areas_de_oportunidad', []))}",
    ]
    for h in review_data.get("hallazgos", []):
        lines.append(
            f"- [{h.get('severidad')}] {h.get('categoria')}: {h.get('hallazgo')} "
            f"(riesgo: {h.get('por_que')}) → {h.get('recomendacion')}"
        )
    return "\n".join(lines)


def _org_evidence(
    per_repo_review_data: dict[str, dict], languages_by_repo: dict[str, list[str]]
) -> str:
    blocks = []
    for repo, review_data in per_repo_review_data.items():
        langs = ", ".join(languages_by_repo.get(repo, [])) or "no detectado"
        if not review_data:
            blocks.append(f"### {repo}\nLenguajes: {langs}\n(sin datos estructurados)")
            continue
        counts = _severity_counts(review_data)
        counts_txt = ", ".join(f"{k}: {v}" for k, v in counts.items()) or "sin hallazgos"
        blocks.append(
            f"### {repo}\n"
            f"Lenguajes: {langs}\n"
            f"Resumen: {review_data.get('resumen_contextualizado', '')}\n"
            f"Hallazgos por severidad: {counts_txt}\n"
            f"Áreas de oportunidad: {'; '.join(review_data.get('areas_de_oportunidad', []))}"
        )
    return "\n\n".join(blocks)


def _synthesize_narrativa(
    prompt: str, settings: Settings, *, max_tokens: int
) -> _NarrativaEjecutiva:
    """``prompt`` ya viene armado por el caller (``_NARRATIVA_PROMPT`` para un solo
    repo, ``_ORG_NARRATIVA_PROMPT`` para el reporte de organización) — mismo motor
    de síntesis con doble try/except, distinto texto de instrucción."""
    try:
        llm = get_chat_model(
            temperature=settings.reviewer_temperature, max_tokens=max_tokens, settings=settings
        )
        return llm.with_structured_output(_NarrativaEjecutiva).invoke(prompt)
    except Exception as exc:
        log_llm_failure("executive.py::_synthesize_narrativa (structured)", exc)
        # Fallback a texto libre si el structured output falla en el modelo local.
        try:
            llm = get_chat_model(
                temperature=settings.reviewer_temperature, max_tokens=max_tokens, settings=settings
            )
            resp = llm.invoke(
                prompt + "\n\nResponde en dos secciones markdown tituladas 'Principales "
                "hallazgos y conclusiones' y 'Recomendaciones'."
            )
            return _NarrativaEjecutiva(
                principales_hallazgos_y_conclusiones=extract_text(resp.content),
                recomendaciones="",
            )
        except Exception as exc2:
            # El backend puede fallar también en el fallback (p. ej. un 5xx
            # transitorio del proveedor) — degradar sin crashear el reporte.
            log_llm_failure("executive.py::_synthesize_narrativa (fallback texto libre)", exc2)
            return _NarrativaEjecutiva(
                principales_hallazgos_y_conclusiones=(
                    "No se pudo generar esta sección: el backend del LLM no respondió "
                    "al momento de generar el reporte."
                ),
                recomendaciones="",
            )


def build_repo_executive_report(
    repo: str,
    *,
    review_data: dict,
    review_markdown: str,
    static_findings: dict,
    used_repomap: bool,
    commit_facts: dict | None = None,
    settings: Settings | None = None,
) -> ExecutiveReport:
    settings = settings or get_settings()
    resumen = review_data.get("resumen_contextualizado", "") if review_data else ""
    evidence = _repo_evidence(review_data, review_markdown)
    prompt = _NARRATIVA_PROMPT.format(evidencia=evidence)
    narrativa = _synthesize_narrativa(
        prompt, settings, max_tokens=settings.report_narrativa_max_tokens
    )
    static_languages = (static_findings or {}).get("languages", [])
    return ExecutiveReport(
        repo=repo,
        introduccion=_template_introduccion(repo, resumen),
        objetivos_generales=_template_objetivos(repo),
        metodologia=_template_metodologia(
            used_repomap=used_repomap, static_languages=static_languages
        ),
        actividad_desarrollo=_template_actividad_desarrollo(repo, commit_facts),
        principales_hallazgos_y_conclusiones=narrativa.principales_hallazgos_y_conclusiones,
        recomendaciones=narrativa.recomendaciones,
    )


def build_org_executive_report(
    per_repo_review_data: dict[str, dict],
    languages_by_repo: dict[str, list[str]],
    *,
    used_repomap: bool,
    commit_facts_by_repo: dict[str, dict] | None = None,
    settings: Settings | None = None,
) -> ExecutiveReport:
    settings = settings or get_settings()
    evidence = _org_evidence(per_repo_review_data, languages_by_repo)
    prompt = _ORG_NARRATIVA_PROMPT.format(evidencia=evidence)
    narrativa = _synthesize_narrativa(
        prompt, settings, max_tokens=settings.report_org_synthesis_max_tokens
    )
    all_langs = sorted({lang for langs in languages_by_repo.values() for lang in langs})
    repos_txt = ", ".join(f"`{r}`" for r in per_repo_review_data) or "(ninguno)"
    return ExecutiveReport(
        repo=None,
        introduccion=(
            "Este documento presenta un panorama consolidado de la revisión de calidad "
            f"de código realizada sobre {len(per_repo_review_data)} repositorios de la "
            f"organización NTech-TRNM ({repos_txt})."
        ),
        objetivos_generales=_template_objetivos(None),
        metodologia=_template_metodologia(used_repomap=used_repomap, static_languages=all_langs),
        actividad_desarrollo=_template_actividad_desarrollo(None, commit_facts_by_repo),
        principales_hallazgos_y_conclusiones=narrativa.principales_hallazgos_y_conclusiones,
        recomendaciones=narrativa.recomendaciones,
    )


def render_executive_report(report: ExecutiveReport) -> str:
    title = (
        f"# Reporte ejecutivo — {report.repo}"
        if report.repo
        else "# Reporte ejecutivo — Organización NTech-TRNM"
    )
    return "\n\n".join(
        [
            title,
            "## Introducción",
            report.introduccion,
            "## Objetivos generales",
            report.objetivos_generales,
            "## Metodología",
            report.metodologia,
            "## Actividad de desarrollo",
            report.actividad_desarrollo,
            "## Principales hallazgos y conclusiones",
            report.principales_hallazgos_y_conclusiones,
            "## Recomendaciones",
            report.recomendaciones,
        ]
    )
