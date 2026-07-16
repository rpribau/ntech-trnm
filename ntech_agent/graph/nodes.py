"""Nodos del grafo: retrieve, static, reviewer (Skill) y synthesize."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from config.settings import Settings, get_settings
from ntech_agent.analysis.static import format_for_prompt, run_static_analysis
from ntech_agent.graph.state import AgentState
from ntech_agent.llm import extract_text, get_chat_model
from ntech_agent.retrieval.hybrid import retrieve_guidelines, search


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_skill(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    skill_path = settings.skills_dir / "code_best_practices" / "SKILL.md"
    rubric_path = settings.skills_dir / "code_best_practices" / "rubric.yaml"
    parts = []
    if skill_path.exists():
        parts.append(skill_path.read_text(encoding="utf-8"))
    if rubric_path.exists():
        parts.append("\n\n# rubric.yaml\n" + rubric_path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _format_context(docs: list[Document], settings: Settings) -> str:
    out = []
    for d in docs[: settings.ctx_max_docs]:
        bc = d.metadata.get("breadcrumb", d.metadata.get("path", "?"))
        out.append(f"### {bc}\n{d.page_content[: settings.ctx_chars]}")
    return "\n\n".join(out) if out else "(sin contexto recuperado)"


def _citations(docs: list[Document]) -> str:
    seen: dict[str, None] = {}
    for d in docs:
        m = d.metadata
        key = f"- `{m.get('repo')}/{m.get('path')}:{m.get('start_line', 1)}`"
        seen.setdefault(key, None)
    return "\n".join(seen.keys()) if seen else "_(sin fuentes)_"


# --------------------------------------------------------------------------- #
# Nodos
# --------------------------------------------------------------------------- #
def retrieve_node(state: AgentState, config=None) -> dict:
    settings = get_settings()
    q = state.get("question", "")
    repo = state.get("repo")
    if state.get("route") == "org_report":
        docs = search(q or "panorama general de la organización", repo=None, settings=settings)
    else:
        docs = search(q, repo=repo, settings=settings)
    return {"retrieved": docs}


def static_node(state: AgentState, config=None) -> dict:
    settings = get_settings()
    repo = state.get("repo")
    if state.get("route") in ("review", "summary") and repo:
        findings = run_static_analysis(settings.repos_dir / repo, settings)
    else:
        findings = {}
    return {"static_findings": findings}


class _Finding(BaseModel):
    ubicacion: str = Field(description="repo/ruta.py:línea o 'repo-wide'")
    categoria: str
    severidad: str = Field(description="critico | mayor | menor | sugerencia")
    hallazgo: str
    por_que: str
    guia: str = Field(description="archivo de guideline citado")
    recomendacion: str


class _Review(BaseModel):
    resumen_contextualizado: str
    fortalezas: list[str]
    areas_de_oportunidad: list[str]
    hallazgos: list[_Finding]


def _render_review(r: _Review) -> str:
    lines = [f"**Resumen:** {r.resumen_contextualizado}", "", "**Fortalezas:**"]
    lines += [f"- {s}" for s in r.fortalezas] or ["- (ninguna destacada)"]
    lines += ["", "**Áreas de oportunidad:**"]
    lines += [f"- {a}" for a in r.areas_de_oportunidad] or ["- (ninguna)"]
    lines += ["", "**Hallazgos:**"]
    if not r.hallazgos:
        lines.append("- (sin hallazgos)")
    for h in r.hallazgos:
        lines.append(
            f"- [{h.severidad}] `{h.ubicacion}` — **{h.categoria}**: {h.hallazgo} "
            f"_({h.por_que}. Guía: {h.guia})_ → {h.recomendacion}"
        )
    return "\n".join(lines)


def reviewer_node(state: AgentState, config=None) -> dict:
    if state.get("route") != "review":
        return {}
    settings = get_settings()
    repo = state.get("repo") or "(sin repo)"
    guidelines = retrieve_guidelines(
        f"buenas prácticas, design patterns y code smells para {repo}",
        k=settings.reviewer_guidelines_k,
        settings=settings,
    )
    static_txt = format_for_prompt(state.get("static_findings", {}))
    code_ctx = _format_context(state.get("retrieved", []), settings)
    guide_ctx = _format_context(guidelines, settings)

    prompt = (
        f"{load_skill(settings)}\n\n"
        "=== CONTEXTO PARA LA REVISIÓN ===\n"
        f"Repo objetivo: {repo}\n\n"
        f"--- Código recuperado ---\n{code_ctx}\n\n"
        f"--- Guidelines relevantes ---\n{guide_ctx}\n\n"
        f"--- Análisis estático (evidencia objetiva) ---\n{static_txt}\n\n"
        "Aplica la Skill y produce la revisión estructurada. Cita el código "
        "(archivo:línea) y la guía en cada hallazgo. No inventes: usa solo el contexto."
    )

    try:
        llm = get_chat_model(
            temperature=settings.reviewer_temperature,
            max_tokens=settings.reviewer_max_tokens,
            settings=settings,
        )
        review: _Review = llm.with_structured_output(_Review).invoke(prompt)
        return {"review": _render_review(review)}
    except Exception:
        # Fallback a texto libre si el structured output falla en el modelo local.
        llm = get_chat_model(
            temperature=settings.reviewer_temperature,
            max_tokens=settings.reviewer_max_tokens,
            settings=settings,
        )
        resp = llm.invoke(prompt + "\n\nResponde en markdown estructurado.")
        return {"review": extract_text(resp.content)}


def synthesize_node(state: AgentState, config=None) -> dict:
    settings = get_settings()
    route = state.get("route", "assist")
    docs = state.get("retrieved", [])
    citations = _citations(docs)

    if route == "review":
        body = state.get("review") or "_(no se generó revisión)_"
        answer = (
            f"# Revisión de código — {state.get('repo') or 'repo'}\n\n{body}\n\n"
            f"## Fuentes\n{citations}"
        )
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    ctx = _format_context(docs, settings)
    static_txt = format_for_prompt(state.get("static_findings", {})) if state.get(
        "static_findings"
    ) else ""

    if route == "summary":
        instr = (
            f"Da un **resumen contextualizado** del repo '{state.get('repo')}': qué es, "
            "en qué se está trabajando, su estado y estructura, y 2-3 áreas de "
            "oportunidad. Basado solo en el contexto y el análisis estático."
        )
    elif route == "org_report":
        instr = (
            "Da un **panorama de la organización**: qué repos hay, temas que abordan, "
            "fortalezas transversales y áreas de oportunidad comunes. Basado en el contexto."
        )
    else:  # assist
        instr = (
            "Responde la pregunta del usuario de forma clara y útil, apoyándote en el "
            "contexto (código y guidelines) cuando aplique."
        )

    prompt = (
        f"Pregunta del usuario: {state.get('question')}\n\n"
        f"{instr}\n\n"
        f"--- Contexto recuperado ---\n{ctx}\n\n"
        + (f"--- Análisis estático ---\n{static_txt}\n\n" if static_txt else "")
        + "Responde en español, en markdown, sin inventar información fuera del contexto."
    )

    llm = get_chat_model(
        temperature=settings.synthesize_temperature,
        max_tokens=settings.synthesize_max_tokens,
        settings=settings,
    )
    resp = llm.invoke(prompt)
    answer = f"{extract_text(resp.content)}\n\n## Fuentes\n{citations}"
    return {"answer": answer, "messages": [AIMessage(content=answer)]}
