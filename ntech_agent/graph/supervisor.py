"""Supervisor / router: clasifica la consulta y detecta el repo objetivo."""

from __future__ import annotations

import difflib
from typing import Literal, Optional

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from config.settings import Settings, get_settings
from ntech_agent.graph.state import AgentState
from ntech_agent.llm import get_chat_model


def list_local_repos(settings: Settings | None = None) -> list[str]:
    """Nombres de repos ya clonados en data/repos/."""
    settings = settings or get_settings()
    if not settings.repos_dir.exists():
        return []
    return sorted(p.name for p in settings.repos_dir.iterdir() if p.is_dir())


class _Route(BaseModel):
    route: Literal["summary", "review", "org_report", "assist"] = Field(
        description=(
            "summary: resumen/estado de un repo. review: revisión de calidad y "
            "design patterns de un repo. org_report: panorama de toda la org. "
            "assist: pregunta general o sobre capacidades."
        )
    )
    repo: Optional[str] = Field(
        default=None, description="Nombre del repo objetivo si la consulta apunta a uno."
    )


def _match_repo(name: str | None, known: list[str]) -> str | None:
    if not name:
        return None
    if name in known:
        return name
    lower = {k.lower(): k for k in known}
    if name.lower() in lower:
        return lower[name.lower()]
    close = difflib.get_close_matches(name.lower(), list(lower.keys()), n=1, cutoff=0.6)
    return lower[close[0]] if close else name


def route_node(state: AgentState, config=None) -> dict:
    """Nodo router del grafo."""
    settings = get_settings()
    question = state.get("question") or (
        state["messages"][-1].content if state.get("messages") else ""
    )
    known = list_local_repos(settings)

    prompt = (
        "Clasifica la consulta del usuario y detecta el repo objetivo si lo hay.\n"
        f"Repos disponibles en la organización: {', '.join(known) or '(ninguno indexado)'}\n\n"
        f"Consulta: {question}\n\n"
        "Reglas: si menciona un repo concreto (p. ej. 'resumen de ws-arg'), pon su "
        "nombre en 'repo'. 'org_report' solo si pide un panorama de TODA la org."
    )

    try:
        llm = get_chat_model(temperature=0.0, max_tokens=256, settings=settings)
        result: _Route = llm.with_structured_output(_Route).invoke(prompt)
        route, repo = result.route, result.repo
    except Exception:
        # Fallback heurístico si el structured output falla.
        route, repo = "assist", None
        q = question.lower()
        if any(w in q for w in ("revisa", "review", "calidad", "design pattern")):
            route = "review"
        elif any(w in q for w in ("resumen", "resume", "summary", "estado")):
            route = "summary"
        elif any(w in q for w in ("organización", "organizacion", "toda la org", "todos los repos")):
            route = "org_report"

    repo = _match_repo(repo, known)
    return {
        "question": question,
        "route": route,
        "repo": repo,
        "messages": [HumanMessage(content=question)],
    }
