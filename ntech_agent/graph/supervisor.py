"""Supervisor / router: clasifica la consulta y detecta el repo objetivo."""

from __future__ import annotations

import difflib
from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from config.settings import Settings, get_settings
from ntech_agent.graph.state import AgentState
from ntech_agent.llm import get_chat_model, log_llm_failure


def list_local_repos(settings: Settings | None = None) -> list[str]:
    """Nombres de repos ya clonados en data/repos/."""
    settings = settings or get_settings()
    if not settings.repos_dir.exists():
        return []
    return sorted(p.name for p in settings.repos_dir.iterdir() if p.is_dir())


class _Route(BaseModel):
    route: Literal["summary", "review", "org_report", "assist", "commits"] = Field(
        description=(
            "summary: resumen/estado de UN repo puntual (nombrado explícitamente). "
            "review: revisión de calidad y design patterns de un repo. org_report: "
            "panorama de TODOS/varios repos de la organización en conjunto — incluye "
            "pedidos como 'resumen de todos los repositorios' o 'resumen de la org' "
            "(la palabra 'resumen' sola NO implica 'summary': si pide un resumen de "
            "TODA la organización o de 'todos los repos', es 'org_report'). "
            "commits: preguntas sobre actividad/historial de commits de un repo "
            "(últimos cambios, quién participa/contribuye, actividad reciente) — "
            "NO calidad de código. assist: pregunta general o sobre capacidades."
        )
    )
    repos: list[str] = Field(
        default_factory=list,
        description=(
            "Nombres de TODOS los repos mencionados explícitamente en la consulta "
            "(0, 1 o varios), aunque la consulta sea una continuación breve de la "
            "conversación (p. ej. 'y con respecto a ws-arg y ws-mex?' menciona "
            "'ws-arg' Y 'ws-mex' — incluí ambos). Si la consulta compara o menciona "
            "más de un repo (p. ej. 'diferencia entre ws-arg y ws-br'), incluí cada uno."
        ),
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
        "Clasifica la consulta del usuario y detecta los repos objetivo si los hay.\n"
        f"Repos disponibles en la organización: {', '.join(known) or '(ninguno indexado)'}\n\n"
        f"Consulta: {question}\n\n"
        "Reglas: si menciona uno o más repos concretos por nombre (p. ej. 'resumen de "
        "ws-arg', 'diferencia entre ws-arg y ws-br', o una continuación corta como 'y "
        "con respecto a ws-arg y ws-mex?'), poné TODOS sus nombres en 'repos', incluso "
        "si la consulta es breve o depende del contexto previo de la conversación. "
        "'org_report' si pide un panorama de TODA la org o de 'todos los repos/"
        "repositorios' (no hace falta listar repos en ese caso) — no confundir con "
        "'summary', que es solo para UN repo puntual nombrado explícitamente."
    )

    try:
        llm = get_chat_model(
            temperature=settings.router_temperature,
            max_tokens=settings.router_max_tokens,
            settings=settings,
        )
        result: _Route = llm.with_structured_output(_Route).invoke(prompt)
        route, raw_repos = result.route, result.repos
    except Exception as exc:
        log_llm_failure("supervisor.py::route_node", exc)
        # Fallback heurístico si el structured output falla. SÍ intenta detectar
        # nombres de repo acá (escaneando contra los repos conocidos) — antes no
        # lo hacía, lo que significaba que cualquier falla transitoria del LLM en
        # una pregunta multi-repo perdía en silencio las garantías de piso por
        # repo en retrieval (ver retrieval/hybrid.py::search), degradando a una
        # búsqueda global sin ninguna fairness entre repos.
        route = "assist"
        q = question.lower()
        raw_repos = [name for name in known if name.lower() in q]
        if any(
            w in q
            for w in (
                "último commit", "ultimo commit", "últimos commits", "ultimos commits",
                "últimos cambios", "ultimos cambios", "quién participa", "quien participa",
                "quienes participan", "quiénes participan", "contribuidores", "colaboradores",
                "actividad reciente", "quién trabaja", "quien trabaja", "quienes trabajan",
                "autores de",
            )
        ):
            route = "commits"
        elif any(w in q for w in ("revisa", "review", "calidad", "design pattern")):
            route = "review"
        # org_report se chequea ANTES que summary: "resumen de todos los repos" contiene
        # la palabra "resumen" y si no, cae en la rama "summary" por error (la de
        # org_report nunca se llega a evaluar por el elif) — ver el bug real que motivó
        # este reordenamiento.
        elif any(
            w in q
            for w in (
                "organización", "organizacion", "toda la org", "todos los repos",
                "todos los repositorios",
            )
        ):
            route = "org_report"
        elif any(w in q for w in ("resumen", "resume", "summary", "estado")):
            route = "summary"

    # Dedupe preservando orden: dos nombres crudos distintos pueden matchear al
    # mismo repo conocido (p. ej. typo + nombre exacto del mismo repo) — sin esto,
    # una pregunta de un solo repo podría terminar con repos=[x, x] y el repo
    # singular quedando None por error (len==2 en vez de 1).
    repos = list(dict.fromkeys(m for m in (_match_repo(n, known) for n in raw_repos) if m))
    repo = repos[0] if len(repos) == 1 else None

    return {
        "question": question,
        "route": route,
        "repo": repo,
        "repos": repos,
        "messages": [HumanMessage(content=question)],
    }
