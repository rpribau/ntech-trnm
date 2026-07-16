"""Ensambla el grafo LangGraph (supervisor + nodos) y expone ``run_agent``.

Flujo: router → retrieve → repomap → static → reviewer → commits → synthesize → END.
Los nodos repomap/static/reviewer/commits se auto-omiten según la ruta.
"""

from __future__ import annotations

import sqlite3
import uuid
from functools import lru_cache

from langgraph.graph import END, StateGraph

from config.settings import Settings, get_settings
from ntech_agent.graph.nodes import (
    commit_insights_node,
    repomap_node,
    retrieve_node,
    reviewer_node,
    static_node,
    synthesize_node,
)
from ntech_agent.graph.state import AgentState
from ntech_agent.graph.supervisor import route_node


def _checkpointer(settings: Settings):
    """SqliteSaver persistente; si no está disponible, MemorySaver en memoria."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        settings.ensure_dirs()
        conn = sqlite3.connect(str(settings.checkpointer_path), check_same_thread=False)
        return SqliteSaver(conn)
    except Exception:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()


@lru_cache
def get_graph(settings: Settings | None = None):
    """Compila (y cachea) el grafo del agente."""
    settings = settings or get_settings()
    g = StateGraph(AgentState)
    g.add_node("router", route_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("repomap", repomap_node)
    g.add_node("static", static_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("commits", commit_insights_node)
    g.add_node("synthesize", synthesize_node)

    g.set_entry_point("router")
    g.add_edge("router", "retrieve")
    g.add_edge("retrieve", "repomap")
    g.add_edge("repomap", "static")
    g.add_edge("static", "reviewer")
    g.add_edge("reviewer", "commits")
    g.add_edge("commits", "synthesize")
    g.add_edge("synthesize", END)

    return g.compile(checkpointer=_checkpointer(settings))


def run_agent(question: str, *, thread_id: str | None = None) -> dict:
    """Ejecuta una consulta contra el agente. Devuelve el estado final."""
    graph = get_graph()
    thread_id = thread_id or uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke({"question": question}, config=config)


if __name__ == "__main__":  # python -m ntech_agent.graph.builder "Dame un resumen de ws-arg"
    import sys

    q = " ".join(sys.argv[1:]) or "¿Qué repos hay en la organización?"
    final = run_agent(q)
    print(f"\n[ruta: {final.get('route')} | repo: {final.get('repo')}]\n")
    print(final.get("answer", "(sin respuesta)"))
