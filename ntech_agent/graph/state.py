"""Estado compartido del grafo del agente."""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # Historial de conversación (memoria por thread vía checkpointer).
    messages: Annotated[list[AnyMessage], add_messages]
    # Pregunta del turno actual.
    question: str
    # Ruta elegida por el supervisor: summary | review | org_report | assist | commits
    route: str
    # Repo objetivo (si la consulta apunta a UN solo repo; None si son 0 o 2+).
    repo: Optional[str]
    # Todos los repos mencionados/matcheados en la consulta (0, 1 o varios) — ver
    # supervisor.py::route_node. "repo" es simplemente repos[0] cuando len(repos)==1.
    repos: list[str]
    # Documentos recuperados por RAG.
    retrieved: list[Document]
    # Señales de análisis estático (ruff/radon) del repo objetivo.
    static_findings: dict
    # Salida del nodo revisor (markdown, compacto para el chat).
    review: str
    # Salida estructurada del nodo revisor (dump de _Review; {} si cayó al fallback
    # de texto libre). La usan los reportes ejecutivos para no re-parsear markdown.
    review_data: dict
    # Métricas de actividad de commits por repo (route == "commits"), leídas de
    # ntech_agent/commits/query.py — ver commit_insights_node.
    commit_facts: dict

    # Respuesta final sintetizada (markdown).
    answer: str
