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
    # Ruta elegida por el supervisor: summary | review | org_report | assist
    route: str
    # Repo objetivo (si la consulta apunta a uno concreto).
    repo: Optional[str]
    # Documentos recuperados por RAG.
    retrieved: list[Document]
    # Señales de análisis estático (ruff/radon) del repo objetivo.
    static_findings: dict
    # Salida del nodo revisor (markdown).
    review: str
    # Respuesta final sintetizada (markdown).
    answer: str
