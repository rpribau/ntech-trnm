"""Rerank de candidatos con el LLM: puntúa 0–1 la relevancia y aplica umbral.

Con modelos pequeños esto puede ser lento; se controla con ``NTECH_RERANK_ENABLED``
y ``NTECH_RERANK_THRESHOLD``. Si falla o se desactiva, devuelve el top-k original
(fallback seguro, sin romper el flujo).
"""

from __future__ import annotations

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from config.settings import Settings, get_settings
from ntech_agent.llm import get_chat_model

_MAX_CANDIDATES = 20  # cota para no saturar al modelo local
_SNIPPET_CHARS = 500


class _ScoredDoc(BaseModel):
    index: int = Field(description="Índice del candidato (empezando en 0).")
    score: float = Field(description="Relevancia de 0.0 (irrelevante) a 1.0 (muy relevante).")


class _RerankResult(BaseModel):
    scores: list[_ScoredDoc]


def _prompt(query: str, docs: list[Document]) -> str:
    lines = [
        "Eres un reranker. Puntúa la relevancia de cada candidato para la consulta.",
        f"\nConsulta: {query}\n",
        "Candidatos:",
    ]
    for i, d in enumerate(docs):
        bc = d.metadata.get("breadcrumb", d.metadata.get("path", "?"))
        snippet = d.page_content[:_SNIPPET_CHARS].replace("\n", " ")
        lines.append(f"[{i}] ({bc}) {snippet}")
    lines.append(
        "\nDevuelve un score de 0.0 a 1.0 para cada índice. Sé estricto: "
        "solo puntúa alto lo directamente relevante."
    )
    return "\n".join(lines)


def rerank(
    query: str,
    docs: list[Document],
    *,
    k: int | None = None,
    settings: Settings | None = None,
) -> list[Document]:
    settings = settings or get_settings()
    k = k or settings.retrieval_k
    if not docs:
        return []
    if not settings.rerank_enabled:
        return docs[:k]

    candidates = docs[:_MAX_CANDIDATES]
    try:
        llm = get_chat_model(temperature=0.0, max_tokens=1024, settings=settings)
        structured = llm.with_structured_output(_RerankResult)
        result: _RerankResult = structured.invoke(_prompt(query, candidates))
    except Exception:
        # Fallback: preserva el orden del retrieval híbrido.
        return docs[:k]

    scored: list[tuple[float, Document]] = []
    for s in result.scores:
        if 0 <= s.index < len(candidates):
            scored.append((s.score, candidates[s.index]))

    kept = [(sc, d) for sc, d in scored if sc >= settings.rerank_threshold]
    kept.sort(key=lambda x: x[0], reverse=True)

    if not kept:  # rescate: nunca devolver vacío si había candidatos
        return docs[:k]
    return [d for _, d in kept[:k]]
