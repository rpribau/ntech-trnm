"""Retrieval híbrido: búsqueda vectorial (Chroma) + léxica (BM25) fusionadas con
Reciprocal Rank Fusion (EnsembleRetriever), con rerank opcional por LLM.
"""

from __future__ import annotations

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config.settings import Settings, get_settings
from ntech_agent.ingest.index import get_vectorstore, load_chunks_jsonl
from ntech_agent.retrieval.rerank import rerank


def _build_retriever(
    settings: Settings, repo: str | None, fetch_k: int
) -> BaseRetriever:
    vs = get_vectorstore(settings)
    search_kwargs: dict = {"k": fetch_k}
    if repo:
        search_kwargs["filter"] = {"repo": repo}
    vector = vs.as_retriever(search_kwargs=search_kwargs)

    docs = load_chunks_jsonl(settings)
    if repo:
        docs = [d for d in docs if d.metadata.get("repo") == repo]

    if not docs:  # sin jsonl BM25: solo vectorial
        return vector

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = fetch_k
    return EnsembleRetriever(retrievers=[vector, bm25], weights=[0.5, 0.5])


def retrieve(
    query: str,
    *,
    repo: str | None = None,
    k: int | None = None,
    fetch_k: int | None = None,
    settings: Settings | None = None,
) -> list[Document]:
    """Recupera los candidatos híbridos (sin rerank)."""
    settings = settings or get_settings()
    k = k or settings.retrieval_k
    fetch_k = fetch_k or settings.retrieval_fetch_k
    retriever = _build_retriever(settings, repo, fetch_k)
    return retriever.invoke(query)[:fetch_k]


def _dedupe_by_id(docs: list[Document]) -> list[Document]:
    seen: set[str] = set()
    out: list[Document] = []
    for d in docs:
        doc_id = d.metadata.get("id")
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(d)
    return out


def search(
    query: str,
    *,
    repo: str | list[str] | None = None,
    k: int | None = None,
    settings: Settings | None = None,
) -> list[Document]:
    """Pipeline completo: retrieve híbrido -> rerank -> top-k.

    ``repo`` acepta un solo nombre, una lista (comparar 2+ repos) o None (sin
    filtro). Con 2+ repos, tanto el retrieval COMO el rerank corren POR REPO (no
    un pool combinado con un solo rerank global): un rerank global vuelve a poder
    dejar un repo entero por debajo del umbral de relevancia (o fuera de
    ``_MAX_CANDIDATES`` según el orden de concatenación) aunque el retrieval por
    repo ya haya garantizado candidatos reales para cada uno — es la misma clase
    de bug (ranking global sin piso por repo), solo que un paso más adelante.
    Rerankear por repo y tomar un piso mínimo de CADA UNO antes de unir es lo que
    garantiza representación real en el resultado final.
    """
    settings = settings or get_settings()
    k = k or settings.retrieval_k
    repos = repo if isinstance(repo, list) else ([repo] if repo else [])

    if len(repos) > 1:
        per_repo_fetch_k = max(
            settings.retrieval_multi_repo_min_fetch_k,
            settings.retrieval_fetch_k // len(repos),
        )
        per_repo_k = max(settings.retrieval_multi_repo_min_k, k // len(repos))
        docs_out: list[Document] = []
        for r in repos:
            repo_candidates = retrieve(query, repo=r, fetch_k=per_repo_fetch_k, settings=settings)
            docs_out.extend(rerank(query, repo_candidates, k=per_repo_k, settings=settings))
        return _dedupe_by_id(docs_out)

    candidates = retrieve(query, repo=repos[0] if repos else None, settings=settings)
    return rerank(query, candidates, k=k, settings=settings)


def retrieve_guidelines(
    query: str, *, k: int = 4, settings: Settings | None = None
) -> list[Document]:
    """Recupera solo chunks de guidelines relevantes a la consulta."""
    settings = settings or get_settings()
    vs = get_vectorstore(settings)
    retriever = vs.as_retriever(
        search_kwargs={"k": k, "filter": {"source_type": "guideline"}}
    )
    return retriever.invoke(query)
