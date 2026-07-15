"""Retrieval híbrido: búsqueda vectorial (Chroma) + léxica (BM25) fusionadas con
Reciprocal Rank Fusion (EnsembleRetriever), con rerank opcional por LLM.
"""

from __future__ import annotations

from langchain.retrievers import EnsembleRetriever
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


def search(
    query: str,
    *,
    repo: str | None = None,
    k: int | None = None,
    settings: Settings | None = None,
) -> list[Document]:
    """Pipeline completo: retrieve híbrido -> rerank -> top-k."""
    settings = settings or get_settings()
    k = k or settings.retrieval_k
    candidates = retrieve(query, repo=repo, settings=settings)
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
