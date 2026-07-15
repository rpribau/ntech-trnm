"""Construye/actualiza el índice vectorial (Chroma) sobre repos + guidelines.

También persiste los chunks en un ``chunks.jsonl`` para reconstruir el índice
BM25 del retrieval híbrido.
"""

from __future__ import annotations

import json

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import Settings, get_settings
from ntech_agent.embeddings import get_embeddings
from ntech_agent.ingest.chunking import iter_all_documents

_BATCH = 128


def _client(settings: Settings) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(settings.index_dir))


def get_vectorstore(settings: Settings | None = None) -> Chroma:
    """Devuelve el vectorstore Chroma persistente (para retrieval)."""
    settings = settings or get_settings()
    return Chroma(
        client=_client(settings),
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
    )


def _write_chunks_jsonl(docs: list[Document], settings: Settings) -> None:
    with settings.bm25_docs_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"page_content": d.page_content, "metadata": d.metadata}) + "\n")


def load_chunks_jsonl(settings: Settings | None = None) -> list[Document]:
    """Carga los chunks persistidos (para el retriever BM25)."""
    settings = settings or get_settings()
    path = settings.bm25_docs_path
    if not path.exists():
        return []
    docs: list[Document] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append(Document(page_content=rec["page_content"], metadata=rec["metadata"]))
    return docs


def build_index(*, reset: bool = True, settings: Settings | None = None) -> dict:
    """Indexa todo el corpus. Con ``reset`` recrea la colección (idempotente)."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    client = _client(settings)

    if reset:
        try:
            client.delete_collection(settings.chroma_collection)
        except Exception:
            pass

    vs = Chroma(
        client=client,
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
    )

    docs = list(iter_all_documents(settings))
    if not docs:
        return {"indexed": 0, "repos": 0, "guidelines": 0, "note": "No hay documentos. ¿Corriste sync_repos?"}

    # Alta en lotes (los embeddings se calculan aquí, en CPU).
    for start in range(0, len(docs), _BATCH):
        batch = docs[start : start + _BATCH]
        vs.add_documents(batch, ids=[d.metadata["id"] for d in batch])

    _write_chunks_jsonl(docs, settings)

    repos = {d.metadata["repo"] for d in docs if d.metadata["repo"] != "__guidelines__"}
    n_guidelines = sum(1 for d in docs if d.metadata["source_type"] == "guideline")
    return {
        "indexed": len(docs),
        "repos": len(repos),
        "guidelines": n_guidelines,
        "collection": settings.chroma_collection,
    }
