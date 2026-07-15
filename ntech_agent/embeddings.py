"""Embeddings locales (corren en tu máquina, sin costo de GCP).

Default: ``BAAI/bge-m3`` (multilingüe, bueno para código + docs en español).
Para máquinas muy limitadas puedes usar ``BAAI/bge-small-en-v1.5`` en el .env.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import get_settings


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    """Devuelve el modelo de embeddings cacheado (se carga una sola vez)."""
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )


if __name__ == "__main__":  # python -m ntech_agent.embeddings
    emb = get_embeddings()
    v = emb.embed_query("def foo(): return 42")
    print(f"modelo={get_settings().embedding_model} dim={len(v)}")
