"""Configuración central del proyecto (Pydantic Settings, config-driven).

Todos los parámetros se leen de variables de entorno con prefijo ``NTECH_``
(ver ``.env.example``). No hay valores hardcodeados fuera de aquí.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raíz del repo = carpeta que contiene este archivo config/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Configuración tipada de la aplicación."""

    model_config = SettingsConfigDict(
        env_prefix="NTECH_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Backend del LLM ----
    # cloudrun  = vLLM propio en GCP Cloud Run (self-hosted, GPU L4)
    # ollama    = modelo local vía Ollama (tu hardware, barato/offline)
    # anthropic = API de Anthropic (Claude), pay-per-token, sin infra propia
    llm_backend: Literal["cloudrun", "ollama", "anthropic"] = "cloudrun"

    # ---- Cloud Run (vLLM) ----
    cloudrun_url: str = ""
    cloudrun_model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    # Se lee sin el prefijo NTECH_ (variable estándar de google-auth).
    google_application_credentials: str = Field(
        default="", validation_alias="GOOGLE_APPLICATION_CREDENTIALS"
    )

    # ---- Ollama (modelo local) ----
    ollama_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5-coder:7b"

    # ---- Anthropic (Claude API) ----
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-4-8"

    # ---- Embeddings ----
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"

    # ---- GitHub ----
    github_token: str = ""
    github_org: str = "NTech-TRNM"
    github_mcp_url: str = "https://api.githubcopilot.com/mcp/"

    # ---- Rutas (relativas a la raíz del repo) ----
    repos_dir: Path = Path("data/repos")
    index_dir: Path = Path("data/index")
    reports_dir: Path = Path("data/reports")
    guidelines_dir: Path = Path("guidelines")
    skills_dir: Path = Path("skills")

    # ---- RAG ----
    chroma_collection: str = "ntech_code"
    chunk_size: int = 1200
    chunk_overlap: int = 150
    retrieval_k: int = 8
    retrieval_fetch_k: int = 24
    rerank_enabled: bool = True
    rerank_threshold: float = 0.3

    # ---- Análisis estático ----
    static_analysis_enabled: bool = True

    @field_validator(
        "repos_dir", "index_dir", "reports_dir", "guidelines_dir", "skills_dir",
        mode="after",
    )
    @classmethod
    def _absolutize(cls, v: Path) -> Path:
        """Convierte rutas relativas a absolutas respecto a la raíz del repo."""
        return v if v.is_absolute() else (PROJECT_ROOT / v).resolve()

    # ---- Helpers derivados ----
    @property
    def bm25_docs_path(self) -> Path:
        """Ruta del jsonl con los chunks (para reconstruir el índice BM25)."""
        return self.index_dir / "chunks.jsonl"

    @property
    def checkpointer_path(self) -> Path:
        """Ruta del checkpointer SQLite de LangGraph (memoria por thread)."""
        return self.index_dir / "checkpoints.sqlite"

    def ensure_dirs(self) -> None:
        """Crea las carpetas de datos si no existen."""
        for d in (self.repos_dir, self.index_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def active_model(self) -> str:
        if self.llm_backend == "cloudrun":
            return self.cloudrun_model
        if self.llm_backend == "anthropic":
            return self.anthropic_model
        return self.ollama_model


@lru_cache
def get_settings() -> Settings:
    """Devuelve una instancia cacheada de Settings."""
    return Settings()
