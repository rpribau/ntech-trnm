"""Chunking estructural de código, docs y guidelines.

Produce ``Document`` de LangChain con metadata (repo, path, línea, breadcrumb…)
e IDs deterministas (uuid5) para que la reindexación sea idempotente.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from config.settings import Settings, get_settings

# Namespace fijo para IDs deterministas.
_NS = uuid.UUID("6f4b2d2a-0000-4000-8000-6e746563680a")

# Extensiones de código -> lenguaje del splitter (los no listados usan el default).
_CODE_LANG: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rb": Language.RUBY,
    ".rs": Language.RUST,
    ".c": Language.CPP,
    ".h": Language.CPP,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".cs": Language.CSHARP,
    ".php": Language.PHP,
    ".scala": Language.SCALA,
    ".kt": Language.KOTLIN,
    ".swift": Language.SWIFT,
}
# Otras extensiones de código/config que indexamos con el splitter genérico.
_OTHER_CODE = {".sql", ".sh", ".ps1", ".r", ".m", ".jl", ".yaml", ".yml", ".toml"}
_DOC_EXT = {".md", ".rst", ".txt"}

# Directorios y archivos a ignorar.
_SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "env", "__pycache__", ".mypy_cache",
    ".ruff_cache", ".pytest_cache", "dist", "build", ".next", "target", ".idea",
    ".vscode", "site-packages", ".ipynb_checkpoints", "data",
}
_MAX_BYTES = 1_000_000  # ignora archivos > 1 MB (probablemente datos/artefactos)


def _det_id(repo: str, path: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NS, f"{repo}::{path}::{chunk_index}"))


def _splitter(ext: str, settings: Settings) -> RecursiveCharacterTextSplitter:
    lang = _CODE_LANG.get(ext)
    if lang is not None:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )


def _read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > _MAX_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, ValueError):
        return None


def _notebook_to_text(raw: str) -> str:
    """Extrae source de celdas de un .ipynb como texto plano."""
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    parts: list[str] = []
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        text = "".join(src) if isinstance(src, list) else str(src)
        kind = cell.get("cell_type", "code")
        parts.append(f"# --- {kind} cell ---\n{text}")
    return "\n\n".join(parts)


def _line_of(text: str, chunk: str, cursor: int) -> tuple[int, int]:
    """Estima la línea de inicio del chunk y devuelve (start_line, nuevo_cursor)."""
    probe = chunk[:60]
    pos = text.find(probe, cursor)
    if pos == -1:
        pos = cursor
    start_line = text.count("\n", 0, pos) + 1
    return start_line, pos + 1


def _chunk_file(path: Path, repo: str, rel: str, settings: Settings) -> Iterator[Document]:
    ext = path.suffix.lower()
    raw = _read_text(path)
    if not raw:
        return

    if ext == ".ipynb":
        text = _notebook_to_text(raw)
        source_type, language = "notebook", "jupyter"
        splitter = _splitter(".txt", settings)
    elif ext in _CODE_LANG or ext in _OTHER_CODE:
        text = raw
        source_type = "code"
        language = _CODE_LANG.get(ext).value if ext in _CODE_LANG else ext.lstrip(".")
        splitter = _splitter(ext, settings)
    elif ext in _DOC_EXT:
        text = raw
        source_type, language = "doc", "markdown"
        splitter = _splitter(".md", settings)
    else:
        return

    if not text.strip():
        return

    breadcrumb = f"{repo} > {rel}"
    cursor = 0
    for i, chunk in enumerate(splitter.split_text(text)):
        start_line, cursor = _line_of(text, chunk, cursor)
        yield Document(
            page_content=f"[{breadcrumb} :{start_line}]\n{chunk}",
            metadata={
                "id": _det_id(repo, rel, i),
                "repo": repo,
                "path": rel,
                "start_line": start_line,
                "language": language or "text",
                "source_type": source_type,
                "breadcrumb": breadcrumb,
                "chunk_index": i,
            },
        )


def iter_repo_documents(settings: Settings | None = None) -> Iterator[Document]:
    """Recorre ``data/repos/`` y produce Documents chunked de cada repo."""
    settings = settings or get_settings()
    repos_dir = settings.repos_dir
    if not repos_dir.exists():
        return
    for repo_path in sorted(p for p in repos_dir.iterdir() if p.is_dir()):
        repo = repo_path.name
        for path in repo_path.rglob("*"):
            if not path.is_file():
                continue
            if any(part in _SKIP_DIRS for part in path.relative_to(repo_path).parts):
                continue
            rel = path.relative_to(repo_path).as_posix()
            yield from _chunk_file(path, repo, rel, settings)


def iter_guideline_documents(settings: Settings | None = None) -> Iterator[Document]:
    """Chunk de los guidelines (Software Dev + Ciencia de Datos)."""
    settings = settings or get_settings()
    gdir = settings.guidelines_dir
    if not gdir.exists():
        return
    splitter = _splitter(".md", settings)
    for path in sorted(gdir.rglob("*.md")):
        raw = _read_text(path)
        if not raw:
            continue
        rel = path.relative_to(settings.guidelines_dir.parent).as_posix()
        category = path.parent.name  # software_dev | data_science
        breadcrumb = f"guideline > {category} > {path.stem}"
        for i, chunk in enumerate(splitter.split_text(raw)):
            yield Document(
                page_content=f"[{breadcrumb}]\n{chunk}",
                metadata={
                    "id": _det_id("__guidelines__", rel, i),
                    "repo": "__guidelines__",
                    "path": rel,
                    "start_line": 1,
                    "language": "markdown",
                    "source_type": "guideline",
                    "breadcrumb": breadcrumb,
                    "chunk_index": i,
                    "category": category,
                },
            )


def iter_all_documents(settings: Settings | None = None) -> Iterator[Document]:
    settings = settings or get_settings()
    yield from iter_repo_documents(settings)
    yield from iter_guideline_documents(settings)
