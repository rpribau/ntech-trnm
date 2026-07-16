"""Construye y persiste el repo map de un repositorio (calculado offline, en
``build_index``, nunca por-consulta): parsea con tree-sitter, arma el grafo de
dependencias entre archivos y calcula un PageRank base."""

from __future__ import annotations

import configparser
import json
from datetime import UTC, datetime
from pathlib import Path

from config.settings import Settings, get_settings
from ntech_agent.ingest.chunking import _MAX_BYTES, _SKIP_DIRS
from ntech_agent.repomap.graph import graph_from_edges, rank_files
from ntech_agent.repomap.tags import (
    extract_definitions,
    extract_references,
    is_notebook,
    language_for_path,
    notebook_code_to_text,
)


def _read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > _MAX_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, ValueError):
        return None


# Nombres de carpeta que casi siempre contienen código de terceros vendorizado,
# no el código propio del repo. Heurística por nombre (no solo `.gitmodules`):
# en ws-br, la librería vendorizada YOLOv8-TensorRT-CPP vive bajo
# cpp-motor/libs/ sin que el ws-br raíz la declare como submódulo — solo el
# .gitmodules ANIDADO dentro de esa librería declara SU PROPIO submódulo interno.
# Sin esta heurística, el ranking PageRank queda dominado por código que no es
# de la organización.
_VENDOR_DIR_NAMES = {
    "libs", "lib", "vendor", "vendored", "third_party", "third-party", "external", "deps",
}


def _submodule_prefixes(repo_dir: Path) -> set[str]:
    """Rutas declaradas en el ``.gitmodules`` raíz del repo (si existe)."""
    gitmodules = repo_dir / ".gitmodules"
    if not gitmodules.exists():
        return set()
    cp = configparser.ConfigParser()
    try:
        cp.read(gitmodules, encoding="utf-8")
    except configparser.Error:
        return set()
    return {
        cp.get(section, "path").strip().replace("\\", "/")
        for section in cp.sections()
        if cp.has_option(section, "path")
    }


def _is_vendored(rel: str, submodule_prefixes: set[str]) -> bool:
    parts = Path(rel).parts
    if any(p in _VENDOR_DIR_NAMES for p in parts):
        return True
    return any(rel == sp or rel.startswith(sp + "/") for sp in submodule_prefixes)


def _is_excluded(rel: str, submodule_prefixes: set[str]) -> bool:
    if any(part in _SKIP_DIRS for part in Path(rel).parts):
        return True
    return _is_vendored(rel, submodule_prefixes)


def _repomap_path(repo: str, settings: Settings) -> Path:
    return settings.index_dir / "repomap" / f"{repo}.json"


def build_repo_map(repo: str, settings: Settings | None = None) -> dict:
    """Parsea el repo, arma el grafo de dependencias y lo persiste en
    ``data/index/repomap/{repo}.json``. Devuelve stats resumidas."""
    settings = settings or get_settings()
    repo_dir = settings.repos_dir / repo
    if not repo_dir.exists():
        return {"repo": repo, "n_files": 0, "n_edges": 0, "languages": [], "excluded_vendored": 0}

    submodule_prefixes = _submodule_prefixes(repo_dir)
    file_text: dict[str, str] = {}
    file_language: dict[str, str] = {}
    file_symbols: dict[str, list] = {}
    defined_index: dict[str, set[str]] = {}
    excluded_vendored = 0

    for path in sorted(repo_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_dir).as_posix()
        if _is_excluded(rel, submodule_prefixes):
            if _is_vendored(rel, submodule_prefixes):
                excluded_vendored += 1
            continue
        lang = language_for_path(path)
        if lang is None:
            continue
        raw = _read_text(path)
        if not raw or not raw.strip():
            continue
        text = notebook_code_to_text(raw) if is_notebook(path) else raw
        if not text.strip():
            continue
        try:
            symbols = extract_definitions(text, lang)
        except Exception:
            symbols = []
        file_text[rel] = text
        file_language[rel] = lang
        file_symbols[rel] = symbols
        for s in symbols:
            defined_index.setdefault(s.name, set()).add(rel)

    all_names = set(defined_index)
    edges: dict[tuple[str, str], int] = {}
    for rel, text in file_text.items():
        for name in extract_references(text, all_names):
            for def_file in defined_index.get(name, ()):
                if def_file == rel:
                    continue
                key = (rel, def_file)
                edges[key] = edges.get(key, 0) + 1

    files_meta = [
        {
            "path": rel,
            "language": file_language[rel],
            "n_defs": len(file_symbols[rel]),
            "symbols": [
                {"name": s.name, "kind": s.kind, "start_line": s.start_line, "end_line": s.end_line}
                for s in file_symbols[rel]
            ],
        }
        for rel in sorted(file_language)
    ]
    edges_meta = [{"from": a, "to": b, "weight": w} for (a, b), w in sorted(edges.items())]

    graph = graph_from_edges(list(file_language), edges_meta)
    rank_uniform = rank_files(graph, seeds=None, alpha=settings.repomap_pagerank_alpha)

    data = {
        "repo": repo,
        "generated_at": datetime.now(UTC).isoformat(),
        "languages_supported": sorted({lang for lang in file_language.values()}),
        "files": files_meta,
        "edges": edges_meta,
        "rank_uniform": rank_uniform,
    }
    out_path = _repomap_path(repo, settings)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "repo": repo,
        "n_files": len(files_meta),
        "n_edges": len(edges_meta),
        "languages": data["languages_supported"],
        "excluded_vendored": excluded_vendored,
    }


def load_repo_map(repo: str, settings: Settings | None = None) -> dict | None:
    settings = settings or get_settings()
    path = _repomap_path(repo, settings)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
