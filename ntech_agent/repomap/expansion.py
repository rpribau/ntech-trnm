"""Expansión dirigida: una única llamada LLM (estructurada, acotada) que puede
pedir la lectura completa de archivos que solo aparecen en el skeleton (ver
``skeleton.py``) — el archivo nunca ganó el top-N por PageRank ni fue traído por
RAG, pero el propio skeleton puede señalar que hace falta verlo completo.

A diferencia de ``reviewer_node``/``_answer_review_question`` (que necesitan un
fallback de DOS capas porque su fallback de texto libre es otra llamada LLM que
también puede fallar), acá el resultado es una mejora opcional del contexto, no
algo que el usuario vea directo: si la llamada falla, alcanza con loguearlo y
tratarlo como "sin expansión" (lista vacía) — no hay una segunda llamada que
proteger.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from config.settings import Settings, get_settings
from ntech_agent.llm import get_chat_model, log_llm_failure


class _ExpansionRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)


def _normalize_path(path: str) -> str:
    """Quita espacios, ``./`` inicial y normaliza separadores — evita rechazar
    sugerencias válidas del LLM por diferencias triviales de formato (los
    backends locales/chicos son poco confiables en output estructurado, ya
    documentado en ``CLAUDE.md``)."""
    p = path.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.strip("/")


def _prompt(question: str, skeleton: str) -> str:
    return (
        "Tenés el índice estructural (solo firmas, sin cuerpo) de un repositorio.\n\n"
        f"Pregunta del usuario: {question}\n\n"
        "--- Índice del repo ---\n"
        f"{skeleton}\n\n"
        "Si para responder bien necesitás ver el CUERPO completo de algún archivo "
        "que aparece en el índice (no solo su firma), listá sus rutas EXACTAS tal "
        "como aparecen ahí. Si el contexto que ya tenés alcanza, devolvé una lista "
        "vacía. No inventes rutas que no estén en el índice."
    )


def select_expansion_files(
    *,
    question: str,
    skeleton: str,
    already_included: set[str],
    valid_paths: set[str],
    settings: Settings | None = None,
) -> list[str]:
    """Hasta ``repomap_expansion_max_files`` rutas, validadas contra
    ``valid_paths`` y sin las de ``already_included`` (ambos normalizados). ``[]``
    sin llamar al LLM si ``repomap_expansion_enabled`` es falso, o si ``skeleton``/
    ``valid_paths`` están vacíos, o si no queda ningún candidato tras descartar
    ``already_included``. ``[]`` si la llamada falla o si todo lo pedido es
    inválido/duplicado. Nunca lanza."""
    settings = settings or get_settings()
    if not settings.repomap_expansion_enabled or not skeleton or not valid_paths:
        return []

    valid_norm = {_normalize_path(p): p for p in valid_paths}
    already_norm = {_normalize_path(p) for p in already_included}
    candidates_norm = set(valid_norm) - already_norm
    if not candidates_norm:
        return []

    try:
        llm = get_chat_model(temperature=0.0, max_tokens=512, settings=settings)
        structured = llm.with_structured_output(_ExpansionRequest)
        result: _ExpansionRequest = structured.invoke(_prompt(question, skeleton))
    except Exception as exc:
        log_llm_failure("expansion.py::select_expansion_files", exc)
        return []

    chosen: list[str] = []
    seen: set[str] = set()
    for raw_path in result.paths:
        norm = _normalize_path(raw_path)
        if norm not in candidates_norm or norm in seen:
            continue
        seen.add(norm)
        chosen.append(valid_norm[norm])
        if len(chosen) >= settings.repomap_expansion_max_files:
            break
    return chosen
