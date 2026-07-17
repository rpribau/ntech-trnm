"""Nodos del grafo: retrieve, static, reviewer (Skill) y synthesize."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from config.settings import Settings, get_settings
from ntech_agent.analysis.static import format_for_prompt, run_static_analysis
from ntech_agent.commits.query import org_insights
from ntech_agent.graph.state import AgentState
from ntech_agent.llm import extract_text, get_chat_model, log_llm_failure
from ntech_agent.repomap.build import load_repo_map
from ntech_agent.repomap.expansion import select_expansion_files
from ntech_agent.repomap.graph import graph_from_edges, rank_files
from ntech_agent.repomap.skeleton import render_repo_skeleton
from ntech_agent.repomap.tags import is_notebook, notebook_code_to_text
from ntech_agent.retrieval.hybrid import retrieve_guidelines, search


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_skill(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    skill_path = settings.skills_dir / "code_best_practices" / "SKILL.md"
    rubric_path = settings.skills_dir / "code_best_practices" / "rubric.yaml"
    parts = []
    if skill_path.exists():
        parts.append(skill_path.read_text(encoding="utf-8"))
    if rubric_path.exists():
        parts.append("\n\n# rubric.yaml\n" + rubric_path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _format_context(docs: list[Document], settings: Settings) -> str:
    # Los docs "repomap_file" (top-N por PageRank o expansión dirigida) son una
    # decisión explícita del pipeline, no un candidato más de RAG — se incluyen
    # SIEMPRE completos, y el resto del cupo ctx_max_docs se llena con chunks
    # normales. Más robusto que depender del orden de inserción para que
    # sobrevivan el corte (un simple prepend es frágil si algo más antepone otra
    # cosa antes de llegar acá).
    priority = [d for d in docs if d.metadata.get("source_type") == "repomap_file"]
    rest = [d for d in docs if d.metadata.get("source_type") != "repomap_file"]
    budget = max(settings.ctx_max_docs - len(priority), 0)
    selected = priority + rest[:budget]

    out = []
    for d in selected:
        bc = d.metadata.get("breadcrumb", d.metadata.get("path", "?"))
        # Los documentos de archivo completo (repo map) usan un presupuesto más
        # grande que los chunks normales — truncarlos a ctx_chars (900) anularía
        # el propósito de darle al reviewer el archivo entero.
        limit = (
            settings.repomap_file_chars
            if d.metadata.get("source_type") == "repomap_file"
            else settings.ctx_chars
        )
        out.append(f"### {bc}\n{d.page_content[:limit]}")
    return "\n\n".join(out) if out else "(sin contexto recuperado)"


def _citations(docs: list[Document], settings: Settings) -> str:
    # Deduplica por archivo (no por línea): varios chunks del mismo archivo en
    # líneas distintas antes generaban listas largas casi-duplicadas.
    seen: dict[str, None] = {}
    for d in docs:
        m = d.metadata
        key = f"- `{m.get('repo')}/{m.get('path')}`"
        seen.setdefault(key, None)
    all_lines = list(seen.keys())
    if not all_lines:
        return "_(sin fuentes)_"
    shown = all_lines[: settings.citations_max]
    remaining = len(all_lines) - len(shown)
    if remaining > 0:
        shown.append(f"_(+{remaining} fuentes más)_")
    return "\n".join(shown)


# --------------------------------------------------------------------------- #
# Nodos
# --------------------------------------------------------------------------- #
def retrieve_node(state: AgentState, config=None) -> dict:
    settings = get_settings()
    q = state.get("question", "")
    if state.get("route") == "org_report":
        docs = search(q or "panorama general de la organización", repo=None, settings=settings)
    else:
        # "repos" es la fuente de verdad (0, 1 o varios); si el caller solo puso
        # "repo" (p. ej. report/generate.py, que arma el state a mano sin "repos"),
        # se cae hacia atrás a ese único valor.
        repos = state.get("repos") or ([state["repo"]] if state.get("repo") else [])
        docs = search(q, repo=repos or None, settings=settings)
    return {"retrieved": docs}


def _read_repo_file(path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    # Un .ipynb crudo es JSON (celdas, metadata, outputs) — inservible para el
    # reviewer. Mostrarle en cambio solo el código de las celdas.
    return notebook_code_to_text(text) if is_notebook(path) else text


_REPOMAP_ROUTES = {"review", "summary", "assist"}


def _read_full_files(repo: str, paths: list[str], settings: Settings) -> list[Document]:
    """Lee COMPLETOS los archivos en ``paths`` (código o, para ``.ipynb``, solo
    las celdas de código) — mismo shape de Document/metadata sin importar si
    ``paths`` vino del top-N por PageRank o de la expansión dirigida, así
    ``_citations()``/``_used_repomap()`` no necesitan distinguir el origen."""
    repo_dir = settings.repos_dir / repo
    docs: list[Document] = []
    for path in paths:
        full_path = repo_dir / path
        text = _read_repo_file(full_path)
        if not text:
            continue
        kind = "notebook, solo celdas de código" if is_notebook(full_path) else "archivo completo"
        breadcrumb = f"{repo} > {path} ({kind}, repo map)"
        docs.append(
            Document(
                page_content=f"[{breadcrumb} :1]\n{text[: settings.repomap_file_chars]}",
                metadata={
                    "repo": repo,
                    "path": path,
                    "start_line": 1,
                    "source_type": "repomap_file",
                    "breadcrumb": breadcrumb,
                },
            )
        )
    return docs


def repomap_node(state: AgentState, config=None) -> dict:
    """Repo map: en review/summary/assist arma un índice de solo firmas
    (``repomap_skeleton``, ver ``repomap/skeleton.py``) con cobertura del 100%
    de los archivos con símbolos del repo — no depende de ranking, así que NO
    usa ``repomap_min_files_for_rank`` (ese umbral protege la calidad del
    PageRank de abajo, no aplica a un índice que no rankea nada).

    Solo en route=review, además reemplaza los chunks de ``retrieved`` por
    archivos COMPLETOS elegidos por PageRank sobre el grafo de dependencias del
    repo con boost a lo que RAG ya trajo como semilla (comportamiento sin
    cambios, sigue exclusivo de review por costo).

    En las tres rutas, si hay skeleton, se intenta una expansión dirigida (una
    sola llamada LLM acotada, ``repomap/expansion.py``) que puede rescatar del
    índice algún archivo que ni ganó el top-N ni fue traído por RAG — se
    antepone a ``retrieved`` como archivo completo.

    Se autosaltea (deja ``retrieved`` intacto) si la ruta no aplica, no hay
    repo, o no hay repo map disponible/habilitado para el repo."""
    settings = get_settings()
    repo = state.get("repo")
    route = state.get("route")
    if not repo or not settings.repomap_enabled or route not in _REPOMAP_ROUTES:
        return {}

    data = load_repo_map(repo, settings)
    if data is None:
        return {}

    result: dict = {}

    if settings.repomap_skeleton_enabled:
        skeleton = render_repo_skeleton(data, settings)
        if skeleton:
            result["repomap_skeleton"] = skeleton

    top_paths: list[str] = []
    if route == "review" and len(data.get("files", [])) >= settings.repomap_min_files_for_rank:
        seeds = {
            d.metadata["path"]: settings.repomap_seed_boost
            for d in state.get("retrieved", [])
            if d.metadata.get("repo") == repo and d.metadata.get("path")
        }
        graph = graph_from_edges([f["path"] for f in data["files"]], data["edges"])
        scores = rank_files(graph, seeds=seeds, alpha=settings.repomap_pagerank_alpha)
        top_paths = sorted(scores, key=lambda p: -scores[p])[: settings.repomap_top_files]
        full_docs = _read_full_files(repo, top_paths, settings)
        if full_docs:
            result["retrieved"] = full_docs

    if settings.repomap_expansion_enabled and result.get("repomap_skeleton"):
        base_retrieved = result.get("retrieved", state.get("retrieved", []))
        if route == "review":
            already_included = set(top_paths)
        else:
            already_included = {
                d.metadata.get("path")
                for d in base_retrieved
                if d.metadata.get("repo") == repo and d.metadata.get("path")
            }
        chosen = select_expansion_files(
            question=state.get("question", ""),
            skeleton=result["repomap_skeleton"],
            already_included=already_included,
            valid_paths={f["path"] for f in data["files"]},
            settings=settings,
        )
        if chosen:
            expansion_docs = _read_full_files(repo, chosen, settings)
            if expansion_docs:
                result["retrieved"] = expansion_docs + base_retrieved

    return result


def static_node(state: AgentState, config=None) -> dict:
    settings = get_settings()
    repo = state.get("repo")
    if state.get("route") in ("review", "summary") and repo:
        findings = run_static_analysis(settings.repos_dir / repo, settings)
    else:
        findings = {}
    return {"static_findings": findings}


def commit_insights_node(state: AgentState, config=None) -> dict:
    """Si route == "commits", carga métricas de actividad de commits (último
    commit, contribuidores, actividad semanal) por cada repo mencionado desde
    ``ntech_agent/commits/query.py`` (SQLite, no RAG). Se autosaltea (mismo
    patrón que repomap/static/reviewer) si la ruta no es "commits"."""
    if state.get("route") != "commits":
        return {}
    settings = get_settings()
    repos = state.get("repos") or ([state["repo"]] if state.get("repo") else [])
    if not repos:
        return {"commit_facts": {}}
    return {"commit_facts": org_insights(repos, settings)}


class _Finding(BaseModel):
    ubicacion: str = Field(description="repo/ruta.py:línea o 'repo-wide'")
    categoria: str
    severidad: str = Field(description="critico | mayor | menor | sugerencia")
    hallazgo: str
    por_que: str
    guia: str = Field(description="archivo de guideline citado")
    recomendacion: str


class _Review(BaseModel):
    resumen_contextualizado: str
    fortalezas: list[str]
    areas_de_oportunidad: list[str]
    hallazgos: list[_Finding]


_SEVERITY_ORDER = {"critico": 0, "mayor": 1, "menor": 2, "sugerencia": 3}


def _render_review(r: _Review, settings: Settings) -> str:
    # Render compacto para el chat: ubicación + severidad + hallazgo + recomendación
    # por línea (sin por_que/guia, que siguen íntegros en review_data para el
    # reporte ejecutivo), topado a review_inline_findings_max y priorizado por
    # severidad para no perder lo más importante al truncar.
    lines = [f"**Resumen:** {r.resumen_contextualizado}", "", "**Fortalezas:**"]
    lines += [f"- {s}" for s in r.fortalezas] or ["- (ninguna destacada)"]
    lines += ["", "**Áreas de oportunidad:**"]
    lines += [f"- {a}" for a in r.areas_de_oportunidad] or ["- (ninguna)"]
    lines += ["", "**Hallazgos:**"]
    if not r.hallazgos:
        lines.append("- (sin hallazgos)")
    else:
        ordered = sorted(r.hallazgos, key=lambda h: _SEVERITY_ORDER.get(h.severidad, 99))
        shown = ordered[: settings.review_inline_findings_max]
        for h in shown:
            lines.append(f"- [{h.severidad}] `{h.ubicacion}` — {h.hallazgo} → {h.recomendacion}")
        remaining = len(ordered) - len(shown)
        if remaining > 0:
            lines.append(f"_(+{remaining} hallazgos más — ver reporte completo con `--repo`)_")
    return "\n".join(lines)


def reviewer_node(state: AgentState, config=None) -> dict:
    if state.get("route") != "review":
        return {}
    settings = get_settings()
    repo = state.get("repo") or "(sin repo)"
    guidelines = retrieve_guidelines(
        f"buenas prácticas, design patterns y code smells para {repo}",
        k=settings.reviewer_guidelines_k,
        settings=settings,
    )
    static_txt = format_for_prompt(state.get("static_findings", {}))
    code_ctx = _format_context(state.get("retrieved", []), settings)
    guide_ctx = _format_context(guidelines, settings)
    skeleton = state.get("repomap_skeleton") or ""
    skeleton_block = (
        f"--- Índice del repo (firmas, sin cuerpo — referencia estructural) ---\n{skeleton}\n\n"
        if skeleton
        else ""
    )

    prompt = (
        f"{load_skill(settings)}\n\n"
        "=== CONTEXTO PARA LA REVISIÓN ===\n"
        f"Repo objetivo: {repo}\n\n"
        f"{skeleton_block}"
        f"--- Código recuperado ---\n{code_ctx}\n\n"
        f"--- Guidelines relevantes ---\n{guide_ctx}\n\n"
        f"--- Análisis estático (evidencia objetiva) ---\n{static_txt}\n\n"
        "Aplica la Skill y produce la revisión estructurada. Cita el código "
        "(archivo:línea) y la guía en cada hallazgo. El índice del repo es solo "
        "referencia estructural (nombres/ubicación de símbolos, SIN cuerpo) — no "
        "generes hallazgos ni afirmaciones de comportamiento basados solo en el "
        "índice, para eso usá el código recuperado. No inventes: usa solo el contexto."
    )

    try:
        llm = get_chat_model(
            temperature=settings.reviewer_temperature,
            max_tokens=settings.reviewer_max_tokens,
            settings=settings,
        )
        review: _Review = llm.with_structured_output(_Review).invoke(prompt)
        return {"review": _render_review(review, settings), "review_data": review.model_dump()}
    except Exception as exc:
        log_llm_failure("nodes.py::reviewer_node (structured)", exc)
        # Fallback a texto libre si el structured output falla en el modelo local.
        try:
            llm = get_chat_model(
                temperature=settings.reviewer_temperature,
                max_tokens=settings.reviewer_max_tokens,
                settings=settings,
            )
            resp = llm.invoke(prompt + "\n\nResponde en markdown estructurado.")
            return {"review": extract_text(resp.content), "review_data": {}}
        except Exception as exc2:
            # El backend puede fallar también en el fallback (p. ej. un 5xx
            # transitorio del proveedor) — degradar sin crashear el grafo/la UI.
            log_llm_failure("nodes.py::reviewer_node (fallback texto libre)", exc2)
            return {
                "review": "_(No se pudo generar la revisión: el backend del LLM "
                "no respondió. Probá de nuevo en un momento.)_",
                "review_data": {},
            }


_REVIEW_ANSWER_PROMPT = (
    "Ya se realizó una revisión de código completa; abajo está la evidencia recopilada "
    "(hallazgos técnicos con ubicación, severidad y recomendación).\n\n"
    "Pregunta del usuario: {question}\n\n"
    "Respondé ESPECÍFICAMENTE lo que se preguntó, en español:\n"
    "- Si la pregunta es puntual (sí/no, cuáles archivos, cuántos hallazgos, etc.), "
    "respondé directo y breve (pocas líneas), citando archivo:línea solo si ayuda a "
    "ubicar la respuesta.\n"
    "- Si la pregunta pide una revisión o resumen general del repo, das un resumen "
    "priorizado por severidad (no hace falta listar cada hallazgo).\n"
    "- NUNCA references un hallazgo por número ('Hallazgo #3', 'el hallazgo 7'): la "
    "lista de evidencia de abajo no está numerada y el usuario nunca la ve, así que "
    "ese número no significa nada para él. Si necesitás señalar un hallazgo "
    "concreto, citá el archivo (y línea si aplica) o describilo en una frase corta.\n"
    "No inventes nada que no esté en la evidencia.\n\n"
    "=== EVIDENCIA ===\n{evidence}"
)


def _review_evidence(review_data: dict, review_markdown: str) -> str:
    if not review_data:
        return review_markdown or "(sin evidencia disponible)"
    lines = [
        f"Resumen: {review_data.get('resumen_contextualizado', '')}",
        f"Fortalezas: {'; '.join(review_data.get('fortalezas', []))}",
        f"Áreas de oportunidad: {'; '.join(review_data.get('areas_de_oportunidad', []))}",
        "Hallazgos:",
    ]
    for h in review_data.get("hallazgos", []):
        lines.append(
            f"- [{h.get('severidad')}] `{h.get('ubicacion')}` — {h.get('categoria')}: "
            f"{h.get('hallazgo')} (riesgo: {h.get('por_que')}. Guía: {h.get('guia')}) "
            f"→ {h.get('recomendacion')}"
        )
    return "\n".join(lines)


def _answer_review_question(
    question: str, review_data: dict, review_markdown: str, settings: Settings
) -> str:
    """Responde lo que el usuario preguntó (puntual o general) usando la evidencia
    ya recopilada por reviewer_node — en vez de siempre volcar la revisión completa
    sin importar qué se haya preguntado (esto último generaba respuestas repetidas
    entre preguntas distintas sobre el mismo repo)."""
    evidence = _review_evidence(review_data, review_markdown)
    prompt = _REVIEW_ANSWER_PROMPT.format(
        question=question or "Revisa este repositorio.", evidence=evidence
    )
    try:
        llm = get_chat_model(
            temperature=settings.synthesize_temperature,
            max_tokens=settings.synthesize_max_tokens,
            settings=settings,
        )
        resp = llm.invoke(prompt)
        return extract_text(resp.content)
    except Exception as exc:
        log_llm_failure("nodes.py::_answer_review_question", exc)
        return review_markdown or "_(no se pudo generar una respuesta)_"


_MULTI_REPO_PROMPT = (
    "Pregunta del usuario: {question}\n\n"
    "Compará y relacioná los repositorios {repos_txt} según lo que se pregunte: qué "
    "hace cada uno, en qué se parecen y en qué se diferencian (propósito, "
    "tecnología/stack, estado, patrones de código repetidos entre ellos). Basate "
    "solo en el contexto recuperado de cada repo.\n\n"
    "- Si alguno de los repos mencionados no tiene contexto recuperado abajo, "
    "decilo explícitamente en vez de inventar o ignorarlo en silencio.\n"
    "- Esto es una comparación basada en el contexto recuperado, no una revisión "
    "formal con rúbrica de cada repo. Si la pregunta pedía revisar formalmente cada "
    "uno, sugerí correr la revisión repo por repo (p. ej. \"revisa {primer_repo}\").\n\n"
    "--- Contexto recuperado ---\n{ctx}\n\n"
    "Responde en español, en markdown, sin inventar información fuera del contexto. "
    "Sé breve y directo (máx. ~250-300 palabras)."
)


def _answer_multi_repo_question(
    question: str, repos: list[str], docs: list[Document], settings: Settings
) -> str:
    """Compara/relaciona 2+ repos mencionados explícitamente. Usa el contexto YA
    recuperado multi-repo (ver retrieval/hybrid.py::search) — no review_data, que
    queda vacío porque repomap_node/static_node/reviewer_node se autosaltean solos
    cuando no hay un único repo (state["repo"] es None con 2+ repos)."""
    ctx = _format_context(docs, settings)
    repos_txt = ", ".join(f"`{r}`" for r in repos)
    prompt = _MULTI_REPO_PROMPT.format(
        question=question or f"Compará {repos_txt}.",
        repos_txt=repos_txt,
        ctx=ctx,
        primer_repo=repos[0],
    )
    try:
        llm = get_chat_model(
            temperature=settings.synthesize_temperature,
            max_tokens=settings.synthesize_max_tokens,
            settings=settings,
        )
        resp = llm.invoke(prompt)
        return extract_text(resp.content)
    except Exception as exc:
        log_llm_failure("nodes.py::_answer_multi_repo_question", exc)
        return (
            "_(No se pudo generar una respuesta: el backend del LLM no respondió. "
            "Probá de nuevo en un momento.)_"
        )


_COMMITS_ANSWER_PROMPT = (
    "Pregunta del usuario: {question}\n\n"
    "Abajo hay datos de actividad de commits YA CALCULADOS (no los inventes ni los cambies, "
    "solo redactá la respuesta en base a ellos) — incluyen un listado commit por commit con "
    "autor, fecha, mensaje y archivos tocados. Si preguntan por un autor puntual, un commit "
    "específico (p. ej. \"el último\") o el detalle de archivos/resúmenes, buscá esa "
    "información en el listado de 'Commits recientes' en vez de decir que no está disponible. "
    "Si el usuario preguntó por un repo que no aparece en los datos, decilo explícitamente en "
    "vez de inventar.\n\n"
    "Responde en español, breve y directo, citando fechas y nombres de autor tal cual "
    "aparecen en los datos.\n\n"
    "=== DATOS DE ACTIVIDAD ===\n{evidence}"
)

_SUMMARY_SNIPPET_CHARS = 200  # truncado por archivo al listar commits, mismo espíritu que ctx_chars


def _recent_commits_block(recent_files: list[dict]) -> str:
    """Agrupa las filas por-archivo de ``repo_insights()["recent"]`` en una
    línea por COMMIT (no por archivo), con sus archivos y resúmenes — sin esto,
    la evidencia solo tenía conteos agregados por autor y el último commit
    suelto, así que el LLM no podía responder "qué commits hizo tal autor" ni
    dar detalle (archivos/resumen) de un commit puntual, aunque el dato ya
    estuviera sincronizado."""
    if not recent_files:
        return "(sin commits recientes)"
    by_commit: dict[str, dict] = {}
    order: list[str] = []
    for row in recent_files:
        sha = row["commit_sha"]
        if sha not in by_commit:
            by_commit[sha] = {
                "author": row["author"],
                "committed_at": row["committed_at"],
                "commit_message": row["commit_message"],
                "files": [],
            }
            order.append(sha)
        file_txt = row["file_path"]
        if row.get("summary"):
            file_txt += f" ({row['summary'][:_SUMMARY_SNIPPET_CHARS]})"
        by_commit[sha]["files"].append(file_txt)

    lines = []
    for sha in order:
        c = by_commit[sha]
        lines.append(
            f"- `{sha[:8]}` {c['committed_at']} por {c['author']}: \"{c['commit_message']}\" "
            f"— archivos: {'; '.join(c['files'])}"
        )
    return "\n".join(lines)


def _commit_facts_evidence(commit_facts: dict) -> str:
    if not commit_facts:
        return "(sin datos de commits sincronizados — correr `python -m scripts.sync_commits`)"
    blocks = []
    for repo, facts in commit_facts.items():
        last = facts.get("last_commit")
        last_txt = (
            f"{last['committed_at']} por {last['author']}: \"{last['commit_message']}\""
            if last
            else "(sin commits sincronizados)"
        )
        contributors = ", ".join(
            f"{c['author']} ({c['commits']})" for c in facts.get("contributors", [])[:10]
        ) or "(sin datos)"
        commits_txt = _recent_commits_block(facts.get("recent", []))
        blocks.append(
            f"### {repo}\n"
            f"Total de commits sincronizados: {facts.get('total_commits', 0)}\n"
            f"Último commit: {last_txt}\n"
            f"Contribuidores (commits): {contributors}\n"
            f"Commits recientes (más nuevo primero):\n{commits_txt}"
        )
    return "\n\n".join(blocks)


def _answer_commits_question(question: str, commit_facts: dict, settings: Settings) -> str:
    """Responde preguntas de actividad/historial de commits usando datos YA
    calculados por ``commit_insights_node`` (SQL determinístico sobre
    ``commit_files``) — el LLM solo redacta, no inventa fechas ni nombres."""
    evidence = _commit_facts_evidence(commit_facts)
    prompt = _COMMITS_ANSWER_PROMPT.format(
        question=question or "Actividad reciente.", evidence=evidence
    )
    try:
        llm = get_chat_model(
            temperature=settings.synthesize_temperature,
            max_tokens=settings.synthesize_max_tokens,
            settings=settings,
        )
        resp = llm.invoke(prompt)
        return extract_text(resp.content)
    except Exception as exc:
        log_llm_failure("nodes.py::_answer_commits_question", exc)
        return (
            "_(No se pudo generar una respuesta: el backend del LLM no respondió. "
            "Probá de nuevo en un momento.)_"
        )


def synthesize_node(state: AgentState, config=None) -> dict:
    settings = get_settings()
    route = state.get("route", "assist")
    docs = state.get("retrieved", [])
    citations = _citations(docs, settings)

    # Ruta "commits": la evidencia es SQL determinístico (commit_facts), no RAG
    # — se chequea antes que todo lo demás y no se le pega el bloque "Fuentes"
    # (que lista archivos de código, ajeno a esta respuesta).
    if route == "commits":
        answer = _answer_commits_question(
            state.get("question", ""), state.get("commit_facts") or {}, settings
        )
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    # Comparación entre 2+ repos mencionados explícitamente — chequear ANTES que
    # "review" (si no, una pregunta multi-repo clasificada como "review" caería en
    # _answer_review_question con evidencia vacía, ya que reviewer_node se
    # autosalteó al no haber un único repo).
    repos = state.get("repos") or ([state["repo"]] if state.get("repo") else [])
    if len(repos) > 1 and route != "org_report":
        answer_body = _answer_multi_repo_question(state.get("question", ""), repos, docs, settings)
        answer = f"{answer_body}\n\n## Fuentes\n{citations}"
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    if route == "review":
        answer_body = _answer_review_question(
            state.get("question", ""),
            state.get("review_data") or {},
            state.get("review") or "_(no se generó revisión)_",
            settings,
        )
        answer = (
            f"# Revisión de código — {state.get('repo') or 'repo'}\n\n{answer_body}\n\n"
            f"## Fuentes\n{citations}"
        )
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    ctx = _format_context(docs, settings)
    skeleton = state.get("repomap_skeleton") or ""
    skeleton_block = (
        f"--- Índice del repo (firmas, sin cuerpo — referencia estructural) ---\n{skeleton}\n\n"
        if skeleton
        else ""
    )
    static_txt = format_for_prompt(state.get("static_findings", {})) if state.get(
        "static_findings"
    ) else ""

    if route == "summary":
        instr = (
            f"Da un **resumen contextualizado** del repo '{state.get('repo')}': qué es, "
            "en qué se está trabajando, su estado y estructura, y 2-3 áreas de "
            "oportunidad. Basado solo en el contexto y el análisis estático."
        )
    elif route == "org_report":
        instr = (
            "Da un **panorama de la organización**: qué repos hay, temas que abordan, "
            "fortalezas transversales y áreas de oportunidad comunes. Identificá "
            "explícitamente patrones que se repitan entre repos (mismo stack "
            "tecnológico, mismos anti-patrones o brechas) y diferencias notables "
            "entre proyectos. Basado en el contexto."
        )
    else:  # assist
        instr = (
            "Responde la pregunta del usuario de forma clara y útil, apoyándote en el "
            "contexto (código y guidelines) cuando aplique."
        )

    prompt = (
        f"Pregunta del usuario: {state.get('question')}\n\n"
        f"{instr}\n\n"
        f"{skeleton_block}"
        f"--- Contexto recuperado ---\n{ctx}\n\n"
        + (f"--- Análisis estático ---\n{static_txt}\n\n" if static_txt else "")
        + "Responde en español, en markdown, sin inventar información fuera del contexto. "
        "El índice del repo (si aparece) es solo referencia estructural, no lo cites "
        "como evidencia de comportamiento. Sé breve y directo (máx. ~200-250 palabras), "
        "sin relleno ni repetición innecesaria."
    )

    try:
        llm = get_chat_model(
            temperature=settings.synthesize_temperature,
            max_tokens=settings.synthesize_max_tokens,
            settings=settings,
        )
        resp = llm.invoke(prompt)
        body = extract_text(resp.content)
    except Exception as exc:
        log_llm_failure(f"nodes.py::synthesize_node (route={route})", exc)
        body = (
            "_(No se pudo generar una respuesta: el backend del LLM no respondió. "
            "Probá de nuevo en un momento.)_"
        )
    answer = f"{body}\n\n## Fuentes\n{citations}"
    return {"answer": answer, "messages": [AIMessage(content=answer)]}
