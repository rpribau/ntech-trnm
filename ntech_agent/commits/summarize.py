"""Fase 2/3: resume con el LLM los archivos modificados de cada commit
pendiente (``summary IS NULL``).

Un LLM call POR COMMIT (no por archivo): el prompt lleva el diff de todos los
archivos tocados y pide una lista estructurada ``{file_path, resumen}`` — misma
tabla final que resumir script por script, mucho más barato.

Mismo patrón de doble try/except que ``reviewer_node``
(``ntech_agent/graph/nodes.py``) y ``_synthesize_narrativa``
(``ntech_agent/report/executive.py``): structured output → fallback a texto
libre → fallback final. A diferencia de esos, en la falla total se deja
``summary`` en NULL (se reintenta en el próximo sync) en vez de escribir un
placeholder — acá el resultado va a una tabla interna, no directo al usuario,
así que reintentar es más barato que degradar con un texto de relleno.
"""

from __future__ import annotations

from github import Auth, Github
from pydantic import BaseModel, Field

from config.settings import Settings, get_settings
from ntech_agent.commits.db import get_connection
from ntech_agent.llm import extract_text, get_chat_model, log_llm_failure
from ntech_agent.repomap.build import _is_excluded, _submodule_prefixes

_PATCH_CHARS = 3000  # truncado por archivo, mismo espíritu que ctx_chars/repomap_file_chars


class _FileSummary(BaseModel):
    file_path: str
    resumen: str = Field(description="1-2 frases: qué cambió en este archivo en este commit")


class _CommitSummary(BaseModel):
    archivos: list[_FileSummary]


def _client(settings: Settings) -> Github:
    if not settings.github_token:
        raise RuntimeError("NTECH_GITHUB_TOKEN no está configurado en .env")
    return Github(auth=Auth.Token(settings.github_token), per_page=100)


def _pending_commits(conn, repo: str, settings: Settings) -> list[str]:
    """Commits con al menos un archivo sin resumir, excluyendo los que tocan
    más de ``commits_max_files_per_commit`` archivos (merges grandes, dumps de
    vendor) — esos se guardan sin resumir para no gastar LLM en ellos."""
    rows = conn.execute(
        "SELECT DISTINCT commit_sha FROM commit_files "
        "WHERE repo = ? AND summary IS NULL "
        "AND commit_sha NOT IN ("
        "  SELECT commit_sha FROM commit_files WHERE repo = ? "
        "  GROUP BY commit_sha HAVING COUNT(*) > ?"
        ")",
        (repo, repo, settings.commits_max_files_per_commit),
    ).fetchall()
    return [r[0] for r in rows]


def _summarize_commit(
    prompt: str, filenames: list[str], settings: Settings
) -> dict[str, str]:
    try:
        llm = get_chat_model(
            temperature=0.1, max_tokens=settings.commits_summary_max_tokens, settings=settings
        )
        result: _CommitSummary = llm.with_structured_output(_CommitSummary).invoke(prompt)
        return {fs.file_path: fs.resumen for fs in result.archivos}
    except Exception as exc:
        log_llm_failure("summarize.py::_summarize_commit (structured)", exc)
        try:
            llm = get_chat_model(
                temperature=0.1, max_tokens=settings.commits_summary_max_tokens, settings=settings
            )
            resp = llm.invoke(prompt + "\n\nResponde en un párrafo breve (sin structured output).")
            text = extract_text(resp.content)
            # Sin structured output no hay mapeo confiable a archivos individuales:
            # se usa el mismo resumen para todos los archivos de este commit (mejor
            # que perder el dato).
            return dict.fromkeys(filenames, text)
        except Exception as exc2:
            # El backend puede fallar también en el fallback (p. ej. un 5xx
            # transitorio del proveedor) — se deja `summary` en NULL, se
            # reintenta en el próximo sync.
            log_llm_failure("summarize.py::_summarize_commit (fallback texto libre)", exc2)
            return {}


def summarize_new_commits(repo: str, settings: Settings | None = None, *, conn=None) -> int:
    """Resume los commits pendientes de ``repo``. Devuelve cuántos se resumieron."""
    settings = settings or get_settings()
    if not settings.commits_summarize_enabled:
        return 0
    owned_conn = conn is None
    conn = conn or get_connection(settings)
    try:
        shas = _pending_commits(conn, repo, settings)
        if not shas:
            return 0
        gh = _client(settings)
        gh_repo = gh.get_repo(f"{settings.github_org}/{repo}")
        submodule_prefixes = _submodule_prefixes(settings.repos_dir / repo)

        summarized = 0
        for sha in shas:
            local_paths = {
                r[0]
                for r in conn.execute(
                    "SELECT file_path FROM commit_files WHERE repo = ? AND commit_sha = ?",
                    (repo, sha),
                ).fetchall()
            }
            commit = gh_repo.get_commit(sha)
            filenames: list[str] = []
            patches: list[str] = []
            for f in commit.files or []:
                if f.filename not in local_paths or _is_excluded(f.filename, submodule_prefixes):
                    continue
                filenames.append(f.filename)
                patch = (f.patch or "(sin diff disponible)")[:_PATCH_CHARS]
                patches.append(f"### {f.filename}\n{patch}")
            if not patches:
                continue

            subject = commit.commit.message.splitlines()[0] if commit.commit.message else ""
            prompt = (
                "Resumí en 1-2 frases QUÉ CAMBIÓ en cada archivo de este commit, en español, "
                "basándote solo en el diff. Sé específico y breve (sin relleno).\n\n"
                f"--- Commit: {subject} ---\n\n" + "\n\n".join(patches)
            )
            resumenes = _summarize_commit(prompt, filenames, settings)
            if not resumenes:
                continue
            for file_path, resumen in resumenes.items():
                conn.execute(
                    "UPDATE commit_files SET summary = ? "
                    "WHERE repo = ? AND commit_sha = ? AND file_path = ?",
                    (resumen, repo, sha, file_path),
                )
            conn.commit()
            summarized += 1
        return summarized
    finally:
        if owned_conn:
            conn.close()
