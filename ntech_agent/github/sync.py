"""Clona / actualiza los repos de la org en ``data/repos/`` para indexarlos.

Usa clones *shallow* (``--depth 1``) para ahorrar espacio y evita persistir el
token en el ``.git/config`` (lo inyecta solo durante la operación y luego deja el
remote sin credenciales).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from config.settings import Settings, get_settings
from ntech_agent.github.client import RepoInfo, list_org_repos


@dataclass
class SyncResult:
    repo: str
    action: str  # "cloned" | "updated" | "skipped" | "error"
    detail: str = ""


def _auth_url(clone_url: str, token: str) -> str:
    """Inserta el token en la URL https solo para la operación de red."""
    return clone_url.replace("https://", f"https://x-access-token:{token}@", 1)


def _run_git(args: list[str], token: str | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        # Nunca dejar el token en el mensaje de error.
        stderr = proc.stderr
        if token:
            stderr = stderr.replace(token, "***")
        raise RuntimeError(stderr.strip() or "git falló")
    return proc


def _clone(repo: RepoInfo, dest: Path, token: str) -> None:
    _run_git(
        ["git", "clone", "--depth", "1", "--branch", repo.default_branch,
         _auth_url(repo.clone_url, token), str(dest)],
        token=token,
    )
    # Quitar el token del remote persistido.
    _run_git(["git", "-C", str(dest), "remote", "set-url", "origin", repo.clone_url])


def _update(repo: RepoInfo, dest: Path, token: str) -> None:
    auth = _auth_url(repo.clone_url, token)
    try:
        _run_git(["git", "-C", str(dest), "remote", "set-url", "origin", auth], token=token)
        _run_git(["git", "-C", str(dest), "fetch", "--depth", "1", "origin",
                  repo.default_branch], token=token)
        _run_git(["git", "-C", str(dest), "reset", "--hard",
                  f"origin/{repo.default_branch}"], token=token)
    finally:
        _run_git(["git", "-C", str(dest), "remote", "set-url", "origin", repo.clone_url])


def sync_all_repos(
    settings: Settings | None = None, *, include_archived: bool = False
) -> list[SyncResult]:
    """Clona o actualiza todos los repos de la org. Devuelve un resumen por repo."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    token = settings.github_token
    results: list[SyncResult] = []

    for repo in list_org_repos(settings, include_archived=include_archived):
        dest = settings.repos_dir / repo.name
        try:
            if (dest / ".git").exists():
                _update(repo, dest, token)
                results.append(SyncResult(repo.name, "updated"))
            else:
                _clone(repo, dest, token)
                results.append(SyncResult(repo.name, "cloned"))
        except Exception as exc:
            results.append(SyncResult(repo.name, "error", str(exc)))
    return results


if __name__ == "__main__":  # python -m ntech_agent.github.sync
    for res in sync_all_repos():
        print(f"[{res.action:8s}] {res.repo} {res.detail}")
