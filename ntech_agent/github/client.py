"""Enumera los repositorios de la org de GitHub vía la API (PyGithub)."""

from __future__ import annotations

from dataclasses import dataclass, field

from github import Auth, Github

from config.settings import Settings, get_settings


@dataclass
class RepoInfo:
    name: str
    full_name: str
    clone_url: str
    default_branch: str
    private: bool
    archived: bool
    language: str | None
    description: str | None
    size_kb: int
    pushed_at: str | None
    topics: list[str] = field(default_factory=list)


def _client(settings: Settings) -> Github:
    if not settings.github_token:
        raise RuntimeError("NTECH_GITHUB_TOKEN no está configurado en .env")
    return Github(auth=Auth.Token(settings.github_token), per_page=100)


def list_org_repos(
    settings: Settings | None = None, *, include_archived: bool = False
) -> list[RepoInfo]:
    """Devuelve la metadata de todos los repos de la org."""
    settings = settings or get_settings()
    gh = _client(settings)
    org = gh.get_organization(settings.github_org)

    repos: list[RepoInfo] = []
    for r in org.get_repos(type="all"):
        if r.archived and not include_archived:
            continue
        repos.append(
            RepoInfo(
                name=r.name,
                full_name=r.full_name,
                clone_url=r.clone_url,
                default_branch=r.default_branch or "main",
                private=r.private,
                archived=r.archived,
                language=r.language,
                description=r.description,
                size_kb=r.size,
                pushed_at=r.pushed_at.isoformat() if r.pushed_at else None,
                topics=list(r.get_topics()) if r.raw_data.get("topics") else [],
            )
        )
    return sorted(repos, key=lambda x: x.name.lower())


if __name__ == "__main__":  # python -m ntech_agent.github.client
    for repo in list_org_repos():
        flag = "🔒" if repo.private else "  "
        print(f"{flag} {repo.name:30s} {repo.language or '-':12s} {repo.description or ''}")
