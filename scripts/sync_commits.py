"""CLI: sincroniza metadata de commits (+ resúmenes LLM) para insights/dashboards.

Uso:
    python -m scripts.sync_commits [--repo NOMBRE] [--no-summarize]
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from config.settings import get_settings
from ntech_agent.commits.extract import sync_commits_for_repo
from ntech_agent.commits.summarize import summarize_new_commits
from ntech_agent.graph.supervisor import list_local_repos

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sincroniza metadata de commits.")
    parser.add_argument("--repo", help="Solo este repo (default: todos los repos clonados).")
    parser.add_argument(
        "--no-summarize", action="store_true",
        help="Solo Fase 1 (metadata): no gasta LLM resumiendo commits.",
    )
    args = parser.parse_args()
    settings = get_settings()

    repos = [args.repo] if args.repo else list_local_repos(settings)
    if not repos:
        console.print(
            "[yellow]No hay repos clonados. Corre `python -m scripts.sync_repos`.[/yellow]"
        )
        return

    console.print("[bold]Sincronizando metadata de commits…[/bold]")
    table = Table(title="Resultado de la sincronización de commits")
    table.add_column("Repo")
    table.add_column("Commits nuevos")
    table.add_column("Archivos nuevos")
    table.add_column("Resumidos")
    table.add_column("Detalle", overflow="fold")

    for repo in repos:
        try:
            extracted = sync_commits_for_repo(repo, settings)
            summarized = (
                0 if args.no_summarize else summarize_new_commits(repo, settings)
            )
            table.add_row(
                repo, str(extracted.new_commits), str(extracted.new_files), str(summarized), ""
            )
        except Exception as exc:
            table.add_row(repo, "-", "-", "-", f"[red]{exc}[/red]")

    console.print(table)


if __name__ == "__main__":
    main()
