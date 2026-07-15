"""CLI: clona/actualiza todos los repos de la org NTech-TRNM en data/repos/.

Uso:
    python -m scripts.sync_repos [--include-archived]
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from ntech_agent.github.sync import sync_all_repos

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sincroniza los repos de la org.")
    parser.add_argument(
        "--include-archived", action="store_true", help="Incluir repos archivados."
    )
    args = parser.parse_args()

    console.print("[bold]Sincronizando repos de la org…[/bold]")
    results = sync_all_repos(include_archived=args.include_archived)

    table = Table(title="Resultado de la sincronización")
    table.add_column("Repo")
    table.add_column("Acción")
    table.add_column("Detalle", overflow="fold")
    style = {"cloned": "green", "updated": "cyan", "skipped": "yellow", "error": "red"}
    for r in results:
        table.add_row(r.repo, f"[{style.get(r.action, '')}]{r.action}[/]", r.detail)
    console.print(table)

    errors = [r for r in results if r.action == "error"]
    console.print(
        f"\n[bold]{len(results)}[/bold] repos · "
        f"[green]{sum(1 for r in results if r.action == 'cloned')}[/] clonados · "
        f"[cyan]{sum(1 for r in results if r.action == 'updated')}[/] actualizados · "
        f"[red]{len(errors)}[/] errores"
    )


if __name__ == "__main__":
    main()
