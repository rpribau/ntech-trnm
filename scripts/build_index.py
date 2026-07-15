"""CLI: indexa código + guidelines en Chroma.

Uso:
    python -m scripts.build_index [--no-reset]
"""

from __future__ import annotations

import argparse

from rich.console import Console

from ntech_agent.ingest.index import build_index

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye el índice vectorial.")
    parser.add_argument(
        "--no-reset", action="store_true",
        help="No recrear la colección (por defecto se recrea para evitar duplicados).",
    )
    args = parser.parse_args()

    console.print("[bold]Indexando corpus (código + guidelines)…[/bold]")
    with console.status("Calculando embeddings e insertando en Chroma…"):
        stats = build_index(reset=not args.no_reset)

    console.print(
        f"[green]Listo.[/green] {stats['indexed']} chunks · "
        f"{stats.get('repos', 0)} repos · {stats.get('guidelines', 0)} chunks de guidelines"
    )
    if stats.get("note"):
        console.print(f"[yellow]{stats['note']}[/yellow]")


if __name__ == "__main__":
    main()
