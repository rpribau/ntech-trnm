"""CLI: genera reportes de revisión o hace una consulta puntual al agente.

Uso:
    python -m scripts.run_review --repo ws-arg [--pdf]
    python -m scripts.run_review --org [--pdf]
    python -m scripts.run_review --ask "Dame un resumen de ws-arg"
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.markdown import Markdown

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Revisión / reportes del agente.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--repo", help="Genera el reporte de revisión de un repo.")
    group.add_argument("--org", action="store_true", help="Genera el reporte de toda la org.")
    group.add_argument("--ask", help="Hace una consulta puntual y muestra la respuesta.")
    parser.add_argument("--pdf", action="store_true", help="Exporta también a PDF (extra 'pdf').")
    args = parser.parse_args()

    if args.ask:
        from ntech_agent.graph.builder import run_agent

        with console.status("Pensando…"):
            final = run_agent(args.ask)
        console.print(f"[dim]ruta={final.get('route')} repo={final.get('repo')}[/dim]\n")
        console.print(Markdown(final.get("answer", "(sin respuesta)")))
        return

    from ntech_agent.report.generate import generate_org_report, generate_repo_report

    if args.org:
        with console.status("Generando reporte de la organización…"):
            path = generate_org_report(pdf=args.pdf)
    else:
        with console.status(f"Generando reporte de {args.repo}…"):
            path = generate_repo_report(args.repo, pdf=args.pdf)

    console.print(f"[green]Reporte escrito en:[/green] {path}")


if __name__ == "__main__":
    main()
