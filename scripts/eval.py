"""CLI: evaluación de retrieval (recall@k y MRR) sobre un set gold.

El set gold es un JSONL en ``tests/gold.jsonl`` (si no existe, usa
``tests/gold.example.jsonl``). Cada línea:
    {"question": "...", "repo": "opcional", "expected_paths": ["ruta/parcial", ...]}

Uso:
    python -m scripts.eval [--k 8]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from config.settings import get_settings
from ntech_agent.retrieval.hybrid import retrieve

console = Console()


def _load_gold() -> tuple[list[dict], Path]:
    root = get_settings().skills_dir.parent  # raíz del repo
    for name in ("tests/gold.jsonl", "tests/gold.example.jsonl"):
        path = root / name
        if path.exists():
            items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return items, path
    return [], root / "tests/gold.jsonl"


def _matches(doc_path: str, expected: list[str]) -> bool:
    return any(e.lower() in doc_path.lower() for e in expected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación de retrieval.")
    parser.add_argument("--k", type=int, default=None, help="Top-k a evaluar.")
    args = parser.parse_args()

    settings = get_settings()
    k = args.k or settings.retrieval_k
    gold, path = _load_gold()
    if not gold:
        console.print("[yellow]No hay set gold. Crea tests/gold.jsonl.[/yellow]")
        return

    console.print(f"[bold]Evaluando {len(gold)} consultas (k={k}) desde {path.name}[/bold]")
    table = Table(title="Resultados por consulta")
    table.add_column("Consulta", overflow="fold")
    table.add_column("recall@k", justify="right")
    table.add_column("RR", justify="right")

    recalls, rrs = [], []
    for item in gold:
        docs = retrieve(item["question"], repo=item.get("repo"), k=k, settings=settings)[:k]
        expected = item.get("expected_paths", [])
        paths = [f"{d.metadata.get('repo')}/{d.metadata.get('path')}" for d in docs]

        found = sum(1 for e in expected if any(_matches(p, [e]) for p in paths))
        recall = found / len(expected) if expected else 0.0

        rr = 0.0
        for rank, p in enumerate(paths, start=1):
            if _matches(p, expected):
                rr = 1.0 / rank
                break

        recalls.append(recall)
        rrs.append(rr)
        table.add_row(item["question"][:60], f"{recall:.2f}", f"{rr:.2f}")

    console.print(table)
    mean_recall = sum(recalls) / len(recalls)
    mrr = sum(rrs) / len(rrs)
    console.print(f"\n[bold]recall@{k} medio:[/bold] {mean_recall:.3f}   [bold]MRR:[/bold] {mrr:.3f}")


if __name__ == "__main__":
    main()
