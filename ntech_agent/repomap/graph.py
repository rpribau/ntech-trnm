"""Grafo de dependencias entre archivos (peso = referencias cruzadas) + PageRank."""

from __future__ import annotations

import networkx as nx


def graph_from_edges(nodes: list[str], edges: list[dict]) -> nx.DiGraph:
    """Reconstruye el grafo desde la forma persistida (usado tanto al indexar
    como al recargar el repo map en cada consulta)."""
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    for e in edges:
        g.add_edge(e["from"], e["to"], weight=e.get("weight", 1))
    return g


def rank_files(
    graph: nx.DiGraph, seeds: dict[str, float] | None, alpha: float
) -> dict[str, float]:
    """PageRank sobre el grafo de archivos. ``seeds`` boostea (personalization) los
    archivos ya traídos por RAG normal, dando prioridad a lo relevante a la consulta
    además de a lo estructuralmente importante."""
    n = graph.number_of_nodes()
    if n == 0:
        return {}
    personalization = dict.fromkeys(graph.nodes, 1.0)
    if seeds:
        for path, boost in seeds.items():
            if path in personalization:
                personalization[path] += boost
    try:
        return nx.pagerank(graph, alpha=alpha, personalization=personalization, weight="weight")
    except nx.PowerIterationFailedConvergence:
        return dict.fromkeys(graph.nodes, 1.0 / n)
