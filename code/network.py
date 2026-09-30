"""Network construction and centrality computation."""

from __future__ import annotations

import numpy as np


def build_network(agent_num: int, topology: str = "watts_strogatz", seed: int | None = 42):
    """Return (adjacency sets, degree, degree_centrality, betweenness)."""
    import networkx as nx

    if topology == "watts_strogatz":
        k = 4 if agent_num >= 4 else agent_num - 1
        graph = nx.watts_strogatz_graph(n=agent_num, k=k, p=0.1, seed=seed)
    elif topology == "barabasi_albert":
        m = 2
        graph = nx.barabasi_albert_graph(n=agent_num, m=m, seed=seed)
    elif topology == "complete":
        graph = nx.complete_graph(n=agent_num)
    elif topology == "erdos_renyi":
        p = max(0.05, 4.0 / max(1, agent_num - 1))
        graph = nx.erdos_renyi_graph(n=agent_num, p=p, seed=seed)
    else:
        raise ValueError(f"unknown topology: {topology}")

    adjacency = {node: set(graph.neighbors(node)) for node in graph.nodes()}
    degrees = dict(graph.degree())
    degree_centrality = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph)

    return adjacency, degrees, degree_centrality, betweenness
