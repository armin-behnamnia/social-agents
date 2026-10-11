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
        # Nearly sparse: average degree ~4. Retry with bumped seeds
        # until the graph is connected (ER with low average degree can
        # otherwise isolate nodes, which would make them unreachable
        # persuasion targets).
        p = max(0.05, 4.0 / max(1, agent_num - 1))
        graph = None
        for bump in range(50):
            cand = nx.erdos_renyi_graph(
                n=agent_num, p=p, seed=None if seed is None else seed + bump
            )
            if nx.is_connected(cand):
                graph = cand
                break
        if graph is None:
            raise RuntimeError(
                f"could not build a connected ER graph after 50 tries "
                f"(n={agent_num}, p={p:.3f})"
            )
    else:
        raise ValueError(f"unknown topology: {topology}")

    adjacency = {node: set(graph.neighbors(node)) for node in graph.nodes()}
    degrees = dict(graph.degree())
    degree_centrality = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph)

    return adjacency, degrees, degree_centrality, betweenness
