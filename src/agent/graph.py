"""Wiring only. What each node does lives in nodes.py, ingest.py, retrieval.py.

Nodes are referenced as ``module.name`` at build time, so a test that patches
``agent.nodes.reflect`` before calling build_pipeline() actually gets its patch
into the graph.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from . import ingest, nodes, retrieval
from .state import ResearchState


def build_pipeline():
    """Compile the research graph.

    plan -> search -> ingest -> retrieve -> reflect, and the conditional edge
    out of reflect routes back to search on a coverage gap, bounded by
    MAX_SEARCH_ROUNDS.
    """
    graph = StateGraph(ResearchState)
    graph.add_node("generate_queries", nodes.generate_queries)
    graph.add_node("web_search", nodes.web_search)
    graph.add_node("ingest", ingest.ingest)
    graph.add_node("retrieve", retrieval.retrieve)
    graph.add_node("reflect", nodes.reflect)
    graph.add_node("synthesize", nodes.synthesize)

    graph.add_edge(START, "generate_queries")
    graph.add_edge("generate_queries", "web_search")
    graph.add_edge("web_search", "ingest")
    graph.add_edge("ingest", "retrieve")
    graph.add_edge("retrieve", "reflect")
    graph.add_conditional_edges(
        "reflect",
        nodes.route_after_reflect,
        {"web_search": "web_search", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)
    return graph.compile()
