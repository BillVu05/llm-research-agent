from agent.graph import Graph, Edge
from agent.nodes.generate_queries import GenerateQueries
from agent.nodes.web_search_tool import WebSearchTool
from agent.nodes.reflect import Reflect
from agent.nodes.synthesize import Synthesize

def build_pipeline() -> Graph:
    nodes = {
        "GenerateQueries": GenerateQueries(),
        "WebSearchTool": WebSearchTool(),
        "Reflect": Reflect(),
        "Synthesize": Synthesize()
    }
    edges = [
        Edge("GenerateQueries", "WebSearchTool"),
        Edge("WebSearchTool", "Reflect"),
        Edge("Reflect", "Synthesize"),
    ]
    return Graph(nodes, edges, entry="GenerateQueries", exit="Synthesize", max_iter=2)
