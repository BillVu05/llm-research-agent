from typing import Any, Dict, List

class Edge:
    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target

class Node:
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses should implement this!")

class Graph:
    def __init__(self, nodes: Dict[str, Node], edges: List[Edge], entry: str, exit: str, max_iter: int = 2):
        self.nodes = nodes
        self.edges = edges
        self.entry = entry
        self.exit = exit
        self.max_iter = max_iter

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        current_node = self.entry
        data = input_data
        iter_count = 0
        while current_node != self.exit and iter_count < self.max_iter:
            node = self.nodes[current_node]
            output = node.run(data)
            if output:
                data.update(output)
            next_nodes = [e.target for e in self.edges if e.source == current_node]
            if not next_nodes:
                break
            current_node = next_nodes[0]
            iter_count += 1
        output = self.nodes[self.exit].run(data)
        if output:
            data.update(output)
        return data
