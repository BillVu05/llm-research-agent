from typing import Any, Dict, List
import argparse
import os
import concurrent.futures
import google.generativeai as genai
print(dir(genai))

import sys
print(sys.executable)

# Set the Gemini API key as an environment variable for google.generativeai
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
print("GEMINI_API_KEY is", os.getenv("GEMINI_API_KEY"))

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-1.5-flash")
response = model.generate_content("Say hello!")
print(response.text)

# Define Edge class
class Edge:
    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target

# Define a base Node class
class Node:
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses should implement this!")

# Minimal Graph class implementation for the pipeline
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
            data = node.run(data)
            # Find next node(s)
            next_nodes = [e.target for e in self.edges if e.source == current_node]
            if not next_nodes:
                break
            # For simplicity, pick the first next node (could be improved)
            current_node = next_nodes[0]
            iter_count += 1
        # Run the exit node
        data = self.nodes[self.exit].run(data)
        return data

MAX_ITER = 2

# Node: GenerateQueries (Gemini)
class GenerateQueries(Node):
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        topic = input_data['topic']
        prompt = (
            f"Break the following research question into 3-5 distinct English search queries. "
            f"Return only a JSON array of the queries.\n\nQuestion: {topic}\nQueries: "
        )
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
        response = model.generate_content(prompt)
        import json
        content = response.text
        if not content:
            raise ValueError("LLM response content is None. Cannot parse queries.")
        queries = json.loads(content)
        return {"queries": queries}

# Node: WebSearchTool
class WebSearchTool(Node):
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        queries = input_data["queries"]
        # Mock Bing Web Search API call
        def mock_bing_search(query):
            # Simulate a search result as a dict with 'title' and 'url'
            return [
                {"title": f"{query} - Result 1", "url": f"https://example.com/{query.replace(' ', '_')}_1"},
                {"title": f"{query} - Result 2", "url": f"https://example.com/{query.replace(' ', '_')}_2"}
            ]
        # Run searches concurrently
        docs = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(mock_bing_search, queries))
        # Flatten and de-duplicate by URL
        seen_urls = set()
        for result_list in results:
            for doc in result_list:
                if doc["url"] not in seen_urls:
                    docs.append(doc)
                    seen_urls.add(doc["url"])
        return {"docs": docs}

# Node: Reflect
class Reflect(Node):
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Simple heuristic: if less than 6 docs, need more
        docs = input_data.get("docs", [])
        queries = input_data.get("queries", [])
        need_more = len(docs) < 6
        new_queries = []
        if need_more:
            # Generate up to 3 refined queries
            new_queries = [q + " (refined)" for q in queries[:3]]
        return {"need_more": need_more, "new_queries": new_queries, "docs": docs}

# Node: Synthesize (Gemini)
class Synthesize(Node):
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        docs = input_data["docs"]
        context = "\n".join(f"[{i+1}] {doc['title']} {doc['url']}" for i, doc in enumerate(docs))
        prompt = (
            "Using the following sources, write a concise English answer to the research question (max 80 words, ~400 characters). "
            "End the answer with Markdown references [1][2]….\n\n"
            f"Sources:\n{context}\n\nAnswer:"
        )
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
        response = model.generate_content(prompt)
        answer = response.text.strip() if response.text else ""
        citations = [doc["url"] for doc in docs]
        return {"answer": answer, "citations": citations}

# Build pipeline
def build_pipeline() -> Graph:
    nodes = {
        "GenerateQueries": GenerateQueries(),
        "WebSearchTool": WebSearchTool(),
        "Reflect": Reflect(),
        "Synthesize": Synthesize(),
    }
    edges = [
        Edge("GenerateQueries", "WebSearchTool"),
        Edge("WebSearchTool", "Reflect"),
        Edge("Reflect", "WebSearchTool"),  # For cycles
        Edge("Reflect", "Synthesize"),
    ]
    pipeline = Graph(nodes, edges, entry="GenerateQueries", exit="Synthesize", max_iter=MAX_ITER)
    return pipeline

# Example CLI entrypoint
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, required=False)
    parser.add_argument("question", nargs="?", help="Research question (positional, for Docker)")
    args = parser.parse_args()

    # Prefer positional argument for Docker Compose
    topic = args.topic or args.question
    if not topic:
        parser.error("A research topic/question must be provided.")

    pipeline = build_pipeline()
    result = pipeline.run({"topic": topic})
    import json
    # Print only the final JSON (from Synthesize)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()