from typing import Any, Dict, List
import argparse
import os
import concurrent.futures
import sys
import json
import google.generativeai as genai

# === Configuration ===
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MAX_ITER = 1  # single pass to stay within quota

# === Initialize once for reuse ===
GEN_MODEL = genai.GenerativeModel("models/gemini-1.5-flash") if USE_LLM else None

# === Graph Support Classes ===
class Edge:
    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target

class Node:
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses should implement this!")

class Graph:
    def __init__(self, nodes: Dict[str, Node], edges: List[Edge], entry: str, exit: str, max_iter: int = 1):
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
            # merge output into data to pass forward
            if output:
                data.update(output)
            next_nodes = [e.target for e in self.edges if e.source == current_node]
            if not next_nodes:
                break
            current_node = next_nodes[0]
            iter_count += 1
        # run exit node
        output = self.nodes[self.exit].run(data)
        if output:
            data.update(output)
        return data

# === Node Implementations ===
class GenerateQueries(Node):
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        topic = input_data['topic']
        if not USE_LLM:
            return {"queries": [f"{topic} query 1", f"{topic} query 2", f"{topic} query 3"]}
        prompt = (
            f"Break the following research question into 3-5 distinct English search queries. "
            f"Return only a JSON array of the queries.\n\nQuestion: {topic}\nQueries: "
        )
        response = GEN_MODEL.generate_content(prompt)
        content = response.text
        if not content:
            raise ValueError("LLM response content is None. Cannot parse queries.")
        
        # Strip markdown code block if present
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        
        try:
            queries = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from LLM response: {e}")
            print(f"Response content:\n{content}")
            raise
        return {"queries": queries}

class WebSearchTool(Node):
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        queries = input_data["queries"]
        def mock_bing_search(query):
            return [
                {"title": f"{query} - Result 1", "url": f"https://example.com/{query.replace(' ', '_')}_1"},
                {"title": f"{query} - Result 2", "url": f"https://example.com/{query.replace(' ', '_')}_2"}
            ]
        docs = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(mock_bing_search, queries))
        seen_urls = set()
        for result_list in results:
            for doc in result_list:
                if doc["url"] not in seen_urls:
                    docs.append(doc)
                    seen_urls.add(doc["url"])
        return {"docs": docs, "queries": queries}

class Reflect(Node):
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        docs = input_data.get("docs", [])
        queries = input_data.get("queries", [])
        need_more = len(docs) < 6
        new_queries = [q + " (refined)" for q in queries[:3]] if need_more else []
        # Always include docs and queries in output so downstream nodes get them
        return {"need_more": need_more, "new_queries": new_queries, "docs": docs, "queries": queries}

class Synthesize(Node):
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        docs = input_data.get("docs", [])
        if not USE_LLM:
            return {
                "answer": "This is a mock answer for development (no LLM used).",
                "citations": [doc["url"] for doc in docs]
            }
        context = "\n".join(f"[{i+1}] {doc['title']} {doc['url']}" for i, doc in enumerate(docs))
        prompt = (
            "Using the following sources, write a concise English answer to the research question (max 80 words, ~400 characters). "
            "End the answer with Markdown references [1][2]….\n\n"
            f"Sources:\n{context}\n\nAnswer:"
        )
        response = GEN_MODEL.generate_content(prompt)
        answer = response.text.strip() if response.text else ""
        citations = [doc["url"] for doc in docs]
        return {"answer": answer, "citations": citations}

# === Pipeline Construction ===
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
        Edge("Reflect", "WebSearchTool"),
        Edge("Reflect", "Synthesize"),
    ]
    return Graph(nodes, edges, entry="GenerateQueries", exit="Synthesize", max_iter=MAX_ITER)

# === CLI Entry Point ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, required=False)
    parser.add_argument("question", nargs="?", help="Research question (positional, for Docker)")
    args = parser.parse_args()
    topic = args.topic or args.question
    if not topic:
        parser.error("A research topic/question must be provided.")
    pipeline = build_pipeline()
    result = pipeline.run({"topic": topic})
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
