from typing import Any, Dict, List
import argparse
import os
import json
import concurrent.futures
from dotenv import load_dotenv
import google.generativeai as genai
from serpapi import GoogleSearch

# === Load environment variables ===
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# === Configure Gemini ===
genai.configure(api_key=GEMINI_KEY)
GEN_MODEL = genai.GenerativeModel("models/gemini-1.5-flash")

# === Graph Support Classes ===
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

# === Node Implementations ===
class GenerateQueries(Node):
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        topic = input_data["topic"]
        prompt = (
            f"Break the following research question into 3-5 distinct English search queries. "
            f"Return only a JSON array of strings. No markdown formatting or code blocks.\n\n"
            f"Question: {topic}\nQueries:"
        )
        response = GEN_MODEL.generate_content(prompt)
        content = response.text.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(line for line in lines if not line.strip().startswith("```")).strip()
        try:
            queries = json.loads(content)
        except Exception as e:
            print("[GenerateQueries] Failed to parse query JSON:", e)
            print("[GenerateQueries] Response content:\n", content)
            raise
        return {"queries": queries}

class WebSearchTool(Node):
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        queries = input_data.get("queries", [])
        docs = []
        seen_urls = set()

        def search(query):
            params = {
                "engine": "google",
                "q": query,
                "api_key": SERPAPI_KEY,
                "num": "10",
                "safe": "active",
                "hl": "en",
                "gl": "us"
            }
            try:
                search = GoogleSearch(params)
                results = search.get_dict()
                return results.get("organic_results", [])
            except Exception as e:
                print(f"[WebSearchTool] Error searching query '{query}': {e}")
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            all_results = executor.map(search, queries)

        total = 0
        for result_list in all_results:
            for item in result_list:
                url = item.get("link")
                title = item.get("title")
                if url and url not in seen_urls:
                    docs.append({"title": title, "url": url})
                    seen_urls.add(url)
                    total += 1

        return {"docs": docs}

class Reflect(Node):
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        docs = input_data.get("docs", [])
        queries = input_data.get("queries", [])
        need_more = len(docs) < 6
        refined = [q + " (refined)" for q in queries[:3]] if need_more else []
        return {"need_more": need_more, "new_queries": refined, "docs": docs, "queries": queries}

class Synthesize(Node):
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        topic = input_data["topic"]
        docs = input_data.get("docs", [])

        if not docs:
            print("[Synthesize] No documents provided. Skipping synthesis.")
            return {
                "answer": "No relevant documents found to answer the question.",
                "citations": []
            }


        source_text = "\n".join(f"[{i+1}] {doc['title']} {doc['url']}" for i, doc in enumerate(docs[:5]))

        prompt = (
            f"Answer this research question: '{topic}'\n"
            f"Using only the sources below, write a concise English answer not exceeding 80 words "
            f"(about 400 characters). End your answer with Markdown-style references like [1][2].\n\n"
            f"Return only the raw JSON object, no markdown formatting or code blocks.\n"
            f"Example:\n"
            f'{{\n  "answer": "...",\n  "citations": [{{"id": 1, "title": "...", "url": "..."}}] }}\n\n'
            f"Sources:\n{source_text}"
        )

        response = GEN_MODEL.generate_content(prompt)
        content = response.text.strip()

        # Strip code block fences if present
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(line for line in lines if not line.strip().startswith("```")).strip()

        try:
            parsed = json.loads(content)
            return {
                "answer": parsed.get("answer", ""),
                "citations": parsed.get("citations", [])
            }
        except json.JSONDecodeError as e:
            print("[Synthesize] Failed to parse JSON from LLM:", e)
            return {
                "answer": "Could not parse LLM response.",
                "citations": []
            }

# === Build Pipeline ===
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

# === CLI Entry Point ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, required=False)
    parser.add_argument("question", nargs="?", help="Research question (positional for Docker)")
    args = parser.parse_args()

    topic = args.topic or args.question
    if not topic:
        parser.error("Please provide a research topic/question.")

    pipeline = build_pipeline()
    result = pipeline.run({"topic": topic})

    # Output just the clean answer + citations
    clean_output = {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", [])
    }
    print(json.dumps(clean_output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
