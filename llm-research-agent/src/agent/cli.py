from typing import Any, Dict, List
import argparse
import os
import concurrent.futures
import sys
import json
import google.generativeai as genai

# === Load Gemini API key from environment ===
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# === Initialize model ===
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
        topic = input_data['topic']
        prompt = (
            f"Break the following research question into 3-5 distinct English search queries. "
            f"Return only a JSON array of the queries.\n\nQuestion: {topic}\nQueries: "
        )
        response = GEN_MODEL.generate_content(prompt)
        content = response.text
        if not content:
            raise ValueError("LLM response content is None. Cannot parse queries.")

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

class Reflect(Node):
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        docs = input_data.get("docs", [])
        queries = input_data.get("queries", [])
        need_more = len(docs) < 6
        new_queries = [q + " (refined)" for q in queries[:3]] if need_more else []
        return {"need_more": need_more, "new_queries": new_queries, "docs": docs, "queries": queries}

class Synthesize(Node):
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            "You are an AI answering the research question: "
            f"'{input_data['topic']}'.\n\n"
            "Using your knowledge, write a concise (max 80 words) answer, and return structured citations "
            "in the following JSON format:\n\n"
            "Answer: <your answer here>\n"
            "Citations (JSON array):\n"
            "[\n"
            "  {\"id\": 1, \"title\": \"<title>\", \"url\": \"<url>\"},\n"
            "  {\"id\": 2, \"title\": \"<title>\", \"url\": \"<url>\"}\n"
            "]\n\n"
            "Answer the question and cite reliable web pages you know."
        )

        response = GEN_MODEL.generate_content(prompt)
        content = response.text.strip()

        try:
            answer_part, citation_part = content.split("Citations", 1)
            answer = answer_part.replace("Answer:", "").strip()
            citations_json = citation_part.replace(":", "", 1).strip()
            citations = json.loads(citations_json)
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            print("Raw response:", content)
            answer = content
            citations = []

        return {
            "answer": answer,
            "citations": citations
        }

# === Pipeline Construction ===
def build_pipeline() -> Graph:
    nodes = {
        "GenerateQueries": GenerateQueries(),
        "Reflect": Reflect(),
        "Synthesize": Synthesize(),
    }
    edges = [
        Edge("GenerateQueries", "Reflect"),
        Edge("Reflect", "Synthesize"),
    ]
    return Graph(nodes, edges, entry="GenerateQueries", exit="Synthesize", max_iter=1)

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
