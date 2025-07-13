import json
from typing import Any, Dict
import google.generativeai as genai
from config import GEMINI_KEY

genai.configure(api_key=GEMINI_KEY)
GEN_MODEL = genai.GenerativeModel("models/gemini-1.5-flash")

class Synthesize:
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
