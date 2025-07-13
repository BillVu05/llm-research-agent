import json
from typing import Any, Dict
import google.generativeai as genai
from config import GEMINI_KEY

genai.configure(api_key=GEMINI_KEY)
GEN_MODEL = genai.GenerativeModel("models/gemini-1.5-flash")

class GenerateQueries:
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
