# tests/test_pipeline.py
import pytest
from src.agent.pipeline import build_pipeline

def test_happy_path(monkeypatch):
    # Patch Gemini and SerpAPI to return dummy data
    def mock_run(self, input_data):
        return {
            "answer": "Mock answer.",
            "citations": [{"id": 1, "title": "Mock Title", "url": "https://example.com"}]
        }

    pipeline = build_pipeline()
    pipeline.nodes["Synthesize"].run = mock_run.__get__(pipeline.nodes["Synthesize"])

    result = pipeline.run({"topic": "Who won the 2022 World Cup?"})
    assert "answer" in result
    assert result["answer"] == "Mock answer."
    assert result["citations"][0]["id"] == 1
