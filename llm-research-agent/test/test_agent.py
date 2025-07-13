import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock

# Ensure src/ is in the Python path so `agent` can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from agent.cli import (
    GenerateQueries, WebSearchTool, Reflect, Synthesize,
    Graph, Edge
)

@pytest.fixture
def dummy_input():
    return {"topic": "Who won the 2022 FIFA World Cup?"}

def test_happy_path(dummy_input):
    def gemini_side_effect(prompt):
        if "Break the following research question" in prompt:
            return MagicMock(text=json.dumps([
                "2022 FIFA World Cup winner",
                "Argentina World Cup final"
            ]))
        elif "Answer this research question" in prompt:
            return MagicMock(text=json.dumps({
                "answer": "Argentina won the 2022 FIFA World Cup.",
                "citations": [{"id": 1, "title": "Argentina wins", "url": "https://example.com/a"}]
            }))
        else:
            return MagicMock(text="{}")

    with patch("agent.cli.GEN_MODEL.generate_content", side_effect=gemini_side_effect), \
         patch("agent.cli.GoogleSearch") as mock_serp:

        mock_search_instance = MagicMock()
        mock_search_instance.get_dict.return_value = {
            "organic_results": [
                {"title": "Argentina wins", "link": "https://example.com/a"},
                {"title": "World Cup final", "link": "https://example.com/b"},
            ]
        }
        mock_serp.return_value = mock_search_instance

        graph = Graph(
            nodes={
                "GenerateQueries": GenerateQueries(),
                "WebSearchTool": WebSearchTool(),
                "Reflect": Reflect(),
                "Synthesize": Synthesize()
            },
            edges=[
                Edge("GenerateQueries", "WebSearchTool"),
                Edge("WebSearchTool", "Reflect"),
                Edge("Reflect", "Synthesize"),
            ],
            entry="GenerateQueries",
            exit="Synthesize",
            max_iter=2
        )

        result = graph.run(dummy_input)

        assert "Argentina" in result["answer"]
        assert isinstance(result["citations"], list)
        assert result["citations"][0]["id"] == 1

def test_no_results(dummy_input):
    with patch("agent.cli.GEN_MODEL.generate_content", return_value=MagicMock(text=json.dumps(["some irrelevant query"]))), \
         patch("agent.cli.GoogleSearch") as mock_serp:

        mock_search_instance = MagicMock()
        mock_search_instance.get_dict.return_value = {"organic_results": []}
        mock_serp.return_value = mock_search_instance

        # Also patch synthesis step
        with patch("agent.cli.GEN_MODEL.generate_content", return_value=MagicMock(text=json.dumps({
            "answer": "No relevant documents found to answer the question.",
            "citations": []
        }))):
            graph = Graph(
                nodes={
                    "GenerateQueries": GenerateQueries(),
                    "WebSearchTool": WebSearchTool(),
                    "Reflect": Reflect(),
                    "Synthesize": Synthesize()
                },
                edges=[
                    Edge("GenerateQueries", "WebSearchTool"),
                    Edge("WebSearchTool", "Reflect"),
                    Edge("Reflect", "Synthesize"),
                ],
                entry="GenerateQueries",
                exit="Synthesize",
                max_iter=2
            )

            result = graph.run(dummy_input)
            assert result["answer"].startswith("No relevant")
            assert result["citations"] == []

def test_http_429(dummy_input):
    with patch("agent.cli.GEN_MODEL.generate_content", return_value=MagicMock(text=json.dumps(["rate limited search"]))), \
         patch("agent.cli.GoogleSearch", side_effect=Exception("429 Too Many Requests")), \
         patch("agent.cli.GEN_MODEL.generate_content", return_value=MagicMock(text=json.dumps({
             "answer": "No relevant documents found to answer the question.",
             "citations": []
         }))):

        graph = Graph(
            nodes={
                "GenerateQueries": GenerateQueries(),
                "WebSearchTool": WebSearchTool(),
                "Reflect": Reflect(),
                "Synthesize": Synthesize()
            },
            edges=[
                Edge("GenerateQueries", "WebSearchTool"),
                Edge("WebSearchTool", "Reflect"),
                Edge("Reflect", "Synthesize"),
            ],
            entry="GenerateQueries",
            exit="Synthesize",
            max_iter=2
        )

        result = graph.run(dummy_input)
        assert result["citations"] == []

def test_timeout(dummy_input):
    with patch("agent.cli.GEN_MODEL.generate_content", return_value=MagicMock(text=json.dumps(["timeout query"]))), \
         patch("agent.cli.GoogleSearch", side_effect=TimeoutError("Timeout")), \
         patch("agent.cli.GEN_MODEL.generate_content", return_value=MagicMock(text=json.dumps({
             "answer": "No relevant documents found to answer the question.",
             "citations": []
         }))):

        graph = Graph(
            nodes={
                "GenerateQueries": GenerateQueries(),
                "WebSearchTool": WebSearchTool(),
                "Reflect": Reflect(),
                "Synthesize": Synthesize()
            },
            edges=[
                Edge("GenerateQueries", "WebSearchTool"),
                Edge("WebSearchTool", "Reflect"),
                Edge("Reflect", "Synthesize"),
            ],
            entry="GenerateQueries",
            exit="Synthesize",
            max_iter=2
        )

        result = graph.run(dummy_input)
        assert result["answer"].startswith("No relevant")

def test_two_round_supplement(dummy_input):
    def gemini_side_effect(prompt):
        if "Break the following research question" in prompt:
            return MagicMock(text=json.dumps(["first query"]))
        elif "Answer this research question" in prompt:
            return MagicMock(text=json.dumps({
                "answer": "No relevant documents found to answer the question.",
                "citations": []
            }))
        return MagicMock(text="{}")

    with patch("agent.cli.GEN_MODEL.generate_content", side_effect=gemini_side_effect), \
         patch("agent.cli.GoogleSearch") as mock_serp:

        mock_search_instance = MagicMock()
        mock_search_instance.get_dict.return_value = {"organic_results": []}
        mock_serp.return_value = mock_search_instance

        graph = Graph(
            nodes={
                "GenerateQueries": GenerateQueries(),
                "WebSearchTool": WebSearchTool(),
                "Reflect": Reflect(),
                "Synthesize": Synthesize()
            },
            edges=[
                Edge("GenerateQueries", "WebSearchTool"),
                Edge("WebSearchTool", "Reflect"),
                Edge("Reflect", "Synthesize"),
            ],
            entry="GenerateQueries",
            exit="Synthesize",
            max_iter=2
        )

        result = graph.run(dummy_input)
        assert result["answer"].startswith("No relevant")
