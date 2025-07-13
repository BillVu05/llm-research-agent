# tests/test_websearch.py
from src.agent.pipeline import WebSearchTool

def test_websearch_no_results(monkeypatch):
    def mock_get_results(*args, **kwargs):
        return []

    monkeypatch.setattr("src.agent.pipeline.GoogleSearch.get_dict", lambda self: {"organic_results": []})
    tool = WebSearchTool()
    result = tool.run({"queries": ["gibberish unfindable query"]})
    assert result["docs"] == []

def test_websearch_rate_limit(monkeypatch):
    def mock_get_dict(self):
        raise Exception("HTTP 429: Too Many Requests")

    monkeypatch.setattr("src.agent.pipeline.GoogleSearch.get_dict", mock_get_dict)
    tool = WebSearchTool()
    result = tool.run({"queries": ["2022 World Cup"]})
    assert result["docs"] == []  # Should gracefully handle
