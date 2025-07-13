import pytest
from unittest.mock import patch, MagicMock
from src.agent.cli import GenerateQueries, WebSearchTool, Reflect, Synthesize

# ① Happy path: LLM returns queries, search returns docs, synthesis works
@patch('google.generativeai.GenerativeModel.generate_content')
def test_happy_path(mock_generate):
    mock_generate.return_value.text = '["Q1", "Q2", "Q3"]'
    gq = GenerateQueries()
    queries = gq.run({'topic': 'AI'})['queries']
    ws = WebSearchTool()
    docs = ws.run({'queries': queries})['docs']
    rf = Reflect()
    reflect_out = rf.run({'queries': queries, 'docs': docs})
    sz = Synthesize()
    with patch('google.generativeai.GenerativeModel.generate_content') as mock_syn:
        mock_syn.return_value.text = "Final answer [1][2]"
        result = sz.run({'docs': docs})
    assert isinstance(result['answer'], str)
    assert isinstance(result['citations'], list)

# ② No result: search returns empty docs, reflect asks for more
def test_no_result():
    queries = ["Q1"]
    docs = []  # Simulate no results
    rf = Reflect()
    reflect_out = rf.run({'queries': queries, 'docs': docs})
    assert reflect_out['need_more'] is True
    assert len(reflect_out['new_queries']) > 0

# ③ HTTP 429: Gemini RateLimit simulation (simulate with RuntimeError)
@patch('google.generativeai.GenerativeModel.generate_content')
def test_http_429(mock_generate):
    mock_generate.side_effect = RuntimeError("429: Rate limit exceeded")
    gq = GenerateQueries()
    with pytest.raises(RuntimeError, match="429"):
        gq.run({'topic': 'AI'})

# ④ Timeout: simulate timeout error from Gemini
@patch('google.generativeai.GenerativeModel.generate_content')
def test_timeout(mock_generate):
    import socket
    mock_generate.side_effect = socket.timeout('timeout')
    gq = GenerateQueries()
    with pytest.raises(socket.timeout):
        gq.run({'topic': 'AI'})

# ⑤ Two-round supplement: reflect triggers a second round
@patch('google.generativeai.GenerativeModel.generate_content')
def test_two_round_supplement(mock_generate):
    # First call returns queries, second and third return answer
    mock_generate.side_effect = [
        MagicMock(text='["Q1", "Q2", "Q3"]'),
        MagicMock(text='Final answer [1][2][3]')
    ]
    gq = GenerateQueries()
    queries = gq.run({'topic': 'AI'})['queries']
    ws = WebSearchTool()
    docs = ws.run({'queries': queries})['docs'][:2]  # Too few docs, will trigger second round
    rf = Reflect()
    reflect_out = rf.run({'queries': queries, 'docs': docs})
    assert reflect_out['need_more'] is True

    new_queries = reflect_out['new_queries']
    docs2 = ws.run({'queries': new_queries})['docs']
    all_docs = docs + docs2

    sz = Synthesize()
    with patch('google.generativeai.GenerativeModel.generate_content') as mock_syn:
        mock_syn.return_value.text = 'Final answer [1][2][3]'
        result = sz.run({'docs': all_docs})
    assert '[1]' in result['answer']
    assert isinstance(result['citations'], list)
