FROM python:3.12-slim

WORKDIR /app

# Build context is the repo root, so requirements.txt is the single source of truth.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY llm-research-agent/src ./src

ENTRYPOINT ["python", "src/agent/cli.py"]
