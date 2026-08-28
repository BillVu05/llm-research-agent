FROM python:3.12-slim

WORKDIR /app

# Build context is the repo root: one requirements.txt, no path gymnastics.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

# Nothing here needs root, so do not run as it.
RUN useradd --create-home --uid 10001 agent && chown -R agent:agent /app
USER agent

ENTRYPOINT ["python", "src/agent/cli.py"]
