FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --force-reinstall \
        openai \
        "google-generativeai>=0.4.0" \
        python-dotenv \
        google-search-results

# Default entrypoint
ENTRYPOINT ["python", "src/agent/cli.py"]
