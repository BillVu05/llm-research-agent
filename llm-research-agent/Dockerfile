FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies, including correct SerpAPI version
RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        openai \
        "google-generativeai>=0.4.0" \
        python-dotenv \
        google-search-results

# Set default entrypoint
ENTRYPOINT ["python", "src/agent/cli.py"]

