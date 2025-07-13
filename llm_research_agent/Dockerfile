FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy everything into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        openai \
        "google-generativeai>=0.4.0" \
        python-dotenv \
        google-search-results

# Set the default entrypoint to run the CLI via __main__.py
ENTRYPOINT ["python", "-m", "llm_research_agent"]
