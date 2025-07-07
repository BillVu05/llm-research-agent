FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy everything
COPY . .

# Install required packages
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --force-reinstall \
        openai \
        "google-generativeai>=0.4.0"

# Debug: check genai module after install
RUN python -c "import google.generativeai as genai; print('✅ genai module ok:', dir(genai))"

# Default entrypoint
ENTRYPOINT ["python", "src/agent/cli.py"]
