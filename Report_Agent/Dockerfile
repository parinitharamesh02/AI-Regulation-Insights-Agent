# Use a small, production-friendly Python image
FROM python:3.11-slim

# Do not run as root in production containers
RUN useradd -m appuser

WORKDIR /app

# Install system dependencies if needed (kept minimal here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src ./src
COPY data ./data
COPY examples ./examples

# Use the non-root user
USER appuser

# Default: run CLI assistant (can be overridden in ECS Task Definition)
CMD ["python", "-m", "src.chat_cli"]
