FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers / faiss (adjust if needed)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Ensure NLTK data is available at build time
RUN python -m nltk.downloader punkt

ENV PYTHONUNBUFFERED=1
ENV OPENAI_API_KEY=""

# Default command: run Streamlit UI
EXPOSE 8501
CMD ["streamlit", "run", "src/app/ui_app.py", "--server.headless=true", "--server.port=8501", "--server.address=0.0.0.0"]
