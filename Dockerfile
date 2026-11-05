# Works when build context is the repository root (Render, Hugging Face Spaces, Railway, Docker)
FROM python:3.11-slim
WORKDIR /app

# 1. Install dependencies first
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy source code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
RUN chmod -R 777 /app

WORKDIR /app/backend
ENV PORT=8000 HF_HOME=/tmp/hf
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
