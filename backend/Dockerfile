# Works on Render (Docker), Hugging Face Spaces (port 7860) and any container host.
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY frontend/ ./frontend/
RUN chmod -R 777 /app
WORKDIR /app/backend
ENV PORT=8000 HF_HOME=/tmp/hf TRANSFORMERS_CACHE=/tmp/hf
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
