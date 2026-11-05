# Works when build context is "backend" (e.g. Render Root Directory = backend)
FROM python:3.11-slim
WORKDIR /app/backend

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod -R 777 /app

ENV PORT=8000 HF_HOME=/tmp/hf
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
