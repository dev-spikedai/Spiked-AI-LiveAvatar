FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV APP_ENV=production
ENV PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src
COPY ./public ./public

EXPOSE 8080

CMD ["sh", "-c", "uvicorn src.live_avatar:app --host 0.0.0.0 --port ${PORT:-8080}"]
