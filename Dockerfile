# ---- Stage 1: Builder ----
FROM python:3.11-slim as builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt


# ---- Stage 2: Final Runtime Image ----
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# No APP_ENV default here on purpose: core/config.py already defaults it to
# "production" (fail-secure) when unset. Local dev sets APP_ENV=development
# via .env; a deploy must NOT bake that in, since core/dependencies.py and
# core/kyc_database.py fall back to DEFAULT_TEST_USER_ID -- a real,
# unauthenticated pass -- for any request whenever APP_ENV=="development".
ENV PORT=8080

WORKDIR /app

COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Fast-embeddings service: only the /ask/regular answer-gen path.
COPY ./main.py .
COPY ./core ./core
COPY ./models ./models
COPY ./services ./services
COPY ./routers ./routers

EXPOSE 8080
ENTRYPOINT ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
