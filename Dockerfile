# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.1 AS uv

FROM node:22-alpine AS frontend-build
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN cd frontend && npm ci
COPY frontend ./frontend
RUN cd frontend && npm run build

FROM python:3.12-slim-bookworm

COPY --from=uv /uv /uvx /bin/
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HF_HUB_CACHE=/models/hub \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv export --frozen --only-group server --no-emit-project --no-hashes --output-file requirements.txt \
    && uv pip install --system --no-cache --requirement requirements.txt \
    && rm requirements.txt

COPY app.py access_control.py audio_search.py core.py thumb.py frame_maps.py ./
COPY --from=frontend-build /build/static/dist ./static/dist

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app \
    && mkdir /cache \
    && chown app:app /cache

USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "64"]
