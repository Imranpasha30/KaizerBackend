# Kaizer backend — Linux/cloud image (Wave 5).
#
# Two-stage build: the builder stage needs a C toolchain because
# requirements.txt contains source-only packages (webrtcvad has no
# manylinux wheel); the runtime stage ships only the venv + ffmpeg, so
# gcc and the pip cache never reach the final image.
#
# NOTE on size: requirements.txt is the FULL dependency set (torch,
# transformers, onnxruntime, …) because workers run the render pipeline
# in-process. For an API-only image, swap in requirements-railway.txt
# (the lean set used by the old Railway deploy) — see nixpacks.toml.
#
# Build:  docker build -t kaizer-backend .
# Run:    docker run -p 8000:8000 --env-file .env.docker kaizer-backend
# Worker: docker run --env-file .env.docker kaizer-backend \
#             python -m services.publish_worker

# ── Stage 1: build the virtualenv ────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────
FROM python:3.11-slim

# ffmpeg: every render/branding/probe path shells out to it.
# curl: container healthchecks against /metrics or /docs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user — uvicorn binds 8000 (unprivileged) so no root needed.
RUN useradd --create-home --uid 1000 kaizer

WORKDIR /app
COPY --chown=kaizer:kaizer . .

# Writable dirs for renders/logs (mount named volumes over these in
# compose so output survives container replacement).
RUN mkdir -p /app/output /app/logs \
    && chown -R kaizer:kaizer /app/output /app/logs

USER kaizer

EXPOSE 8000

# Single uvicorn worker by design — the app runs an embedded publish
# worker + cron leader when KAIZER_DURABLE_QUEUE=1; scale horizontally
# with more containers, not --workers (see docs/DEPLOYMENT.md).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
