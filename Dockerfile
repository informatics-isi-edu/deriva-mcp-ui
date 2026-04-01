# deriva-mcp-ui
# Multi-stage build for smaller final image.
# This Dockerfile is used for standalone builds from the repo root.
# For deriva-docker integration builds see deriva-docker/deriva/mcp-ui/Dockerfile.

# Build stage: install dependencies into an isolated venv
FROM python:3.13-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
COPY src/ ./src/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --frozen --no-dev --no-editable


# Runtime stage: minimal image with only what the server needs to run
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --retries=4 --start-period=15s \
  CMD curl -f http://localhost:8001/health || exit 1

CMD ["deriva-mcp-ui"]

LABEL org.opencontainers.image.title="deriva-mcp-ui"
LABEL org.opencontainers.image.description="Browser-based chatbot UI for the DERIVA platform"
LABEL org.opencontainers.image.licenses="Apache-2.0"