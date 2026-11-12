# Dev image (docker compose): all deps incl. pytest, source bind-mounted, auto-reload.
# A slim production target comes in Phase 6.

FROM python:3.12-slim AS base
RUN pip install --no-cache-dir uv==0.13.7
# Keep the venv OUTSIDE /app so the bind-mounted source (and any Windows .venv) can't clash.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    EMBED_CACHE_DIR=/opt/models
WORKDIR /app


FROM base AS dev
# Deps first (cached layer: only re-runs when deps change).
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project
# Bake the embedding model into the image: no download at runtime, same weights every run.
# Change EMBED_MODEL here and in app/config.py together.
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
RUN /opt/venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBED_MODEL}', cache_dir='/opt/models')"
COPY . .
EXPOSE 8000
CMD ["uv", "run", "--frozen", "fastapi", "dev", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
