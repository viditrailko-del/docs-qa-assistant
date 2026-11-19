# Two targets from one file:
#   dev  (docker compose): all deps incl. pytest, source bind-mounted, auto-reload
#   prod (default):        runtime deps only, no uv, non-root, JSON logs, healthcheck
#
#   docker build -t docs-qa-api .                  # prod (last stage = default)
#   docker build --target dev -t docs-qa-api:dev .
#
# Both bake the embedding model into the image: no download at runtime, same weights every
# run (prod sets HF_HUB_OFFLINE=1: it never contacts the model hub). Change EMBED_MODEL here and in app/config.py together (and re-embed every chunk).
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5

FROM python:3.12-slim AS base
RUN pip install --no-cache-dir uv==0.13.7
# Keep the venv OUTSIDE /app so the bind-mounted source (and any Windows .venv) can't clash.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    EMBED_CACHE_DIR=/opt/models
WORKDIR /app


FROM base AS dev
ARG EMBED_MODEL
# Deps first (cached layer: only re-runs when deps change).
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project
RUN /opt/venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBED_MODEL}', cache_dir='/opt/models')"
COPY . .
EXPOSE 8000
CMD ["uv", "run", "--frozen", "fastapi", "dev", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]


FROM base AS prod-deps
ARG EMBED_MODEL
COPY pyproject.toml uv.lock .python-version ./
# --no-dev: no pytest/httpx in production. Bytecode compiled at build time.
RUN UV_COMPILE_BYTECODE=1 uv sync --frozen --no-dev --no-install-project
# a+rX: the non-root user must read every cached model file (some are created 0600).
RUN /opt/venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBED_MODEL}', cache_dir='/opt/models')" \
    && chmod -R a+rX /opt/models


FROM python:3.12-slim AS prod
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    EMBED_CACHE_DIR=/opt/models \
    HF_HUB_OFFLINE=1 \
    LOG_FORMAT=json
RUN useradd --create-home --uid 10001 app
WORKDIR /app
# Same base image as prod-deps, so the venv's python symlinks stay valid.
COPY --from=prod-deps /opt/venv /opt/venv
COPY --from=prod-deps /opt/models /opt/models
# Only what runs: the app (prompts and the demo index included), migrations, and the frozen corpus
# (~1 MB: the demo seeds its database from it). No tests, eval, scripts.
COPY app ./app
COPY corpus ./corpus
COPY docker/demo-start.sh ./docker/demo-start.sh
COPY alembic ./alembic
COPY alembic.ini ./
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
# Migrations are a separate deploy step (`alembic upgrade head`), not run on every start.
CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
