#!/bin/sh
# Start command of the public demo (render.yaml). Runs on every boot: a free instance sleeps when idle and boots
# again on the next visit, and the database may be brand new.
#   1. migrations (no-op when up to date)
#   2. the document index: corpus + exported vectors (idempotent: skips documents already there; ~1407 rows
#      the first time, no embedding model needed)
#   3. the API on $PORT, trusting the host's proxy for the client IP (the live-mode per-visitor cap needs it)
# One ONNX thread: on a 0.1-CPU instance, a thread per host core stalls the embedding model completely.
set -e
export EMBED_THREADS="${EMBED_THREADS:-1}"
alembic upgrade head
python -m app.demo.index seed
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips '*'
