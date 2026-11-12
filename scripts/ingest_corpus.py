"""Ingest the frozen corpus (corpus/handbook, per corpus/manifest.json) into the database.

    docker compose run --rm -T api uv run --frozen python -m scripts.ingest_corpus

Idempotent: files already ingested (same bytes) are skipped. Uses the same service as
POST /documents, so the corpus goes through exactly the production path.
"""

from __future__ import annotations

import json
import sys
import time

from app.config import get_settings
from app.db.session import SessionLocal
from app.ingestion.embeddings import get_embedder
from app.ingestion.service import ingest_document
from scripts.snapshot_corpus import DOCS, MANIFEST, verify


def main() -> None:
    verify()  # never ingest a corpus that doesn't match its manifest
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    settings = get_settings()
    t0 = time.perf_counter()
    embedder = get_embedder()
    print(f"embedder {embedder.model_name} ({embedder.dim} dims) loaded in {time.perf_counter() - t0:.1f}s")

    created = skipped = chunks = 0
    t0 = time.perf_counter()
    with SessionLocal() as db:
        for f in manifest["files"]:
            raw = (DOCS / f["path"]).read_bytes()
            result = ingest_document(db, embedder, raw, path=f["path"], source_url=f["url"],
                                     max_chars=settings.chunk_max_chars)
            n = len(result.document.chunks)
            if result.created:
                created += 1
                chunks += n
                print(f"  + {f['path']} ({n} chunks)")
            else:
                skipped += 1
    print(f"done in {time.perf_counter() - t0:.1f}s: {created} ingested ({chunks} chunks), {skipped} already present")


if __name__ == "__main__":
    sys.exit(main())
