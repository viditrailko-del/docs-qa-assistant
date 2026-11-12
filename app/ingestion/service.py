"""Ingestion: raw Markdown bytes -> Document + embedded Chunks. Idempotent by content hash."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document
from app.ingestion.embeddings import Embedder
from app.ingestion.markdown import chunk_markdown, embedding_input


class InvalidDocument(ValueError):
    """The upload can't be ingested (not UTF-8 text, empty...). Maps to a 4xx, never a 500."""


@dataclass
class IngestResult:
    document: Document
    created: bool  # False = these exact bytes were already ingested


def ingest_document(
    db: Session,
    embedder: Embedder,
    raw: bytes,
    *,
    path: str,
    source_url: str | None = None,
    max_chars: int = 1200,
) -> IngestResult:
    content_hash = hashlib.sha256(raw).hexdigest()
    if existing := db.scalar(select(Document).where(Document.content_hash == content_hash)):
        return IngestResult(existing, created=False)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise InvalidDocument("file is not UTF-8 text") from e
    if not text.strip():
        raise InvalidDocument("file is empty")

    title, drafts = chunk_markdown(text, fallback_title=PurePosixPath(path).stem, max_chars=max_chars)
    if not drafts:
        raise InvalidDocument("no readable text after cleaning")

    vectors = embedder.embed_passages([embedding_input(d.heading_path, d.text) for d in drafts])
    doc = Document(content_hash=content_hash, path=path, title=title, source_url=source_url, raw_text=text)
    doc.chunks = [
        Chunk(seq=d.seq, heading_path=d.heading_path, text=d.text, char_count=len(d.text),
              embedding=v, embed_model=embedder.model_name)
        for d, v in zip(drafts, vectors, strict=True)
    ]
    db.add(doc)
    try:
        db.commit()
    except IntegrityError:
        # Same file uploaded twice at the same moment: the other request won. Return its row.
        db.rollback()
        existing = db.scalar(select(Document).where(Document.content_hash == content_hash))
        if existing is None:
            raise
        return IngestResult(existing, created=False)
    return IngestResult(doc, created=True)
