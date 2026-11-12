"""Document routes. Thin: validation and HTTP status only; the work is in app/ingestion."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.models import Chunk, Document
from app.db.session import get_db
from app.ingestion.embeddings import Embedder, get_embedder
from app.ingestion.service import InvalidDocument, ingest_document
from app.schemas.document import ChunkOut, DocumentOut

router = APIRouter(prefix="/documents", tags=["documents"])

DbDep = Annotated[Session, Depends(get_db)]


def _out(doc: Document, chunk_count: int) -> DocumentOut:
    return DocumentOut(
        id=doc.id, path=doc.path, title=doc.title, source_url=doc.source_url,
        content_hash=doc.content_hash, chunk_count=chunk_count, created_at=doc.created_at,
    )


def _count(db: Session, doc_id: uuid.UUID) -> int:
    return db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == doc_id)) or 0


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED,
             responses={200: {"description": "Already ingested (same bytes): existing document"}})
def upload_document(
    response: Response,
    db: DbDep,
    file: Annotated[UploadFile, File(description="A Markdown (.md) file")],
    path: Annotated[str | None, Form(description="Logical path, e.g. finance/expenses.md")] = None,
    source_url: Annotated[str | None, Form(description="Public URL citations should link to")] = None,
    embedder: Embedder = Depends(get_embedder),
    settings: Settings = Depends(get_settings),
) -> DocumentOut:
    name = path or file.filename or "document.md"
    if not name.lower().endswith((".md", ".markdown")):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "only Markdown (.md) files are supported")
    limit = settings.max_upload_mb * 1024 * 1024
    raw = file.file.read(limit + 1)
    if len(raw) > limit:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"file is larger than {settings.max_upload_mb} MB")
    try:
        result = ingest_document(db, embedder, raw, path=name, source_url=source_url,
                                 max_chars=settings.chunk_max_chars)
    except InvalidDocument as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
    if not result.created:
        response.status_code = status.HTTP_200_OK
    return _out(result.document, _count(db, result.document.id))


@router.get("", response_model=list[DocumentOut])
def list_documents(db: DbDep) -> list[DocumentOut]:
    counts = (
        select(Chunk.document_id, func.count().label("n")).group_by(Chunk.document_id).subquery()
    )
    rows = db.execute(
        select(Document, func.coalesce(counts.c.n, 0))
        .outerjoin(counts, counts.c.document_id == Document.id)
        .order_by(Document.path)
    ).all()
    return [_out(doc, n) for doc, n in rows]


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: uuid.UUID, db: DbDep) -> DocumentOut:
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    return _out(doc, _count(db, doc.id))


@router.get("/{doc_id}/chunks", response_model=list[ChunkOut])
def get_chunks(doc_id: uuid.UUID, db: DbDep) -> list[Chunk]:
    if db.get(Document, doc_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    return list(db.scalars(select(Chunk).where(Chunk.document_id == doc_id).order_by(Chunk.seq)))
