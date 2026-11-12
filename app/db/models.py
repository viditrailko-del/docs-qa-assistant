import uuid
from datetime import datetime

from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# bge-small-en-v1.5 returns 384 numbers per text. A different model needs a migration.
EMBED_DIM = 384

# Heading words weigh less (B) than the passage's own words (A) in keyword ranking.
SEARCH_TSV_SQL = (
    "setweight(to_tsvector('english', coalesce(heading_path, '')), 'B') || "
    "setweight(to_tsvector('english', coalesce(text, '')), 'A')"
)


JsonType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class Document(Base):
    """One uploaded source file. Same bytes -> same record (content_hash is unique)."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True)  # sha256 of the raw bytes
    path: Mapped[str] = mapped_column(String(500), index=True)  # e.g. finance/expenses.md
    title: Mapped[str | None] = mapped_column(String(500))
    source_url: Mapped[str | None] = mapped_column(String(1000))  # where a citation links to
    raw_text: Mapped[str] = mapped_column(Text)  # stored as uploaded, never modified
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Chunk.seq"
    )


class Chunk(Base):
    """A passage of a document: the unit we search, cite and verify quotes against."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "seq"),
        Index("ix_chunks_search_tsv", "search_tsv", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer)  # position in the document, from 0
    heading_path: Mapped[str] = mapped_column(Text)  # "Page title > Section > Subsection"
    text: Mapped[str] = mapped_column(Text)  # cleaned passage text (what the LLM will see)
    char_count: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM))
    embed_model: Mapped[str] = mapped_column(String(200))  # vectors from different models don't mix
    # Keyword index for full-text search, computed by Postgres from heading + text (never written
    # by us). 'english' stems words: "reimbursed" matches "reimbursable".
    search_tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed(SEARCH_TSV_SQL, persisted=True), deferred=True
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


class Ask(Base):
    """One question asked through POST /ask: what was retrieved and what came back."""

    __tablename__ = "asks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    question: Mapped[str] = mapped_column(Text)
    # pending -> answered | answered_with_flags | refused | failed  (plain string: no enum to migrate)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    retrieval_method: Mapped[str] = mapped_column(String(20))
    k: Mapped[int] = mapped_column(Integer)
    # [{label, chunk_id, similarity, path, heading_path, source_url}] in the order the LLM saw them
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JsonType, default=list)
    answer_json: Mapped[dict[str, Any] | None] = mapped_column(JsonType)  # validated LLMAnswer
    # [{code, severity, reason, statement, citation}] from app/checks; recomputable from the rest
    flags: Mapped[list[dict[str, Any]]] = mapped_column(JsonType, default=list, server_default=text("'[]'"))
    prompt_version: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    llm_calls: Mapped[list["LlmCall"]] = relationship(
        back_populates="ask", cascade="all, delete-orphan", order_by="LlmCall.id"
    )


class LlmCall(Base):
    """One LLM call. The raw response is saved BEFORE parsing, so nothing the model said is lost."""

    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    ask_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("asks.id", ondelete="CASCADE"), index=True)
    model: Mapped[str] = mapped_column(String(100))
    prompt_version: Mapped[str] = mapped_column(String(20))
    raw_response: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)  # API error, or "validation: ..." for bad output
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    thinking_tokens: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))  # null = unknown price, never 0
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    ask: Mapped[Ask] = relationship(back_populates="llm_calls")
