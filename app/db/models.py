import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# bge-small-en-v1.5 returns 384 numbers per text. A different model needs a migration.
EMBED_DIM = 384


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
    __table_args__ = (UniqueConstraint("document_id", "seq"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer)  # position in the document, from 0
    heading_path: Mapped[str] = mapped_column(Text)  # "Page title > Section > Subsection"
    text: Mapped[str] = mapped_column(Text)  # cleaned passage text (what the LLM will see)
    char_count: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM))
    embed_model: Mapped[str] = mapped_column(String(200))  # vectors from different models don't mix

    document: Mapped[Document] = relationship(back_populates="chunks")
