"""The demo index: the frozen corpus + exported vectors seed a fresh database, exactly and idempotently."""

import pytest
from sqlalchemy import func, select

from app.config import Settings
from app.db.models import Chunk, Document
from app.demo import index
from tests.conftest import test_settings


@pytest.mark.parametrize("url", ["postgres://u:p@host/db?sslmode=require", "postgresql://u:p@host/db?sslmode=require"])
def test_hosted_postgres_urls_use_psycopg3(url):
    assert Settings(database_url=url).database_url == "postgresql+psycopg://u:p@host/db?sslmode=require"


def test_driver_urls_are_left_alone():
    assert Settings(database_url="postgresql+psycopg://u:p@h/d").database_url == "postgresql+psycopg://u:p@h/d"


def test_seed_loads_the_whole_corpus_with_the_exported_vectors(db):
    meta, vectors = index.load_index()
    result = index.seed_index(db, test_settings())
    assert result == {"documents_added": 60, "chunks_added": len(meta["chunks"]), "documents_present": 0}
    assert db.scalar(select(func.count()).select_from(Chunk)) == len(meta["chunks"]) == 1407

    c = db.scalars(select(Chunk).join(Document).where(Document.path == "finance/expenses.md", Chunk.seq == 3)).one()
    key, vec = vectors[("finance/expenses.md", 3)]
    assert key == index.chunk_key(c.heading_path, c.text)
    assert [float(x) for x in c.embedding] == vec  # float32 in, float32 out: bit for bit
    assert c.embed_model == meta["embed_model"] == "BAAI/bge-small-en-v1.5"
    assert c.document.source_url.startswith("https://handbook.gitlab.com/")


def test_seed_is_idempotent(db):
    index.seed_index(db, test_settings())
    assert index.seed_index(db, test_settings()) == {"documents_added": 0, "chunks_added": 0, "documents_present": 60}
    assert db.scalar(select(func.count()).select_from(Document)) == 60


def test_seed_refuses_vectors_that_belong_to_other_text(db, monkeypatch):
    meta, vectors = index.load_index()
    k = ("finance/expenses.md", 0)
    vectors[k] = ("0" * 16, vectors[k][1])
    monkeypatch.setattr(index, "load_index", lambda: (meta, vectors))
    with pytest.raises(index.IndexMismatch, match="finance/expenses.md chunk 0"):
        index.seed_index(db, test_settings())


def test_seed_refuses_a_chunker_with_other_settings(db):
    with pytest.raises(index.IndexMismatch):
        index.seed_index(db, test_settings(chunk_max_chars=600))
