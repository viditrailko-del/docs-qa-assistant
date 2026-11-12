"""Test setup: a separate `rag_test` database, wiped between tests, and a fake embedder.

Tests never touch the dev database, never load the real embedding model, never call Gemini.
"""

import hashlib
import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.db.models import EMBED_DIM, Base
from app.db.session import get_db
from app.ingestion.embeddings import get_embedder
from app.main import app

TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", "postgresql+psycopg://rag:rag@localhost:5433/rag_test")

engine = create_engine(TEST_DB_URL)
TestSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class FakeEmbedder:
    """Deterministic stand-in: same text -> same unit vector, different texts -> different ones."""

    model_name = "fake-embedder"
    dim = EMBED_DIM

    def __init__(self) -> None:
        self.passage_calls = 0

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        raw = [(digest[i % len(digest)] - 127.5) / 127.5 for i in range(self.dim)]
        norm = sum(x * x for x in raw) ** 0.5
        return [x / norm for x in raw]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.passage_calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture(scope="session", autouse=True)
def _schema() -> Iterator[None]:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture(autouse=True)
def _clean_tables() -> Iterator[None]:
    yield
    with engine.begin() as conn:
        tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
        conn.execute(text(f"TRUNCATE {tables} CASCADE"))


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def client(fake_embedder) -> Iterator[TestClient]:
    def _db() -> Iterator[Session]:
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    app.dependency_overrides[get_settings] = lambda: Settings(database_url=TEST_DB_URL, max_upload_mb=1)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def db() -> Iterator[Session]:
    with TestSession() as s:
        yield s
