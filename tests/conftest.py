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

from app.answering.client import LLMError, LLMResponse, get_llm
from app.config import Settings, get_settings
from app.db.models import EMBED_DIM, Base
from app.db.session import get_db
from app.ingestion.embeddings import get_embedder
from app.main import app

ANSWER_JSON = """{"answerable": true, "statements": [{"text": "Books are limited to $60 a year.",
  "citations": [{"source": "S1", "quote": "Limit is set to $60/year."}]}],
  "refusal_reason": null, "sources_disagree": false}"""

REFUSAL_JSON = """{"answerable": false, "statements": [],
  "refusal_reason": "The sources do not say when salary is paid.", "sources_disagree": false}"""


class FakeLLM:
    """Stands in for Gemini. Each call pops the next scripted reply: a str (returned as the
    model's text) or an LLMError (raised). With no script left, it returns ANSWER_JSON."""

    def __init__(self) -> None:
        self.replies: list[str | LLMError] = []
        self.calls: list[dict] = []

    def generate(self, *, prompt, schema, model) -> LLMResponse:
        self.calls.append({"model": model, "prompt": prompt})
        reply = self.replies.pop(0) if self.replies else ANSWER_JSON
        if isinstance(reply, LLMError):
            raise reply
        return LLMResponse(text=reply, model=model, latency_ms=5, input_tokens=1000, output_tokens=100)


def test_settings(**overrides) -> Settings:
    base = dict(database_url=TEST_DB_URL, max_upload_mb=1, gemini_api_key="fake",
                gemini_model="gemini-3.5-flash", gemini_fallback_model="gemini-3.5-flash-lite",
                llm_retry_backoff_s=0, retrieval_k=3)
    return Settings(**{**base, **overrides})


test_settings.__test__ = False  # a helper, not a test

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
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(fake_embedder, fake_llm) -> Iterator[TestClient]:
    def _db() -> Iterator[Session]:
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_settings] = lambda: test_settings()  # FastAPI reads the signature
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def db() -> Iterator[Session]:
    with TestSession() as s:
        yield s
