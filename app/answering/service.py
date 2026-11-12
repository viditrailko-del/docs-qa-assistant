"""POST /ask, with the database: retrieve, answer, and record everything that happened."""

from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.answering.client import LLMClient
from app.answering.core import Attempt, answer_question
from app.config import Settings
from app.db.models import Ask, LlmCall
from app.ingestion.embeddings import Embedder
from app.retrieval.search import Method, search


class DbRecorder:
    """Writes each LLM call as an llm_calls row and commits at once: a crash later still
    leaves the raw model output in the database."""

    def __init__(self, db: Session, ask: Ask) -> None:
        self.db, self.ask = db, ask
        self._rows: dict[int, LlmCall] = {}

    def record(self, attempt: Attempt) -> None:
        row = LlmCall(ask_id=self.ask.id, **_fields(attempt))
        self.db.add(row)
        self.db.commit()
        self._rows[id(attempt)] = row

    def update(self, attempt: Attempt) -> None:
        row = self._rows[id(attempt)]
        row.error = attempt.error
        self.db.commit()


def _fields(a: Attempt) -> dict:
    return {
        "model": a.model, "prompt_version": a.prompt_version, "raw_response": a.raw_response,
        "error": a.error, "latency_ms": a.latency_ms, "input_tokens": a.input_tokens,
        "output_tokens": a.output_tokens, "thinking_tokens": a.thinking_tokens, "cost_usd": a.cost_usd,
    }


def ask_question(
    db: Session,
    embedder: Embedder,
    llm: LLMClient,
    settings: Settings,
    question: str,
    *,
    k: int | None = None,
    method: Method = Method.vector,
) -> Ask:
    k = k or settings.retrieval_k
    ask = Ask(question=question, status="pending", retrieval_method=method.value, k=k,
              prompt_version=settings.prompt_version)
    db.add(ask)
    db.commit()  # the question is on record before any slow or failing work

    hits = search(db, embedder, question, k=k, method=method)
    outcome = answer_question(question, hits, llm, settings, DbRecorder(db, ask))

    ask.sources = [
        {"label": s.label, "chunk_id": s.hit.chunk_id, "similarity": round(s.hit.similarity, 4),
         "path": s.hit.path, "heading_path": s.hit.heading_path, "source_url": s.hit.source_url}
        for s in outcome.sources
    ]
    ask.answer_json = outcome.answer.model_dump() if outcome.answer else None
    ask.flags = [f.to_dict() for f in outcome.flags]
    ask.status = outcome.status.value
    ask.completed_at = func.now()
    db.commit()
    db.refresh(ask)
    return ask


