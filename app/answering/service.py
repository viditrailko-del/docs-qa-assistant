"""POST /ask, with the database: retrieve, answer, and record everything that happened."""

from __future__ import annotations

import logging
import time

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.answering.client import LLMClient
from app.answering.core import Attempt, answer_question
from app.config import Settings
from app.db.models import Ask, LlmCall
from app.ingestion.embeddings import Embedder
from app.logging_setup import log_event
from app.retrieval.search import Method, search

log = logging.getLogger(__name__)


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
    start = time.monotonic()
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
    _log_finished(ask, outcome, int((time.monotonic() - start) * 1000))
    return ask


def _log_finished(ask: Ask, outcome, wall_ms: int) -> None:
    """One event per question with everything needed to monitor quality and cost. The question
    text is NOT logged (it may hold personal details): ask_id links to it in the database."""
    a = outcome.attempts
    costs = [x.cost_usd for x in a if x.cost_usd is not None]
    used = [x.model for x in a if x.raw_response is not None and not x.error]
    log_event(
        log, logging.INFO, "ask_finished",
        ask_id=ask.id, status=ask.status, retrieval_method=ask.retrieval_method, k=ask.k,
        sources=len(outcome.sources),
        top_similarity=round(max((s.hit.similarity for s in outcome.sources), default=0.0), 4),
        model=used[-1] if used else None, prompt_version=ask.prompt_version,
        llm_calls=len(a), failed_calls=sum(x.error is not None for x in a),
        llm_latency_ms=sum(x.latency_ms or 0 for x in a), wall_ms=wall_ms,
        input_tokens=sum(x.input_tokens or 0 for x in a),
        output_tokens=sum((x.output_tokens or 0) + (x.thinking_tokens or 0) for x in a),
        cost_usd=sum(costs) if costs else None,
        flags=sorted({f.code for f in outcome.flags}),
    )


