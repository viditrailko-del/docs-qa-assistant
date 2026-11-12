"""API models for /ask. The LLM's own output schema is app/schemas/answer.py."""

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field

from app.db.models import Ask


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int | None = Field(default=None, ge=1, le=20, description="Chunks to retrieve (default from settings)")


class SourceOut(BaseModel):
    label: str
    chunk_id: int
    similarity: float
    path: str
    heading_path: str
    source_url: str | None


class CitationOut(BaseModel):
    source: str  # label as the model wrote it
    quote: str
    chunk_id: int | None  # None: the label matches no retrieved source (Phase 4 flags this)
    path: str | None
    heading_path: str | None
    source_url: str | None


class FlagOut(BaseModel):
    code: str  # e.g. quote_not_in_source, number_unsupported, scope_mismatch
    severity: str  # error | warning
    reason: str
    statement: int | None  # index into statements (None: about the whole answer)
    citation: int | None


class StatementOut(BaseModel):
    text: str
    citations: list[CitationOut]
    verified: bool  # every check passed for this statement
    flags: list[FlagOut]


class LlmCallOut(BaseModel):
    model: str
    prompt_version: str
    error: str | None
    latency_ms: int | None
    input_tokens: int | None
    output_tokens: int | None
    thinking_tokens: int | None
    cost_usd: Decimal | None


class AskOut(BaseModel):
    id: uuid.UUID
    question: str
    status: str  # answered | answered_with_flags | refused | failed
    answer: str | None  # the statements joined: convenient for display
    statements: list[StatementOut]
    flags: list[FlagOut]  # all flags; each statement also lists its own
    sources_disagree: bool
    refusal_reason: str | None
    sources: list[SourceOut]
    prompt_version: str
    created_at: datetime
    llm_calls: list[LlmCallOut]

    @classmethod
    def from_ask(cls, ask: Ask) -> "AskOut":
        by_label = {s["label"]: s for s in ask.sources}
        data = ask.answer_json or {}
        flags = [FlagOut(**f) for f in ask.flags or []]
        statements = [
            StatementOut(
                text=st["text"],
                flags=(own := [f for f in flags if f.statement == i]),
                verified=not own,
                citations=[
                    CitationOut(
                        source=c["source"], quote=c["quote"],
                        chunk_id=(src := by_label.get(c["source"].strip(), {})).get("chunk_id"),
                        path=src.get("path"), heading_path=src.get("heading_path"),
                        source_url=src.get("source_url"),
                    )
                    for c in st["citations"]
                ],
            )
            for i, st in enumerate(data.get("statements", []))
        ]
        refusal = data.get("refusal_reason") if ask.status == "refused" else None
        if ask.status == "refused" and not refusal and not ask.sources:
            refusal = "No documents matched the question."
        return cls(
            id=ask.id, question=ask.question, status=ask.status,
            answer=" ".join(s.text for s in statements) or None,
            statements=statements, flags=flags, sources_disagree=bool(data.get("sources_disagree")),
            refusal_reason=refusal, sources=[SourceOut(**s) for s in ask.sources],
            prompt_version=ask.prompt_version, created_at=ask.created_at,
            llm_calls=[LlmCallOut.model_validate(c, from_attributes=True) for c in ask.llm_calls],
        )
