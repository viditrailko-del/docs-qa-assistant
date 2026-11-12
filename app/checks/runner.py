"""Run every check on an answer. Pure: same inputs, same flags."""

from __future__ import annotations

from collections.abc import Sequence

from app.checks.base import Flag, Severity, SourceText
from app.checks.citations import check_citations
from app.checks.numbers import check_numbers
from app.checks.scope import check_scope
from app.schemas.answer import LLMAnswer


def run_checks(
    question: str, answer: LLMAnswer, sources: Sequence[SourceText], *, min_similarity: float = 0.0
) -> list[Flag]:
    """min_similarity > 0 adds a `weak_retrieval` warning when no source is that close to the
    question. Off by default: the Phase 2 eval found no threshold that separates answerable from
    unanswerable questions without overfitting the 61 labels (see CLAUDE.md)."""
    if not answer.answerable:
        return []  # a refusal makes no claims to verify
    flags = check_citations(answer, sources) + check_numbers(answer, sources) + check_scope(question, answer, sources)
    best = max((s.similarity for s in sources), default=0.0)
    if min_similarity and best < min_similarity:
        flags.append(Flag("weak_retrieval", Severity.warning,
                          f"best source similarity {best:.3f} is below {min_similarity:.3f}"))
    return flags
