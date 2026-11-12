"""The LLM part of answering, with no database: shared by the API and the eval harness.

answer_question() owns the "the model may be wrong, slow, or down" logic:
  - no retrieved sources: refuse without calling the LLM
  - transient errors (overload, timeout): retry with backoff, then the fallback model
  - output that doesn't fit the schema: retry ONCE, telling the model what was wrong
  - nothing worked: status `failed` (never an exception to the caller)

Every LLM call is reported to a `RunRecorder` right after it returns, BEFORE its output is
parsed: the API's recorder saves llm_calls rows, the eval's keeps them in memory. So the eval
measures exactly the code the API runs.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from app.answering.client import LLMClient, LLMError
from app.answering.pricing import cost_usd
from app.checks.base import Flag, SourceText
from app.checks.runner import run_checks
from app.config import Settings
from app.retrieval.search import Hit
from app.schemas.answer import LLMAnswer

PROMPTS_DIR = Path(__file__).parent / "prompts"
MAX_TRANSIENT_RETRIES = 2  # per model, so up to 3 calls each
MAX_VALIDATION_RETRIES = 1

RETRY_FEEDBACK = """

Your previous reply was rejected because it did not match the required format.
Problems:
{errors}

Reply again to the same question with the same sources, fixing these problems."""

NO_SOURCES_REASON = "No documents matched the question."


class AnswerStatus(StrEnum):
    answered = "answered"  # every check passed
    answered_with_flags = "answered_with_flags"  # shown, but marked: something couldn't be verified
    refused = "refused"  # the model (or the system) says the sources don't answer it
    failed = "failed"  # the LLM call or its output failed every attempt


@dataclass
class Attempt:
    """One LLM call and what came of it (one llm_calls row in the API)."""

    model: str
    prompt_version: str
    raw_response: str | None = None
    error: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    cost_usd: Decimal | None = None


class RunRecorder(Protocol):
    def record(self, attempt: Attempt) -> None:
        """Called right after each LLM call, BEFORE its output is parsed."""

    def update(self, attempt: Attempt) -> None:
        """Called when a recorded attempt changes (its output failed validation)."""


@dataclass
class Source:
    """A retrieved chunk as the LLM sees it: labeled S1, S2... in retrieval order."""

    label: str
    hit: Hit


@dataclass
class AnswerOutcome:
    status: AnswerStatus
    sources: list[Source]
    answer: LLMAnswer | None = None
    attempts: list[Attempt] = field(default_factory=list)
    flags: list[Flag] = field(default_factory=list)

    @property
    def refusal_reason(self) -> str | None:
        if self.status is AnswerStatus.refused:
            return self.answer.refusal_reason if self.answer else NO_SOURCES_REASON
        return None


@lru_cache
def load_prompt(version: str) -> str:
    return (PROMPTS_DIR / f"{version}.txt").read_text(encoding="utf-8")


@lru_cache
def response_schema() -> dict:
    return LLMAnswer.model_json_schema()


def label_sources(hits: Sequence[Hit]) -> list[Source]:
    return [Source(label=f"S{i}", hit=h) for i, h in enumerate(hits, start=1)]


def format_sources(sources: Sequence[Source]) -> str:
    blocks = []
    for s in sources:
        h = s.hit
        blocks.append(f"[{s.label}] {h.title or h.path}\nSection: {h.heading_path}\n\n{h.text}")
    return "\n\n---\n\n".join(blocks)


def build_prompt(version: str, question: str, sources: Sequence[Source]) -> str:
    # str.replace, not str.format: handbook text may contain { } braces.
    return load_prompt(version).replace("{sources}", format_sources(sources)).replace("{question}", question)


def answer_question(
    question: str,
    hits: Sequence[Hit],
    llm: LLMClient,
    settings: Settings,
    recorder: RunRecorder,
    sleep: Callable[[float], None] = time.sleep,
) -> AnswerOutcome:
    sources = label_sources(hits)
    if not sources:
        return AnswerOutcome(AnswerStatus.refused, sources)

    outcome = AnswerOutcome(AnswerStatus.failed, sources)
    base_prompt = build_prompt(settings.prompt_version, question, sources)
    prompt = base_prompt
    for _ in range(1 + MAX_VALIDATION_RETRIES):
        attempt = _call_llm(prompt, llm, settings, recorder, sleep, outcome.attempts)
        if attempt is None:
            return outcome  # every model/attempt failed at the API level
        try:
            answer = LLMAnswer.model_validate_json(attempt.raw_response or "")
        except ValidationError as e:
            attempt.error = f"validation: {e}"[:4000]
            recorder.update(attempt)
            prompt = base_prompt + RETRY_FEEDBACK.format(errors=_short_errors(e))
            continue
        outcome.answer = answer
        if not answer.answerable:
            outcome.status = AnswerStatus.refused
            return outcome
        outcome.flags = run_checks(question, answer, source_texts(sources), min_similarity=settings.min_similarity)
        outcome.status = AnswerStatus.answered_with_flags if outcome.flags else AnswerStatus.answered
        return outcome
    return outcome


def source_texts(sources: Sequence[Source]) -> list[SourceText]:
    return [SourceText(label=s.label, text=s.hit.text, heading_path=s.hit.heading_path, title=s.hit.title,
                       path=s.hit.path, similarity=s.hit.similarity) for s in sources]


def _call_llm(
    prompt: str,
    llm: LLMClient,
    settings: Settings,
    recorder: RunRecorder,
    sleep: Callable[[float], None],
    attempts: list[Attempt],
) -> Attempt | None:
    models = [m for m in (settings.gemini_model, settings.gemini_fallback_model) if m]
    for model in models:
        for attempt_no in range(1 + MAX_TRANSIENT_RETRIES):
            attempt = Attempt(model=model, prompt_version=settings.prompt_version)
            attempts.append(attempt)
            try:
                resp = llm.generate(prompt=prompt, schema=response_schema(), model=model)
            except LLMError as e:
                attempt.error = str(e)[:4000]
                attempt.latency_ms = e.latency_ms
                recorder.record(attempt)
                if not e.transient:
                    break  # e.g. bad request or daily quota: same model won't fix it, try the fallback
                if attempt_no < MAX_TRANSIENT_RETRIES:
                    sleep(settings.llm_retry_backoff_s * (2**attempt_no))
                continue

            attempt.raw_response = resp.text
            attempt.latency_ms = resp.latency_ms
            attempt.input_tokens = resp.input_tokens
            attempt.output_tokens = resp.output_tokens
            attempt.thinking_tokens = resp.thinking_tokens
            attempt.cost_usd = cost_usd(model, resp.input_tokens, resp.output_tokens, resp.thinking_tokens)
            recorder.record(attempt)  # raw output is saved before we even try to parse it
            return attempt
    return None


def _short_errors(e: ValidationError) -> str:
    return "\n".join(
        f"- {'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}" for err in e.errors()[:20]
    )
