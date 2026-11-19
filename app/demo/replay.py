"""Replay mode for the public demo: the real pipeline, with Gemini's answers pre-recorded.

A scenario (app/demo/scenarios/<name>.json, built by `python -m demo.build_replay`) holds one labeled question
and what flash-lite actually answered in a frozen eval run. Replaying it runs exactly what POST /ask runs
(pgvector search on the demo DB, the prompt, schema validation, the CURRENT citation checks, the stored ask and
llm_calls rows); only two inputs are recorded instead of computed:

- `RecordedEmbedder` returns the question's vector as the eval machine computed it. The local model's floats
  can differ in the last bits on another CPU, which could swap two near-tied chunks; the recorded vector makes
  the search on the demo DB (same chunk vectors, see app/demo/index.py) return exactly what the eval saw.
- `RecordedLLM` returns the recorded model outputs, in order. It refuses to improvise: the prompt it receives
  (question + retrieved chunks + prompt version) must hash to the recorded one. Any difference (a chunk or the
  prompt file changed, retrieval returned something else) raises a non-transient LLMError, so the ask ends
  `failed` with "replay diverged" instead of showing an answer to a prompt the model never saw.

Replayed calls are stored under the model name `replay:<recorded model>`: no price is known for that name, so
their cost is None, and the live-mode quota (counted on real model names) never counts them.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import delete, exists, select
from sqlalchemy.orm import Session

from app.answering.client import LLMError, LLMResponse
from app.answering.service import ask_question
from app.config import Settings
from app.db.models import Ask, LlmCall

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
REPLAY_PREFIX = "replay:"


@dataclass(frozen=True)
class Scenario:
    name: str
    data: dict[str, Any]

    @property
    def question(self) -> str:
        return self.data["question"]

    def summary(self) -> dict[str, Any]:
        """What the demo page lists: no recorded outputs, no expected result."""
        return {k: self.data[k] for k in ("name", "group", "title", "blurb", "label_id", "question")}


@lru_cache
def scenarios() -> dict[str, Scenario]:
    """name -> scenario, in the order the demo page lists them (the builder's `order`)."""
    loaded = [json.loads(p.read_text(encoding="utf-8")) for p in SCENARIOS_DIR.glob("*.json")]
    return {d["name"]: Scenario(name=d["name"], data=d) for d in sorted(loaded, key=lambda d: d["order"])}


def prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class RecordedEmbedder:
    """Embeds only the recorded question, with the recorded vector."""

    def __init__(self, model_name: str, question: str, vector: list[float]) -> None:
        self.model_name, self.dim = model_name, len(vector)
        self._question, self._vector = question, vector

    def embed_query(self, text: str) -> list[float]:
        if text != self._question:
            raise ValueError("replay: no recorded embedding for this text")
        return self._vector

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        raise ValueError("replay: passages are never embedded")


class RecordedLLM:
    """An LLMClient that answers with a scenario's recorded outputs, in order, and only to the recorded prompt.

    A validation retry sends the same prompt plus feedback, so each call's prompt must START with the recorded
    prompt (its first `prompt_chars` characters must hash to `prompt_sha256`)."""

    def __init__(self, recorded: dict[str, Any]) -> None:
        self._calls: Iterator[dict] = iter(recorded["calls"])
        self._sha, self._chars = recorded["prompt_sha256"], recorded["prompt_chars"]

    def generate(self, *, prompt: str, schema: dict[str, Any], model: str) -> LLMResponse:
        if prompt_sha(prompt[: self._chars]) != self._sha:
            raise LLMError("replay diverged: the retrieved sources or the prompt differ from the recording",
                           transient=False)
        rec = next(self._calls, None)
        if rec is None:
            raise LLMError("replay: no recorded answer left", transient=False)
        return LLMResponse(text=rec["raw_response"], model=model, latency_ms=rec["latency_ms"] or 0,
                           input_tokens=rec["input_tokens"], output_tokens=rec["output_tokens"],
                           thinking_tokens=rec["thinking_tokens"])


def replay_settings(settings: Settings, recorded: dict[str, Any]) -> Settings:
    """The app's settings with the recorded model, prompt version and k: calls are labeled `replay:<model>`,
    there is no fallback model, and no retry waits."""
    return settings.model_copy(update={
        "gemini_model": REPLAY_PREFIX + recorded["model"], "gemini_fallback_model": "",
        "prompt_version": recorded["prompt_version"], "retrieval_k": recorded["k"], "llm_retry_backoff_s": 0,
    })


# One replay of a question at a time (the demo runs as one process): a replay deletes earlier replays of the
# same question, which must not happen under one that is still running.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock(name: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(name, threading.Lock())


def clear_earlier_replays(db: Session, scenario: Scenario) -> int:
    """Delete this scenario's earlier replays (so the table stays bounded however often visitors click).
    Live asks with the same question (no replay calls) are never touched."""
    replayed = exists().where(LlmCall.ask_id == Ask.id, LlmCall.model.like(f"{REPLAY_PREFIX}%"))
    ids = list(db.scalars(select(Ask.id).where(Ask.question == scenario.question, replayed)))
    if ids:
        db.execute(delete(Ask).where(Ask.id.in_(ids)))  # llm_calls cascade
        db.commit()
    return len(ids)


def replay(db: Session, scenario: Scenario, settings: Settings) -> Ask:
    rec = scenario.data["recorded"]
    embedder = RecordedEmbedder(rec["embed_model"], scenario.question, rec["query_vector"])
    with _lock(scenario.question):  # by question: two scenarios may share one (u07 with prompts v1 and v2)
        clear_earlier_replays(db, scenario)
        return ask_question(db, embedder, RecordedLLM(rec), replay_settings(settings, rec), scenario.question,
                            k=rec["k"])
