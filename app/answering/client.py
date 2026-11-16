"""Talking to the LLM. This file knows about Gemini; nothing else in the app does.

The rest of the app depends only on the `LLMClient` protocol, so tests plug in a fake and
never call the real API (rule: tests never hit Gemini). Adapted from the invoice project.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import httpx
from google import genai
from google.genai import errors, types

from app.config import get_settings

TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}  # rate limit, server errors, overloaded

# A 429 is either "too many per MINUTE" (retry soon) or "too many per DAY" (this model is done
# until the quota resets). Retrying a daily quota only burns time: go straight to the fallback.
DAILY_QUOTA = re.compile(r"per day|PerDay", re.IGNORECASE)


@dataclass
class LLMResponse:
    text: str | None
    model: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None  # billed as output, but not included in output_tokens


class LLMError(Exception):
    """A failed LLM call. `transient=True` means retrying might work (overload, timeout)."""

    def __init__(self, message: str, *, transient: bool, latency_ms: int = 0):
        super().__init__(message)
        self.transient = transient
        self.latency_ms = latency_ms


class LLMClient(Protocol):
    def generate(self, *, prompt: str, schema: dict[str, Any], model: str) -> LLMResponse: ...


class GeminiClient:
    def __init__(self, api_key: str, *, timeout_s: int = 60, thinking_level: str = "low"):
        # A missing key must not become an HTTP 500: every call fails as a permanent LLMError
        # and the question ends `failed`.
        self._client = (
            genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_s * 1000))
            if api_key
            else None
        )
        self._thinking_level = thinking_level

    def generate(self, *, prompt: str, schema: dict[str, Any], model: str) -> LLMResponse:
        if self._client is None:
            raise LLMError("GEMINI_API_KEY is not set", transient=False)
        start = time.monotonic()
        try:
            resp = self._client.models.generate_content(
                model=model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=schema,
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_level=self._thinking_level),
                ),
            )
        except errors.APIError as e:
            raise LLMError(f"{e.code}: {str(e)[:500]}", transient=is_transient(e.code, str(e)),
                           latency_ms=_ms_since(start)) from e
        except httpx.TimeoutException as e:
            raise LLMError("timeout", transient=True, latency_ms=_ms_since(start)) from e
        except httpx.TransportError as e:
            # Network down, DNS failure, connection reset (found by the eval: a DNS outage
            # crashed 13 questions, and would have been an HTTP 500 in the API). Retryable.
            raise LLMError(f"network: {type(e).__name__}: {e}"[:500], transient=True,
                           latency_ms=_ms_since(start)) from e

        usage = resp.usage_metadata
        return LLMResponse(
            text=resp.text,
            model=model,
            latency_ms=_ms_since(start),
            input_tokens=usage.prompt_token_count if usage else None,
            output_tokens=usage.candidates_token_count if usage else None,
            thinking_tokens=usage.thoughts_token_count if usage else None,
        )


def is_transient(code: int, message: str) -> bool:
    """Could the SAME model succeed if we retry it soon?"""
    if code == 429 and DAILY_QUOTA.search(message):
        return False
    return code in TRANSIENT_CODES


def _ms_since(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


@lru_cache
def get_llm() -> LLMClient:
    """FastAPI dependency; tests override it with a fake."""
    s = get_settings()
    return GeminiClient(s.gemini_api_key, timeout_s=s.gemini_timeout_s, thinking_level=s.gemini_thinking_level)
