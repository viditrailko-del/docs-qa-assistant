"""Shared types for the verification checks.

Checks are pure functions: (question, answer, sources) -> list[Flag]. No DB, no network, no LLM.
They never fix an answer; they say what could not be verified, and where.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from enum import StrEnum


class Severity(StrEnum):
    error = "error"  # the answer says something its sources don't support
    warning = "warning"  # supported, but possibly misleading (scope, weak retrieval)


@dataclass(frozen=True)
class Flag:
    code: str  # check id, e.g. "quote_not_in_source"
    severity: Severity
    reason: str  # human-readable, specific
    statement: int | None = None  # index into answer.statements
    citation: int | None = None  # index into that statement's citations

    def to_dict(self) -> dict:
        return {k: (v.value if isinstance(v, Severity) else v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class SourceText:
    """What a check needs to know about one retrieved chunk the LLM saw."""

    label: str  # S1, S2...
    text: str
    heading_path: str
    title: str | None
    path: str
    similarity: float


_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', " ": " ",
                         "–": "-", "—": "-"})
_MARKUP = re.compile(r"\*\*|__|`")


def normalize(text: str) -> str:
    """Comparable text: same characters regardless of curly/straight quotes, dashes, Markdown
    bold/code markers, case and runs of whitespace. Nothing else is forgiven."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = _MARKUP.sub("", text)
    return " ".join(text.split()).lower()
