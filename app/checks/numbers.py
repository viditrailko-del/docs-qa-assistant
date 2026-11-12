"""Never trust LLM numbers: every number in a statement must appear in a source that statement
cites (its text or section heading).

Numbers are compared by value, so "$1,500", "1500" and "1,500.00" are the same number. A number
the model computed (a sum, a conversion) is flagged on purpose: arithmetic is the code's job,
not the model's. Numbers written as words ("sixteen") are not checked.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation

from app.checks.base import Flag, Severity, SourceText
from app.schemas.answer import LLMAnswer

# A number not glued to a letter on its left (so "S1", "401(k)"'s k-side, "v2" don't count),
# with optional thousands separators and decimals: 16, 1,500, 0.50, 2.79
_NUMBER = re.compile(r"(?<![A-Za-z\d.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![A-Za-z\d.,])\d+(?:\.\d+)?")


def numbers_in(text: str) -> set[Decimal]:
    out: set[Decimal] = set()
    for m in _NUMBER.finditer(text):
        try:
            out.add(Decimal(m.group().replace(",", "")).normalize())
        except InvalidOperation:  # pragma: no cover - the regex only matches valid numbers
            continue
    return out


def check_numbers(answer: LLMAnswer, sources: Sequence[SourceText]) -> list[Flag]:
    by_label = {s.label: s for s in sources}
    flags: list[Flag] = []
    for i, statement in enumerate(answer.statements):
        cited = [by_label[c.source.strip()] for c in statement.citations if c.source.strip() in by_label]
        if not cited:
            continue  # nothing to compare against; check_citations already flags unknown labels
        supported: set[Decimal] = set()
        for src in cited:
            supported |= numbers_in(src.text) | numbers_in(src.heading_path)
        missing = sorted(numbers_in(statement.text) - supported)
        if missing:
            shown = ", ".join(format(n, "f") for n in missing)
            flags.append(Flag("number_unsupported", Severity.error,
                              f"{shown} not found in the cited source(s) "
                              f"{', '.join(s.label for s in cited)}", i))
    return flags
