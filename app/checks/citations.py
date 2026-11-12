"""Is every citation real? The label must name a source the model was shown, and the quote
must actually be in that source.

Why this matters: a model can produce a fluent, plausible "quote" that exists nowhere. A citation
we can't find is worse than no citation, because it makes an invented answer look checked.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.checks.base import Flag, Severity, SourceText, normalize
from app.schemas.answer import LLMAnswer


def check_citations(answer: LLMAnswer, sources: Sequence[SourceText]) -> list[Flag]:
    by_label = {s.label: s for s in sources}
    flags: list[Flag] = []
    for i, statement in enumerate(answer.statements):
        for j, cit in enumerate(statement.citations):
            label = cit.source.strip()
            src = by_label.get(label)
            if src is None:
                flags.append(Flag("unknown_source", Severity.error,
                                  f"cites {label}, but the model was only shown "
                                  f"{', '.join(by_label) or 'no sources'}", i, j))
                continue
            if normalize(cit.quote) not in normalize(src.text):
                hint = " (contains '...': passages were joined)" if "..." in cit.quote or "…" in cit.quote else ""
                flags.append(Flag("quote_not_in_source", Severity.error,
                                  f"quote not found in {label} ({src.path}){hint}: \"{cit.quote[:120]}\"", i, j))
    return flags
