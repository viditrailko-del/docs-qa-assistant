"""What the LLM must return. This Pydantic model is the single source of truth: its JSON schema
is sent to Gemini as the response schema, and the same model validates the reply.

Design: the answer is a list of *statements*, each backed by citations with an exact quote
from a numbered source ([S1], [S2]...). Statement-level citations let Phase 4 verify every
sentence on its own, instead of trusting a paragraph with a few footnotes at the end.

Constraints live in validators, not in schema keywords (minItems, pattern...): Gemini supports
only part of JSON Schema, and the validation retry tells the model exactly what was wrong.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, model_validator

SOURCE_LABEL = re.compile(r"^S\d+$")


class Citation(BaseModel):
    source: str = Field(description="Label of the source that supports the statement, e.g. S2")
    quote: str = Field(
        description="Words copied EXACTLY, character for character, from that source: one contiguous "
        "passage of at most ~30 words that supports the statement. No paraphrase, no ellipsis."
    )


class Statement(BaseModel):
    text: str = Field(description="One sentence of the answer, in your own words")
    citations: list[Citation] = Field(description="At least one source supporting this sentence")


class LLMAnswer(BaseModel):
    answerable: bool = Field(
        description="true only if the sources contain what is needed to answer the question"
    )
    statements: list[Statement] = Field(
        description="The answer, one sentence per item, each with citations. Empty if not answerable."
    )
    refusal_reason: str | None = Field(
        default=None,
        description="If not answerable: what the sources lack (and any related link they mention). Else null.",
    )
    sources_disagree: bool = Field(
        default=False,
        description="true if two sources give different values or rules for the same thing; then state "
        "both, each with its own citation",
    )

    @model_validator(mode="after")
    def _consistent(self) -> LLMAnswer:
        problems = []
        if self.answerable and not self.statements:
            problems.append("answerable is true but statements is empty")
        if not self.answerable:
            if self.statements:
                problems.append("answerable is false but statements is not empty")
            if not (self.refusal_reason or "").strip():
                problems.append("answerable is false but refusal_reason is empty")
        for i, s in enumerate(self.statements):
            if not s.text.strip():
                problems.append(f"statements[{i}].text is empty")
            if not s.citations:
                problems.append(f"statements[{i}] has no citations")
            for j, c in enumerate(s.citations):
                if not SOURCE_LABEL.match(c.source.strip()):
                    problems.append(f"statements[{i}].citations[{j}].source must look like S1, S2...")
                if not c.quote.strip():
                    problems.append(f"statements[{i}].citations[{j}].quote is empty")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.statements)
