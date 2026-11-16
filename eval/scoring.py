"""Answer eval scoring. Pure functions over frozen answers: no DB, no network, no LLM.

Each question ends in exactly one OUTCOME:

    answerable question                     unanswerable question
    -------------------                     ---------------------
    correct          right, no flags        correct_refusal   refused
    correct_flagged  right, but flagged     confident_wrong   answered, no flags   <- headline
    confident_wrong  wrong, no flags  <-    wrong_flagged     answered, flagged (checks caught it)
    wrong_flagged    wrong, flagged         failed
    over_refusal     refused
    failed           LLM never produced a valid answer

The headline is **confident_wrong**: a wrong answer shown to the user as verified. A flagged
wrong answer is bad but honest (marked "could not verify"); a refusal is never wrong-and-
confident. That is the whole point of the reliability layer, so it is what we measure first.

Correctness ("verdict") is deterministic: every `key_fact` of the label must appear in the
answer text (numbers compared by value, words case/spacing/hyphen-insensitive). Short facts
like "no" or "not" are weak evidence, and a correct answer can phrase a fact differently, so a
human verdict in the run's judgments.json always wins over the automatic one.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from app.checks.base import SourceText, normalize
from app.checks.numbers import numbers_in
from app.checks.runner import run_checks
from app.ingestion.markdown import clean_markdown
from app.schemas.answer import LLMAnswer
from scripts.check_questions import normalize as label_normalize

ANSWERED = {"answered", "answered_with_flags"}
VERDICTS = ("correct", "partial", "wrong")
OUTCOMES = ("correct", "correct_flagged", "confident_wrong", "wrong_flagged",
            "over_refusal", "correct_refusal", "failed")
CITATION_ERRORS = {"quote_not_in_source", "unknown_source"}

_HYPHENS = re.compile(r"[-‐‑–—/]")
_WORD = re.compile(r"[a-z]+")


# --- key facts -------------------------------------------------------------------


_NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
_NUMBER_WORD = re.compile(rf"\b({'|'.join(_NUMBER_WORDS)})\b", re.IGNORECASE)
# A phrase may end in a common inflection: "comply" matches "complying", but "no" never "not".
_SUFFIX = r"(?:s|es|d|ed|ing)?(?!\w)"


def _digits(s: str) -> str:
    """'five business days' -> '5 business days' (applied to fact AND answer, so it's symmetric)."""
    return _NUMBER_WORD.sub(lambda m: _NUMBER_WORDS[m.group().lower()], s)


def _fact_text(s: str) -> str:
    return " ".join(_HYPHENS.sub(" ", normalize(_digits(s))).split())


def fact_found(fact: str, answer_text: str) -> bool:
    """Does the answer state this key fact?

    Number words up to twelve count as digits ("six months" = "6 months"). Then:
    1. The fact as a phrase (word-bounded, ignoring case, spacing, hyphens, a plural/-ing/-ed
       ending): "four year" matches "four-year", "comply" matches "complying", "no" not "not".
    2. Otherwise, if the fact has numbers: every number appears by value ($1,500 = 1500 USD) and
       every word appears as a word prefix ("90 days" matches "90 calendar days" and "90-day").
    """
    fact_n, text_n = _fact_text(fact), _fact_text(answer_text)
    if re.search(rf"(?<!\w){re.escape(fact_n)}{_SUFFIX}", text_n):
        return True
    nums = numbers_in(_digits(fact))
    if not nums or not nums <= numbers_in(_digits(answer_text)):
        return False
    stems = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _WORD.findall(fact_n)]
    return all(re.search(rf"\b{re.escape(stem)}", text_n) for stem in stems)


def auto_verdict(key_facts: Sequence[str], answer_text: str) -> tuple[str, list[str]]:
    """-> (correct | partial | wrong, missing facts). No key facts (unanswerable) -> wrong:
    the only right reply to an unanswerable question is a refusal."""
    missing = [f for f in key_facts if not fact_found(f, answer_text)]
    if not key_facts or len(missing) == len(key_facts):
        return "wrong", missing
    return ("partial" if missing else "correct"), missing


def outcome_for(expect: str, status: str, verdict: str | None, flagged: bool) -> str:
    if status == "failed":
        return "failed"
    if status == "refused":
        return "correct_refusal" if expect == "refuse" else "over_refusal"
    if verdict == "correct":
        return "correct_flagged" if flagged else "correct"
    return "wrong_flagged" if flagged else "confident_wrong"


# --- evidence (did the LLM see / cite the labeled passage?) -------------------------


def evidence_labels(quotes: Sequence[str], sources: Sequence[SourceText]) -> list[set[str]]:
    """For each label quote, the source labels (S1...) whose chunk text contains it. Same
    normalization as the retrieval eval, so "shown" here = "retrieved" there."""
    texts = {s.label: label_normalize(s.text) for s in sources}
    return [{lab for lab, t in texts.items() if label_normalize(clean_markdown(q)) in t} for q in quotes]


# --- one question ----------------------------------------------------------------


@dataclass
class QuestionScore:
    id: str
    type: str
    expect: str
    question: str
    status: str  # recomputed: checks are re-run on the frozen answer
    stored_status: str  # what the run itself produced
    outcome: str
    verdict: str | None = None  # correct | partial | wrong, None if not answered
    verdict_by: str | None = None  # "auto" or the human's name
    missing_facts: list[str] = field(default_factory=list)
    answer_text: str = ""
    refusal_reason: str | None = None
    flags: list[str] = field(default_factory=list)  # check codes
    citations: int = 0
    valid_citations: int = 0
    evidence_shown: bool | None = None  # any label quote in the sources shown (None: unanswerable)
    evidence_full: bool | None = None  # every label quote in the sources shown
    evidence_cited: bool | None = None  # the answer cites at least one evidence source
    sources_disagree: bool = False
    answered_by: str | None = None
    calls: int = 0
    llm_latency_ms: int = 0
    cost_usd: Decimal | None = None


def score_question(q: dict, raw: dict, judgment: dict | None = None, *, min_similarity: float = 0.0) -> QuestionScore:
    """q: one entry of questions.json. raw: one frozen result from `eval.run answer`.
    judgment: optional human verdict {"verdict": "correct"|"partial"|"wrong", "by": "..."}."""
    sources = [SourceText(label=s["label"], text=s["text"], heading_path=s["heading_path"], title=s["title"],
                          path=s["path"], similarity=s["similarity"]) for s in raw["sources"]]
    answer = LLMAnswer.model_validate(raw["answer"]) if raw.get("answer") else None

    # Re-run the (pure) checks so a check change is measured without new LLM calls.
    flags = run_checks(q["question"], answer, sources, min_similarity=min_similarity) if answer else []
    if raw["status"] == "failed" or answer is None:
        status = "failed" if raw["status"] == "failed" else raw["status"]  # no-source refusal
    elif not answer.answerable:
        status = "refused"
    else:
        status = "answered_with_flags" if flags else "answered"

    s = QuestionScore(id=q["id"], type=q["type"], expect=q["expect"], question=q["question"], status=status,
                      stored_status=raw["status"], outcome="failed", flags=sorted({f.code for f in flags}),
                      answered_by=raw.get("answered_by"))
    s.refusal_reason = answer.refusal_reason if answer and not answer.answerable else None
    s.sources_disagree = bool(answer and answer.sources_disagree)

    if status in ANSWERED:
        s.answer_text = answer.text
        s.verdict, s.missing_facts = auto_verdict(q.get("key_facts", []), answer.text)
        s.verdict_by = "auto"
        if judgment:
            if judgment.get("verdict") not in VERDICTS:
                raise ValueError(f"{q['id']}: judgment verdict must be one of {VERDICTS}")
            s.verdict, s.verdict_by = judgment["verdict"], judgment.get("by") or "human"
        bad = {(f.statement, f.citation) for f in flags if f.code in CITATION_ERRORS}
        cites = [(i, j, c) for i, st in enumerate(answer.statements) for j, c in enumerate(st.citations)]
        s.citations = len(cites)
        s.valid_citations = sum((i, j) not in bad for i, j, _ in cites)

    quotes = [src["quote"] for src in q.get("sources", [])]
    if quotes:
        found = evidence_labels(quotes, sources)
        s.evidence_shown = any(found)
        s.evidence_full = all(found)
        if status in ANSWERED:
            cited = {c.source.strip() for st in answer.statements for c in st.citations}
            s.evidence_cited = bool(cited & set().union(*found))

    s.outcome = outcome_for(q["expect"], status, s.verdict, bool(flags))
    _add_costs(s, raw.get("attempts", []))
    return s


def _add_costs(s: QuestionScore, attempts: list[dict]) -> None:
    s.calls = len(attempts)
    s.llm_latency_ms = sum(a.get("latency_ms") or 0 for a in attempts)
    ok = [a for a in attempts if a.get("raw_response") is not None]
    costs = [Decimal(str(a["cost_usd"])) for a in ok if a.get("cost_usd") is not None]
    # "unknown" (None) is not "free" (0): only sum when every successful call had a price.
    s.cost_usd = sum(costs, Decimal(0)) if ok and len(costs) == len(ok) else None


# --- the whole run -----------------------------------------------------------------


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def summarize(scores: Sequence[QuestionScore]) -> dict:
    n = len(scores)
    out = Counter(s.outcome for s in scores)
    answerable = [s for s in scores if s.expect != "refuse"]
    unanswerable = [s for s in scores if s.expect == "refuse"]
    scored = [s for s in scores if s.outcome != "failed"]
    answered = [s for s in scores if s.status in ANSWERED]
    wrong = out["confident_wrong"] + out["wrong_flagged"]
    flagged = out["wrong_flagged"] + out["correct_flagged"]
    cites = sum(s.citations for s in answered)
    a_answered = [s for s in answered if s.expect != "refuse"]
    latencies = [s.llm_latency_ms for s in scores if s.calls]
    costs = [s.cost_usd for s in scores if s.cost_usd is not None]

    def outcome_counts(group: Sequence[QuestionScore]) -> dict:
        c = Counter(s.outcome for s in group)
        return {"n": len(group), **{o: c[o] for o in OUTCOMES if c[o]}}

    return {
        "n": n,
        "outcomes": {o: out[o] for o in OUTCOMES},
        # Headline: share of (non-failed) questions where a wrong answer was shown as verified.
        "confident_wrong": {"count": out["confident_wrong"], "rate": _rate(out["confident_wrong"], len(scored)),
                            "ids": sorted(s.id for s in scores if s.outcome == "confident_wrong")},
        "answerable": {
            "n": len(answerable),
            "correct": _rate(sum(s.verdict == "correct" for s in answerable), len(answerable)),
            "correct_unflagged": _rate(out["correct"], len(answerable)),
            "over_refusal": _rate(out["over_refusal"], len(answerable)),
            "over_refusal_ids": sorted(s.id for s in answerable if s.outcome == "over_refusal"),
            "partial_ids": sorted(s.id for s in answerable if s.verdict == "partial"),
        },
        "unanswerable": {
            "n": len(unanswerable),
            "correct_refusal": _rate(out["correct_refusal"], len(unanswerable)),
            "answered_ids": sorted(s.id for s in unanswerable if s.status in ANSWERED),
        },
        "flags": {
            # Of the wrong answers, how many did the checks catch? Of the flagged ones, how many were wrong?
            "catch_rate": _rate(out["wrong_flagged"], wrong),
            "precision": _rate(out["wrong_flagged"], flagged),
            "by_code": dict(Counter(code for s in answered for code in s.flags).most_common()),
        },
        "citations": {
            "total": cites,
            "valid_rate": _rate(sum(s.valid_citations for s in answered), cites),
            # Right answer from the wrong evidence (q30 smoke test) shows up here.
            "cites_labeled_evidence": _rate(sum(bool(s.evidence_cited) for s in a_answered), len(a_answered)),
        },
        "retrieval": {
            "evidence_shown": _rate(sum(bool(s.evidence_shown) for s in answerable), len(answerable)),
            "all_evidence_shown": _rate(sum(bool(s.evidence_full) for s in answerable), len(answerable)),
        },
        "conflict_disagree_reported": _rate(sum(s.sources_disagree for s in scores if s.type == "conflict" and s.status in ANSWERED),
                                            sum(s.type == "conflict" and s.status in ANSWERED for s in scores)),
        "by_type": {t: outcome_counts([s for s in scores if s.type == t]) for t in sorted({s.type for s in scores})},
        "answered_by": dict(Counter(s.answered_by or "-" for s in scores)),
        # Auto verdicts worth a human look: facts missing (maybe just worded differently), or facts
        # present but the labeled evidence not cited (maybe right from the wrong passage, like q30).
        "to_review": sorted(s.id for s in answered if s.verdict_by == "auto"
                            and (s.verdict != "correct" or s.evidence_cited is False)),
        "cost": {
            "total_usd": str(sum(costs, Decimal(0))) if costs else None,
            "per_question_usd": str((sum(costs, Decimal(0)) / len(costs)).quantize(Decimal("0.000001"))) if costs else None,
            "unknown": n - len(costs),
        },
        "latency_ms": {
            "mean": round(statistics.mean(latencies)) if latencies else None,
            "median": round(statistics.median(latencies)) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "llm_calls": sum(s.calls for s in scores),
        "failed": out["failed"],
    }
