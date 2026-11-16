"""Answer eval: key-fact matching, outcomes, metrics, and the freeze -> score round trip.

No DB, no real LLM: answers come from the fake LLM or are written by hand."""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from app.answering.core import answer_question
from eval.run import MemoryRecorder, freeze, out_of_quota
from eval.scoring import auto_verdict, fact_found, outcome_for, score_question, summarize
from tests.conftest import REFUSAL_JSON, test_settings
from tests.test_answering import hit

BOOKS = "Books and e-books are reimbursable. Limit is set to $60/year."
Q = {"id": "q13", "type": "simple", "expect": "answer", "question": "How much can I spend on books per year?",
     "key_facts": ["$60"], "sources": [{"path": "finance/expenses.md", "quote": "Limit is set to $60/year."}]}
U = {"id": "u02", "type": "unanswerable", "expect": "refuse", "question": "What day is payday?",
     "key_facts": [], "sources": []}


def answer_json(text: str, quote: str = "Limit is set to $60/year.", source: str = "S1") -> str:
    return json.dumps({"answerable": True, "sources_disagree": False, "refusal_reason": None,
                       "statements": [{"text": text, "citations": [{"source": source, "quote": quote}]}]})


def frozen(fake_llm, reply: str | None, texts=(BOOKS,), question=Q) -> dict:
    """Run the real answering core with a scripted reply and freeze it like `eval.run answer`."""
    if reply is not None:
        fake_llm.replies = [reply]
    hits = [hit(i, t) for i, t in enumerate(texts, start=1)]
    rec = MemoryRecorder()
    outcome = answer_question(question["question"], hits, fake_llm, test_settings(), rec)
    return json.loads(json.dumps(freeze(question["id"], hits, outcome, rec.attempts, None, 10), default=str))


# --- key facts --------------------------------------------------------------------


@pytest.mark.parametrize(("fact", "text", "found"), [
    ("$1,500", "up to 1500 USD per year", True),  # numbers by value
    ("$80", "the limit is $800", False),
    ("90 days", "within 90 calendar days", True),  # words as prefixes, any order around numbers
    ("90 days", "within a 90-day window", True),
    ("90 days", "within 9 days", False),
    ("16 weeks", "16 days of leave", False),
    ("four year", "a four-year vesting schedule", True),  # hyphens = spaces
    ("no", "you are not eligible", False),  # word-bounded: "no" is not "not"
    ("no", "No, you are not eligible yet.", True),
    ("not eligible", "You are NOT   eligible.", True),
    ("50+", "teams of 50+ people", True),
    ("HealthKick", "via healthkick discounts", True),
    ("comply", "GitLab recommends complying with the request", True),  # inflected ending
    ("5 business days", "within five business days", True),  # number words
    ("6 months", "six months in role", True),
    ("6 months", "sixteen months", False),
    ("one year", "a 1-year cliff", True),
])
def test_fact_found(fact, text, found):
    assert fact_found(fact, text) is found


def test_auto_verdict():
    assert auto_verdict(["$100", "$80"], "$100 on the card, $80 reimbursed") == ("correct", [])
    assert auto_verdict(["$100", "$80"], "$100 per person") == ("partial", ["$80"])
    assert auto_verdict(["$100"], "no limit") == ("wrong", ["$100"])
    assert auto_verdict([], "payday is the 25th") == ("wrong", [])  # unanswerable: any answer is wrong


@pytest.mark.parametrize(("expect", "status", "verdict", "flagged", "outcome"), [
    ("answer", "answered", "correct", False, "correct"),
    ("answer", "answered_with_flags", "correct", True, "correct_flagged"),
    ("answer", "answered", "partial", False, "confident_wrong"),  # incomplete counts as wrong
    ("answer", "answered_with_flags", "wrong", True, "wrong_flagged"),
    ("answer", "refused", None, False, "over_refusal"),
    ("answer_both", "failed", None, False, "failed"),
    ("refuse", "refused", None, False, "correct_refusal"),
    ("refuse", "answered", "wrong", False, "confident_wrong"),
    ("refuse", "answered_with_flags", "wrong", True, "wrong_flagged"),
])
def test_outcome_for(expect, status, verdict, flagged, outcome):
    assert outcome_for(expect, status, verdict, flagged) == outcome


# --- one question, through the real answering core -----------------------------------


def test_correct_clean_answer(fake_llm):
    s = score_question(Q, frozen(fake_llm, answer_json("Books are limited to $60 a year.")))
    assert (s.outcome, s.verdict, s.verdict_by) == ("correct", "correct", "auto")
    assert (s.citations, s.valid_citations) == (1, 1)
    assert (s.evidence_shown, s.evidence_full, s.evidence_cited) == (True, True, True)
    assert s.answered_by == "gemini-3.5-flash" and s.cost_usd == Decimal("0.002250")


def test_wrong_answer_with_valid_quote_is_confident_wrong(fake_llm):
    # Exact quote and a number that IS in the source, so every check passes; but the label wants
    # $50. This is the failure mode the checks can't see: the headline metric.
    q = {**Q, "key_facts": ["$50"]}
    s = score_question(q, frozen(fake_llm, answer_json("Books are limited to $60 a year.")))
    assert s.outcome == "confident_wrong"
    assert s.missing_facts == ["$50"]


def test_invented_quote_is_caught(fake_llm):
    raw = frozen(fake_llm, answer_json("Books are limited to $75 a year.", quote="Limit is $75."))
    s = score_question(Q, raw)
    assert s.outcome == "wrong_flagged"
    assert s.flags == ["number_unsupported", "quote_not_in_source"]
    assert (s.citations, s.valid_citations) == (1, 0)


def test_right_answer_from_wrong_evidence(fake_llm):
    other = "Learning budget: books are covered, limit is set to $60/year for courses too."
    raw = frozen(fake_llm, answer_json("Books: $60 a year.", quote="limit is set to $60/year", source="S2"),
                 texts=(BOOKS, other))
    s = score_question(Q, raw)
    assert s.outcome == "correct" and s.evidence_shown is True and s.evidence_cited is False
    assert summarize([s])["to_review"] == ["q13"]  # auto-correct, but worth a human look


def test_refusals_and_failures(fake_llm):
    assert score_question(Q, frozen(fake_llm, REFUSAL_JSON)).outcome == "over_refusal"
    assert score_question(U, frozen(fake_llm, REFUSAL_JSON, question=U)).outcome == "correct_refusal"
    assert score_question(U, frozen(fake_llm, None, texts=(), question=U)).outcome == "correct_refusal"  # no sources
    fake_llm.replies = ["not json", "still not json"]
    s = score_question(Q, frozen(fake_llm, None))
    assert s.outcome == "failed" and s.calls == 2


def test_answered_unanswerable_is_confident_wrong(fake_llm):
    s = score_question(U, frozen(fake_llm, answer_json("Books are limited to $60 a year."), question=U))
    assert s.outcome == "confident_wrong" and s.evidence_shown is None


def test_human_judgment_overrides_auto_verdict(fake_llm):
    q = {**Q, "key_facts": ["sixty dollars"]}
    raw = frozen(fake_llm, answer_json("Books are limited to $60 a year."))
    assert score_question(q, raw).outcome == "confident_wrong"
    s = score_question(q, raw, {"verdict": "correct", "by": "Vidit"})
    assert (s.outcome, s.verdict_by) == ("correct", "Vidit")
    with pytest.raises(ValueError):
        score_question(q, raw, {"verdict": "fine"})


def test_checks_are_rerun_on_frozen_answers(fake_llm):
    raw = frozen(fake_llm, answer_json("Books are limited to $60 a year."))
    raw["sources"][0]["text"] = "Something else entirely."  # as if a check (or source) had changed
    s = score_question(Q, raw)
    assert (s.stored_status, s.status) == ("answered", "answered_with_flags")
    assert s.flags == ["number_unsupported", "quote_not_in_source"]
    assert s.outcome == "correct_flagged"  # the verdict is about the answer text, which still says $60


# --- whole run -------------------------------------------------------------------------


def test_summarize(fake_llm):
    scores = [
        score_question(Q, frozen(fake_llm, answer_json("Books are limited to $60 a year."))),  # correct
        score_question({**Q, "id": "q99", "key_facts": ["$50"]},
                       frozen(fake_llm, answer_json("Books are limited to $60 a year."))),  # confident wrong
        score_question(Q, frozen(fake_llm, answer_json("It is $75.", quote="Limit is $75."))),  # wrong, flagged
        score_question(U, frozen(fake_llm, REFUSAL_JSON, question=U)),  # correct refusal
    ]
    s = summarize(scores)
    assert s["confident_wrong"] == {"count": 1, "rate": 0.25, "ids": ["q99"]}
    assert s["answerable"]["correct"] == round(1 / 3, 3)
    assert s["unanswerable"]["correct_refusal"] == 1.0
    assert s["flags"]["catch_rate"] == 0.5 and s["flags"]["precision"] == 1.0
    assert s["citations"] == {"total": 3, "valid_rate": round(2 / 3, 3), "cites_labeled_evidence": 1.0}
    assert s["to_review"] == ["q13", "q99"]
    assert s["llm_calls"] == 4 and s["failed"] == 0


def test_judgments_carry_over_only_to_identical_answers(fake_llm, tmp_path):
    from eval.run import carry_judgments, write_json

    def make_run(name: str, answers: dict[str, str]) -> Path:
        run = tmp_path / name
        write_json(run / "manifest.json", {"questions": list(answers), "min_similarity": 0.0})
        for qid, text in answers.items():
            write_json(run / "raw" / f"{qid}.json", frozen(fake_llm, answer_json(text), question={**Q, "id": qid}))
        return run

    old = make_run("old", {"q13": "Books are limited to $60 a year.", "q12": "Books are limited to $60 a year."})
    write_json(old / "judgments.json", {"q13": {"verdict": "correct", "by": "Vidit"},
                                        "q12": {"verdict": "wrong", "by": "Vidit"}})
    new = make_run("new", {"q13": "Books are limited to $60 a year.", "q12": "Up to $60 a year for books."})
    assert carry_judgments(old, new) == ["q13"]  # q12's answer changed: it needs a fresh human look
    carried = json.loads((new / "judgments.json").read_text(encoding="utf-8"))
    assert carried == {"q13": {"verdict": "correct", "by": "Vidit", "carried_from": "old"}}


def test_out_of_quota():
    quota = {"status": "failed", "attempts": [{"error": "429: Quota exceeded ... PerDay limit"}]}
    assert out_of_quota(quota)
    assert not out_of_quota({**quota, "attempts": [{"error": "503: overloaded"}]})
    assert not out_of_quota({"status": "answered", "attempts": quota["attempts"]})
