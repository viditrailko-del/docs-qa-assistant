"""Answering: the no-DB core (retries, fallback, validation) and POST /ask, with a fake LLM."""

import json
import uuid
from decimal import Decimal

import pytest

from app.answering.client import LLMError, is_transient
from app.answering.core import AnswerStatus, answer_question, build_prompt, response_schema
from app.answering.pricing import cost_usd
from app.retrieval.search import Hit
from app.schemas.answer import LLMAnswer
from tests.conftest import ANSWER_JSON, REFUSAL_JSON, test_settings


def hit(n: int, text: str = "Limit is set to $60/year.") -> Hit:
    return Hit(rank=n, chunk_id=100 + n, document_id=uuid.uuid4(), path="finance/expenses.md",
               title="Expenses", source_url="https://h.example/expenses", heading_path="Expenses > Books",
               text=text, similarity=0.8, score=0.8)


class ListRecorder:
    def __init__(self):
        self.events = []

    def record(self, a):
        self.events.append(("record", a.model, a.raw_response, a.error))

    def update(self, a):
        self.events.append(("update", a.model, a.raw_response, a.error))


def run(fake_llm, hits=None, **settings):
    rec, sleeps = ListRecorder(), []
    out = answer_question("How much for books?", [hit(1)] if hits is None else hits, fake_llm,
                          test_settings(**settings), rec, sleep=sleeps.append)
    return out, rec, sleeps


# --- core -------------------------------------------------------------------------------------

def test_answer_is_validated_and_recorded_before_parsing(fake_llm):
    out, rec, _ = run(fake_llm)
    assert out.status is AnswerStatus.answered
    assert out.answer.statements[0].citations[0].quote == "Limit is set to $60/year."
    assert [s.label for s in out.sources] == ["S1"]
    assert rec.events == [("record", "gemini-3.5-flash", ANSWER_JSON, None)]
    assert out.attempts[0].cost_usd == Decimal("0.002250")  # 1000 in x 1.50 + 100 out x 7.50, per 1M


def test_model_refusal(fake_llm):
    fake_llm.replies = [REFUSAL_JSON]
    out, _, _ = run(fake_llm)
    assert out.status is AnswerStatus.refused
    assert out.refusal_reason == "The sources do not say when salary is paid."


def test_no_sources_refuses_without_calling_the_llm(fake_llm):
    out, rec, _ = run(fake_llm, hits=[])
    assert out.status is AnswerStatus.refused and out.refusal_reason
    assert fake_llm.calls == [] and rec.events == []


@pytest.mark.parametrize("bad", [
    "not json at all",
    '{"answerable": true, "statements": [], "sources_disagree": false}',  # answerable but empty
    '{"answerable": false, "statements": [], "refusal_reason": ""}',  # refusal without a reason
    '{"answerable": true, "statements": [{"text": "x", "citations": []}]}',  # uncited statement
    '{"answerable": true, "statements": [{"text": "x", "citations": [{"source": "doc 1", "quote": "q"}]}]}',
])
def test_invalid_output_is_retried_once_with_the_errors(fake_llm, bad):
    fake_llm.replies = [bad, ANSWER_JSON]
    out, rec, _ = run(fake_llm)
    assert out.status is AnswerStatus.answered
    assert len(fake_llm.calls) == 2
    assert "rejected" in fake_llm.calls[1]["prompt"] and "rejected" not in fake_llm.calls[0]["prompt"]
    assert rec.events[0][2] == bad  # the bad raw output is kept...
    assert rec.events[1][0] == "update" and rec.events[1][3].startswith("validation:")  # ...and marked


def test_invalid_output_twice_fails(fake_llm):
    fake_llm.replies = ["{}", "{}"]
    out, _, _ = run(fake_llm)
    assert out.status is AnswerStatus.failed and out.answer is None and len(out.attempts) == 2


def test_transient_errors_retry_with_backoff_then_fall_back(fake_llm):
    fake_llm.replies = [LLMError("503", transient=True)] * 3 + [ANSWER_JSON]
    out, _, sleeps = run(fake_llm, llm_retry_backoff_s=2.0)
    assert out.status is AnswerStatus.answered
    assert [c["model"] for c in fake_llm.calls] == ["gemini-3.5-flash"] * 3 + ["gemini-3.5-flash-lite"]
    assert sleeps == [2.0, 4.0]


def test_permanent_error_skips_to_fallback(fake_llm):
    fake_llm.replies = [LLMError("429 per day", transient=False), ANSWER_JSON]
    out, _, sleeps = run(fake_llm)
    assert out.status is AnswerStatus.answered and sleeps == []
    assert [c["model"] for c in fake_llm.calls] == ["gemini-3.5-flash", "gemini-3.5-flash-lite"]


def test_everything_down_fails_without_raising(fake_llm):
    fake_llm.replies = [LLMError("no key", transient=False)] * 2
    out, rec, _ = run(fake_llm)
    assert out.status is AnswerStatus.failed
    assert [e[3] for e in rec.events] == ["no key", "no key"]


def test_daily_quota_is_not_transient():
    assert not is_transient(429, "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel")
    assert is_transient(429, "Quota exceeded: GenerateRequestsPerMinute")
    assert is_transient(503, "overloaded") and not is_transient(400, "bad request")


def test_prompt_labels_sources_and_survives_braces():
    from app.answering.core import label_sources

    p = build_prompt("v2", "What about {weird} braces?", label_sources([hit(1), hit(2, "Use {{% x %}} {0}")]))
    assert "[S1] Expenses\nSection: Expenses > Books" in p and "[S2]" in p
    assert "Use {{% x %}} {0}" in p and p.rstrip().endswith("Question: What about {weird} braces?")
    assert "{sources}" not in p


def test_response_schema_avoids_unsupported_keywords():
    schema = json.dumps(response_schema())
    for kw in ('"pattern"', '"minItems"', '"minLength"'):
        assert kw not in schema
    assert {"answerable", "statements"} <= set(response_schema()["required"])


def test_answer_text_joins_statements():
    a = LLMAnswer.model_validate_json(ANSWER_JSON)
    assert a.text == "Books are limited to $60 a year."


def test_pricing_unknown_model_is_none_not_zero():
    assert cost_usd("some-new-model", 10, 10) is None
    assert cost_usd("gemini-3.5-flash-lite", 1_000_000, 0, thinking_tokens=1_000_000) == Decimal("2.800000")


# --- API --------------------------------------------------------------------------------------

DOC = b"""---
title: Expenses
---

## Books

Limit is set to $60/year.
"""


@pytest.fixture
def ingested(client):
    assert client.post("/documents", files={"file": ("expenses.md", DOC, "text/markdown")},
                       data={"source_url": "https://h.example/expenses"}).status_code == 201


def test_ask_answers_with_resolved_citations(client, ingested, fake_llm):
    r = client.post("/ask", json={"question": "How much can I spend on books?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered"
    assert body["answer"] == "Books are limited to $60 a year."
    cit = body["statements"][0]["citations"][0]
    assert cit["source"] == "S1" and cit["chunk_id"] == body["sources"][0]["chunk_id"]
    assert cit["source_url"] == "https://h.example/expenses"
    assert body["sources"][0]["heading_path"] == "Expenses > Books"
    assert body["llm_calls"][0]["model"] == "gemini-3.5-flash" and body["llm_calls"][0]["cost_usd"] is not None
    assert "Limit is set to $60/year." in fake_llm.calls[0]["prompt"]
    assert client.get(f"/ask/{body['id']}").json() == body  # stored, retrievable


def test_citation_to_unknown_label_is_kept_but_unresolved(client, ingested, fake_llm):
    fake_llm.replies = [ANSWER_JSON.replace('"S1"', '"S9"')]
    body = client.post("/ask", json={"question": "books?"}).json()
    cit = body["statements"][0]["citations"][0]
    assert cit["source"] == "S9" and cit["chunk_id"] is None
    assert body["status"] == "answered_with_flags"
    assert [f["code"] for f in body["statements"][0]["flags"]] == ["unknown_source"]


def test_clean_answer_is_verified(client, ingested):
    body = client.post("/ask", json={"question": "How much can I spend on books?"}).json()
    assert body["status"] == "answered" and body["flags"] == []
    assert body["statements"][0]["verified"] is True


def test_invented_quote_and_number_are_flagged_per_statement(client, ingested, fake_llm):
    fake_llm.replies = ["""{"answerable": true, "sources_disagree": false, "statements": [
      {"text": "Books are limited to $60 a year.", "citations": [{"source": "S1", "quote": "Limit is set to $60/year."}]},
      {"text": "Audiobooks get $90 extra.", "citations": [{"source": "S1", "quote": "Audiobooks get an extra $90."}]}]}"""]
    body = client.post("/ask", json={"question": "books?"}).json()
    assert body["status"] == "answered_with_flags"
    first, second = body["statements"]
    assert first["verified"] is True and first["flags"] == []
    assert second["verified"] is False
    assert sorted(f["code"] for f in second["flags"]) == ["number_unsupported", "quote_not_in_source"]
    assert client.get(f"/ask/{body['id']}").json()["flags"] == body["flags"]  # stored


def test_ask_refusal(client, ingested, fake_llm):
    fake_llm.replies = [REFUSAL_JSON]
    body = client.post("/ask", json={"question": "When is payday?"}).json()
    assert body["status"] == "refused" and body["statements"] == [] and body["answer"] is None
    assert body["refusal_reason"] == "The sources do not say when salary is paid."


def test_llm_outage_is_failed_not_500(client, ingested, fake_llm):
    fake_llm.replies = [LLMError("GEMINI_API_KEY is not set", transient=False)] * 2
    r = client.post("/ask", json={"question": "books?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failed" and [c["error"] for c in body["llm_calls"]] == ["GEMINI_API_KEY is not set"] * 2


def test_ask_with_no_documents_refuses_without_llm(client, fake_llm):
    body = client.post("/ask", json={"question": "anything at all?"}).json()
    assert body["status"] == "refused" and body["sources"] == [] and fake_llm.calls == []
    assert body["refusal_reason"] == "No documents matched the question."


def test_ask_validation_and_404(client):
    assert client.post("/ask", json={"question": "x"}).status_code == 422
    assert client.post("/ask", json={"question": "books?", "k": 0}).status_code == 422
    assert client.get(f"/ask/{uuid.uuid4()}").status_code == 404


# --- the real Gemini client's error mapping (no network: the SDK call is replaced) ----------

def _gemini_raising(exc):
    from app.answering.client import GeminiClient

    client = GeminiClient("fake-key")

    class Models:
        def generate_content(self, **kwargs):
            raise exc

    client._client = type("C", (), {"models": Models()})()
    return client


@pytest.mark.parametrize("exc", [
    __import__("httpx").ConnectError("[Errno -2] Name or service not known"),  # DNS outage (eval, 2026-11-16)
    __import__("httpx").ReadError("connection reset"),
    __import__("httpx").ReadTimeout("slow"),
    # google-genai 3.x uses httpx2, whose errors are not httpx subclasses (sister project, 2026-11-17)
    __import__("httpx2").RemoteProtocolError("Server disconnected without sending a response."),
    __import__("httpx2").ConnectError("dns"),
    __import__("httpx2").ReadTimeout("slow"),
])
def test_network_errors_become_transient_llm_errors(exc):
    with pytest.raises(LLMError) as info:
        _gemini_raising(exc).generate(prompt="p", schema={}, model="m")
    assert info.value.transient
    expected = "timeout" if "Timeout" in type(exc).__name__ else "network: "
    assert str(info.value).startswith(expected)


def test_unexpected_sdk_error_is_a_permanent_llm_error():
    with pytest.raises(LLMError) as info:
        _gemini_raising(ValueError("weird")).generate(prompt="p", schema={}, model="m")
    assert not info.value.transient
    assert str(info.value).startswith("unexpected: ValueError")


def test_network_outage_ends_failed_not_crash(client, ingested):  # noqa: F811
    from app.answering.client import get_llm
    from app.main import app

    app.dependency_overrides[get_llm] = lambda: _gemini_raising(__import__("httpx").ConnectError("dns"))
    r = client.post("/ask", json={"question": "How much can I spend on books?"})
    assert r.status_code == 200 and r.json()["status"] == "failed"
