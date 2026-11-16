"""Structured logging: the formatters, and the one-event-per-question summary."""

import json
import logging
import sys
import uuid
from decimal import Decimal

from app.answering.client import LLMError
from app.logging_setup import JsonFormatter, TextFormatter
from tests.test_answering import ingested  # noqa: F401  (fixture)


def _record(event="ask_finished", **fields) -> logging.LogRecord:
    record = logging.getLogger("test").makeRecord("test", logging.INFO, __file__, 1, event, (), None)
    record.fields = fields
    return record


def test_json_formatter_one_object_per_line():
    ask_id = uuid.uuid4()
    line = JsonFormatter().format(_record(ask_id=ask_id, cost_usd=Decimal("0.0031"), flags=["a"]))
    entry = json.loads(line)
    assert "\n" not in line
    assert entry["event"] == "ask_finished" and entry["level"] == "info"
    assert entry["ask_id"] == str(ask_id)
    assert entry["cost_usd"] == "0.0031"  # Decimal stays exact (a string), never a float
    assert entry["flags"] == ["a"]


def test_text_formatter_key_values():
    assert TextFormatter().format(_record(status="answered", llm_calls=1)).endswith(
        "ask_finished status=answered llm_calls=1")


def test_exception_is_included():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = logging.getLogger("test").makeRecord("test", logging.ERROR, __file__, 1, "x", (),
                                                      exc_info=sys.exc_info())
    assert "RuntimeError: boom" in json.loads(JsonFormatter().format(record))["exception"]


def _events(caplog, name: str) -> list[dict]:
    return [r.fields for r in caplog.records if r.getMessage() == name]


def test_one_summary_event_per_question(client, ingested, caplog):  # noqa: F811
    caplog.set_level(logging.INFO)
    body = client.post("/ask", json={"question": "How much can I spend on books?"}).json()
    [event] = _events(caplog, "ask_finished")
    assert event["ask_id"] == uuid.UUID(body["id"]) and event["status"] == "answered"
    assert event["llm_calls"] == 1 and event["failed_calls"] == 0
    assert event["model"] == "gemini-3.5-flash" and event["prompt_version"] == "v2"
    assert event["input_tokens"] == 1000 and event["output_tokens"] == 100
    assert event["cost_usd"] == Decimal("0.002250") and event["flags"] == []
    assert "question" not in event  # may contain personal details: only the ask_id is logged


def test_failures_and_invalid_output_are_logged(client, ingested, fake_llm, caplog):  # noqa: F811
    caplog.set_level(logging.INFO)
    fake_llm.replies = [LLMError("503", transient=True)] * 6
    client.post("/ask", json={"question": "books?"})
    [event] = _events(caplog, "ask_finished")
    assert event["status"] == "failed" and event["model"] is None
    assert event["llm_calls"] == event["failed_calls"] == 6
    assert len(_events(caplog, "llm_call_failed")) == 6

    caplog.clear()
    fake_llm.replies = ["not json"]
    client.post("/ask", json={"question": "books?"})
    [invalid] = _events(caplog, "llm_output_invalid")
    assert invalid["errors"] == 1
    assert _events(caplog, "ask_finished")[0]["status"] == "answered"  # the retry succeeded


def test_ingestion_is_logged(client, caplog):
    caplog.set_level(logging.INFO)
    client.post("/documents", files={"file": ("a.md", b"# A\n\nSome text.", "text/markdown")})
    [event] = _events(caplog, "document_ingested")
    assert event["path"] == "a.md" and event["chunks"] == 1
