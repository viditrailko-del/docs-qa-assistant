"""Live mode and the demo guards: real-model questions are capped, flash-lite only, and the documents are fixed."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from app.answering.client import LLMError
from app.config import get_settings
from app.db.models import Ask, Document, LlmCall
from app.demo import live
from app.demo.index import seed_index
from app.main import app
from tests.conftest import test_settings

Q = {"question": "How much is the hotel limit per night?"}


def use(**overrides):
    app.dependency_overrides[get_settings] = lambda: test_settings(demo_mode=True, **overrides)


@pytest.fixture(autouse=True)
def _fresh_quota():
    live.quota.reset()
    yield
    live.quota.reset()


@pytest.fixture
def demo(client, db):
    seed_index(db, test_settings())
    use()
    return client


def test_live_routes_are_hidden_unless_demo_mode(client):
    assert client.post("/demo/ask", json=Q).status_code == 404
    assert client.get("/demo/status").status_code == 404


def test_live_question_runs_on_flash_lite_only_and_returns_source_texts(demo, fake_llm):
    r = demo.post("/demo/ask", json=Q)
    assert r.status_code == 200
    a = r.json()
    assert a["status"] in ("answered", "answered_with_flags")
    assert [c["model"] for c in fake_llm.calls] == ["gemini-3.5-flash-lite"]
    assert set(a["source_texts"]) == {s["label"] for s in a["sources"]}


def test_overloaded_gemini_ends_failed_without_trying_the_main_model(demo, fake_llm):
    fake_llm.replies = [LLMError("503: overloaded", transient=True)] * 3
    a = demo.post("/demo/ask", json=Q).json()
    assert a["status"] == "failed"
    assert {c["model"] for c in fake_llm.calls} == {"gemini-3.5-flash-lite"}
    assert len(fake_llm.calls) == 3


def test_question_length_is_capped(demo):
    assert demo.post("/demo/ask", json={"question": "x" * 301}).status_code == 422
    use(demo_question_max_chars=50)
    assert demo.post("/demo/ask", json={"question": "y" * 60}).status_code == 422
    assert demo.post("/ask", json={"question": "y" * 60}).status_code == 422


def test_daily_cap_for_the_whole_demo(demo):
    use(demo_live_daily_limit=2, demo_live_per_ip_per_hour=10)
    assert [demo.post("/demo/ask", json=Q).status_code for _ in range(3)] == [200, 200, 429]
    s = demo.get("/demo/status").json()
    assert (s["live_used_today"], s["live_left_today"], s["live_model"]) == (2, 0, "gemini-3.5-flash-lite")


def test_per_visitor_hourly_cap(demo):
    use(demo_live_daily_limit=10, demo_live_per_ip_per_hour=2)
    codes = [demo.post("/demo/ask", json=Q).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert "per hour" in demo.post("/demo/ask", json=Q).json()["detail"]


def test_per_visitor_cap_frees_up_after_an_hour(db):
    s = test_settings(demo_live_daily_limit=10, demo_live_per_ip_per_hour=1)
    now = datetime.now(UTC).replace(hour=12)
    live.quota.take(db, "1.2.3.4", s, now)
    with pytest.raises(live.QuotaExceeded):
        live.quota.take(db, "1.2.3.4", s, now + timedelta(minutes=59))
    live.quota.take(db, "5.6.7.8", s, now)  # another visitor
    live.quota.take(db, "1.2.3.4", s, now + timedelta(minutes=61))


def test_a_restart_does_not_hand_out_a_fresh_day(demo, db):
    use(demo_live_daily_limit=2)
    demo.post("/demo/ask", json=Q)
    demo.post("/demo/replay/payday")  # replays never count
    live.quota.reset()  # what a restart does to the in-memory counter
    assert demo.get("/demo/status").json()["live_used_today"] == 1
    assert [demo.post("/demo/ask", json=Q).status_code for _ in range(2)] == [200, 429]


def test_post_ask_spends_the_same_quota_in_demo_mode(demo, fake_llm):
    use(demo_live_daily_limit=1)
    assert demo.post("/ask", json=Q).status_code == 200
    assert fake_llm.calls[-1]["model"] == "gemini-3.5-flash-lite"
    assert demo.post("/demo/ask", json=Q).status_code == 429
    assert demo.post("/ask", json=Q).status_code == 429


def test_post_ask_has_no_cap_outside_demo_mode(client, db):
    seed_index(db, test_settings())
    app.dependency_overrides[get_settings] = lambda: test_settings(demo_live_daily_limit=0)
    assert client.post("/ask", json=Q).status_code == 200


def test_uploads_are_off_in_demo_mode(demo):
    r = demo.post("/documents", files={"file": ("x.md", b"# Title\n\nSome text here.", "text/markdown")})
    assert r.status_code == 403


def test_reset_is_off_without_a_token(demo):
    assert demo.post("/demo/reset", headers={"X-Reset-Token": ""}).status_code == 404


def test_reset_needs_the_right_token(demo):
    use(demo_reset_token="s3cret")
    assert demo.post("/demo/reset").status_code == 403
    assert demo.post("/demo/reset", headers={"X-Reset-Token": "wrong"}).status_code == 403


def test_reset_deletes_questions_and_keeps_documents(demo, db):
    use(demo_reset_token="s3cret", demo_live_daily_limit=1)
    demo.post("/demo/ask", json=Q)
    demo.post("/demo/replay/payday")
    assert demo.post("/demo/ask", json=Q).status_code == 429
    r = demo.post("/demo/reset", headers={"X-Reset-Token": "s3cret"})
    assert r.status_code == 200 and r.json() == {"asks_deleted": 2, "documents_added": 0}
    assert db.scalar(select(func.count()).select_from(Ask)) == 0
    assert db.scalar(select(func.count()).select_from(LlmCall)) == 0
    assert db.scalar(select(func.count()).select_from(Document)) == 60
    assert demo.post("/demo/ask", json=Q).status_code == 200  # a fresh day's quota
