"""Replay mode: every recorded question runs through the real pipeline and ends where the eval scored it.

If a check, a prompt file or the chunker changes what a scenario does, these fail: re-run
`python -m demo.build_replay` and re-read the scenario's blurb.
"""

import copy
import json

import pytest
from sqlalchemy import func, select

from app.config import get_settings
from app.db.models import Ask, LlmCall
from app.demo.index import seed_index
from app.demo.replay import RecordedEmbedder, Scenario, replay, scenarios
from app.main import app
from tests.conftest import test_settings

QUESTIONS = {q["id"]: q for q in json.loads(open("eval/questions/questions.json", encoding="utf-8").read())["questions"]}


@pytest.fixture
def demo(client):
    app.dependency_overrides[get_settings] = lambda: test_settings(demo_mode=True)
    return client


@pytest.fixture
def seeded(db):
    seed_index(db, test_settings())
    return db


def test_demo_routes_are_hidden_unless_demo_mode(client):
    assert client.get("/demo/scenarios").status_code == 404
    assert client.post("/demo/replay/payday").status_code == 404


def test_lists_scenarios_without_recordings(demo):
    listed = demo.get("/demo/scenarios").json()
    assert [s["name"] for s in listed] == list(scenarios())
    assert len(listed) >= 12
    assert all("recorded" not in s and "expected" not in s for s in listed)


def test_unknown_scenario_is_404(demo):
    assert demo.post("/demo/replay/nope").status_code == 404


@pytest.mark.parametrize("name", list(scenarios()))
def test_scenario_question_is_its_verified_label(name):
    sc = scenarios()[name]
    label = QUESTIONS[sc.data["label_id"]]
    assert label.get("verified_by"), f"{label['id']} must be human-verified before it is shown"
    assert sc.question == label["question"]


def test_scenarios_tell_the_whole_story():
    groups = {s.data["group"] for s in scenarios().values()}
    outcomes = {s.data["expected"]["outcome"] for s in scenarios().values()}
    assert len(groups) >= 5
    assert {"correct", "correct_refusal", "wrong_flagged", "confident_wrong"} <= outcomes


def test_every_replay_reproduces_its_scored_result(demo, seeded, fake_embedder, fake_llm):
    # One test for all scenarios: the full corpus index is seeded once (1407 chunks).
    wrong = {}
    for name, sc in scenarios().items():
        exp = sc.data["expected"]
        r = demo.post(f"/demo/replay/{name}")
        assert r.status_code == 200, (name, r.text)
        a = r.json()
        got = {"status": a["status"], "flags": sorted({f["code"] for f in a["flags"]})}
        if got != {"status": exp["status"], "flags": exp["flags"]}:
            wrong[name] = (got, exp, [c["error"] for c in a["llm_calls"] if c["error"]])
        assert a["llm_calls"] and all(c["model"].startswith("replay:") for c in a["llm_calls"])
        assert all(c["cost_usd"] is None for c in a["llm_calls"])
        assert len(a["sources"]) == sc.data["recorded"]["k"]
        assert set(a["source_texts"]) == {s["label"] for s in a["sources"]} and all(a["source_texts"].values())
    assert not wrong, wrong
    assert fake_llm.calls == []  # Gemini was never asked anything
    assert fake_embedder.passage_calls == 0


def test_replaying_again_replaces_the_earlier_replay(demo, seeded):
    first = demo.post("/demo/replay/payday").json()
    second = demo.post("/demo/replay/payday").json()
    assert first["id"] != second["id"]
    assert seeded.scalar(select(func.count()).select_from(Ask)) == 1
    assert seeded.scalar(select(func.count()).select_from(LlmCall)) == 1


def test_a_live_ask_with_the_same_question_is_not_cleared(demo, seeded):
    sc = scenarios()["payday"]
    live = Ask(question=sc.question, status="answered", retrieval_method="vector", k=12, prompt_version="v2")
    live.llm_calls = [LlmCall(model="gemini-3.5-flash-lite", prompt_version="v2")]
    seeded.add(live)
    seeded.commit()
    demo.post("/demo/replay/payday")
    demo.post("/demo/replay/payday")
    seeded.expire_all()
    assert seeded.get(Ask, live.id) is not None


def test_different_sources_fail_the_replay_instead_of_improvising(seeded):
    sc = scenarios()["home_office_budget"]
    data = copy.deepcopy(sc.data)
    data["recorded"]["k"] = data["recorded"]["k"] - 1  # one source fewer: not the prompt the model answered
    ask = replay(seeded, Scenario(name="diverged", data=data), test_settings())
    assert ask.status == "failed"
    assert any("replay diverged" in (c.error or "") for c in ask.llm_calls)


def test_a_changed_question_vector_fails_the_replay(seeded):
    sc = scenarios()["home_office_budget"]
    data = copy.deepcopy(sc.data)
    data["recorded"]["query_vector"] = list(reversed(data["recorded"]["query_vector"]))
    ask = replay(seeded, Scenario(name="diverged", data=data), test_settings())
    assert ask.status == "failed"


def test_recorded_embedder_only_knows_its_question():
    e = RecordedEmbedder("m", "q?", [0.1, 0.2])
    assert e.embed_query("q?") == [0.1, 0.2]
    with pytest.raises(ValueError):
        e.embed_query("another question")
