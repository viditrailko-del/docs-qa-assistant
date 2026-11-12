import pytest

from app.retrieval.search import rrf
from eval.metrics import (
    QuestionResult,
    first_relevant_rank,
    full_coverage_rank,
    is_trapped,
    refusal_signal,
    summarize,
)

# --- RRF fusion ---------------------------------------------------------------------------

def test_rrf_rewards_items_ranked_well_by_both_lists():
    fused = [i for i, _ in rrf([[1, 2, 3], [3, 1, 4]])]
    assert fused[:2] == [1, 3]  # in both lists, near the top
    assert set(fused) == {1, 2, 3, 4}


def test_rrf_scores_are_reciprocal_ranks():
    scores = dict(rrf([[7, 8]], k=60))
    assert scores[7] == pytest.approx(1 / 61) and scores[8] == pytest.approx(1 / 62)


def test_rrf_weights_shift_the_balance():
    vector, keyword = [1, 2], [2, 1]
    assert rrf([vector, keyword], weights=[1.0, 0.1])[0][0] == 1
    assert rrf([vector, keyword], weights=[0.1, 1.0])[0][0] == 2


def test_rrf_ties_are_deterministic_and_empty_is_empty():
    assert [i for i, _ in rrf([[5, 6], [6, 5]])] == [5, 6]
    assert rrf([[], []]) == []


def test_rrf_rejects_wrong_number_of_weights():
    with pytest.raises(ValueError):
        rrf([[1], [2]], weights=[1.0])


# --- metrics --------------------------------------------------------------------------------

def test_first_relevant_and_full_coverage():
    ranked = [10, 11, 12, 13]
    sources = [{12}, {13, 99}]
    assert first_relevant_rank(ranked, sources) == 3
    assert full_coverage_rank(ranked, sources) == 4
    assert full_coverage_rank(ranked, [{12}, {42}]) is None  # one piece of evidence never retrieved
    assert first_relevant_rank(ranked, []) is None


def test_trap_detection():
    assert is_trapped([5, 1], [{1}], near_miss={5})  # decoy above the evidence
    assert not is_trapped([1, 5], [{1}], near_miss={5})
    assert is_trapped([5], [], near_miss={5})  # unanswerable: decoy retrieved
    assert not is_trapped([1, 2], [{1}], near_miss={1})  # a chunk that is also evidence isn't a decoy


def test_summarize_rates_and_misses():
    rs = [
        QuestionResult("a", "simple", True, first_rank=1, full_rank=1, top_similarity=0.9),
        QuestionResult("b", "simple", True, first_rank=4, full_rank=4, top_similarity=0.8),
        QuestionResult("c", "multi", True, first_rank=None, full_rank=None, top_similarity=0.7),
        QuestionResult("u", "unanswerable", False, trapped=True, top_similarity=0.5),
    ]
    s = summarize(rs)
    a = s["answerable"]
    assert a["n"] == 3
    assert a["hit@1"] == pytest.approx(1 / 3) and a["hit@5"] == pytest.approx(2 / 3)
    assert a["mrr"] == pytest.approx((1 + 1 / 4) / 3)
    assert a["missed@10"] == ["c"]
    assert s["by_type"]["simple"]["hit@3"] == 0.5
    assert s["trapped"] == ["u"]


def test_refusal_signal_finds_a_separating_threshold():
    rs = [QuestionResult(str(i), "simple", True, top_similarity=s) for i, s in enumerate([0.8, 0.85, 0.9])]
    rs += [QuestionResult("u1", "unanswerable", False, top_similarity=0.6)]
    sig = refusal_signal(rs)
    assert sig["best_accuracy"] == 1.0
    assert 0.6 < sig["best_threshold"] <= 0.8
    assert sig["unanswerable_refused"] == "1/1" and sig["answerable_wrongly_refused"] == "0/3"


# --- search API (fake embedder: vectors carry no meaning, so test mechanics only) -------------

DOC = b"""---
title: Expenses
---

## Equipment

Walking pads and treadmills are not reimbursable.

## Books

Limit is set to $60/year.

## Hotels

Hotel stays are reimbursable up to $300 per night.
"""


@pytest.fixture
def ingested(client):
    r = client.post("/documents", files={"file": ("expenses.md", DOC, "text/markdown")},
                    data={"source_url": "https://handbook.example/expenses"})
    assert r.status_code == 201


def test_keyword_search_finds_stemmed_words(client, ingested):
    hits = client.get("/search", params={"q": "is a treadmill reimbursed?", "method": "keyword"}).json()
    assert hits[0]["heading_path"] == "Expenses > Equipment"  # treadmill~treadmills, reimbursed~reimbursable
    assert hits[0]["source_url"] == "https://handbook.example/expenses"


def test_vector_search_ranks_by_similarity(client, ingested):
    hits = client.get("/search", params={"q": "anything", "k": 3}).json()
    assert [h["rank"] for h in hits] == [1, 2, 3]
    sims = [h["similarity"] for h in hits]
    assert sims == sorted(sims, reverse=True) and all(-1 <= s <= 1 for s in sims)


def test_hybrid_search_merges_both_lists(client, ingested):
    hits = client.get("/search", params={"q": "hotel per night", "method": "hybrid", "k": 3}).json()
    assert len(hits) == 3
    assert "Expenses > Hotels" in [h["heading_path"] for h in hits]


def test_keyword_search_with_only_stop_words_returns_nothing(client, ingested):
    assert client.get("/search", params={"q": "the a of", "method": "keyword"}).json() == []


def test_search_validates_params_and_handles_empty_db(client):
    assert client.get("/search", params={"q": "x"}).json() == []
    assert client.get("/search", params={"q": ""}).status_code == 422
    assert client.get("/search", params={"q": "x", "k": 0}).status_code == 422
    assert client.get("/search", params={"q": "x", "method": "magic"}).status_code == 422
