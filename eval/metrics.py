"""Retrieval metrics. Pure functions over ranked chunk ids: no DB, no model.

Vocabulary:
- A question's *source sets*: for each evidence quote, the chunk ids that contain it. Usually
  one chunk per quote; two quotes can share a chunk.
- hit@k ("recall@k"): at least one relevant chunk is in the top k.
- full@k: EVERY source set has a chunk in the top k (multi-passage and conflict questions
  need all their evidence, not just one piece).
- MRR (mean reciprocal rank): average of 1/rank of the first relevant chunk (0 if missing).
  1.0 = always first, 0.5 = typically second.
- trap: a near-miss chunk is ranked ABOVE the first relevant chunk (or there is no relevant
  chunk in the list at all), so the LLM would read the decoy first.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

KS = (1, 3, 5, 10)


def first_rank(ranked: Sequence[int], wanted: Iterable[int]) -> int | None:
    wanted = set(wanted)
    return next((i for i, cid in enumerate(ranked, start=1) if cid in wanted), None)


def first_relevant_rank(ranked: Sequence[int], source_sets: Sequence[set[int]]) -> int | None:
    return first_rank(ranked, set().union(*source_sets)) if source_sets else None


def full_coverage_rank(ranked: Sequence[int], source_sets: Sequence[set[int]]) -> int | None:
    """Smallest k whose top-k covers every source set, or None."""
    ranks = [first_rank(ranked, s) for s in source_sets]
    return None if not ranks or None in ranks else max(ranks)


def is_trapped(ranked: Sequence[int], source_sets: Sequence[set[int]], near_miss: set[int]) -> bool:
    decoys = near_miss - set().union(*source_sets) if source_sets else near_miss
    trap = first_rank(ranked, decoys)
    if trap is None:
        return False
    rel = first_relevant_rank(ranked, source_sets)
    return rel is None or trap < rel


@dataclass
class QuestionResult:
    id: str
    type: str
    answerable: bool
    first_rank: int | None = None
    full_rank: int | None = None
    trapped: bool = False
    top_similarity: float = 0.0
    ranked: list[int] = field(default_factory=list)


def summarize(results: Sequence[QuestionResult]) -> dict:
    answerable = [r for r in results if r.answerable]

    def rates(rs: Sequence[QuestionResult]) -> dict:
        if not rs:
            return {}
        n = len(rs)
        return {
            "n": n,
            **{f"hit@{k}": sum(r.first_rank is not None and r.first_rank <= k for r in rs) / n for k in KS},
            **{f"full@{k}": sum(r.full_rank is not None and r.full_rank <= k for r in rs) / n for k in KS},
            "mrr": sum(1 / r.first_rank for r in rs if r.first_rank) / n,
            "missed@10": sorted(r.id for r in rs if r.first_rank is None),
        }

    by_type = {t: rates([r for r in answerable if r.type == t]) for t in sorted({r.type for r in answerable})}
    with_traps = [r for r in results if r.trapped is not None]
    return {
        "answerable": rates(answerable),
        "by_type": by_type,
        "trapped": sorted(r.id for r in with_traps if r.trapped),
        "refusal_signal": refusal_signal(results),
    }


def refusal_signal(results: Sequence[QuestionResult]) -> dict:
    """Could 'top similarity below a threshold -> refuse' separate unanswerable questions?

    Tries every observed similarity as a threshold and reports the best accuracy. With few
    unanswerable questions this is a rough indicator, not a tuned setting.
    """
    ans = [r.top_similarity for r in results if r.answerable]
    una = [r.top_similarity for r in results if not r.answerable]
    if not ans or not una:
        return {}
    best = (0.0, 0.0, 0, 0)
    for t in sorted(set(ans + una)) + [max(ans + una) + 1e-9]:
        refused_una = sum(s < t for s in una)
        refused_ans = sum(s < t for s in ans)
        acc = (refused_una + len(ans) - refused_ans) / (len(ans) + len(una))
        if acc > best[0]:
            best = (acc, t, refused_una, refused_ans)

    def dist(xs: list[float]) -> dict:
        return {"min": round(min(xs), 3), "median": round(statistics.median(xs), 3), "max": round(max(xs), 3)}

    return {
        "answerable_top_sim": dist(ans),
        "unanswerable_top_sim": dist(una),
        "best_threshold": round(best[1], 3),
        "best_accuracy": round(best[0], 3),
        "unanswerable_refused": f"{best[2]}/{len(una)}",
        "answerable_wrongly_refused": f"{best[3]}/{len(ans)}",
    }
