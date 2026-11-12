"""Find the chunks most likely to answer a question.

Three methods, compared by the retrieval eval (eval/retrieval.py):
- vector:  meaning-based. Embed the question, rank chunks by cosine similarity. Good at
           paraphrases ("time off for a new baby" ~ "parental leave"), weak on exact terms.
- keyword: Postgres full-text search. Good at exact words and numbers ("walking pad",
           "USERRA"), blind to synonyms.
- hybrid:  both lists merged with Reciprocal Rank Fusion (RRF). Each chunk scores
           sum(weight / (RRF_K + rank)) over the lists it appears in, so a chunk ranked well by
           either method rises, and one ranked well by both rises most. RRF only uses ranks,
           so the two methods' incomparable scores never need calibrating.

Default is VECTOR, decided by the eval (2026-11-12, 61 questions): vector hit@5 0.92 vs
hybrid 0.81 (equal weights) / 0.89 (keyword weight 0.3), keyword alone 0.68. Postgres ts_rank
has no IDF (a common word like "team" counts as much as a rare one like "USERRA"), so the
keyword list mostly adds noise. Re-run the eval before changing the default.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from sqlalchemy import Text, cast, func, select
from sqlalchemy.dialects.postgresql import TSQUERY
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document
from app.ingestion.embeddings import Embedder

RRF_K = 60  # standard constant from the RRF paper; damps the advantage of rank 1 over rank 2
POOL = 50  # candidates taken from each method before fusing


class Method(StrEnum):
    vector = "vector"
    keyword = "keyword"
    hybrid = "hybrid"


@dataclass(frozen=True)
class Hit:
    rank: int  # 1-based
    chunk_id: int
    document_id: uuid.UUID
    path: str
    title: str | None
    source_url: str | None
    heading_path: str
    text: str
    similarity: float  # cosine similarity to the question, always computed (refusal signal)
    score: float  # the method's own ranking score (similarity, keyword rank or RRF)


def rrf(
    ranked_lists: Sequence[Sequence[int]], weights: Sequence[float] | None = None, k: int = RRF_K
) -> list[tuple[int, float]]:
    """(Weighted) Reciprocal Rank Fusion of ranked id lists -> [(id, fused score)], best first.

    Each list contributes weight / (k + rank). Ties keep first-seen order (deterministic).
    """
    weights = weights or [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("one weight per ranked list")
    scores: dict[int, float] = {}
    for ranked, w in zip(ranked_lists, weights):
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def _or_query(question: str):
    # plainto_tsquery drops stop words and stems ("reimbursed" -> 'reimburs'), but ANDs the
    # terms: a natural question would then match almost nothing. OR them instead, and let the
    # ranking reward chunks that contain more of the terms.
    anded = cast(func.plainto_tsquery("english", question), Text)
    return cast(func.replace(anded, "&", "|"), TSQUERY)


def vector_ids(db: Session, query_vec: list[float], limit: int) -> list[tuple[int, float]]:
    sim = (1 - Chunk.embedding.cosine_distance(query_vec)).label("sim")
    rows = db.execute(select(Chunk.id, sim).order_by(Chunk.embedding.cosine_distance(query_vec), Chunk.id).limit(limit))
    return [(r.id, float(r.sim)) for r in rows]


def keyword_ids(db: Session, question: str, limit: int) -> list[tuple[int, float]]:
    tsq = _or_query(question)
    rank = func.ts_rank(Chunk.search_tsv, tsq, 1).label("rank")  # 1 = normalize by length
    rows = db.execute(
        select(Chunk.id, rank).where(Chunk.search_tsv.op("@@")(tsq)).order_by(rank.desc(), Chunk.id).limit(limit)
    )
    return [(r.id, float(r.rank)) for r in rows]


def search(
    db: Session,
    embedder: Embedder,
    question: str,
    *,
    k: int = 5,
    method: Method = Method.vector,
    keyword_weight: float = 1.0,
) -> list[Hit]:
    """keyword_weight: the keyword list's weight in hybrid fusion (vector list = 1.0)."""
    query_vec = embedder.embed_query(question)
    if method is Method.vector:
        ranked = vector_ids(db, query_vec, k)
    elif method is Method.keyword:
        ranked = keyword_ids(db, question, k)
    else:
        fused = rrf([[i for i, _ in vector_ids(db, query_vec, POOL)],
                     [i for i, _ in keyword_ids(db, question, POOL)]],
                    weights=[1.0, keyword_weight])
        ranked = fused[:k]
    if not ranked:
        return []

    ids = [i for i, _ in ranked]
    sim = (1 - Chunk.embedding.cosine_distance(query_vec)).label("sim")
    rows = {
        r.Chunk.id: r
        for r in db.execute(select(Chunk, Document, sim).join(Document).where(Chunk.id.in_(ids)))
    }
    return [
        Hit(rank=n, chunk_id=cid, document_id=rows[cid].Document.id, path=rows[cid].Document.path,
            title=rows[cid].Document.title, source_url=rows[cid].Document.source_url,
            heading_path=rows[cid].Chunk.heading_path, text=rows[cid].Chunk.text,
            similarity=float(rows[cid].sim), score=score)
        for n, (cid, score) in enumerate(ranked, start=1)
    ]
