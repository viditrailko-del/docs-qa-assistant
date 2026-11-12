"""GET /search: the retrieval step on its own, for inspection and debugging (no LLM)."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.ingestion.embeddings import Embedder, get_embedder
from app.retrieval.search import Method, search
from app.schemas.search import SearchHit

router = APIRouter(tags=["search"])


@router.get("/search", response_model=list[SearchHit])
def search_chunks(
    q: Annotated[str, Query(min_length=1, max_length=500, description="The question")],
    db: Annotated[Session, Depends(get_db)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    k: Annotated[int, Query(ge=1, le=50)] = 5,
    method: Method = Method.vector,
    keyword_weight: Annotated[float, Query(ge=0, le=5, description="hybrid only")] = 1.0,
) -> list[SearchHit]:
    hits = search(db, embedder, q, k=k, method=method, keyword_weight=keyword_weight)
    return [SearchHit.model_validate(h, from_attributes=True) for h in hits]
