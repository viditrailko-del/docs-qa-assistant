import uuid

from pydantic import BaseModel


class SearchHit(BaseModel):
    rank: int
    chunk_id: int
    document_id: uuid.UUID
    path: str
    title: str | None
    source_url: str | None
    heading_path: str
    text: str
    similarity: float  # cosine similarity to the question (0..1, higher = closer in meaning)
    score: float  # the method's own ranking score
