"""Local text embeddings.

An embedding turns a text into a list of numbers (384 here) such that texts with similar
meaning get nearby vectors. Search = embed the question, find the chunks whose vectors are
closest (cosine distance). It runs on the CPU with ONNX, so it's free and unlimited.

bge models are trained with an instruction prefix on the *query* side, so questions and
passages are embedded slightly differently: always use embed_query for questions.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from app.config import get_settings


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    def __init__(self, model_name: str, cache_dir: str | None = None, threads: int | None = None) -> None:
        from fastembed import TextEmbedding  # imported here: tests never load the real model

        self.model_name = model_name
        self._model = TextEmbedding(model_name, cache_dir=cache_dir, threads=threads)
        self.dim = len(self.embed_query("dimension probe"))

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


@lru_cache
def get_embedder() -> Embedder:
    """FastAPI dependency. Loading the model takes ~1s, so it's done once per process."""
    s = get_settings()
    return FastEmbedder(s.embed_model, s.embed_cache_dir, s.embed_threads)
