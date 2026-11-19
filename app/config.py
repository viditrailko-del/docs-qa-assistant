from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All config comes from environment variables / .env. No hard-coded secrets."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://rag:rag@localhost:5433/rag"
    max_upload_mb: int = 5
    log_format: str = "text"  # "json" in the production image: one JSON object per line
    log_level: str = "INFO"

    # Embeddings run locally (fastembed / ONNX). The model is baked into the Docker image.
    # Changing the model changes every vector: re-embed all chunks (and EMBED_DIM + a migration).
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_cache_dir: str | None = None  # /opt/models in the image
    # ONNX threads. None = one per CPU core the machine has. On a small CPU quota (Render free: 0.1 CPU)
    # that many threads stall each other: embedding one question never finished in 10 min; with 1 thread
    # it takes ~0.15 s. The demo start script sets 1.
    embed_threads: int | None = None

    # Chunking: ~1200 chars is ~250-300 tokens, a paragraph or two of policy text.
    chunk_max_chars: int = 1200

    # Retrieval: how many chunks the LLM sees. Answer eval (2026-11-16): 4 of the 6 confident-wrong
    # answers at k=8 lacked evidence ranked 9-11th. All evidence shown (full@k, 53 questions):
    # k=8 46, k=10 48, k=12 49, k=20 still 49. k=12: confident-wrong 5 -> 4, ~+20% cost.
    retrieval_k: int = 12
    # > 0 adds a weak_retrieval warning below this cosine similarity. Off: no threshold separated
    # answerable from unanswerable questions in the Phase 2 eval without overfitting.
    min_similarity: float = 0.0

    # Answering (Gemini). Free tier: ~20 requests/day on the main model, then the fallback answers.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    gemini_fallback_model: str = "gemini-3.5-flash-lite"
    gemini_thinking_level: str = "low"  # higher levels were 25-50x slower in the invoice project
    gemini_timeout_s: int = 60
    # v2 (eval 2026-11-16, k=12): confident-wrong 4 -> 2; refuses general questions answered only
    # by country pages (u07 fixed, q30 now an over-refusal), states conditions and pointers.
    prompt_version: str = "v2"
    llm_retry_backoff_s: float = 2.0  # waits 2s, then 4s, between retries

    # Public demo (Phase 8). Off: every /demo route answers 404.
    demo_mode: bool = False
    # Live questions run on flash-lite with NO fallback: the main model's ~20 requests/day are too few to share
    # between demos. One question is usually one call (eval: 65 calls for 61 questions), at most 3 (retries).
    demo_live_model: str = "gemini-3.5-flash-lite"
    demo_live_daily_limit: int = 30  # whole demo, per UTC day (the support demo's 30 tickets share the key)
    demo_live_per_ip_per_hour: int = 5
    demo_question_max_chars: int = 300
    demo_reset_token: str = ""  # empty: POST /demo/reset is off

    @field_validator("database_url")
    @classmethod
    def _psycopg_driver(cls, url: str) -> str:
        """Hosted Postgres (Neon, Render) hands out postgres:// or postgresql:// URLs; SQLAlchemy would pick
        psycopg2 for those, which isn't installed. Use psycopg 3."""
        for prefix in ("postgres://", "postgresql://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url[len(prefix):]
        return url


@lru_cache
def get_settings() -> Settings:
    return Settings()
