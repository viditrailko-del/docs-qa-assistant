from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All config comes from environment variables / .env. No hard-coded secrets."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://rag:rag@localhost:5433/rag"
    max_upload_mb: int = 5

    # Embeddings run locally (fastembed / ONNX). The model is baked into the Docker image.
    # Changing the model changes every vector: re-embed all chunks (and EMBED_DIM + a migration).
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_cache_dir: str | None = None  # /opt/models in the image

    # Chunking: ~1200 chars is ~250-300 tokens, a paragraph or two of policy text.
    chunk_max_chars: int = 1200

    # Retrieval: how many chunks the LLM sees. Eval: hit@5 0.92 vs hit@10 0.96, and full@10
    # (all evidence present, matters for conflicts) 0.91 vs full@5 0.83. 8 is the compromise.
    retrieval_k: int = 8

    # Answering (Gemini). Free tier: ~20 requests/day on the main model, then the fallback answers.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    gemini_fallback_model: str = "gemini-3.5-flash-lite"
    gemini_thinking_level: str = "low"  # higher levels were 25-50x slower in the invoice project
    gemini_timeout_s: int = 60
    prompt_version: str = "v1"
    llm_retry_backoff_s: float = 2.0  # waits 2s, then 4s, between retries


@lru_cache
def get_settings() -> Settings:
    return Settings()
