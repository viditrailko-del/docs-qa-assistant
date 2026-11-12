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


@lru_cache
def get_settings() -> Settings:
    return Settings()
