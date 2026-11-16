import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import ask, documents, search
from app.config import get_settings
from app.logging_setup import adopt_uvicorn_loggers, configure_logging, log_event

configure_logging(get_settings().log_format, get_settings().log_level)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    adopt_uvicorn_loggers()  # uvicorn configured its loggers after our import
    s = get_settings()
    if not s.gemini_api_key:
        # Not fatal: search and ingestion still work, /ask answers `failed` (never a 500).
        log_event(log, logging.ERROR, "gemini_api_key_missing", effect="POST /ask will return status failed")
    log_event(log, logging.INFO, "startup", model=s.gemini_model, fallback_model=s.gemini_fallback_model,
              prompt_version=s.prompt_version, embed_model=s.embed_model, retrieval_k=s.retrieval_k)
    yield


app = FastAPI(
    title="Docs Q&A Assistant",
    description="Answers questions from company documents with verified citations, or refuses.",
    version="0.2.0",
    lifespan=lifespan,
)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(ask.router)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}
