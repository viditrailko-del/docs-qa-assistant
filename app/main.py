from fastapi import FastAPI

from app.api import ask, documents, search

app = FastAPI(
    title="Docs Q&A Assistant",
    description="Answers questions from company documents with verified citations, or refuses.",
    version="0.1.0",
)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(ask.router)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}
