"""POST /ask and GET /ask/{id}. Thin: the work is in app/answering."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.answering.client import LLMClient, get_llm
from app.answering.service import ask_question
from app.api.demo import admit_live
from app.config import Settings, get_settings
from app.db.models import Ask
from app.db.session import get_db
from app.ingestion.embeddings import Embedder, get_embedder
from app.schemas.ask import AskIn, AskOut

router = APIRouter(prefix="/ask", tags=["ask"])


@router.post("", response_model=AskOut, responses={429: {"description": "Demo: live quota used up"}})
def ask(
    body: AskIn,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    llm: Annotated[LLMClient, Depends(get_llm)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AskOut:
    """Answer from the documents with cited quotes, or refuse. An LLM outage gives
    status `failed` (HTTP 200), never a 500."""
    question = body.question.strip()
    if settings.demo_mode:  # the public demo: same caps and model as its live mode, whichever route is used
        if len(question) > settings.demo_question_max_chars:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                                f"questions are limited to {settings.demo_question_max_chars} characters")
        settings = admit_live(db, request, settings)
    return AskOut.from_ask(ask_question(db, embedder, llm, settings, question, k=body.k))


@router.get("/{ask_id}", response_model=AskOut)
def get_ask(ask_id: uuid.UUID, db: Annotated[Session, Depends(get_db)]) -> AskOut:
    row = db.get(Ask, ask_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "ask not found")
    return AskOut.from_ask(row)
