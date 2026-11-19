"""Demo routes. Thin: the work is in app/demo. Every route answers 404 unless DEMO_MODE is on."""

import hmac
import logging
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.answering.client import LLMClient, get_llm
from app.answering.service import ask_question
from app.config import Settings, get_settings
from app.db.session import get_db
from app.demo import live
from app.demo.present import demo_ask_out
from app.demo.replay import replay, scenarios
from app.demo.reset import reset_demo
from app.ingestion.embeddings import Embedder, get_embedder
from app.logging_setup import log_event
from app.schemas.demo import DemoAskOut, DemoStatus, LiveAskIn


def demo_enabled(settings: Annotated[Settings, Depends(get_settings)]) -> None:
    if not settings.demo_mode:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not Found")


log = logging.getLogger(__name__)
router = APIRouter(prefix="/demo", tags=["demo"], dependencies=[Depends(demo_enabled)])

Db = Annotated[Session, Depends(get_db)]
Cfg = Annotated[Settings, Depends(get_settings)]


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def admit_live(db: Session, request: Request, settings: Settings) -> Settings:
    """Take one live slot (or 429) and return the settings a live question runs with. Also used by POST /ask
    while DEMO_MODE is on, so /docs can't get around the cap."""
    try:
        live.quota.take(db, client_ip(request), settings, datetime.now(UTC))
    except live.QuotaExceeded as e:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, e.message) from None
    return live.live_settings(settings)


@router.get("/scenarios")
def list_scenarios() -> list[dict[str, Any]]:
    """The recorded questions a visitor can replay, and what each one shows."""
    return [s.summary() for s in scenarios().values()]


@router.post("/replay/{name}", response_model=DemoAskOut)
def replay_scenario(name: str, db: Db, settings: Cfg) -> DemoAskOut:
    """Run a recorded question through the real pipeline (search, validation, citation checks) with Gemini's
    answer pre-recorded: no Gemini call. Replaces this scenario's earlier replay."""
    sc = scenarios().get(name)
    if sc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown scenario")
    return demo_ask_out(db, replay(db, sc, settings))


@router.get("/status", response_model=DemoStatus)
def demo_status(db: Db, settings: Cfg) -> DemoStatus:
    used = live.quota.used_today(db, datetime.now(UTC))
    return DemoStatus(
        live_daily_limit=settings.demo_live_daily_limit, live_used_today=used,
        live_left_today=max(settings.demo_live_daily_limit - used, 0),
        live_per_ip_per_hour=settings.demo_live_per_ip_per_hour, live_model=settings.demo_live_model,
        question_max_chars=settings.demo_question_max_chars,
    )


@router.post("/ask", response_model=DemoAskOut, responses={429: {"description": "Live quota used up"}})
def live_ask(body: LiveAskIn, request: Request, db: Db, settings: Cfg,
             embedder: Annotated[Embedder, Depends(get_embedder)],
             llm: Annotated[LLMClient, Depends(get_llm)]) -> DemoAskOut:
    """Your own question about the demo documents, answered by real Gemini (flash-lite) and checked like every
    answer. Capped per day and per visitor. A Gemini outage gives status `failed`, never an error."""
    question = body.question.strip()
    if len(question) > settings.demo_question_max_chars:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                            f"questions are limited to {settings.demo_question_max_chars} characters")
    run_settings = admit_live(db, request, settings)
    return demo_ask_out(db, ask_question(db, embedder, llm, run_settings, question))


@router.post("/reset", include_in_schema=False)
def reset(db: Db, settings: Cfg, x_reset_token: Annotated[str, Header()] = "") -> dict[str, int]:
    """Nightly (GitHub Actions): delete visitors' questions. Off unless DEMO_RESET_TOKEN is set."""
    if not settings.demo_reset_token:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not Found")
    if not hmac.compare_digest(x_reset_token.encode(), settings.demo_reset_token.encode()):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "bad reset token")
    result = reset_demo(db, settings)
    live.quota.reset()  # re-read from the (now empty) DB on the next request
    log_event(log, logging.INFO, "demo_reset", **result)
    return result
