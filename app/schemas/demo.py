"""API models for the public demo (/demo/*)."""

from pydantic import BaseModel, Field

from app.schemas.ask import AskOut


class DemoAskOut(AskOut):
    """An ask as the demo page shows it: also the text of every source the model was given, so the visitor can
    see each quote inside its chunk (POST /ask returns only the sources' locations)."""

    source_texts: dict[str, str] = Field(default_factory=dict)  # label -> chunk text


class LiveAskIn(BaseModel):
    question: str = Field(min_length=3, max_length=300)


class DemoStatus(BaseModel):
    live_daily_limit: int
    live_used_today: int
    live_left_today: int
    live_per_ip_per_hour: int
    live_model: str
    question_max_chars: int
