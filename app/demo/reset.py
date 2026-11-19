"""Put the demo database back to its starting state: no asks (so no llm_calls), the document index complete.
Used by the nightly POST /demo/reset on the public demo.

Documents are never deleted (visitors can't upload in demo mode); the seed only adds what is missing.
"""

from __future__ import annotations

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import Ask, LlmCall
from app.demo.index import seed_index


def reset_demo(db: Session, settings: Settings) -> dict[str, int]:
    asks = db.scalar(select(func.count()).select_from(Ask)) or 0
    db.execute(text(f"TRUNCATE {Ask.__tablename__}, {LlmCall.__tablename__}"))
    db.commit()
    seeded = seed_index(db, settings)
    return {"asks_deleted": asks, "documents_added": seeded["documents_added"]}
