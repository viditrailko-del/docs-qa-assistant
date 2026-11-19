"""What the demo page gets back for an ask: the normal AskOut plus each source's text."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Ask, Chunk
from app.schemas.demo import DemoAskOut


def demo_ask_out(db: Session, ask: Ask) -> DemoAskOut:
    ids = [s["chunk_id"] for s in ask.sources]
    texts = {cid: text for cid, text in db.execute(select(Chunk.id, Chunk.text).where(Chunk.id.in_(ids)))} if ids else {}
    out = DemoAskOut.from_ask(ask)
    return out.model_copy(update={"source_texts": {s["label"]: texts.get(s["chunk_id"], "") for s in ask.sources}})
