"""Re-run the verification checks on stored answers (no LLM call, free).

    docker compose run --rm -T api uv run --frozen python -m scripts.recheck_asks [--dry-run]

Use after adding or changing a check. Idempotent: an answer whose flags don't change is left
alone. Only answered / answered_with_flags rows change (refused and failed have nothing to check).
"""

from __future__ import annotations

import argparse

from sqlalchemy import select

from app.checks.base import SourceText
from app.checks.runner import run_checks
from app.config import get_settings
from app.db.models import Ask, Chunk, Document
from app.db.session import SessionLocal
from app.schemas.answer import LLMAnswer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    settings = get_settings()
    changed = 0
    with SessionLocal() as db:
        asks = db.scalars(select(Ask).where(Ask.status.in_(["answered", "answered_with_flags"]))
                          .order_by(Ask.created_at)).all()
        for ask in asks:
            answer = LLMAnswer.model_validate(ask.answer_json)
            sources = []
            for s in ask.sources:
                chunk, doc = db.execute(select(Chunk, Document).join(Document).where(Chunk.id == s["chunk_id"])).one()
                sources.append(SourceText(label=s["label"], text=chunk.text, heading_path=chunk.heading_path,
                                          title=doc.title, path=doc.path, similarity=s["similarity"]))
            flags = [f.to_dict() for f in run_checks(ask.question, answer, sources,
                                                     min_similarity=settings.min_similarity)]
            status = "answered_with_flags" if flags else "answered"
            mark = "  " if (flags == ask.flags and status == ask.status) else "* "
            print(f"{mark}{status:<20} {ask.question[:60]}")
            for f in flags:
                print(f"      {f['severity']:<7} {f['code']:<20} stmt {f['statement']}: {f['reason'][:110]}")
            if mark == "* ":
                changed += 1
                if not args.dry_run:
                    ask.flags, ask.status = flags, status
        if not args.dry_run:
            db.commit()
    print(f"\n{len(asks)} answers checked, {changed} {'would change' if args.dry_run else 'updated'}")


if __name__ == "__main__":
    main()
