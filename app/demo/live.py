"""Live mode for the public demo: a visitor asks their own question about the fixed demo documents, answered by
real Gemini. Everything after admission is the normal POST /ask pipeline; this module only decides which model
answers and whether there is quota left.

Quota: live questions spend the free tier, so there is a daily cap for the whole demo and an hourly cap per
visitor IP. The counter lives in memory (the demo runs as one process) and is re-read from the DB at the first
request of each UTC day, so a restart doesn't hand out a fresh day. The per-IP cap trusts X-Forwarded-For
(spoofable); the daily cap is what protects the quota.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import Ask, LlmCall
from app.demo.replay import REPLAY_PREFIX


class QuotaExceeded(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class LiveQuota:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day: date | None = None
        self._used = 0
        self._by_ip: dict[str, deque[datetime]] = {}

    def reset(self) -> None:
        with self._lock:
            self._day, self._used, self._by_ip = None, 0, {}

    def used_today(self, db: Session, now: datetime) -> int:
        with self._lock:
            self._roll(db, now)
            return self._used

    def take(self, db: Session, ip: str, settings: Settings, now: datetime) -> None:
        """Reserve one live question for this IP, or raise QuotaExceeded."""
        with self._lock:
            self._roll(db, now)
            if self._used >= settings.demo_live_daily_limit:
                raise QuotaExceeded("Today's live questions are used up (the demo runs on a free Gemini tier). "
                                    "The recorded questions still work; live mode is back tomorrow (UTC).")
            recent = self._by_ip.setdefault(ip, deque())
            while recent and now - recent[0] >= timedelta(hours=1):
                recent.popleft()
            if len(recent) >= settings.demo_live_per_ip_per_hour:
                raise QuotaExceeded("You've asked the most live questions allowed per hour. Try the recorded "
                                    "questions, or come back in a little while.")
            recent.append(now)
            self._used += 1

    def _roll(self, db: Session, now: datetime) -> None:
        if self._day != now.date():
            self._day, self._by_ip = now.date(), {}
            self._used = count_live_today(db, now)


def count_live_today(db: Session, now: datetime) -> int:
    """Asks created today (UTC) that called a real model (not a replay)."""
    start = datetime.combine(now.date(), time(), UTC)
    return db.scalar(
        select(func.count(distinct(LlmCall.ask_id)))
        .join(Ask, Ask.id == LlmCall.ask_id)
        .where(Ask.created_at >= start, LlmCall.model.not_like(f"{REPLAY_PREFIX}%"))
    ) or 0


quota = LiveQuota()


def live_settings(settings: Settings) -> Settings:
    """Live questions: flash-lite only, so a day of visitors fits the free tier."""
    return settings.model_copy(update={"gemini_model": settings.demo_live_model, "gemini_fallback_model": ""})
