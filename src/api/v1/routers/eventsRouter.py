"""
REST API for UFC upcoming events.

Wraps the Playwright scraper in a FastAPI service with a simple in-memory
TTL cache, since scraping takes a few seconds and the source page only
changes occasionally (not something you want to re-run on every request).

Run:
    pip install fastapi uvicorn playwright
    playwright install chromium
    uvicorn api:app --reload

Then:
GET http://localhost:8000/events/upcoming
POST http://localhost:8000/events/upcoming/sync
    GET http://localhost:8000/health
"""

import asyncio
from typing import Annotated
from datetime import datetime, timezone

from fastapi import Depends, APIRouter, HTTPException
from sqlalchemy.orm import Session

from src.api.v1.database import get_db
from src.api.v1.schemas.eventsSchema import EventResponse, EventSyncResponse
from src.api.v1.services.eventsServices import list_upcoming_events, sync_upcoming_events
from src.scrapers.upcoming_events_scraper import scrape_upcoming_events

router = APIRouter()

# Only one browser/database synchronization may run per API process.
_refresh_lock = asyncio.Lock()


@router.get("/upcoming", response_model=list[EventResponse])
def get_upcoming_events(db: Annotated[Session, Depends(get_db)]):
    utc_now = datetime.now(timezone.utc)
    return list_upcoming_events(db, utc_now.date())


@router.post("/upcoming/sync", response_model=EventSyncResponse)
async def sync_events(db: Annotated[Session, Depends(get_db)]):
    """
    Runs the scraper to get update upcoming events. 
    Inserts new events into the events table, and updates already stored events to past,
    if event_date < current_date.
    """
    async with _refresh_lock:
        try:
            raw_events = await scrape_upcoming_events(headless=True)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Could not scrape UFCStats: {exc}") from exc

        utc_now = datetime.now(timezone.utc)
        today = utc_now.date()
        try:
            result = sync_upcoming_events(db, raw_events, today)
            db.commit()
        except Exception:
            db.rollback()
            raise

        events = list_upcoming_events(db, today)
        synchronized_at = datetime.now(timezone.utc)
        return EventSyncResponse(
            **result.__dict__, synchronized_at=synchronized_at, events=events
        )
