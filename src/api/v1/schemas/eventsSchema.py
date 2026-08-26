from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class EventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    event_date: date | None
    location: str | None
    source_url: str | None
    status: str


class EventSyncResponse(BaseModel):
    scraped: int
    inserted: int
    updated: int
    completed: int
    malformed: int
    synchronized_at: datetime
    events: list[EventResponse]
