from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.models import Event, Fight
from app.repositories.fighters import upsert_fighter
from app.schemas import ScrapedEvent


def upsert_events_with_fights(db: Session, events: list[ScrapedEvent]) -> int:
    stored = 0
    for scraped_event in events:
        event = db.scalar(select(Event).where(Event.name == scraped_event.name))
        if event is None:
            event = Event(name=scraped_event.name)
            db.add(event)

        event.event_date = scraped_event.event_date
        event.location = scraped_event.location
        event.source_url = scraped_event.source_url
        event.status = "upcoming"
        db.flush()

        for scraped_fight in scraped_event.fights:
            red = upsert_fighter(db, scraped_fight.red_fighter)
            blue = upsert_fighter(db, scraped_fight.blue_fighter)
            fight = db.scalar(
                select(Fight).where(
                    Fight.event_id == event.id,
                    Fight.red_fighter_id == red.id,
                    Fight.blue_fighter_id == blue.id,
                )
            )
            if fight is None:
                fight = Fight(event_id=event.id, red_fighter_id=red.id, blue_fighter_id=blue.id)
                db.add(fight)
            fight.weight_class = scraped_fight.weight_class
            fight.bout_type = scraped_fight.bout_type
            fight.source_url = scraped_fight.source_url

        stored += 1

    db.commit()
    return stored


def list_upcoming_events(db: Session) -> list[Event]:
    return list(
        db.scalars(
            select(Event)
            .where(Event.status == "upcoming")
            .options(
                joinedload(Event.fights).joinedload(Fight.red_fighter),
                joinedload(Event.fights).joinedload(Fight.blue_fighter),
            )
            .order_by(Event.event_date.asc().nulls_last())
        )
        .unique()
        .all()
    )


def get_fight(db: Session, fight_id: int) -> Fight | None:
    return db.scalar(
        select(Fight)
        .where(Fight.id == fight_id)
        .options(
            joinedload(Fight.event),
            joinedload(Fight.red_fighter),
            joinedload(Fight.blue_fighter),
        )
    )

