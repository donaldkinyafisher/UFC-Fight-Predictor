from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Fighter
from app.schemas import ScrapedFighter


def upsert_fighter(db: Session, scraped: ScrapedFighter) -> Fighter:
    fighter = db.scalar(select(Fighter).where(Fighter.name == scraped.name))
    if fighter is None:
        fighter = Fighter(name=scraped.name)
        db.add(fighter)

    fighter.profile_url = scraped.profile_url or fighter.profile_url
    fighter.height = scraped.height or fighter.height
    fighter.weight = scraped.weight or fighter.weight
    fighter.reach = scraped.reach or fighter.reach
    fighter.stance = scraped.stance or fighter.stance
    fighter.dob = scraped.dob or fighter.dob
    fighter.wins = scraped.wins if scraped.wins is not None else fighter.wins
    fighter.losses = scraped.losses if scraped.losses is not None else fighter.losses
    fighter.draws = scraped.draws if scraped.draws is not None else fighter.draws
    fighter.last_synced_at = datetime.utcnow()
    return fighter


def list_fighters(db: Session, search: str | None = None) -> list[Fighter]:
    statement = select(Fighter).order_by(Fighter.name)
    if search:
        statement = statement.where(Fighter.name.ilike(f"%{search}%"))
    return list(db.scalars(statement).all())

