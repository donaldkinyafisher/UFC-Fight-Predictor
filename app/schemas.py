from datetime import date

from pydantic import BaseModel


class ScrapedFighter(BaseModel):
    name: str
    profile_url: str | None = None
    height: str | None = None
    weight: str | None = None
    reach: str | None = None
    stance: str | None = None
    dob: str | None = None
    wins: int | None = None
    losses: int | None = None
    draws: int | None = None


class ScrapedFight(BaseModel):
    red_fighter: ScrapedFighter
    blue_fighter: ScrapedFighter
    weight_class: str | None = None
    bout_type: str | None = None
    source_url: str | None = None


class ScrapedEvent(BaseModel):
    name: str
    event_date: date | None = None
    location: str | None = None
    source_url: str | None = None
    fights: list[ScrapedFight] = []


class PredictionResult(BaseModel):
    predicted_winner_id: int | None
    red_win_probability: float
    blue_win_probability: float
    model_name: str
    feature_snapshot: dict[str, float | int | str | None]

