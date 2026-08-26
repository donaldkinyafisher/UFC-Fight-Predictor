from datetime import date, datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.api.v1.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Fighter(Base):
    __tablename__ = "fighters"
    __table_args__ = (UniqueConstraint("profile_url", name="uq_fighters_profile_url"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(160), index=True)
    profile_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    height: Mapped[str | None] = mapped_column(String(40), nullable=True)
    weight: Mapped[str | None] = mapped_column(String(40), nullable=True)
    reach: Mapped[str | None] = mapped_column(String(40), nullable=True)
    stance: Mapped[str | None] = mapped_column(String(80), nullable=True)
    dob: Mapped[str | None] = mapped_column(String(80), nullable=True)
    wins: Mapped[int | None] = mapped_column(Integer, nullable=True)
    losses: Mapped[int | None] = mapped_column(Integer, nullable=True)
    draws: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_synced_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    red_fights: Mapped[list["Fight"]] = relationship(
        foreign_keys="Fight.red_fighter_id", back_populates="red_fighter"
    )
    blue_fights: Mapped[list["Fight"]] = relationship(
        foreign_keys="Fight.blue_fighter_id", back_populates="blue_fighter"
    )
    fight_statistics: Mapped[list["FightStatistic"]] = relationship(
        back_populates="fighter"
    )
    career_stats: Mapped["FighterCareerStats | None"] = relationship(
        back_populates="fighter", cascade="all, delete-orphan", uselist=False
    )


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (UniqueConstraint("source_url", name="uq_events_source_url"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(220), index=True)
    event_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    location: Mapped[str | None] = mapped_column(String(220), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="upcoming")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, onupdate=utc_now)

    fights: Mapped[list["Fight"]] = relationship(back_populates="event", cascade="all, delete-orphan")


class Fight(Base):
    __tablename__ = "fights"
    __table_args__ = (UniqueConstraint("event_id", "red_fighter_id", "blue_fighter_id", name="uq_event_fighters"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), index=True)
    red_fighter_id: Mapped[int] = mapped_column(ForeignKey("fighters.id"), index=True)
    blue_fighter_id: Mapped[int] = mapped_column(ForeignKey("fighters.id"), index=True)
    weight_class: Mapped[str | None] = mapped_column(String(120), nullable=True)
    bout_type: Mapped[str | None] = mapped_column(String(80), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    winner_side: Mapped[str] = mapped_column(String(20))
    red_result: Mapped[str | None] = mapped_column(String(4), nullable=True)
    blue_result: Mapped[str | None] = mapped_column(String(4), nullable=True)
    method: Mapped[str | None] = mapped_column(String(120), nullable=True)
    round: Mapped[int | None] = mapped_column(Integer, nullable=True)
    time: Mapped[str | None] = mapped_column(String(20), nullable=True)
    time_format: Mapped[str | None] = mapped_column(String(80), nullable=True)

    event: Mapped[Event] = relationship(back_populates="fights")
    red_fighter: Mapped[Fighter] = relationship(
        foreign_keys=[red_fighter_id], back_populates="red_fights"
    )
    blue_fighter: Mapped[Fighter] = relationship(
        foreign_keys=[blue_fighter_id], back_populates="blue_fights"
    )

    statistics: Mapped[list["FightStatistic"]] = relationship(
        back_populates="fight", cascade="all, delete-orphan"
    )
    prediction: Mapped["Prediction | None"] = relationship(
        back_populates="fight",
        cascade="all, delete-orphan",
        single_parent=True,
        uselist=False,
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    fight_id: Mapped[int] = mapped_column(ForeignKey("fights.id"), unique=True, index=True)
    predicted_winner_id: Mapped[int | None] = mapped_column(ForeignKey("fighters.id"), nullable=True)
    red_win_probability: Mapped[float] = mapped_column(Float)
    blue_win_probability: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(120), default="baseline_record_heuristic")
    feature_snapshot: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    fight: Mapped[Fight] = relationship(back_populates="prediction")
    predicted_winner: Mapped[Fighter | None] = relationship(foreign_keys=[predicted_winner_id])

class FightStatistic(Base):
    """One fighter's performance statistics from one fight."""

    __tablename__ = "fight_statistics"
    __table_args__ = (
        UniqueConstraint("fight_id", "fighter_id", name="uq_fight_stats_fighter"),
        UniqueConstraint("fight_id", "corner", name="uq_fight_stats_corner"),
        CheckConstraint("corner IN ('red', 'blue')", name="ck_fight_stats_corner"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    fight_id: Mapped[int] = mapped_column(
        ForeignKey("fights.id", ondelete="CASCADE"), index=True
    )
    fighter_id: Mapped[int] = mapped_column(
        ForeignKey("fighters.id", ondelete="CASCADE"), index=True
    )
    corner: Mapped[str] = mapped_column(String(4))
    knockdowns: Mapped[float | None] = mapped_column(Float, nullable=True)
    significant_strikes_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_strikes_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    takedowns_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    submission_attempts: Mapped[float | None] = mapped_column(Float, nullable=True)
    reversals: Mapped[float | None] = mapped_column(Float, nullable=True)
    control_time: Mapped[str | None] = mapped_column(String(20), nullable=True)
    head_strikes_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_strikes_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    leg_strikes_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    distance_strikes_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    clinch_strikes_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    ground_strikes_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    fight: Mapped[Fight] = relationship(back_populates="statistics")
    fighter: Mapped[Fighter] = relationship(back_populates="fight_statistics")


class FighterCareerStats(Base):
    """Latest aggregate statistics for a fighter.

    This is intentionally the current state only; historical snapshots are
    not stored in this version.
    """

    __tablename__ = "fighter_career_stats"

    fighter_id: Mapped[int] = mapped_column(
        ForeignKey("fighters.id", ondelete="CASCADE"), primary_key=True
    )
    strikes_landed_per_minute: Mapped[float | None] = mapped_column(Float, nullable=True)
    striking_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    strikes_absorbed_per_minute: Mapped[float | None] = mapped_column(Float, nullable=True)
    striking_defense: Mapped[float | None] = mapped_column(Float, nullable=True)
    takedown_average: Mapped[float | None] = mapped_column(Float, nullable=True)
    takedown_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    takedown_defense: Mapped[float | None] = mapped_column(Float, nullable=True)
    submission_average: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, onupdate=utc_now)

    fighter: Mapped[Fighter] = relationship(back_populates="career_stats")
