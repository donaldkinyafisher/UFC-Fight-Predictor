from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Fighter(Base):
    __tablename__ = "fighters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    profile_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    height: Mapped[str | None] = mapped_column(String(40), nullable=True)
    weight: Mapped[str | None] = mapped_column(String(40), nullable=True)
    reach: Mapped[str | None] = mapped_column(String(40), nullable=True)
    stance: Mapped[str | None] = mapped_column(String(80), nullable=True)
    dob: Mapped[str | None] = mapped_column(String(80), nullable=True)
    wins: Mapped[int | None] = mapped_column(Integer, nullable=True)
    losses: Mapped[int | None] = mapped_column(Integer, nullable=True)
    draws: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_synced_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(220), unique=True, index=True)
    event_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    location: Mapped[str | None] = mapped_column(String(220), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="upcoming")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

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

    event: Mapped[Event] = relationship(back_populates="fights")
    red_fighter: Mapped[Fighter] = relationship(foreign_keys=[red_fighter_id])
    blue_fighter: Mapped[Fighter] = relationship(foreign_keys=[blue_fighter_id])
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    fight: Mapped[Fight] = relationship(back_populates="prediction")
    predicted_winner: Mapped[Fighter | None] = relationship(foreign_keys=[predicted_winner_id])
