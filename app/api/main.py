import json
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.serializers import event_to_dict, fighter_to_dict, prediction_to_dict
from app.core.database import Base, engine, get_db
from app.ml.predictor import predict_fight
from app.ml.training import DEFAULT_METRICS_PATH, train_model
from app.repositories import events as event_repo
from app.repositories import fighters as fighter_repo
from app.repositories import predictions as prediction_repo
from app.scrapers.ufcstats import UFCStatsScraper

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="UFC Analytics API",
    description="Scrapes upcoming UFC data, stores it in SQL, and exposes analytics-ready endpoints.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/scrape/upcoming")
def scrape_upcoming(db: Session = Depends(get_db)) -> dict[str, int]:
    scraper = UFCStatsScraper()
    try:
        events = scraper.get_upcoming_events()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not scrape UFCStats: {exc}") from exc

    stored_events = event_repo.upsert_events_with_fights(db, events)
    return {
        "events_found": len(events),
        "events_stored": stored_events,
    }


@app.get("/events/upcoming")
def upcoming_events(db: Session = Depends(get_db)):
    return [event_to_dict(event) for event in event_repo.list_upcoming_events(db)]


@app.get("/fighters")
def fighters(search: str | None = Query(default=None), db: Session = Depends(get_db)):
    return [fighter_to_dict(fighter) for fighter in fighter_repo.list_fighters(db, search=search)]


@app.post("/predictions/run")
def run_prediction(fight_id: int, db: Session = Depends(get_db)):
    fight = event_repo.get_fight(db, fight_id)
    if fight is None:
        raise HTTPException(status_code=404, detail="Fight not found")

    result = predict_fight(fight)
    prediction = prediction_repo.store_prediction(db, fight_id=fight_id, result=result)
    return prediction_to_dict(prediction)


@app.get("/predictions")
def predictions(db: Session = Depends(get_db)):
    return [prediction_to_dict(prediction) for prediction in prediction_repo.list_predictions(db)]


@app.post("/predictions/train")
def train_prediction_model():
    data_path = Path("app/data/historical_fights.csv")
    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Historical fights data not found")

    try:
        historical_fights_df = pd.read_csv(data_path)
        saved_model = train_model(historical_fights_df, 
                                  compare_models=["pytorch_mlp", "logistic_regression"]
                                  )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not train model: {exc}") from exc

    return saved_model.metrics


@app.get("/predictions/metrics")
def prediction_model_metrics():
    if not DEFAULT_METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Model metrics not found")

    with DEFAULT_METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)
