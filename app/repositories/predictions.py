import json

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.models import Fight, Prediction
from app.schemas import PredictionResult


def store_prediction(db: Session, fight_id: int, result: PredictionResult) -> Prediction:
    prediction = db.scalar(select(Prediction).where(Prediction.fight_id == fight_id))
    if prediction is None:
        prediction = Prediction(fight_id=fight_id)
        db.add(prediction)

    prediction.predicted_winner_id = result.predicted_winner_id
    prediction.red_win_probability = result.red_win_probability
    prediction.blue_win_probability = result.blue_win_probability
    prediction.model_name = result.model_name
    prediction.feature_snapshot = json.dumps(result.feature_snapshot)
    db.commit()
    db.refresh(prediction)
    return prediction


def list_predictions(db: Session) -> list[Prediction]:
    return list(
        db.scalars(
            select(Prediction)
            .options(
                joinedload(Prediction.predicted_winner),
                joinedload(Prediction.fight).joinedload(Fight.event),
                joinedload(Prediction.fight).joinedload(Fight.red_fighter),
                joinedload(Prediction.fight).joinedload(Fight.blue_fighter),
            )
            .order_by(Prediction.created_at.desc())
        ).all()
    )

