import json

from app.models import Event, Fight, Fighter, Prediction


def fighter_to_dict(fighter: Fighter | None) -> dict | None:
    if fighter is None:
        return None
    return {
        "id": fighter.id,
        "name": fighter.name,
        "profile_url": fighter.profile_url,
        "height": fighter.height,
        "weight": fighter.weight,
        "reach": fighter.reach,
        "stance": fighter.stance,
        "dob": fighter.dob,
        "wins": fighter.wins,
        "losses": fighter.losses,
        "draws": fighter.draws,
        "last_synced_at": fighter.last_synced_at.isoformat() if fighter.last_synced_at else None,
    }


def fight_to_dict(fight: Fight) -> dict:
    return {
        "id": fight.id,
        "event_id": fight.event_id,
        "red_fighter": fighter_to_dict(fight.red_fighter),
        "blue_fighter": fighter_to_dict(fight.blue_fighter),
        "weight_class": fight.weight_class,
        "bout_type": fight.bout_type,
        "source_url": fight.source_url,
    }


def event_to_dict(event: Event) -> dict:
    return {
        "id": event.id,
        "name": event.name,
        "event_date": event.event_date.isoformat() if event.event_date else None,
        "location": event.location,
        "source_url": event.source_url,
        "status": event.status,
        "fights": [fight_to_dict(fight) for fight in event.fights],
    }


def prediction_to_dict(prediction: Prediction) -> dict:
    fight = fight_to_dict(prediction.fight)
    fight["event"] = {
        "id": prediction.fight.event.id,
        "name": prediction.fight.event.name,
        "event_date": prediction.fight.event.event_date.isoformat() if prediction.fight.event.event_date else None,
    }
    return {
        "id": prediction.id,
        "fight_id": prediction.fight_id,
        "fight": fight,
        "predicted_winner": fighter_to_dict(prediction.predicted_winner),
        "red_win_probability": prediction.red_win_probability,
        "blue_win_probability": prediction.blue_win_probability,
        "model_name": prediction.model_name,
        "feature_snapshot": json.loads(prediction.feature_snapshot),
        "created_at": prediction.created_at.isoformat() if prediction.created_at else None,
    }

