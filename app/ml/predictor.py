from app.models import Fight, Fighter
from app.schemas import PredictionResult


def predict_fight(fight: Fight) -> PredictionResult:
    red_score = _record_score(fight.red_fighter)
    blue_score = _record_score(fight.blue_fighter)
    total = red_score + blue_score

    if total <= 0:
        red_probability = 0.5
    else:
        red_probability = red_score / total

    red_probability = max(0.05, min(0.95, red_probability))
    blue_probability = 1 - red_probability
    predicted_winner_id = fight.red_fighter_id if red_probability >= blue_probability else fight.blue_fighter_id

    return PredictionResult(
        predicted_winner_id=predicted_winner_id,
        red_win_probability=round(red_probability, 4),
        blue_win_probability=round(blue_probability, 4),
        model_name="baseline_record_heuristic",
        feature_snapshot={
            "red_fighter": fight.red_fighter.name,
            "blue_fighter": fight.blue_fighter.name,
            "red_wins": fight.red_fighter.wins,
            "red_losses": fight.red_fighter.losses,
            "blue_wins": fight.blue_fighter.wins,
            "blue_losses": fight.blue_fighter.losses,
            "weight_class": fight.weight_class,
        },
    )


def _record_score(fighter: Fighter) -> float:
    wins = fighter.wins or 0
    losses = fighter.losses or 0
    draws = fighter.draws or 0
    total = wins + losses + draws
    if total == 0:
        return 1.0
    return 1.0 + wins + (wins / total) - (losses * 0.25)

