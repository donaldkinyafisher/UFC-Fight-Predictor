"""Prediction helpers for upcoming UFC fights.

The persisted estimators are trained on scaled numeric features and one-hot
encoded categorical features.  This module accepts the raw fight dataframe
used by the app and recreates that feature transformation before predicting.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.api.v1.models import Fight
from src.api.v1.schemas import PredictionResult


HISTORICAL_FIGHTS_PATH = Path(__file__).resolve().parents[1] / "data" / "historical_fights.csv"
MODELS_DIR = Path(__file__).resolve().parent / "artifacts" / "models"
# Logistic regression has the highest recorded validation accuracy in the
# bundled metrics and does not require optional XGBoost runtime support.
DEFAULT_MODEL_NAME = "logistic_regression"
MODEL_FILE_SUFFIX = ".joblib"
TARGET_COLUMN = "winner"
LABEL_MAPPING = {"blue": 0, "red": 1}
NUMERIC_FEATURES = [
    "red_fighter_height", "red_fighter_reach", "red_fighter_slpm_cs", "red_fighter_str_acc_cs",
    "red_fighter_sapm_cs", "red_fighter_str_def_cs", "red_fighter_td_avg_cs", "red_fighter_td_acc_cs",
    "red_fighter_td_def_cs", "red_fighter_sub_avg_cs", "blue_fighter_height", "blue_fighter_reach",
    "blue_fighter_slpm_cs", "blue_fighter_str_acc_cs", "blue_fighter_sapm_cs", "blue_fighter_str_def_cs",
    "blue_fighter_td_avg_cs", "blue_fighter_td_acc_cs", "blue_fighter_td_def_cs", "blue_fighter_sub_avg_cs",
    "weight_class",
]
CATEGORICAL_FEATURES = ["red_fighter_stance", "blue_fighter_stance", "sex"]


def predict_fights(fights: pd.DataFrame, model_name: str = DEFAULT_MODEL_NAME) -> pd.DataFrame:
    """Predict the winner of each fight in a raw upcoming-fights dataframe.

    ``fights`` may contain additional columns; only the features used during
    training are selected.  The returned dataframe preserves its index and
    includes red and blue win probabilities plus ``predicted_winner``.
    """
    if not isinstance(fights, pd.DataFrame):
        raise TypeError("fights must be a pandas DataFrame.")
    if fights.empty:
        return _empty_prediction_frame(fights.index)

    model = load_trained_model(model_name)
    transformed_features = _preprocess_fights(fights)
    red_probabilities = _red_win_probabilities(model, transformed_features)
    blue_probabilities = 1.0 - red_probabilities

    predictions = pd.DataFrame(index=fights.index)
    if "red_fighter_name" in fights:
        predictions["red_fighter_name"] = fights["red_fighter_name"]
    if "blue_fighter_name" in fights:
        predictions["blue_fighter_name"] = fights["blue_fighter_name"]
    predictions["red_win_probability"] = red_probabilities.round(4)
    predictions["blue_win_probability"] = blue_probabilities.round(4)
    predictions["predicted_winner"] = np.where(red_probabilities >= 0.5, "red", "blue")
    predictions["model_name"] = model_name
    return predictions


def load_trained_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Load a named estimator from ``app/ml/artifacts/models``."""
    if not model_name or Path(model_name).name != model_name or model_name.endswith(MODEL_FILE_SUFFIX):
        raise ValueError("model_name must be an artifact name without a file extension.")

    model_path = MODELS_DIR / f"{model_name}{MODEL_FILE_SUFFIX}"
    if not model_path.is_file():
        available_models = sorted(path.stem for path in MODELS_DIR.glob(f"*{MODEL_FILE_SUFFIX}"))
        raise FileNotFoundError(
            f"Trained model '{model_name}' was not found in {MODELS_DIR}. "
            f"Available models: {', '.join(available_models) or 'none'}"
        )

    try:
        return joblib.load(model_path)
    except AttributeError as exc:
        # Older PyTorch artifacts were saved while FightWinnerNet lived in a
        # script's __main__ module.  Supply that compatibility alias on load.
        if model_name != "pytorch_mlp" or "FightWinnerNet" not in str(exc):
            raise
        setattr(importlib.import_module("__main__"), "FightWinnerNet", _fight_winner_net_class())
        return joblib.load(model_path)


@lru_cache(maxsize=1)
def _training_preprocessor():
    """Recreate the preprocessor fitted to the deterministic training split."""
    if not HISTORICAL_FIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Historical fights data not found: {HISTORICAL_FIGHTS_PATH}")

    historical_fights = _prepare_training_frame(pd.read_csv(HISTORICAL_FIGHTS_PATH))
    features = _raw_feature_frame(historical_fights)
    labels = historical_fights[TARGET_COLUMN].map(LABEL_MAPPING).astype(int)
    x_train, _x_test = train_test_split(
        features,
        test_size=0.2,
        random_state=42,
        stratify=labels if labels.nunique() > 1 else None,
    )
    preprocessor = _build_preprocessor(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    preprocessor.fit(x_train)
    return preprocessor


def _preprocess_fights(fights: pd.DataFrame) -> np.ndarray:
    transformed = _training_preprocessor().transform(_raw_feature_frame(fights))
    return transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)


def _raw_feature_frame(fights: pd.DataFrame) -> pd.DataFrame:
    """Select features in the exact raw order used during model training."""
    frame = pd.DataFrame(index=fights.index)
    for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        frame[column] = fights[column] if column in fights.columns else np.nan
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].where(frame[column].notna(), np.nan)
    return frame


def _red_win_probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    """Return probabilities for label 1 (the red fighter) for any saved model."""
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        classes = np.asarray(model.classes_)
        red_index = _red_class_index(classes)
        return probabilities[:, red_index]

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        probabilities = expit(scores)
        classes = np.asarray(model.classes_)
        return probabilities if classes[-1] == 1 else 1.0 - probabilities

    # FightWinnerNet is the only supplied estimator without the sklearn API.
    try:
        import torch

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(features, dtype=torch.float32))
        return torch.sigmoid(logits).cpu().numpy()
    except Exception as exc:
        raise TypeError("The selected model does not expose a supported prediction interface.") from exc


def _red_class_index(classes: np.ndarray) -> int:
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError(f"Model classes must include label 1 for the red fighter; got {classes.tolist()}.")
    return int(matches[0])


def _empty_prediction_frame(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "red_win_probability": pd.Series(index=index, dtype=float),
            "blue_win_probability": pd.Series(index=index, dtype=float),
            "predicted_winner": pd.Series(index=index, dtype=str),
            "model_name": pd.Series(index=index, dtype=str),
        },
        index=index,
    )


def _prepare_training_frame(fights: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN not in fights.columns:
        raise ValueError(f"Historical fights data must include a '{TARGET_COLUMN}' column.")
    prepared = fights.copy()
    prepared[TARGET_COLUMN] = prepared[TARGET_COLUMN].astype(str).str.lower().str.strip()
    prepared = prepared[prepared[TARGET_COLUMN].isin(LABEL_MAPPING)].copy()
    if prepared.empty:
        raise ValueError("Historical fights data contains no red/blue winner rows.")
    return prepared


def _build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_features = [column for column in NUMERIC_FEATURES if column in feature_columns]
    categorical_features = [column for column in CATEGORICAL_FEATURES if column in feature_columns]
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ],
        remainder="drop",
    )


def _fight_winner_net_class():
    """Define the legacy network class only when its serialized artifact is used."""
    import torch
    from torch import nn

    class FightWinnerNet(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (64, 32), dropout: float = 0.2) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            previous_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)))
                previous_dim = hidden_dim
            layers.append(nn.Linear(previous_dim, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, features: Any):
            return self.network(features).squeeze(1)

    return FightWinnerNet


# def predict_fight(fight: Fight, model_name: str = DEFAULT_MODEL_NAME) -> PredictionResult:
#     """Compatibility wrapper for API callers that hold a database ``Fight``."""
#     fight_frame = pd.DataFrame(
#         [
#             {
#                 "red_fighter_name": fight.red_fighter.name,
#                 "blue_fighter_name": fight.blue_fighter.name,
#                 "red_fighter_height": _measurement_to_cm(fight.red_fighter.height),
#                 "red_fighter_reach": _measurement_to_cm(fight.red_fighter.reach),
#                 "red_fighter_stance": fight.red_fighter.stance,
#                 "blue_fighter_height": _measurement_to_cm(fight.blue_fighter.height),
#                 "blue_fighter_reach": _measurement_to_cm(fight.blue_fighter.reach),
#                 "blue_fighter_stance": fight.blue_fighter.stance,
#                 "weight_class": _weight_class_to_lbs(fight.weight_class),
#             }
#         ]
#     )
#     prediction = predict_fights(fight_frame, model_name=model_name).iloc[0]
#     red_probability = float(prediction["red_win_probability"])
#     blue_probability = float(prediction["blue_win_probability"])
#     return PredictionResult(
#         predicted_winner_id=fight.red_fighter_id if red_probability >= blue_probability else fight.blue_fighter_id,
#         red_win_probability=red_probability,
#         blue_win_probability=blue_probability,
#         model_name=model_name,
#         feature_snapshot=fight_frame.iloc[0].to_dict(),
#     )


def _measurement_to_cm(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = value.strip().lower()
    try:
        if "cm" in cleaned:
            return float(cleaned.replace("cm", "").strip())
        if "'" in cleaned:
            feet, inches = cleaned.replace('"', "").split("'", 1)
            return (float(feet.strip()) * 12 + float(inches.strip() or 0)) * 2.54
        if cleaned.endswith('"'):
            return float(cleaned[:-1].strip()) * 2.54
        return float(cleaned)
    except ValueError:
        return None


def _weight_class_to_lbs(value: str | None) -> float | None:
    if not value:
        return None
    weight_classes = {
        "strawweight": 115.0, "flyweight": 125.0, "bantamweight": 135.0,
        "featherweight": 145.0, "lightweight": 155.0, "welterweight": 170.0,
        "middleweight": 185.0, "light heavyweight": 205.0, "heavyweight": 265.0,
    }
    cleaned = value.lower()
    for name, pounds in weight_classes.items():
        if name in cleaned:
            return pounds
    return None
