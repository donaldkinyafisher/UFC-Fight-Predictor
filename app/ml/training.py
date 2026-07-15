from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODELS_DIR = ARTIFACT_DIR / "models"
DATA_DIR = ARTIFACT_DIR / "data"
METRICS_DIR = ARTIFACT_DIR / "metrics"
DATASET_PATH = DATA_DIR / "ufc_split_data.npz"
DEFAULT_METRICS_PATH = METRICS_DIR/ "model_metrics.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "winner"
LABEL_MAPPING = {"blue": 0, "red": 1}
INVERSE_LABEL_MAPPING = {value: key for key, value in LABEL_MAPPING.items()}

DEFAULT_COMPARISON_MODELS = [
    "pytorch_mlp",
    "logistic_regression",
    "svm",
    "knn",
    "random_forest",
    "xgboost",
]

NUMERIC_FEATURES = [
    "red_fighter_height",
    "red_fighter_reach",
    "red_fighter_slpm_cs",
    "red_fighter_str_acc_cs",
    "red_fighter_sapm_cs",
    "red_fighter_str_def_cs",
    "red_fighter_td_avg_cs",
    "red_fighter_td_acc_cs",
    "red_fighter_td_def_cs",
    "red_fighter_sub_avg_cs",
    "blue_fighter_height",
    "blue_fighter_reach",
    "blue_fighter_slpm_cs",
    "blue_fighter_str_acc_cs",
    "blue_fighter_sapm_cs",
    "blue_fighter_str_def_cs",
    "blue_fighter_td_avg_cs",
    "blue_fighter_td_acc_cs",
    "blue_fighter_td_def_cs",
    "blue_fighter_sub_avg_cs",
    "weight_class",
]

CATEGORICAL_FEATURES = [
    "red_fighter_stance",
    "blue_fighter_stance",
    "sex",
]


class FightWinnerNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (64, 32), dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)

def load_model(model_name: str):
    return joblib.load(MODELS_DIR / f"{model_name}.joblib")

def load_preprocessed_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list]:
    """

    Args:

    Returns:
        _type_: _description_
    """

    # 1. Open the archive wrapper
    data_archive = np.load(DATASET_PATH)

    # 2. Unpack them directly into variables
    X_train = data_archive['X_train']
    X_test = data_archive['X_test']
    y_train = data_archive['y_train']
    y_test = data_archive['y_test']
    feature_names = data_archive['feature_names']

    # 3. Always close the archive file when finished unpacking
    data_archive.close()

    return X_train, y_train, X_test, y_test, feature_names

def train_model(
    models: list = ["pytorch_mlp"],
    model_configs: dict[str, dict[str, Any]] | None = None,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train one or more fight-winner classifiers. The models are written to disk in the artifacts/models directory, 
    and their evaluation metrics are written to a JSON file in the artifacts/metrics directory.
    """

    model_configs = model_configs or {}
    candidate_model_types = models
    unknown_models = set(candidate_model_types) - set(_MODEL_TRAINERS)
    if unknown_models:
        raise ValueError(f"Unknown model type(s): {', '.join(sorted(unknown_models))}")

    if not DATASET_PATH.exists():

        historical_fights_df = pd.read_csv("app/data/historical_fights.csv")
        prepared = _prepare_training_frame(historical_fights_df)
        feature_columns = _available_feature_columns(prepared)
        if not feature_columns:
            raise ValueError("No usable training features found in historical_fights_df.")

        X = _select_features(prepared, feature_columns)
        y = prepared[TARGET_COLUMN].map(LABEL_MAPPING).astype(int)

        stratify = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        preprocessor = _build_preprocessor(feature_columns)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        feature_names = preprocessor.get_feature_names_out().tolist()

        np.savez_compressed(
            DATASET_PATH, 
            X_train=_as_dense_float32(X_train_processed),
            X_test=_as_dense_float32(X_test_processed),
            y_train=y_train, 
            y_test=y_test,
            feature_names = feature_names
            )
        
    else:
        X_train_processed, y_train, X_test_processed, y_test, _ = load_preprocessed_data()
        
    results = {}
    for candidate in candidate_model_types:
        trainer = _MODEL_TRAINERS[candidate]
        model = trainer(
            X_train_processed,
            y_train,
            X_test_processed,
            y_test,
            random_state=random_state,
            config=model_configs.get(candidate, {}),
        )
        joblib.dump(model, MODELS_DIR / f"{candidate}.joblib")

        metrics = _evaluate_model(candidate, model, X_test_processed, y_test)
        results[candidate] = metrics
        #metrics["train_rows"] = int(len(X_train))
        #metrics["validation_rows"] = int(len(X_test))
        #metrics["feature_columns"] = feature_columns
        #metrics["compared_models"] = candidate_model_types

        # results.append(
        #     SavedFightModel(
        #         model_name=candidate,
        #         preprocessor=preprocessor,
        #         model=model,
        #         feature_columns=feature_columns,
        #         numeric_features=[column for column in NUMERIC_FEATURES if column in feature_columns],
        #         categorical_features=[column for column in CATEGORICAL_FEATURES if column in feature_columns],
        #         metrics=metrics,
        #         label_mapping=LABEL_MAPPING,
        #         model_config=model_configs.get(candidate, {}),
        #     )
        # )

    #Write metrics to file
    _write_metrics(results, metrics_path)

    return results

def _prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"historical_fights_df must include a '{TARGET_COLUMN}' column.")

    prepared = df.copy()
    prepared[TARGET_COLUMN] = prepared[TARGET_COLUMN].astype(str).str.lower().str.strip()
    prepared = prepared[prepared[TARGET_COLUMN].isin(LABEL_MAPPING)].copy()
    if prepared.empty:
        raise ValueError("No red/blue winner rows found after filtering draws and no-contests.")

    return prepared


def _available_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES if column in df.columns]


def _select_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    selected = pd.DataFrame(index=df.index)
    for column in feature_columns:
        selected[column] = df[column] if column in df.columns else np.nan
        if column in NUMERIC_FEATURES:
            selected[column] = pd.to_numeric(selected[column], errors="coerce")
    return selected


def _build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_features = [column for column in NUMERIC_FEATURES if column in feature_columns]
    categorical_features = [column for column in CATEGORICAL_FEATURES if column in feature_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def _train_pytorch_mlp(
    x_train: Any,
    y_train: pd.Series,
    x_val: Any,
    y_val: pd.Series,
    random_state: int,
    config: dict[str, Any],
) -> FightWinnerNet:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    x_train_array = _as_dense_float32(x_train)
    y_train_array = _as_dense_float32(y_train)
    x_val_array = _as_dense_float32(x_val)
    y_val_array = _as_dense_float32(y_val)

    hidden_dims = tuple(config.get("hidden_dims", (64, 32)))
    dropout = float(config.get("dropout", 0.2))
    learning_rate = float(config.get("learning_rate", 0.001))
    batch_size = int(config.get("batch_size", 64))
    epochs = int(config.get("epochs", 12))
    patience = int(config.get("patience", 5))

    model = FightWinnerNet(input_dim=x_train_array.shape[1], hidden_dims=hidden_dims, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train_array), torch.tensor(y_train_array)),
        batch_size=batch_size,
        shuffle=True,
    )

    best_loss = float("inf")
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    epochs_without_improvement = 0

    val_features = torch.tensor(x_val_array)
    val_targets = torch.tensor(y_val_array)

    for _epoch in range(epochs):
        model.train()
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_features), batch_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_features), val_targets).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    return model


def _train_logistic_regression(
    x_train: Any,
    y_train: pd.Series,
    _x_val: Any,
    _y_val: pd.Series,
    random_state: int,
    config: dict[str, Any],
) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=int(config.get("max_iter", 1000)),
        class_weight=config.get("class_weight"),
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return model


def _train_svm(
    x_train: Any,
    y_train: pd.Series,
    _x_val: Any,
    _y_val: pd.Series,
    random_state: int,
    config: dict[str, Any],
) -> SVC:
    model = SVC(
        C=float(config.get("C", 1.0)),
        kernel=config.get("kernel", "rbf"),
        gamma=config.get("gamma", "scale"),
        class_weight=config.get("class_weight"),
        probability=True,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return model


def _train_knn(
    x_train: Any,
    y_train: pd.Series,
    _x_val: Any,
    _y_val: pd.Series,
    random_state: int,
    config: dict[str, Any],
) -> KNeighborsClassifier:
    # Retained for a consistent trainer interface; KNN itself is deterministic.
    _ = random_state
    model = KNeighborsClassifier(
        n_neighbors=int(config.get("n_neighbors", 15)),
        weights=config.get("weights", "distance"),
        p=int(config.get("p", 2)),
        n_jobs=int(config.get("n_jobs", -1)),
    )
    model.fit(x_train, y_train)
    return model


def _train_random_forest(
    x_train: Any,
    y_train: pd.Series,
    _x_val: Any,
    _y_val: pd.Series,
    random_state: int,
    config: dict[str, Any],
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=int(config.get("n_estimators", 300)),
        max_depth=config.get("max_depth"),
        min_samples_leaf=int(config.get("min_samples_leaf", 1)),
        class_weight=config.get("class_weight"),
        n_jobs=int(config.get("n_jobs", -1)),
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return model


def _train_xgboost(
    x_train: Any,
    y_train: pd.Series,
    _x_val: Any,
    _y_val: pd.Series,
    random_state: int,
    config: dict[str, Any],
) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required to train model_type='xgboost'. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc

    model = XGBClassifier(
        n_estimators=int(config.get("n_estimators", 300)),
        max_depth=int(config.get("max_depth", 4)),
        learning_rate=float(config.get("learning_rate", 0.05)),
        subsample=float(config.get("subsample", 0.8)),
        colsample_bytree=float(config.get("colsample_bytree", 0.8)),
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=int(config.get("n_jobs", -1)),
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return model

def _train_transformer_model(
    x_train: Any,
    y_train: pd.Series,
    x_val: Any,
    y_val: pd.Series,
    random_state: int,
    config: dict[str, Any],
) -> Any:
    """
    Placeholder for training a transformer-based model.
    This function should be implemented with the specific transformer architecture and training logic.
    """
    raise NotImplementedError("Transformer model training is not yet implemented.")

def _evaluate_model(model_name: str, model: Any, x_val: Any, y_val: pd.Series) -> dict[str, Any]:
    if model_name == "pytorch_mlp":
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(_as_dense_float32(x_val), dtype=torch.float32))
            probabilities = torch.sigmoid(logits).cpu().numpy()
        predictions = (probabilities >= 0.5).astype(int)
    else:
        predictions = model.predict(x_val)

    precision, recall, f1_score, _support = precision_recall_fscore_support(
        y_val,
        predictions,
        average="binary",
        pos_label=LABEL_MAPPING["red"],
        zero_division=0,
    )
    report = classification_report(
        y_val,
        predictions,
        target_names=["blue", "red"],
        zero_division=0,
        output_dict=True,
    )

    return {
        "accuracy": float(accuracy_score(y_val, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "classification_report": report,
    }


def _as_dense_float32(features: Any) -> np.ndarray:
    if hasattr(features, "toarray"):
        features = features.toarray()
    return np.asarray(features, dtype=np.float32)


def _write_metrics(metrics: dict[str, Any], metrics_path: str | Path) -> None:
    path = Path(metrics_path)

    # Load existing metrics file if present and valid
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                existing = json.load(f) or {}
        except (json.JSONDecodeError, OSError):
            existing = {}

    # Get all results keys
    models = list(metrics.keys())

    #Store metrics under top-level keys by model_key
    for model_key in models:
        existing[model_key] = metrics[model_key]

    with path.open("w", encoding="utf-8") as metrics_file:
        json.dump(existing, metrics_file, indent=2)

    return


_MODEL_TRAINERS: dict[str, Callable[..., Any]] = {
    "pytorch_mlp": _train_pytorch_mlp,
    "logistic_regression": _train_logistic_regression,
    "svm": _train_svm,
    "knn": _train_knn,
    "random_forest": _train_random_forest,
    "xgboost": _train_xgboost,
    "transformer": _train_transformer_model,
}

if __name__ == "__main__":

    #historical_fights_df = pd.read_csv("app/data/historical_fights.csv")
    results = train_model(
        models=DEFAULT_COMPARISON_MODELS
    )
    print("Training completed. Results:\n", results)
