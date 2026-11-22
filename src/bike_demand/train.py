"""Model training with automatic hyperparameter selection, MLflow tracking and a guarded model registry.

Automatic model improvement / degradation protection:

1. every candidate hyperparameter set is evaluated with cross-validation (nested MLflow runs),
   the best one is selected automatically,
2. the selected model is evaluated on a held-out test set and registered in the MLflow model registry,
3. the new version only gets the ``champion`` alias (= is served by the API) if its test RMSE is not
   worse than the current champion's - so a retraining can never degrade the served model.
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from bike_demand.config import DATE_COLUMN, RANDOM_STATE, Settings
from bike_demand.data import clean, load_csv, split_features_target, train_test_split_df
from bike_demand.features import build_pipeline

MODEL_ALIAS = "champion"
PRIMARY_METRIC = "rmse"
REFERENCE_ARTIFACT_DIR = "reference"
REFERENCE_FILE_NAME = "reference.csv"

ParamGrid = Sequence[dict[str, Any]]

DEFAULT_PARAM_GRID: ParamGrid = (
    {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2},
)


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of one training run."""

    run_id: str
    model_version: str
    params: dict[str, Any]
    metrics: dict[str, float]
    promoted: bool
    previous_champion_rmse: float | None


def build_model(params: dict[str, Any]) -> Pipeline:
    """Full pipeline (features + random forest) for one hyperparameter set."""
    regressor = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    return build_pipeline(regressor)


def evaluate(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    """Regression metrics of a fitted model."""
    predictions = model.predict(features)
    return {
        "rmse": float(root_mean_squared_error(target, predictions)),
        "mae": float(mean_absolute_error(target, predictions)),
        "r2": float(r2_score(target, predictions)),
    }


def cross_validated_rmse(params: dict[str, Any], features: pd.DataFrame, target: pd.Series, folds: int = 3) -> float:
    """Mean cross-validated RMSE of one hyperparameter set."""
    scores = cross_val_score(
        build_model(params), features, target, cv=folds, scoring="neg_root_mean_squared_error"
    )
    return float(-scores.mean())


def select_best_params(
    param_grid: ParamGrid, features: pd.DataFrame, target: pd.Series, folds: int = 3
) -> tuple[dict[str, Any], float]:
    """Evaluate every candidate (as a nested MLflow run) and return the one with the lowest CV RMSE."""
    best_params: dict[str, Any] = {}
    best_rmse = float("inf")
    for index, params in enumerate(param_grid):
        with mlflow.start_run(run_name=f"candidate-{index}", nested=True):
            rmse = cross_validated_rmse(params, features, target, folds)
            mlflow.log_params(params)
            mlflow.log_metric("cv_rmse", rmse)
        if rmse < best_rmse:
            best_params, best_rmse = dict(params), rmse
    return best_params, best_rmse


def to_api_format(features: pd.DataFrame) -> pd.DataFrame:
    """Convert the date column to ISO strings - the format used by the REST API."""
    result = features.copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN]).dt.strftime("%Y-%m-%d")
    return result


def get_champion_rmse(client: MlflowClient, model_name: str) -> float | None:
    """Test RMSE of the current champion model version, or ``None`` if there is no champion yet."""
    try:
        version = client.get_model_version_by_alias(model_name, MODEL_ALIAS)
    except MlflowException:
        return None
    if version.run_id is None:  # a version registered without a run has no metrics to compare
        return None
    run = client.get_run(version.run_id)
    return run.data.metrics.get(PRIMARY_METRIC)


def promote_if_better(client: MlflowClient, model_name: str, version: str, rmse: float) -> tuple[bool, float | None]:
    """Give ``version`` the champion alias only if it is not worse than the current champion."""
    current_rmse = get_champion_rmse(client, model_name)
    promoted = current_rmse is None or rmse <= current_rmse
    if promoted:
        client.set_registered_model_alias(model_name, MODEL_ALIAS, version)
    client.set_model_version_tag(model_name, version, "promoted", str(promoted).lower())
    if current_rmse is not None:
        client.set_model_version_tag(model_name, version, "previous_champion_rmse", f"{current_rmse:.3f}")
    return promoted, current_rmse


def train_and_log(df: pd.DataFrame, settings: Settings, param_grid: ParamGrid | None = None) -> TrainingResult:
    """Train on a cleaned dataframe, log everything to MLflow, register and (maybe) promote the model."""
    grid = DEFAULT_PARAM_GRID if param_grid is None else param_grid
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    client = MlflowClient()

    train, test = train_test_split_df(df)
    train_features, train_target = split_features_target(train)
    test_features, test_target = split_features_target(test)

    with mlflow.start_run(run_name="training") as run:
        params, cv_rmse = select_best_params(grid, train_features, train_target)
        model = build_model(params).fit(train_features, train_target)
        metrics = evaluate(model, test_features, test_target)

        mlflow.log_params(params)
        mlflow.log_params({"n_train_rows": len(train), "n_test_rows": len(test), "n_candidates": len(grid)})
        mlflow.log_metrics({**metrics, "cv_rmse": cv_rmse})

        api_features = to_api_format(test_features)
        signature = infer_signature(api_features, model.predict(api_features))
        model_info = mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)

        # Training data is saved as the reference dataset for the Evidently drift report
        with tempfile.TemporaryDirectory() as tmp_dir:
            reference_path = Path(tmp_dir) / REFERENCE_FILE_NAME
            to_api_format(train).to_csv(reference_path, index=False)
            mlflow.log_artifact(str(reference_path), artifact_path=REFERENCE_ARTIFACT_DIR)

    version = str(mlflow.register_model(model_info.model_uri, settings.model_name).version)
    promoted, previous_rmse = promote_if_better(client, settings.model_name, version, metrics["rmse"])

    return TrainingResult(
        run_id=run.info.run_id,
        model_version=version,
        params=params,
        metrics=metrics,
        promoted=promoted,
        previous_champion_rmse=previous_rmse,
    )


def run_training(settings: Settings, param_grid: ParamGrid | None = None) -> TrainingResult:
    """Load and clean the dataset from ``settings.data_path``, then train."""
    return train_and_log(clean(load_csv(settings.data_path)), settings, param_grid)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point: ``python -m bike_demand.train [--data PATH]``."""
    parser = argparse.ArgumentParser(description="Train the bike demand model and log it to MLflow.")
    parser.add_argument("--data", type=Path, default=None, help="dataset CSV (default: DATA_PATH env var)")
    parser.add_argument(
        "--skip-if-champion", action="store_true", help="do nothing if a champion model exists (container start-up)"
    )
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if args.data is not None:
        settings = replace(settings, data_path=args.data)
    if args.skip_if_champion:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        if get_champion_rmse(MlflowClient(), settings.model_name) is not None:
            print("Champion model already exists - skipping training")
            return 0
    result = run_training(settings)
    status = "promoted to champion" if result.promoted else "NOT promoted (worse than champion)"
    print(f"Model version {result.model_version}: rmse={result.metrics['rmse']:.1f}, {status}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin entry point, main() itself is tested
    raise SystemExit(main())
