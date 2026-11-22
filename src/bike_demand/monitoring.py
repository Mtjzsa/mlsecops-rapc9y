"""Data drift monitoring with Evidently and a request simulator for the Streamlit dashboard.

Reference data: the training data of the current champion (MLflow artifact ``reference/reference.csv``).
Current data: the features of the requests received by ``/predict`` (``logs/predictions.csv``).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Any, cast

import mlflow
import mlflow.artifacts
import pandas as pd
import requests
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from bike_demand.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, Settings
from bike_demand.predict import ModelNotAvailableError
from bike_demand.request_log import RequestLog
from bike_demand.train import MODEL_ALIAS, REFERENCE_ARTIFACT_DIR, REFERENCE_FILE_NAME, to_api_format

MONITORED_FEATURES: tuple[str, ...] = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
MIN_CURRENT_ROWS = 10


class NotEnoughDataError(ValueError):
    """Raised when there are too few logged requests for a meaningful drift report."""


@dataclass(frozen=True)
class DriftSummary:
    """The key numbers of an Evidently data drift report."""

    dataset_drift: bool
    n_columns: int
    n_drifted_columns: int
    share_of_drifted_columns: float
    drifted_columns: list[str]


def monitoring_experiment_name(settings: Settings) -> str:
    return f"{settings.experiment_name}-monitoring"


def load_reference_data(settings: Settings) -> pd.DataFrame:
    """Download the training data of the current champion model from MLflow."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        version = MlflowClient().get_model_version_by_alias(settings.model_name, MODEL_ALIAS)
    except MlflowException as error:
        raise ModelNotAvailableError("No champion model - train a model first.") from error
    if version.run_id is None:  # a version registered without a run has no reference data
        raise ModelNotAvailableError("The champion model has no training run.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = mlflow.artifacts.download_artifacts(
            run_id=version.run_id, artifact_path=f"{REFERENCE_ARTIFACT_DIR}/{REFERENCE_FILE_NAME}", dst_path=tmp_dir
        )
        return pd.read_csv(path)


def build_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> Report:
    """Run Evidently's data drift preset on the monitored feature columns."""
    columns = list(MONITORED_FEATURES)
    mapping = ColumnMapping(
        numerical_features=list(NUMERIC_FEATURES),
        categorical_features=list(CATEGORICAL_FEATURES),
        target=None,
        prediction=None,
    )
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=cast(pd.DataFrame, reference[columns]),
        current_data=cast(pd.DataFrame, current[columns]),
        column_mapping=mapping,
    )
    return report


def summarize_drift(report: Report) -> DriftSummary:
    """Extract the dataset-level result and the drifted columns from the report."""
    results: dict[str, Any] = {item["metric"]: item["result"] for item in report.as_dict()["metrics"]}
    dataset = results["DatasetDriftMetric"]
    by_column: dict[str, Any] = results["DataDriftTable"]["drift_by_columns"]
    return DriftSummary(
        dataset_drift=bool(dataset["dataset_drift"]),
        n_columns=int(dataset["number_of_columns"]),
        n_drifted_columns=int(dataset["number_of_drifted_columns"]),
        share_of_drifted_columns=float(dataset["share_of_drifted_columns"]),
        drifted_columns=sorted(name for name, info in by_column.items() if info["drift_detected"]),
    )


def log_drift_to_mlflow(settings: Settings, summary: DriftSummary, html: str, n_current_rows: int) -> str:
    """Store the drift metrics and the HTML report in the monitoring experiment. Returns the run id."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(monitoring_experiment_name(settings))
    with mlflow.start_run(run_name="drift-report") as run:
        mlflow.log_metrics(
            {
                "dataset_drift": float(summary.dataset_drift),
                "n_drifted_columns": float(summary.n_drifted_columns),
                "share_of_drifted_columns": summary.share_of_drifted_columns,
                "n_current_rows": float(n_current_rows),
            }
        )
        mlflow.log_param("drifted_columns", ",".join(summary.drifted_columns) or "-")
        mlflow.log_text(html, "drift_report.html")
    return run.info.run_id


def generate_drift_report(settings: Settings) -> tuple[DriftSummary, str]:
    """Compare the logged requests with the champion's training data; log the report to MLflow."""
    current = RequestLog(settings.log_dir).read_predictions()
    if len(current) < MIN_CURRENT_ROWS:
        raise NotEnoughDataError(f"At least {MIN_CURRENT_ROWS} logged requests are needed, found {len(current)}.")
    report = build_drift_report(load_reference_data(settings), current)
    summary = summarize_drift(report)
    html = report.get_html()
    log_drift_to_mlflow(settings, summary, html, len(current))
    return summary, html


def simulate_records(
    df: pd.DataFrame, n_records: int, temperature_shift: float = 0.0, seed: int | None = None
) -> pd.DataFrame:
    """Sample API records from a cleaned dataset; a temperature shift simulates data drift."""
    sample = df.sample(n=n_records, replace=n_records > len(df), random_state=seed).drop(columns=[TARGET])
    sample["temperature"] = sample["temperature"] + temperature_shift
    sample["dew_point_temperature"] = sample["dew_point_temperature"] + temperature_shift
    return to_api_format(sample).reset_index(drop=True)


def send_prediction_requests(api_url: str, records: pd.DataFrame, batch_size: int = 50) -> int:
    """POST the records to the ``/predict`` endpoint in batches. Returns the number of records sent."""
    sent = 0
    for start in range(0, len(records), batch_size):
        batch = records.iloc[start:start + batch_size]
        response = requests.post(f"{api_url}/predict", json={"records": batch.to_dict(orient="records")}, timeout=60)
        response.raise_for_status()
        sent += len(batch)
    return sent
