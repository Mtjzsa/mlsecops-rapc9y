"""Feature engineering and the scikit-learn model pipeline."""

from __future__ import annotations

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from bike_demand.config import CATEGORICAL_FEATURES, DATE_COLUMN, NUMERIC_FEATURES

DATE_FEATURES: tuple[str, ...] = ("month", "day_of_week", "is_weekend")


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month, day of week and weekend flag derived from the date column.

    The date may be a datetime column or ISO formatted strings (``YYYY-MM-DD``),
    e.g. as it arrives in a REST API request.
    """
    result = df.copy()
    dates = pd.to_datetime(result[DATE_COLUMN], format="ISO8601")
    result["month"] = dates.dt.month
    result["day_of_week"] = dates.dt.dayofweek
    result["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    return result


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric (and derived date) features, one-hot encode categorical features."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), [*NUMERIC_FEATURES, *DATE_FEATURES]),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )


def build_pipeline(regressor: RegressorMixin) -> Pipeline:
    """Full model pipeline: date features -> preprocessing -> regressor.

    The pipeline accepts cleaned (or API) records including the ``date`` column,
    so the logged MLflow model can be called directly on incoming requests.
    """
    return Pipeline(
        steps=[
            ("date_features", FunctionTransformer(add_date_features)),
            ("preprocess", build_preprocessor()),
            ("regressor", regressor),
        ]
    )
