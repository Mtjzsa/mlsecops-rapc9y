"""Tests for feature engineering and the model pipeline."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression

from bike_demand.config import DATE_COLUMN
from bike_demand.data import split_features_target
from bike_demand.features import DATE_FEATURES, add_date_features, build_pipeline, build_preprocessor


def test_add_date_features_from_datetime(clean_df: pd.DataFrame) -> None:
    result = add_date_features(clean_df)
    assert set(DATE_FEATURES) <= set(result.columns)
    first = result.iloc[0]  # 2018-01-01 is a Monday
    assert (first["month"], first["day_of_week"], first["is_weekend"]) == (1, 0, 0)


def test_add_date_features_from_iso_strings() -> None:
    df = pd.DataFrame({DATE_COLUMN: ["2018-06-16", "2018-06-18"]})  # Saturday, Monday
    result = add_date_features(df)
    assert result["is_weekend"].tolist() == [1, 0]
    assert result["month"].tolist() == [6, 6]


def test_add_date_features_does_not_modify_input(clean_df: pd.DataFrame) -> None:
    columns_before = list(clean_df.columns)
    add_date_features(clean_df)
    assert list(clean_df.columns) == columns_before


def test_preprocessor_output_shape(clean_df: pd.DataFrame) -> None:
    features = add_date_features(clean_df)
    transformed = build_preprocessor().fit_transform(features)
    # 9 numeric + 3 date features + one-hot encoded categories
    n_categories = sum(features[c].nunique() for c in ("seasons", "holiday", "functioning_day"))
    assert transformed.shape == (len(clean_df), 12 + n_categories)


def test_pipeline_fits_and_predicts_on_raw_records(clean_df: pd.DataFrame) -> None:
    features, target = split_features_target(clean_df)
    pipeline = build_pipeline(LinearRegression()).fit(features, target)

    request = features.head(3).assign(date=features[DATE_COLUMN].head(3).dt.strftime("%Y-%m-%d"))
    predictions = pipeline.predict(request)

    assert len(predictions) == 3


def test_pipeline_ignores_unknown_category(clean_df: pd.DataFrame) -> None:
    features, target = split_features_target(clean_df)
    pipeline = build_pipeline(LinearRegression()).fit(features, target)
    unseen = features.head(1).assign(seasons="Monsoon")
    assert len(pipeline.predict(unseen)) == 1
