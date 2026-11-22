"""Streamlit monitoring dashboard - a thin UI layer, all logic lives in ``bike_demand.monitoring``.

Run locally:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st
import streamlit.components.v1 as components

from bike_demand.config import Settings
from bike_demand.data import clean, load_csv
from bike_demand.monitoring import (
    NotEnoughDataError,
    generate_drift_report,
    send_prediction_requests,
    simulate_records,
)
from bike_demand.predict import ModelNotAvailableError

API_URL = os.environ.get("API_URL", "http://localhost:8000")
settings = Settings.from_env()

st.set_page_config(page_title="Seoul Bike Demand - Monitoring", layout="wide")
st.title("Seoul Bike Demand - Monitoring")

# --- API status -------------------------------------------------------------------------------
st.header("1. API status")
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    st.success(f"API is running - model loaded: {health['model_loaded']}, version: {health['model_version']}")
except requests.RequestException as error:
    st.error(f"API is not reachable at {API_URL}: {error}")

# --- Traffic simulation -----------------------------------------------------------------------
st.header("2. Simulate prediction requests")
st.write("Random rows of the dataset are sent to `/predict`. A temperature shift simulates data drift.")
col_n, col_shift = st.columns(2)
n_records = col_n.slider("Number of records", min_value=10, max_value=1000, value=200, step=10)
shift = col_shift.slider("Temperature shift (°C)", min_value=-20.0, max_value=20.0, value=0.0, step=1.0)
if st.button("Send requests"):
    records = simulate_records(clean(load_csv(settings.data_path)), n_records, temperature_shift=shift)
    try:
        sent = send_prediction_requests(API_URL, records)
        st.success(f"{sent} records sent to the API.")
    except requests.RequestException as error:
        st.error(f"Sending requests failed: {error}")

# --- Drift report -----------------------------------------------------------------------------
st.header("3. Evidently data drift report")
st.write("Reference: training data of the champion model. Current: features of all logged `/predict` requests.")
if st.button("Generate drift report"):
    try:
        with st.spinner("Running Evidently..."):
            summary, html = generate_drift_report(settings)
    except (NotEnoughDataError, ModelNotAvailableError) as error:
        st.warning(str(error))
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Dataset drift", "YES" if summary.dataset_drift else "no")
        m2.metric("Drifted columns", f"{summary.n_drifted_columns} / {summary.n_columns}")
        m3.metric("Share of drifted columns", f"{summary.share_of_drifted_columns:.0%}")
        if summary.drifted_columns:
            st.write("Drifted columns: " + ", ".join(summary.drifted_columns))
        st.caption("The report is also saved to MLflow (experiment: seoul-bike-demand-monitoring).")
        components.html(html, height=1200, scrolling=True)
