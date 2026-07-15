import streamlit as st
import pandas as pd
import os
import json
from pathlib import Path
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MODEL_METRICS_PATH = Path("app/ml/model_metrics.json")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("app/data/historical_fights.csv")
    except FileNotFoundError:
        st.error("Historical fights data not found. Run the import_ufcdata.py script to fetch the data.")
        st.stop()
    return df

def api_get(path: str):
    response = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()

def api_post(path: str, params: dict | None = None):
    response = requests.post(f"{API_BASE_URL}{path}", params=params, timeout=120)
    response.raise_for_status()
    return response.json()

def load_model_metrics() -> dict:
    if MODEL_METRICS_PATH.exists():
        with MODEL_METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
            return json.load(metrics_file)
    else:
        raise FileNotFoundError(f"Model metrics file not found at {MODEL_METRICS_PATH}.")