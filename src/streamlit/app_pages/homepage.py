from __future__ import annotations 

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import json
from pathlib import Path

from src.utils import load_data, get_upcoming_events, load_upcoming_events, load_scheduled_fights
 
API_BASE_URL="http://127.0.0.1:8000/api/v1"

st.set_page_config(page_title="UFC Analytics", layout="wide")

events_df = pd.DataFrame()
scheduled_fights_df = load_scheduled_fights()
historical_fights_df = load_data()
with (Path(__file__).resolve().parents[2] / "data" / "fighters.json").open("r") as f:
    fighters_data = json.load(f)
    
st.title("UFC Data Analytics")

st.subheader("Upcoming Events")
if st.button("Sync upcoming events"):
    sync_response = requests.post(
        f"{API_BASE_URL}/events/upcoming/sync",
        timeout=120,
    )
    sync_response.raise_for_status()
    st.success("Events synced.")

try:
    response = requests.get(
        f"{API_BASE_URL}/events/upcoming",
        timeout=30,
    )
    response.raise_for_status()

    events = response.json()
    events_df = pd.DataFrame(events)
    st.dataframe(events_df, use_container_width=True, hide_index=True)

except requests.exceptions.Timeout:
    events_df = pd.DataFrame()
    st.error("The API took too long to respond.")

except requests.exceptions.RequestException as error:
    events_df = pd.DataFrame()
    st.error(f"API request failed: {error}")

metric_cols = st.columns(2)
metric_cols[0].metric("Upcoming Events", len(events_df))
metric_cols[1].metric("Scheduled Fights", len(scheduled_fights_df))


st.subheader("Historical Data Overview")
ml_data_cols = st.columns(2)
ml_data_cols[0].metric("Total Fights", len(historical_fights_df))
ml_data_cols[1].metric("Unique Fighters", len(set(historical_fights_df['red_fighter_name']).union(set(historical_fights_df['blue_fighter_name']))))