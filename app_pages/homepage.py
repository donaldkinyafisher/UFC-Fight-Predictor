import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from app.utils import load_data, api_get, load_upcoming_events, load_scheduled_fights
import json 

st.set_page_config(page_title="UFC Analytics", layout="wide")

events_df = load_upcoming_events()
scheduled_fights_df = load_scheduled_fights()
historical_fights_df = load_data()
with open("app/data/fighters.json", "r") as f:
    fighters_data = json.load(f)
    
st.title("UFC Data Analytics")

metric_cols = st.columns(4)
metric_cols[0].metric("Upcoming Events", len(events_df))
metric_cols[1].metric("Scheduled Fights", len(scheduled_fights_df))
metric_cols[2].metric("Tracked Fighters", len(fighters_data))

if scheduled_fights_df.empty:
    st.info("No Scheduled fights stored yet. Use Sync Scheduled Fights in the sidebar.")
else:
    st.subheader("Upcoming Fights")
    st.dataframe(
        scheduled_fights_df[
            [
                "event_date",
                "event_name",
                "red_fighter_name",
                "blue_fighter_name",
                "bout_type"
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.subheader("ML Training Data Statistics")
ml_data_cols = st.columns(2)
ml_data_cols[0].metric("Total Fights", len(historical_fights_df))
ml_data_cols[1].metric("Unique Fighters", len(set(historical_fights_df['red_fighter_name']).union(set(historical_fights_df['blue_fighter_name']))))