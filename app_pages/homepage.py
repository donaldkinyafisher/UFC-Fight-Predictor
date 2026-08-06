import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from app.utils import load_data, api_get
import json 

st.set_page_config(page_title="UFC Analytics", layout="wide")




def flatten_fights(events: list[dict]) -> pd.DataFrame:
    rows = []
    for event in events:
        for fight in event.get("fights", []):
            red = fight.get("red_fighter", {})
            blue = fight.get("blue_fighter", {})
            rows.append(
                {
                    "fight_id": fight["id"],
                    "event": event["name"],
                    "date": event.get("event_date"),
                    "location": event.get("location"),
                    "red_fighter": red.get("name"),
                    "blue_fighter": blue.get("name"),
                    "weight_class": fight.get("weight_class"),
                    "red_record": f"{red.get('wins') or 0}-{red.get('losses') or 0}-{red.get('draws') or 0}",
                    "blue_record": f"{blue.get('wins') or 0}-{blue.get('losses') or 0}-{blue.get('draws') or 0}",
                    "red_wins": red.get("wins") or 0,
                    "blue_wins": blue.get("wins") or 0,
                }
            )
    return pd.DataFrame(rows)

events_df = pd.read_csv("app/data/upcoming_events.csv")
scheduled_fights_df = pd.read_csv("app/data/scheduled_fights.csv")
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