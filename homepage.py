import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from utils import load_data, api_get

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


fights_df = pd.DataFrame()
historical_fights_df = load_data()
    
st.title("UFC Data Analytics")

metric_cols = st.columns(4)
metric_cols[0].metric("Upcoming Events", len(fights_df))
metric_cols[1].metric("Scheduled Fights", len(fights_df))
metric_cols[2].metric("Tracked Fighters", len(set(fights_df.get("red_fighter", [])) | set(fights_df.get("blue_fighter", []))) if not fights_df.empty else 0)
metric_cols[3].metric("Weight Classes", fights_df["weight_class"].nunique() if not fights_df.empty else 0)

if fights_df.empty:
    st.info("No Upcoming fights stored yet. Use Sync Upcoming Fights in the sidebar.")
else:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Upcoming Fight Cards")
        st.dataframe(
            fights_df[
                [
                    "date",
                    "event",
                    "red_fighter",
                    "blue_fighter",
                    "weight_class",
                    "red_record",
                    "blue_record",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    with right:
        counts = fights_df.groupby("weight_class", dropna=False).size().reset_index(name="fights")
        fig = px.bar(counts, x="fights", y="weight_class", orientation="h", title="Fights by Weight Class")
        st.plotly_chart(fig, use_container_width=True)

st.subheader("ML Training Data Statistics")
ml_data_cols = st.columns(2)
ml_data_cols[0].metric("Total Fights", len(historical_fights_df))
ml_data_cols[1].metric("Unique Fighters", len(set(historical_fights_df['red_fighter_name']).union(set(historical_fights_df['blue_fighter_name']))))