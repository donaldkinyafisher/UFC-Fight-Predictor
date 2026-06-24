import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="UFC Analytics", layout="wide")


def api_get(path: str):
    response = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, params: dict | None = None):
    response = requests.post(f"{API_BASE_URL}{path}", params=params, timeout=120)
    response.raise_for_status()
    return response.json()


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


st.sidebar.title("UFC Analytics")
page = st.sidebar.radio("Page", ["Dashboard", "Predictions"], label_visibility="collapsed")

if st.sidebar.button("Sync Upcoming Fights", use_container_width=True):
    with st.spinner("Scraping upcoming UFCStats events..."):
        try:
            result = api_post("/scrape/upcoming")
            st.sidebar.success(f"Stored {result['events_stored']} events")
        except requests.RequestException as exc:
            st.sidebar.error(f"Sync failed: {exc}")

try:
    events = api_get("/events/upcoming")
except requests.RequestException as exc:
    st.error(f"Could not reach FastAPI at {API_BASE_URL}. Start it with `uvicorn app.api.main:app --reload`.")
    st.stop()

fights_df = flatten_fights(events)

if page == "Dashboard":
    st.title("UFC Data Analytics")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Upcoming Events", len(events))
    metric_cols[1].metric("Scheduled Fights", len(fights_df))
    metric_cols[2].metric("Tracked Fighters", len(set(fights_df.get("red_fighter", [])) | set(fights_df.get("blue_fighter", []))) if not fights_df.empty else 0)
    metric_cols[3].metric("Weight Classes", fights_df["weight_class"].nunique() if not fights_df.empty else 0)

    if fights_df.empty:
        st.info("No fights stored yet. Use Sync Upcoming Fights in the sidebar.")
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

elif page == "Predictions":
    st.title("Fight Predictions")

    if fights_df.empty:
        st.info("Sync upcoming fights before running predictions.")
        st.stop()

    fight_labels = {
        f"{row.red_fighter} vs {row.blue_fighter} | {row.event}": int(row.fight_id)
        for row in fights_df.itertuples()
    }
    selected_label = st.selectbox("Fight", list(fight_labels.keys()))

    if st.button("Run Prediction", type="primary"):
        with st.spinner("Running baseline model..."):
            try:
                prediction = api_post("/predictions/run", params={"fight_id": fight_labels[selected_label]})
                st.success("Prediction saved")
                st.json(prediction)
            except requests.RequestException as exc:
                st.error(f"Prediction failed: {exc}")

    st.subheader("Saved Predictions")
    try:
        predictions = api_get("/predictions")
    except requests.RequestException:
        predictions = []

    if predictions:
        pred_rows = []
        for prediction in predictions:
            fight = prediction.get("fight", {})
            pred_rows.append(
                {
                    "event": fight.get("event", {}).get("name"),
                    "fight": f"{fight.get('red_fighter', {}).get('name')} vs {fight.get('blue_fighter', {}).get('name')}",
                    "winner": prediction.get("predicted_winner", {}).get("name") if prediction.get("predicted_winner") else None,
                    "red_probability": prediction.get("red_win_probability"),
                    "blue_probability": prediction.get("blue_win_probability"),
                    "model": prediction.get("model_name"),
                }
            )
        st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No saved predictions yet.")
