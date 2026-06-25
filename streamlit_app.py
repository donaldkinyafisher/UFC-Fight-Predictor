import os
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MODEL_METRICS_PATH = Path("app/ml/model_metrics.json")

st.set_page_config(page_title="UFC Analytics", layout="wide")


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

def load_data():
    try:
        df = pd.read_csv("app/data/historical_fights.csv")
    except FileNotFoundError:
        st.error("Historical fights data not found. Run the import_ufcdata.py script to fetch the data.")
        st.stop()

    return df

historical_fights_df = load_data()

st.sidebar.title("UFC Analytics")
page = st.sidebar.radio("Page", ["Dashboard", "Predictions", "Historical Fighter Analysis"], label_visibility="collapsed")

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

elif page == "Historical Fighter Analysis":
    st.title("Historical Fighter Analysis")

    #Win Method distribution over time:
    st.subheader("Win Method Distribution Over Time")
    # Group by year and method, the plot a line chart with the count of fights for each method over time
    # Present as a percentage of total fights per year
    # Add a filter to view averages per weight class
    historical_fights_df["year"] = pd.to_datetime(historical_fights_df["event_date"]).dt.year

    weight_classes = ["All"] + sorted(historical_fights_df["weight_class"].dropna().unique())
    selected_weight = st.selectbox("Weight class (lbs)", weight_classes)

    if selected_weight == "All":
        filtered_fights = historical_fights_df
    else:
        filtered_fights = historical_fights_df[historical_fights_df["weight_class"] == selected_weight]

    method_counts = filtered_fights.groupby(["year", "method"]).size().reset_index(name="count")
    method_counts_pct = method_counts.groupby("year").apply(lambda x: x.assign(pct=x["count"] / x["count"].sum() * 100)).reset_index(drop=True)
    fig = px.line(method_counts_pct, x="year", y="pct", color="method")
    st.plotly_chart(fig, use_container_width=True)

    ## View Balance of red vs blue wins
    st.subheader("Red vs Blue Wins")
    red_blue_wins = historical_fights_df['fight_outcome'].value_counts().reset_index()
    fig = px.pie(red_blue_wins, names='fight_outcome', values='count', 
                 color_discrete_map={'red_win':'red',
                                     'blue_win':'blue',
                                     'draw':'gray',
                                     'no_contest':'black'},
                                 )
    st.plotly_chart(fig)

    ## View Results by Height
    st.subheader("Wins by Height Difference")
    # Create a new column for height difference (red_height - blue_height), and evaluate wins for each colour based on this difference.
    # Create a  plot with red/blue wins compared height difference. Lets bucket the height difference into 2.5cm inteervals, show in a bar chart
    historical_fights_df['height_diff'] = historical_fights_df['red_fighter_height'] - historical_fights_df['blue_fighter_height']
    fig = px.histogram(historical_fights_df, x='height_diff', color='fight_outcome', nbins=30)
    st.plotly_chart(fig, use_container_width=True)

    ### 
    fighter_names = sorted(set(historical_fights_df["red_fighter_name"]).union(set(historical_fights_df["blue_fighter_name"])))

    selected_fighter = st.selectbox("Select Fighter", fighter_names)

    if selected_fighter:
        fighter_fights = historical_fights_df[
            (historical_fights_df["red_fighter_name"] == selected_fighter) | (historical_fights_df["blue_fighter_name"] == selected_fighter)
        ]
        st.subheader(f"Fights for {selected_fighter}")

        #
        st.dataframe(
            fighter_fights[
                [
                    "event_date",
                    "event_name",
                    "event_location",
                    "red_fighter_name",
                    "blue_fighter_name",
                    "bout_type",
                    "fight_outcome",
                    "method",
                    "round",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

elif page == "Predictions":
    st.title("Fight Predictions")

    st.subheader("Training Data Statistics")
    #Table Showing summary statistics from historical_fights.csv - number of entries, number of unique fighters
    st.write(f"Total Fights: {len(historical_fights_df)}")
    st.write(f"Unique Fighters: {len(set(historical_fights_df['red_fighter_name']).union(set(historical_fights_df['blue_fighter_name'])))}")
    #Model metrics - accuracy, precision, recall, f1-score - from model_metrics.json
    
    #Display Metrics - reload model_metrics if user presses button to train model
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            try:
                result = api_post("/train")
                st.success(f"Model trained and metrics saved: {result}")
            except requests.RequestException as exc:
                st.error(f"Model training failed: {exc}")
                
    st.subheader("Model Metrics")
    try:
        model_metrics = load_model_metrics()
    except (OSError, json.JSONDecodeError, requests.RequestException) as exc:
        st.warning(f"Could not load or train model metrics: {exc}")
        model_metrics = {}

    #Show metrics in a table format
    metrics_df = pd.DataFrame.from_dict(
        model_metrics["candidate_results"], 
        orient="index"
        )
    st.table(metrics_df)

    st.subheader("Most Important Features")
    st.selectbox("Select Model", options = [model_metrics["model_type"]])

    #Show Shapely 

    st.subheader("Run Prediction for Upcoming Fights")
    if fights_df.empty:
        st.info("Sync upcoming fights before running predictions.")
        st.stop()

    #TODO: Generate fight label keys
    fight_labels = {}
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
