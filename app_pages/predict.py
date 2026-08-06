import streamlit as st
from app.utils import api_post, api_get
import requests
import pandas as pd
from app.ml.training import DEFAULT_COMPARISON_MODELS

#TODO: Generate fight label keys
fight_labels = {}
selected_label = st.selectbox("Fight", list(fight_labels.keys()))

selected_model = st.selectbox("Select Model", options = DEFAULT_COMPARISON_MODELS)


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