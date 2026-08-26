import streamlit as st
from src.utils import load_upcoming_events, load_scheduled_fights
import requests
import pandas as pd
from src.ml.training import DEFAULT_COMPARISON_MODELS
from src.ml.predictor import predict_fights

#Select Event to generate predictione for
events_df = load_upcoming_events()
scheduled_fights_df = load_scheduled_fights()
selected_event = st.selectbox("Event", events_df['EVENT'] )

selected_model = st.selectbox("Select Model", options = DEFAULT_COMPARISON_MODELS)

if st.button("Predict", type="primary"):
    with st.spinner("Generating Predictions..."):
        try:
            selected_fights_df = scheduled_fights_df[scheduled_fights_df['event_name'] == selected_event]
            predictions = predict_fights(selected_fights_df, selected_model)
            st.dataframe(predictions, use_container_width=True, hide_index=True)
        except requests.RequestException as exc:
            st.error(f"Prediction failed: {exc}")

st.subheader("Saved Predictions")
saved_predictions = []

if saved_predictions:
    pred_rows = []
    for prediction in saved_predictions:
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