import os
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from utils import load_data



historical_fights_df = load_data()

st.title("Historical Fighter Analysis")

#Win Method distribution over time:
st.subheader("Win Method Distribution Over Time")
# Group by year and method, the plot a line chart with the count of fights for each method over time
# Present as a percentage of total fights per year
# Add a filter to view averages per weight class
historical_fights_df["year"] = pd.to_datetime(historical_fights_df["event_date"], dayfirst=True).dt.year

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