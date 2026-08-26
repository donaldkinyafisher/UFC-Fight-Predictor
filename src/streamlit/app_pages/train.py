import streamlit as st
from src.utils import load_data, api_post, api_get, load_model_metrics, load_model, load_preprocessed_data
import pandas as pd
import requests
import streamlit as st
from src.ml.training import DEFAULT_COMPARISON_MODELS, DEFAULT_METRICS_PATH, train_model
import shap
from matplotlib import pyplot as plt
import json

#st.title("Train and select model")

historical_fights_df = load_data()

#Preview data 
st.subheader("Preview Trainining Data")
st.dataframe(historical_fights_df.head())

#Train model
st.subheader("Train Model")

#Display Metrics - reload model_metrics if user presses button to train model
selected_model_to_train = st.multiselect("Select model to train", options=['All'] + DEFAULT_COMPARISON_MODELS)
if selected_model_to_train and 'All' in selected_model_to_train:
    selected_model_to_train = DEFAULT_COMPARISON_MODELS

#Tune Hyper-paramaters
tune_hyperparameters = st.checkbox("Tune Hyper-parameters")
st.warning(" Hyper-parameter tuning is currently not availbale for the Pytorch MLP model. Tuning may take a long time depending on the model and the number of trials.")
tune = True if tune_hyperparameters else False

if st.button("Train", type="primary") and selected_model_to_train:
    with st.spinner("Training model..."):
        try:
            results = train_model(
                models=selected_model_to_train,
                tune = tune
            )
        except Exception as exc:
            raise ValueError(f"Could not train model: {exc}")
        st.success(f"Model trained and metrics saved.")
        for m in selected_model_to_train:
            st.write(f"Classification report for {m}")
            classifcation_report = pd.DataFrame.from_dict(results[m]['classification_report'])
            st.table(classifcation_report)
                                                      
st.subheader("Model Metrics")
try:
    model_metrics = load_model_metrics()
except (OSError, json.JSONDecodeError, requests.RequestException) as exc:
    st.warning(f"Could not load model metrics: {exc}")
    model_metrics = {}

#Show metrics in a table format
metrics_df = pd.DataFrame.from_dict(model_metrics, orient="index")
st.table(metrics_df.drop(columns=["classification_report"], errors="ignore"))

### -----------------

st.subheader("View Feature Importance")
selected_model_name = st.selectbox("Select Model", options=DEFAULT_COMPARISON_MODELS)
X_train, y_train, X_test, y_test, feature_names = load_preprocessed_data()
feature_names_simple = [name.split("__")[-1] for name in feature_names]
if selected_model_name:

    model = load_model(selected_model_name)
    

    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    #fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names_simple, show=False)
    fig = plt.gcf()
    ax = plt.gca()

    # Transparent background
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # White text for axis labels, title, ticks
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(colors='white', which='both')

    # White text for the feature name labels on the y-axis (these are separate text artists)
    for text in ax.get_yticklabels():
        text.set_color('white')
    for text in ax.get_xticklabels():
        text.set_color('white')

    # SHAP summary_plot also creates a colorbar — its ticks/labels need updating too
    # It's usually the last axes added to the figure
    for axis in fig.axes:
        if axis is not ax:  # this is the colorbar axis
            axis.patch.set_alpha(0.0)
            axis.tick_params(colors='white')
            for text in axis.get_yticklabels():
                text.set_color('white')
            if axis.yaxis.label:
                axis.yaxis.label.set_color('white')

    st.pyplot(fig, transparent=True)
