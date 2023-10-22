# UFC-Fight-Predictor

This is a model based on UFC data which uses fighter statistics to determine who will win a future fight.

## Install

pip install

## Guide

- To get seperated training data from the current UFC data set:
  **from ufc_data import X_train, X_test, y_train, y_test**

- Train and save new models from ufc_model.py

- Hypertune parameters in optune_tuning.py

- Trained models are stored in ./models
