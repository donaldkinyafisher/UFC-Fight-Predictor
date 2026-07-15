import streamlit as st

# 1. Define the pages pointing to your file paths
home_page = st.Page("homepage.py", title="Home", icon="🏠", default=True)
data_page = st.Page("analytics_dashboard.py", title="Analytics Dashboard", icon="📊")
ml_page = st.Page("train_models.py", title="ML Model Training", icon="🤖")
predictions_page = st.Page("predictions.py", title="Predictions", icon="📈")
betting_page = st.Page("betting.py", title="Betting", icon="💰")

# 2. Group them into a navigation menu
pg = st.navigation([home_page, data_page, ml_page, predictions_page, betting_page])

# 3. Run the selected page
pg.run()



