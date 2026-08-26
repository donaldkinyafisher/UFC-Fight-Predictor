import sys
from pathlib import Path

import streamlit as st


# Make the repository root importable when Streamlit executes page modules.
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
	sys.path.insert(0, str(project_root))

# 1. Define the pages pointing to your file paths
home_page = st.Page("app_pages/homepage.py", title="Home", icon="🏠", default=True)
data_page = st.Page("app_pages/analytics_dashboard.py", title="Analytics Dashboard", icon="📊")
ml_page = st.Page("app_pages/train.py", title="ML Model Training", icon="🤖")
predictions_page = st.Page("app_pages/predict.py", title="Predictions", icon="📈")
betting_page = st.Page("app_pages/bet.py", title="Betting", icon="💰")

# 2. Group them into a navigation menu
pg = st.navigation([home_page, data_page, ml_page, predictions_page, betting_page])

# 3. Run the selected page
pg.run()



