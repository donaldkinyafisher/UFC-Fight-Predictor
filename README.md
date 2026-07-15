# UFC Data Analytics Dashboard

Streamlit project for collecting UFC fight data, displaying analytics, and storing feature-ready records for prediction models.

## Setup

Environment is handled through uv package manager.

```
git clone [repo-name]
uv sync

```

## Run

Start the dashboard in the terminal:

```bash
streamlit run main.py
```

## Project roadmap

> **Last updated:** 2026-07-15  

- [x] ML Model Tuning
    - [ ] Display Feature Importance
    - [ ] Hyper-parameter Tuning
- [ ] Build "Upcoming Fight" predictor
  - [ ] Fetch Fight data for upcoming event
  - [ ] Predict Winner
- [ ] Betting Performance Tracker
    - [ ] Update model to compute odds/probabilites instead of winning.
    - [ ] Scrape betting odds for upcoming event.
    - [ ] Predictor analyzes odds and profitable oppurtunities based on betting odds.

### Legend
- 🟢 On Track  🟡 At Risk  🔴 Blocked  ⚪ Not Started  ✅ Complete