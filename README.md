# UFC Data Analytics Dashboard

Streamlit project for collecting UFC fight data, displaying analytics, and storing feature-ready records for prediction models.

## Setup

Environment is handled through `uv` package manager. 

```bash
git clone <repo-name>
uv sync
```

## Run

Start the dashboard in the terminal:

```bash
streamlit run main.py
```

## Project roadmap

> **Last updated:** 2026-07-15

### Phase 1

- [ ] ML Model Tuning 🟡
    - [ ] Display Feature Importance 🟡
    - [ ] Hyper-parameter Tuning 🟡
- [ ] Build "Upcoming Fight" predictor ⚪
  - [ ] Fetch Fight data for upcoming event
  - [ ] Predict Winner
- [ ] Deploy Beta on Streamlit Server ⚪
- [ ] Betting Performance Tracker
    - [ ] Update model to compute odds/probabilites instead of winning.
    - [ ] Scrape betting odds for upcoming event.
    - [ ] Predictor analyzes odds and profitable oppurtunities based on betting odds.

### Phase 2
- [ ] Expansion for BJJ

### Legend
-  🟡 In Progress  🔴 Blocked  ⚪ Not Started  ✅ Complete