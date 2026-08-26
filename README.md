# UFC Data Analytics Dashboard

Streamlit project for collecting UFC fight data, displaying analytics, and storing feature-ready records for prediction models.

## Setup

Environment is handled through `uv` package manager. 

```bash
git clone <repo-name>
uv sync
```

## Run

To test the API backend

```
uv run fastapi dev --entrypoint src.api.v1.main:app
```

To run the application, you must initialize the API and Streamlit on separate terminal windows:

```
uv run uvicorn --entrypoint src.api.v1.main:app --reload --port 8000
```

Start the dashboard:

```
streamlit run src/streamlit/main.py
```

## Project roadmap

> **Last updated:** 2026-07-15

### Upcoming event synchronization

The API stores events in the canonical `events` table. Upcoming events are
those with `status="upcoming"` and an `event_date` after the current UTC date.

Refresh the store with:

```bash
curl -X POST http://localhost:8000/api/v1/events/upcoming/sync
curl http://localhost:8000/api/v1/events/upcoming
```

Synchronization marks expired events as `completed`, inserts new events, and
updates matching event metadata without deleting events absent from one scrape.

### Phase 1

- [x] ML Model Tuning 🟡
    - [x] Display Feature Importance 🟡
    - [x] Hyper-parameter Tuning 🟡
- [ ] Build "Upcoming Fight" predictor ⚪
  - [ ] Fetch Fight data for upcoming event
  - [x] Predict Winner
- [ ] Deploy Beta on Streamlit Server ⚪
- [ ] Betting Performance Tracker
    - [ ] Update model to compute odds/probabilites instead of winning.
    - [ ] Scrape betting odds for upcoming event.
    - [ ] Predictor analyzes odds and profitable oppurtunities based on betting odds.

### Phase 2
- [ ] Expansion for BJJ

### Legend
-  🟡 In Progress  🔴 Blocked  ⚪ Not Started  ✅ Complete
