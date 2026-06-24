# UFC Data Analytics Dashboard

Streamlit + FastAPI project for collecting UFC fight data, displaying analytics, and storing feature-ready records for prediction models.

## Stack

- Python
- FastAPI API and scraper service
- Streamlit front-end
- SQL database through SQLAlchemy
- SQLite by default, configurable with `DATABASE_URL`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Start the API:

```bash
uvicorn app.api.main:app --reload
```

Start the dashboard in another terminal:

```bash
streamlit run streamlit_app.py
```

## Useful API Routes

- `GET /health`
- `POST /scrape/upcoming` - scrape and store upcoming events/fights
- `GET /events/upcoming`
- `GET /fighters`
- `POST /predictions/run?fight_id=1`
- `GET /predictions`

## Configuration

Create a `.env` file when you want to override defaults:

```bash
DATABASE_URL=sqlite:///./ufc_analytics.db
UFC_STATS_BASE_URL=http://ufcstats.com
```

For Postgres later:

```bash
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/ufc_analytics
```

