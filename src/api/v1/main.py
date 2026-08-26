from fastapi import FastAPI

from src.api.v1.database import Base, engine
from src.api.v1.routers import eventsRouter, fighters

# Start with synchrnours API
Base.metadata.create_all(bind=engine)

# Base.metadata.create_all(bind=engine)  # Create tables in the database if they don't exist - Synchronous
app = FastAPI(title="UFC Analytics API", version="1.0.0")



app.include_router(fighters.router, prefix="/api/v1/fighters", tags=["fighters"])
app.include_router(eventsRouter.router, prefix="/api/v1/events", tags=["events"])

@app.get("/", include_in_schema=False, name="home")
def home():
    return {"message": "Backend is running!"}

@app.get("/health", name='health')
async def health():
    return {"status": "ok"}

## Exception Handling
