# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import zones, training_ws  # ⬅️ import your new WebSocket router

app = FastAPI(title="SentinelAI Backend")

# Allow Streamlit, React, or other frontends to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include your API routes
app.include_router(zones.router, prefix="/api/zones", tags=["Zones"])

# Include your WebSocket training metrics route
app.include_router(training_ws.router, tags=["Training Metrics"])  # ⬅️ add this

@app.get("/")
def root():
    return {"message": "SentinelAI backend is running successfully"}
