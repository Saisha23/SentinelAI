# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import zones, metrics

app = FastAPI(title="SentinelAI Backend")

# Allow Streamlit or other frontends to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include your API routes
app.include_router(zones.router, prefix="/api/zones", tags=["Zones"])
app.include_router(metrics.router, prefix="/api", tags=["Metrics"])

@app.get("/")
def root():
    return {"message": "SentinelAI backend is running successfully"}
