# app/api/metrics.py
from fastapi import APIRouter
from pydantic import BaseModel
import json, os

router = APIRouter()

# ======= MODEL =======
class TrainingMetrics(BaseModel):
    current_epoch: int
    train_loss: float
    val_loss: float
    accuracy: float
    status: str
    history: list = []

# ======= FILE TO STORE METRICS =======
METRICS_FILE = "training_metrics.json"

# ======= LOAD EXISTING METRICS =======
def load_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return {
        "current_epoch": 0,
        "train_loss": 0.0,
        "val_loss": 0.0,
        "accuracy": 0.0,
        "status": "Idle",
        "history": []
    }

# ======= SAVE METRICS =======
def save_metrics(data):
    with open(METRICS_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ======= ROUTE: GET CURRENT METRICS =======
@router.get("/metrics")
def get_metrics():
    return load_metrics()

# ======= ROUTE: UPDATE METRICS (from Weapon_detection.ipynb) =======
@router.post("/metrics/update")
def update_metrics(metrics: TrainingMetrics):
    save_metrics(metrics.dict())
    return {"message": "Metrics updated successfully"}
