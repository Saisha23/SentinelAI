from pydantic import BaseModel
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class Detection(BaseModel):
    id: str
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x, y, width, height
    timestamp: datetime

class Alert(BaseModel):
    type: str  # e.g., "zone_violation", "anomaly_detected", "weapon_detected"
    severity: str  # "low", "medium", "high"
    timestamp: datetime
    details: Dict

class TrackedObject(BaseModel):
    id: int
    class_name: str
    track_history: List[Tuple[float, float]]  # List of x,y coordinates
    current_bbox: Tuple[float, float, float, float]
    last_seen: datetime

class AnomalyEvent(BaseModel):
    id: str
    type: str
    confidence: float
    location: Tuple[float, float]
    timestamp: datetime
    affected_objects: List[int]  # List of tracked object IDs involved