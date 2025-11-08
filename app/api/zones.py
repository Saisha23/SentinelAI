from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
from app.core import detection_service
import torch

router = APIRouter()

# ==========================================
# 🧩 Data Models
# ==========================================
class ZoneCreate(BaseModel):
    id: str
    name: str
    polygon: List[Tuple[float, float]]

class ZoneUpdate(BaseModel):
    name: str | None = None
    polygon: List[Tuple[float, float]] | None = None

class ZoneOut(BaseModel):
    id: str
    name: str
    polygon: List[Tuple[int, int]]

# ==========================================
# 🧠 Load YOLOv8 model
# ==========================================
try:
    model = YOLO("weapon_yolov8.pt")  # adjust path if inside models/
    print("✅ YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"⚠️ Model load failed: {e}")
    model = None

# ==========================================
# 📦 CRUD: Zone Management
# ==========================================
@router.get("/zones", response_model=List[ZoneOut])
async def list_zones():
    return detection_service.zone_manager.list_zones()

@router.post("/zones", response_model=ZoneOut)
async def create_zone(zone: ZoneCreate):
    existing = [z for z in detection_service.zone_manager.zones if z["id"] == zone.id]
    if existing:
        raise HTTPException(status_code=409, detail="Zone with this id already exists")

    detection_service.zone_manager.add_zone(zone.id, zone.name, zone.polygon)
    z = next((z for z in detection_service.zone_manager.zones if z["id"] == zone.id), None)
    if z is None:
        raise HTTPException(status_code=500, detail="Failed to create zone")

    return {"id": z["id"], "name": z["name"], "polygon": z["polygon"].tolist()}

@router.get("/zones/{zone_id}", response_model=ZoneOut)
async def get_zone(zone_id: str):
    z = next((z for z in detection_service.zone_manager.zones if z["id"] == zone_id), None)
    if not z:
        raise HTTPException(status_code=404, detail="Zone not found")
    return {"id": z["id"], "name": z["name"], "polygon": z["polygon"].tolist()}

@router.put("/zones/{zone_id}", response_model=ZoneOut)
async def update_zone(zone_id: str, upd: ZoneUpdate):
    ok = detection_service.zone_manager.update_zone(zone_id, name=upd.name, polygon=upd.polygon)
    if not ok:
        raise HTTPException(status_code=404, detail="Zone not found")
    z = next((z for z in detection_service.zone_manager.zones if z["id"] == zone_id), None)
    return {"id": z["id"], "name": z["name"], "polygon": z["polygon"].tolist()}

@router.delete("/zones/{zone_id}")
async def delete_zone(zone_id: str):
    ok = detection_service.zone_manager.remove_zone(zone_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Zone not found")
    return {"status": "deleted"}

# ==========================================
# 🚨 YOLOv8 Detection Endpoint
# ==========================================
@router.post("/zones/detect")
async def detect_objects(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="YOLOv8 model not loaded.")

    # Convert image file to numpy array
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Run YOLO prediction
    results = model.predict(source=frame, conf=0.4, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            detections.append({
                "type": label,
                "confidence": f"{conf * 100:.1f}%"
            })

    return {"detections": detections}
