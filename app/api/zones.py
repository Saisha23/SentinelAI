from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
<<<<<<< HEAD

=======
from app.core import detection_service
>>>>>>> 6196de6dec23bb873a6d00cd8be7d35b9d35a0a5

router = APIRouter()

class ZoneCreate(BaseModel):
    id: str
    name: str
    polygon: List[Tuple[float,float]]

class ZoneUpdate(BaseModel):
    name: str | None = None
    polygon: List[Tuple[float,float]] | None = None

class ZoneOut(BaseModel):
    id: str
    name: str
    polygon: List[Tuple[int,int]]

@router.get("/zones", response_model=List[ZoneOut])
async def list_zones():
    return detection_service.zone_manager.list_zones()

@router.post("/zones", response_model=ZoneOut)
async def create_zone(zone: ZoneCreate):
    # prevent duplicate id
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
