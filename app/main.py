from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import json
import asyncio
import cv2
import numpy as np
from app.services.connection_manager import ConnectionManager
from app.services.detection_service import DetectionService
from app.models.detection import Detection, Alert
from app.api.zones import router as zones_router

app = FastAPI(title="Sentinel AI Backend")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
connection_manager = ConnectionManager()
detection_service = DetectionService()

# include API router for zones
app.include_router(zones_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Sentinel AI Backend Service"}

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Process received frame data
            detection_results = await detection_service.process_frame(data)
            
            # Send back detection results
            await websocket.send_json(detection_results)
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)

@app.websocket("/ws/alerts")
async def alert_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            alert = await detection_service.get_alerts()
            if alert:
                await websocket.send_json(alert.dict())
            await asyncio.sleep(0.1)  # Small delay to prevent CPU overload
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)