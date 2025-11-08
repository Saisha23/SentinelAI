from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.services.connection_manager import ConnectionManager
from app.services.detection_service import DetectionService
<<<<<<< HEAD
import asyncio
=======
from app.models.detection import Detection, Alert
from app.api.zones import router as zones_router
>>>>>>> 1aae5ae8accadd37937e33ae75d59dc057e08560

app = FastAPI(title="Sentinel AI Backend")

# ✅ Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


# 🎥 WebSocket for live video/detection stream
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()  # expecting base64 image
            detection_results = await detection_service.process_frame(data)
            await websocket.send_json(detection_results)
            await asyncio.sleep(0.1)  # reduce CPU load
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


# 🚨 WebSocket for alerts
@app.websocket("/ws/alerts")
async def alert_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            alert = await detection_service.get_alerts()
            if alert:
                await websocket.send_json(alert.dict())
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
