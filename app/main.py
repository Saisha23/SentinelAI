from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio

# --- Local imports ---
from app.services.connection_manager import ConnectionManager
from app.services.detection_service import DetectionService
from app.models.detection import Detection, Alert
from app.api.zones import router as zones_router

app = FastAPI(title="Sentinel AI Backend")

# --- CORS setup ---
# Later, restrict allow_origins to deployed Streamlit URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize core services ---
connection_manager = ConnectionManager()
detection_service = DetectionService()

# --- Include API routers ---
app.include_router(zones_router, prefix="/api")

# --- Health check ---
@app.get("/")
async def root():
    return {"message": "Sentinel AI Backend Service Active"}


# ============================================================
# 🎥  WebSocket: /ws/stream  →  live frames + detections
# ============================================================
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Client connected to /ws/stream")

    detection_service = DetectionService()

    try:
        while True:
            frame_data = await detection_service.process_frame()
            if "frame" in frame_data:
                await websocket.send_json(frame_data)
            await asyncio.sleep(0.05)
    except Exception as e:
        print(f"❌ Stream error: {e}")
    finally:
        print("⚠️ Closing stream connection")
        await websocket.close()


# ============================================================
# 🚨  WebSocket: /ws/alerts  →  threat / anomaly notifications
# ============================================================
@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            alert = await detection_service.get_alerts()
            if alert:
                # Alert may be a pydantic model or dict
                payload = alert.dict() if hasattr(alert, "dict") else alert
                await websocket.send_json(payload)

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        print(f"[ERROR] WebSocket /ws/alerts → {e}")
        connection_manager.disconnect(websocket)


# --- Run directly in dev mode ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
