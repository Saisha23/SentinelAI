from fastapi import APIRouter, WebSocket
import asyncio, json, random

router = APIRouter()

@router.websocket("/ws/training")
async def training_metrics(websocket: WebSocket):
    await websocket.accept()
    try:
        # Simulate sending metrics during training
        for epoch in range(1, 11):
            metrics = {
                "epoch": epoch,
                "accuracy": round(random.uniform(0.80, 0.98), 3),
                "loss": round(random.uniform(0.1, 0.5), 3),
                "precision": round(random.uniform(0.75, 0.95), 3),
                "recall": round(random.uniform(0.70, 0.94), 3),
                "f1_score": round(random.uniform(0.72, 0.96), 3)
            }
            await websocket.send_text(json.dumps(metrics))
            await asyncio.sleep(2)
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
    finally:
        await websocket.close()
