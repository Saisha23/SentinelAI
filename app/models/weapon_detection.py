# app/models/weapon_detection.py
import os
from ultralytics import YOLO

# Build an absolute path that works no matter where uvicorn is started
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "weapon_yolov8.pt")

print(f"🔍 Checking for model at: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print("Model file NOT found at path above!")
    model = None
else:
    print("Model file found. Attempting to load...")
    try:
        model = YOLO(MODEL_PATH)
        print("OLOv8 model loaded successfully!")
    except Exception as e:
        print(f"Model load failed with error: {e}")
        model = None


def detect_objects(frame):
    """Detect objects in a given frame using YOLOv8"""
    if model is None:
        return []

    results = model(frame)
    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            if conf > 0.5:
                detections.append({
                    "type": label,
                    "confidence": f"{conf * 100:.0f}%"
                })
    return detections
