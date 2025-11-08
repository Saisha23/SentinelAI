import cv2
import base64
import asyncio
import numpy as np
from ultralytics import YOLO

class DetectionService:
    def __init__(self):
        # Load YOLOv8 model
        try:
            self.model = YOLO("app/models/weapon_yolov8.pt")
            print("✅ YOLOv8 model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Model load failed: {e}")
            self.model = None

        # Initialize webcam
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise Exception("❌ Could not access webcam")

    async def process_frame(self, data=None):
        """Capture frame, run YOLOv8 detection, and return annotated frame"""
        ret, frame = self.cap.read()
        if not ret:
            return {"error": "Frame not captured"}

        frame = cv2.resize(frame, (640, 480))
        detections = []

        if self.model:
            results = self.model(frame, verbose=False)
            for result in results:
                for box in result.boxes:
                    # Extract bounding box and confidence
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = self.model.names[cls] if hasattr(self.model, "names") else f"Class {cls}"

                    # Append detection data
                    detections.append({
                        "type": label,
                        "confidence": round(conf * 100, 2)
                    })

                    # Draw box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Fallback dummy output if model fails
            cv2.putText(frame, "Model not loaded", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode frame as base64 to send to frontend
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        return {
            "frame": encoded_image,
            "detections": detections
        }

    async def get_alerts(self):
        """Simulate backend alert refresh"""
        await asyncio.sleep(5)
        return {"message": "✅ Camera feed running with YOLOv8 anomaly detection"}
