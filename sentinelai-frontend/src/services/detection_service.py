import cv2
import base64
import asyncio
import numpy as np
import random
from datetime import datetime

class DetectionService:
    def __init__(self):
        # Open webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")

        # Dummy labels
        self.labels = ["person", "car", "dog", "bicycle"]

    async def process_frame(self, data=None):
        """Capture a frame, simulate detections, and return encoded image."""
        ret, frame = self.cap.read()
        if not ret:
            return {"error": "Failed to capture frame"}

        # Resize frame for consistent performance
        frame = cv2.resize(frame, (640, 480))

        detections = []
        # Generate 1–3 fake objects
        for _ in range(random.randint(1, 3)):
            x = random.randint(50, 400)
            y = random.randint(50, 300)
            w = random.randint(50, 120)
            h = random.randint(50, 120)
            label = random.choice(self.labels)
            detections.append({
                "bbox": [x, y, w, h],
                "label": label,
                "confidence": round(random.uniform(0.5, 0.99), 2)
            })

        # Encode frame for streaming (if needed)
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        return {
            "frame": encoded_image,
            "detections": detections
        }

    async def get_alerts(self):
        """Simulate random alerts."""
        await asyncio.sleep(random.uniform(3, 7))
        if random.random() > 0.7:
            return {"message": f"⚠️ Suspicious movement detected at {datetime.now().strftime('%H:%M:%S')}"}
        return {"message": "No threats detected"}
