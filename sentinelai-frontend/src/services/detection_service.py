import cv2
import base64
import asyncio
import numpy as np

class DetectionService:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise Exception("❌ Could not access webcam")

    async def process_frame(self, data=None):
        """Capture and send webcam frame in base64"""
        ret, frame = self.cap.read()
        if not ret:
            return {"error": "Frame not captured"}

        # Resize to avoid lag
        frame = cv2.resize(frame, (640, 480))

        # Draw bounding boxes (temporary simulation)
        cv2.putText(frame, "Live Camera Feed", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

        # Encode as base64
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        # Send detections (currently dummy)
        return {
            "frame": encoded_image,
            "detections": []
        }

    async def get_alerts(self):
        await asyncio.sleep(5)
        return {"message": "✅ Camera feed running smoothly"}
