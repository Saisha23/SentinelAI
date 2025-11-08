import base64
import cv2
import numpy as np
import json
from typing import List, Dict, Optional
from app.models.detection import Detection, Alert
import asyncio
from datetime import datetime
from app.services.tracker import Tracker
from app.services.zone_manager import ZoneManager


class DetectionService:
    def __init__(self):
        self.alert_queue = asyncio.Queue()
        self.last_detections = {}
        # Use a lightweight tracker when DeepSort isn't available
        self.tracker = Tracker(max_age=30, iou_threshold=0.3)
        self.zone_manager = ZoneManager()
        # Example: add a default restricted zone (placeholder coordinates)
        # Zones should be configured by an API in a real system
        self.zone_manager.add_zone("zone_1", "Entrance", [(100,100),(400,100),(400,400),(100,400)])

    async def process_frame(self, frame_data: str, detections: Optional[List[Dict]] = None) -> Dict:
        """
        Process incoming frame data and return detection & tracking results.

        - frame_data: base64-encoded image bytes (kept for compatibility)
        - detections: optional list produced by YOLOv8: [{"bbox":(x,y,w,h), "class":str, "confidence":float}]

        If `detections` is not provided, this function will attempt to decode the frame
        and run a model (left as an integration point for Member 1).
        """
        # Keep compatibility with previous signature
        frame = None
        if frame_data:
            try:
                frame = self._decode_frame(frame_data)
            except Exception:
                frame = None

        # If detections not provided, call placeholder detector
        if detections is None:
            detections = self._process_detections(frame)

        # Update tracker with detections
        tracking_results = self._process_tracking(detections)

        # Check for anomalies (placeholder for anomaly detection integration)
        anomalies = self._check_anomalies(tracking_results)

        # Generate alerts if necessary
        await self._generate_alerts(tracking_results, anomalies)

        return {
            "detections": detections,
            "tracks": tracking_results,
            "anomalies": anomalies,
        }

    def _decode_frame(self, frame_data: str) -> np.ndarray:
        """
        Decode base64 frame data to numpy array
        """
        img_data = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def _process_detections(self, frame: Optional[np.ndarray]) -> List[Dict]:
        """
        Placeholder for YOLOv8 detection integration.
        Return detections as list of dicts: {"bbox":(x,y,w,h), "class":str, "confidence":float}
        """
        # Member 1 will replace this with actual model inference.
        return []

    def _process_tracking(self, detections: List[Dict]) -> List[Dict]:
        """
        Use the Tracker service to maintain object identities across frames.
        """
        # Ensure detections are in expected dict format
        normalized = []
        for d in detections:
            if isinstance(d, dict) and "bbox" in d:
                normalized.append({
                    "bbox": tuple(map(float, d["bbox"])),
                    "class": d.get("class", "unknown"),
                    "confidence": float(d.get("confidence", 0.0)),
                })

        tracks = self.tracker.update(normalized)
        # Attach zone info
        for t in tracks:
            zones = self.zone_manager.check_bbox_violations(tuple(t["bbox"]))
            t["zones"] = [z["id"] for z in zones]
        return tracks

    def _check_anomalies(self, tracking_results: List[Dict]) -> List[Dict]:
        """
        Placeholder for anomaly detection integration.
        Returns list of anomaly dicts.
        """
        return []

    async def _generate_alerts(self, tracking_results: List[Dict], anomalies: List[Dict]):
        """
        Generate alerts based on tracking results, zone violations, and anomalies.
        """
        # Zone violations
        for track in tracking_results:
            if track.get("zones"):
                alert = Alert(
                    type="zone_violation",
                    severity="high",
                    timestamp=datetime.now(),
                    details={
                        "object_id": track.get("id"),
                        "object_type": track.get("class"),
                        "location": track.get("bbox"),
                        "zones": track.get("zones"),
                    },
                )
                await self.alert_queue.put(alert)

        # Anomalies
        for anomaly in anomalies:
            alert = Alert(
                type="anomaly_detected",
                severity=anomaly.get("severity", "medium"),
                timestamp=datetime.now(),
                details=anomaly,
            )
            await self.alert_queue.put(alert)

    async def get_alerts(self) -> Optional[Alert]:
        """
        Get the next alert from the queue, or None if none available.
        """
        try:
            return await self.alert_queue.get()
        except asyncio.QueueEmpty:
            return None