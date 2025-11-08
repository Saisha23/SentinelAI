"""
Simple IoU-based multi-object tracker.
This is a lightweight tracker to provide persistent IDs across frames when
DeepSort is not available. It performs greedy IoU matching and keeps a history
of centroids per track.

Interface:
- tracker = Tracker(max_age=30, iou_threshold=0.3)
- tracks = tracker.update(detections)

Where `detections` is a list of dicts: {"bbox": (x,y,w,h), "class": str, "confidence": float}
Returns list of tracks: {"id": int, "bbox": (x,y,w,h), "class": str, "confidence": float, "history": [(cx,cy), ...]}
"""
from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np

def iou_bbox(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    """Compute IoU of two bboxes in (x,y,w,h) format."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a1x, a1y = ax, ay
    a2x, a2y = ax + aw, ay + ah
    b1x, b1y = bx, by
    b2x, b2y = bx + bw, by + bh

    inter_x1 = max(a1x, b1x)
    inter_y1 = max(a1y, b1y)
    inter_x2 = min(a2x, b2x)
    inter_y2 = min(a2y, b2y)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

class Track:
    def __init__(self, tid: int, bbox: Tuple[float,float,float,float], class_name: str, confidence: float):
        self.id = tid
        self.bbox = bbox
        self.class_name = class_name
        self.confidence = confidence
        self.age = 0  # frames since creation
        self.time_since_update = 0
        self.history = [self._centroid(bbox)]

    def _centroid(self, bbox):
        x,y,w,h = bbox
        return (float(x + w/2.0), float(y + h/2.0))

    def update(self, bbox, confidence=None):
        self.bbox = bbox
        if confidence is not None:
            self.confidence = confidence
        self.history.append(self._centroid(bbox))
        self.time_since_update = 0

class Tracker:
    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks: List[Track] = []

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        detections: list of {"bbox":(x,y,w,h), "class": str, "confidence": float}
        returns list of tracks dicts
        """
        # If no detections, age tracks and remove old ones
        if len(detections) == 0:
            for tr in self.tracks:
                tr.time_since_update += 1
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return [self._to_dict(t) for t in self.tracks]

        # Build IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for i, tr in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = iou_bbox(tr.bbox, det["bbox"]) if len(self.tracks) > 0 else 0.0

        assigned_tracks = set()
        assigned_dets = set()

        # Greedy matching by highest IoU
        if iou_matrix.size > 0:
            pairs = []
            for i in range(iou_matrix.shape[0]):
                for j in range(iou_matrix.shape[1]):
                    pairs.append((iou_matrix[i,j], i, j))
            pairs.sort(reverse=True, key=lambda x: x[0])
            for score, ti, di in pairs:
                if score < self.iou_threshold:
                    break
                if ti in assigned_tracks or di in assigned_dets:
                    continue
                # Assign
                self.tracks[ti].update(detections[di]["bbox"], confidence=detections[di].get("confidence"))
                self.tracks[ti].class_name = detections[di].get("class", self.tracks[ti].class_name)
                assigned_tracks.add(ti)
                assigned_dets.add(di)

        # Create new tracks for unassigned detections
        for j, det in enumerate(detections):
            if j in assigned_dets:
                continue
            new_track = Track(self.next_id, det["bbox"], det.get("class","unknown"), det.get("confidence", 0.0))
            self.next_id += 1
            self.tracks.append(new_track)

        # Age unassigned tracks
        for i, tr in enumerate(self.tracks):
            if i not in assigned_tracks:
                tr.time_since_update += 1

        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [self._to_dict(t) for t in self.tracks]

    def _to_dict(self, tr: Track) -> Dict:
        return {
            "id": tr.id,
            "bbox": tuple(map(float, tr.bbox)),
            "class": tr.class_name,
            "confidence": float(tr.confidence),
            "history": list(tr.history)
        }