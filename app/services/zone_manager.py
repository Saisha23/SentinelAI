"""
ZoneManager: store polygonal restricted zones and check violations.
Zones are defined as lists of (x,y) points in image coordinate space.
"""
from typing import List, Tuple, Dict
import numpy as np
import cv2

class ZoneManager:
    def __init__(self, zones: List[Dict] = None):
        """zones: list of dict {"id":str, "name":str, "polygon":[(x,y), ...]}"""
        self.zones = zones or []

    def add_zone(self, zone_id: str, name: str, polygon: List[Tuple[float,float]]):
        self.zones.append({"id": zone_id, "name": name, "polygon": np.array(polygon, dtype=np.int32)})

    def remove_zone(self, zone_id: str) -> bool:
        """Remove zone by id. Returns True if removed, False if not found."""
        for i, z in enumerate(self.zones):
            if z["id"] == zone_id:
                self.zones.pop(i)
                return True
        return False

    def update_zone(self, zone_id: str, name: str = None, polygon: List[Tuple[float,float]] = None) -> bool:
        """Update zone name and/or polygon. Returns True if updated."""
        for z in self.zones:
            if z["id"] == zone_id:
                if name is not None:
                    z["name"] = name
                if polygon is not None:
                    z["polygon"] = np.array(polygon, dtype=np.int32)
                return True
        return False

    def check_point_in_zones(self, point: Tuple[float,float]) -> List[Dict]:
        """Return a list of zone dicts the point is inside."""
        px, py = int(round(point[0])), int(round(point[1]))
        hits = []
        for z in self.zones:
            poly = z["polygon"]
            if cv2.pointPolygonTest(poly, (px, py), False) >= 0:
                hits.append(z)
        return hits

    def check_bbox_violations(self, bbox: Tuple[float,float,float,float]) -> List[Dict]:
        """Check centroid of bbox against zones and return matching zones."""
        x,y,w,h = bbox
        cx = x + w/2.0
        cy = y + h/2.0
        return self.check_point_in_zones((cx, cy))

    def list_zones(self):
        return [{"id": z["id"], "name": z["name"], "polygon": z["polygon"].tolist()} for z in self.zones]