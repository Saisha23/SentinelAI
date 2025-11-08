"""
Demo script for Member 2 - Object Tracking and Zone Management
Tests the IoU tracker and zone management functionality
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.tracker import Tracker
from app.services.zone_manager import ZoneManager
from app.models.detection import Detection

def create_sample_detections(frame_number):
    """Create sample detections that simulate a person moving across the frame"""
    x = 100 + frame_number * 20  # Move right
    y = 200 + frame_number * 5   # Move slightly down
    
    # Create a Detection object
    detection = Detection(
        id=str(frame_number),
        class_name="person",
        confidence=0.95,
        bbox=(float(x), float(y), 50.0, 100.0),  # x, y, width, height
        timestamp=datetime.now()
    )
    return [detection]

def demo_tracking():
    """Demonstrate the IoU tracker functionality"""
    print("\n=== Testing Tracking Service ===")
    
    # Initialize tracker
    tracker = Tracker()
    
    # Simulate 5 frames of detection
    for frame in range(5):
        detections = create_sample_detections(frame)
        # Convert Detection objects to dictionaries for tracker
        detection_dicts = [{
            "bbox": det.bbox,
            "class": det.class_name,
            "confidence": det.confidence
        } for det in detections]
        
        tracks = tracker.update(detection_dicts)
        
        print(f"\nFrame {frame}:")
        for track in tracks:
            print(f"Track ID: {track['id']}")
            print(f"Current position: {track['bbox']}")
            print(f"Track history length: {len(track['history'])}")

def demo_zone_manager():
    """Demonstrate the zone management functionality"""
    print("\n=== Testing Zone Manager ===")
    
    # Initialize zone manager
    zone_manager = ZoneManager()
    
    # Add a sample restricted zone (rectangle from (150,150) to (350,350))
    zone_polygon = [(150,150), (350,150), (350,350), (150,350)]
    zone_manager.add_zone("zone1", "Restricted Area", zone_polygon)
    
    # Test a point inside the zone
    inside_point = (200, 200)
    print(f"\nTesting point {inside_point}:")
    # Create a bounding box around the test point
    bbox = (inside_point[0]-25, inside_point[1]-25, 50, 50)  # x, y, width, height
    violations = zone_manager.check_bbox_violations(bbox)
    print(f"Violations detected: {len(violations) > 0}")
    
    # Test a point outside the zone
    outside_point = (400, 400)
    print(f"\nTesting point {outside_point}:")
    bbox = (outside_point[0]-25, outside_point[1]-25, 50, 50)  # x, y, width, height
    violations = zone_manager.check_bbox_violations(bbox)
    print(f"Violations detected: {len(violations) > 0}")

def main():
    print("Member 2 - Tracking & Zone Management Demo")
    print("=========================================")
    
    demo_tracking()
    demo_zone_manager()
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()