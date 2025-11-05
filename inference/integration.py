"""
Integration module for backend communication (Member 4)
Sends anomaly detection results to FastAPI backend via WebSocket
"""

import json
import asyncio
import websockets
import requests
from typing import Dict, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class BackendIntegration:
    """
    Integration with FastAPI backend (Member 4)
    Sends detection results via REST API and WebSocket
    """
    
    def __init__(self, 
                 api_url: str = None,
                 websocket_url: str = None):
        self.api_url = api_url or Config.BACKEND_API_URL
        self.websocket_url = websocket_url or Config.WEBSOCKET_URL
        self.websocket = None
    
    def send_detection_rest(self, detection_result: Dict) -> bool:
        """
        Send detection result via REST API
        
        Args:
            detection_result: Dictionary containing detection data
        
        Returns:
            True if successful, False otherwise
        """
        try:
            payload = {
                'timestamp': detection_result.get('timestamp'),
                'is_anomalous': detection_result['is_anomalous'],
                'anomaly_score': detection_result['anomaly_score'],
                'ae_score': detection_result['ae_score'],
                'lstm_score': detection_result['lstm_score'],
                'predicted_class': detection_result['predicted_class'],
                'confidence': detection_result['confidence'],
                'frame_number': detection_result.get('frame_number'),
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                return True
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"Error sending to REST API: {e}")
            return False
    
    async def connect_websocket(self):
        """Establish WebSocket connection"""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            print(f"WebSocket connected to {self.websocket_url}")
            return True
        except Exception as e:
            print(f"WebSocket connection error: {e}")
            return False
    
    async def send_detection_websocket(self, detection_result: Dict):
        """
        Send detection result via WebSocket for real-time updates
        
        Args:
            detection_result: Dictionary containing detection data
        """
        if not self.websocket:
            await self.connect_websocket()
        
        try:
            message = json.dumps({
                'type': 'anomaly_detection',
                'data': {
                    'timestamp': detection_result.get('timestamp'),
                    'is_anomalous': detection_result['is_anomalous'],
                    'anomaly_score': detection_result['anomaly_score'],
                    'ae_score': detection_result['ae_score'],
                    'lstm_score': detection_result['lstm_score'],
                    'predicted_class': detection_result['predicted_class'],
                    'confidence': detection_result['confidence'],
                    'frame_number': detection_result.get('frame_number'),
                }
            })
            
            await self.websocket.send(message)
            
        except Exception as e:
            print(f"WebSocket send error: {e}")
            # Try to reconnect
            self.websocket = None
    
    async def close_websocket(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            print("WebSocket connection closed")
    
    def send_alert(self, detection_result: Dict) -> bool:
        """
        Send high-priority alert for severe anomalies
        Uses both REST and WebSocket for redundancy
        """
        if not detection_result['is_anomalous']:
            return False
        
        # Send via REST API
        rest_success = self.send_detection_rest(detection_result)
        
        # Send via WebSocket (async)
        try:
            asyncio.create_task(self.send_detection_websocket(detection_result))
        except Exception as e:
            print(f"Error creating WebSocket task: {e}")
        
        return rest_success


class StreamCallback:
    """
    Callback handler for real-time detection stream
    Integrates detector with backend
    """
    
    def __init__(self, backend: BackendIntegration):
        self.backend = backend
        self.detection_count = 0
        self.anomaly_count = 0
    
    def __call__(self, detection_result: Dict, frame=None):
        """
        Handle detection result
        
        Args:
            detection_result: Detection output from AnomalyDetector
            frame: Video frame (optional)
        """
        self.detection_count += 1
        
        # Log detection
        if detection_result['is_anomalous']:
            self.anomaly_count += 1
            print(f"\n🚨 ANOMALY ALERT #{self.anomaly_count}")
            print(f"Frame: {detection_result.get('frame_number', 'N/A')}")
            print(f"Score: {detection_result['anomaly_score']:.3f}")
            print(f"Type: {detection_result['predicted_class']}")
            print(f"Confidence: {detection_result['confidence']:.3f}")
            
            # Send to backend
            self.backend.send_alert(detection_result)
        
        # Periodic updates (every 30 frames)
        if self.detection_count % 30 == 0:
            print(f"\nStatus: {self.detection_count} detections, {self.anomaly_count} anomalies")
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        return {
            'total_detections': self.detection_count,
            'total_anomalies': self.anomaly_count,
            'anomaly_rate': self.anomaly_count / self.detection_count if self.detection_count > 0 else 0
        }


# Example API endpoint structure for Member 4 (FastAPI)
EXAMPLE_FASTAPI_CODE = """
# Member 4: FastAPI Backend Example
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

class AnomalyDetection(BaseModel):
    timestamp: str
    is_anomalous: bool
    anomaly_score: float
    ae_score: float
    lstm_score: float
    predicted_class: str
    confidence: float
    frame_number: int

@app.post("/api/anomaly")
async def receive_anomaly(detection: AnomalyDetection):
    # Process anomaly detection
    print(f"Received anomaly: {detection.predicted_class} - Score: {detection.anomaly_score}")
    
    # Store in database, trigger alerts, etc.
    if detection.is_anomalous:
        # Trigger alert system
        pass
    
    return {"status": "success", "received_at": datetime.now()}

@app.websocket("/ws/anomaly")
async def websocket_anomaly(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            print(f"WebSocket received: {data}")
            
            # Broadcast to dashboard (Member 5)
            # await broadcast_to_dashboard(data)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
"""


def demo_integration():
    """Demo backend integration"""
    print("Backend Integration Demo")
    print("=" * 50)
    
    # Initialize integration
    integration = BackendIntegration()
    
    print(f"\nAPI URL: {integration.api_url}")
    print(f"WebSocket URL: {integration.websocket_url}")
    
    # Example detection result
    example_result = {
        'timestamp': '2024-01-15T10:30:45',
        'is_anomalous': True,
        'anomaly_score': 0.87,
        'ae_score': 0.82,
        'lstm_score': 0.91,
        'predicted_class': 'suspicious_loitering',
        'confidence': 0.87,
        'frame_number': 1234
    }
    
    print("\nExample detection result:")
    print(json.dumps(example_result, indent=2))
    
    print("\nTo integrate with Member 4's backend:")
    print("1. Member 4 creates FastAPI endpoints (see example below)")
    print("2. Use BackendIntegration to send detection results")
    print("3. Member 5 receives updates via WebSocket for dashboard")
    
    print("\n" + "="*50)
    print("Example FastAPI Backend Code for Member 4:")
    print("="*50)
    print(EXAMPLE_FASTAPI_CODE)


if __name__ == "__main__":
    demo_integration()
