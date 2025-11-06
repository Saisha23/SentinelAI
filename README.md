# Sentinel AI Backend Service

This is the backend service for the Sentinel AI surveillance system. It handles real-time video processing, object detection, tracking, and anomaly detection using YOLOv8, DeepSORT, and custom anomaly detection models.

## Features

- Real-time video stream processing
- Object detection using YOLOv8
- Multi-object tracking using DeepSORT
- Anomaly detection using custom models
- WebSocket-based real-time communication
- Alert system for security events

## Setup

1. Create a conda environment:
```bash
conda create -p .conda python=3.11 -y
```

2. Activate the environment:
```bash
conda activate ./.conda
```

3. Install dependencies:
```bash
pip install fastapi uvicorn websockets opencv-python numpy python-multipart python-jose[cryptography] passlib[bcrypt]
```

## Running the Service

To run the service:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

- `GET /` - Health check endpoint
- `WebSocket /ws/stream` - Real-time video stream processing
- `WebSocket /ws/alerts` - Real-time alert notifications

## Integration Points

### 1. YOLOv8 Integration (Member 1)
- Detection service in `app/services/detection_service.py`
- Integration point in `_process_detections` method

### 2. DeepSORT Integration (Member 2)
- Tracking service in `app/services/detection_service.py`
- Integration point in `_process_tracking` method

### 3. Anomaly Detection Integration (Member 3)
- Anomaly detection in `app/services/detection_service.py`
- Integration point in `_check_anomalies` method

## Security Considerations

- CORS is currently set to allow all origins (`*`). In production, specify allowed origins.
- Implement authentication for WebSocket connections
- Use SSL/TLS in production
- Implement rate limiting

## Performance Optimization

- Frame processing is done asynchronously
- Implement frame dropping if processing falls behind
- Use GPU acceleration where available
- Optimize model inference times