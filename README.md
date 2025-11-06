# SentinelAI - Surveillance Anomaly Detection System

## Backend Service (Member 4)

This is the backend service for the Sentinel AI surveillance system. It handles real-time video processing, object detection, tracking, and anomaly detection using YOLOv8, DeepSORT, and custom anomaly detection models.

### Features

- Real-time video stream processing
- Object detection using YOLOv8
- Multi-object tracking using DeepSORT
- Anomaly detection using custom models
- WebSocket-based real-time communication
- Alert system for security events

### Setup

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

### Running the Service

To run the service:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

- `GET /` - Health check endpoint
- `WebSocket /ws/stream` - Real-time video stream processing
- `WebSocket /ws/alerts` - Real-time alert notifications

### Integration Points

#### 1. YOLOv8 Integration (Member 1)
- Detection service in `app/services/detection_service.py`
- Integration point in `_process_detections` method

#### 2. DeepSORT Integration (Member 2)
- Tracking service in `app/services/detection_service.py`
- Integration point in `_process_tracking` method

#### 3. Anomaly Detection Integration (Member 3)
- Anomaly detection in `app/services/detection_service.py`
- Integration point in `_check_anomalies` method

### Security Considerations

- CORS is currently set to allow all origins (`*`). In production, specify allowed origins.
- Implement authentication for WebSocket connections
- Use SSL/TLS in production
- Implement rate limiting

### Performance Optimization

- Frame processing is done asynchronously
- Implement frame dropping if processing falls behind
- Use GPU acceleration where available
- Optimize model inference times

## Anomaly Detection Module (Member 3)

### Overview
This module implements deep learning models to detect abnormal activities and suspicious crowd behavior from video surveillance footage.

### Architecture
1. **Autoencoder**: Learns normal video patterns and detects reconstruction errors for anomalies
2. **CNN-LSTM**: Analyzes temporal patterns across video frames to identify suspicious sequences

### Project Structure
```
anomaly_detection/
├── models/
│   ├── autoencoder.py       # Autoencoder architecture
│   ├── cnn_lstm.py           # CNN-LSTM architecture
│   └── combined_model.py     # Integrated anomaly detection
├── data/
│   ├── preprocessing.py      # Frame extraction and preprocessing
│   └── dataset.py            # PyTorch dataset classes
├── training/
│   ├── train_autoencoder.py  # Training script for autoencoder
│   ├── train_cnn_lstm.py     # Training script for CNN-LSTM
│   └── train_combined.py     # Combined training
├── inference/
│   ├── detector.py           # Real-time anomaly detection
│   └── integration.py        # Backend integration (Member 4)
├── utils/
│   ├── metrics.py            # Evaluation metrics
│   └── visualization.py      # Plotting and visualization
├── config.py                 # Configuration parameters
└── requirements.txt          # Dependencies
```

### Usage

#### Training
```python
# Train autoencoder on normal video data
python training/train_autoencoder.py --data_path ./data/normal_videos

# Train CNN-LSTM for temporal analysis
python training/train_cnn_lstm.py --data_path ./data/labeled_videos

# Train combined model
python training/train_combined.py
```

#### Inference
```python
from inference.detector import AnomalyDetector

detector = AnomalyDetector(
    autoencoder_weights='weights/autoencoder.pth',
    cnn_lstm_weights='weights/cnn_lstm.pth'
)

anomaly_score = detector.detect(video_frames)
```

### Integration with Other Members
- **Member 4 (Backend)**: Provides API endpoint for anomaly scores
- **Member 5 (Dashboard)**: Sends alerts and visualization data

### Performance Targets
- Real-time processing: <200ms per frame
- Anomaly detection accuracy: >85%
- False positive rate: <10%

### Implementation Timeline (10 Days)
- Day 3-4: Build Autoencoder on normal video data
- Day 5-6: Add CNN-LSTM for temporal analysis
- Day 7: Integrate anomaly score output
- Day 8: Deliver final model
