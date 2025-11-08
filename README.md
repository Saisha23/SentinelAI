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

🧠 Tracking & Zone Management (Member 2)

This backend includes a lightweight IoU-based tracker and a polygonal zone manager as starter implementations.
Member 2 is responsible for reviewing or replacing these with DeepSort or another robust tracker if needed.

📁 Files Overview
File	Description
app/services/tracker.py	Implements a simple IoU tracker that assigns persistent IDs and stores centroid history.
app/services/zone_manager.py	Provides polygon-based zone management. Use add_zone(id, name, polygon) to register restricted areas.
⚙️ Example Usage
from app.services.detection_service import DetectionService

ds = DetectionService()

# Add or modify zones at runtime
ds.zone_manager.add_zone(
    'zone_entrance',
    'Main Entrance',
    [(100, 100), (400, 100), (400, 400), (100, 400)]
)

# Pass YOLO detections directly to process_frame
sample_detections = [
    {"bbox": (120, 130, 50, 100), "class": "person", "confidence": 0.92},
]

result = await ds.process_frame("", detections=sample_detections)
print(result)  # returns a dict with 'tracks' and 'anomalies'

🧩 Notes

Zone violations are detected by checking if a bounding box centroid lies inside any registered polygon.

Detected violations enqueue a zone_violation alert into the backend alert queue.

To replace the default IoU tracker with DeepSort, implement a wrapper exposing the same interface:

update(detections) -> list of tracks


Then integrate it into DetectionService.tracker.

🔒 Security Considerations

CORS currently allows all origins (*); restrict it in production.

Implement authentication for WebSocket connections.

Use SSL/TLS for all production deployments.

Add rate limiting to protect against abuse.

⚡ Performance Optimization

Frame processing is asynchronous.

Implement frame dropping when processing falls behind.

Utilize GPU acceleration where available.

Optimize model inference for faster performance.

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

## Integrated Violence Detection (Third-Party)

We have integrated the reference implementation of the paper "Efficient Two-Stream Network for Violence Detection Using Separable Convolutional LSTM" (IJCNN 2021) as a third-party module for research and benchmarking.

- Source: https://github.com/zahid58/TwoStreamSepConvLSTM_ViolenceDetection
- Path in this repo: `third_party/TwoStreamSepConvLSTM_ViolenceDetection`
- License: MIT (original copyright © 2021 Zahidul Islam). The original license file is included in `third_party/TwoStreamSepConvLSTM_ViolenceDetection/license`.

### Quick start (research use)
- Install its dependencies:
    - From the project root: `pip install -r third_party/TwoStreamSepConvLSTM_ViolenceDetection/requirements.txt`
- Run training/evaluation from inside the third-party folder or adapt paths as needed (see its `README.md`).

### Notes
- This code is included for evaluation and potential integration with our anomaly pipeline. It is not actively maintained by this project. Refer to the upstream repository for original documentation and updates.

