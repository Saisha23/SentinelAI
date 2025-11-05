# Member 3 - Anomaly Detection Module
## Autoencoder + CNN-LSTM for Suspicious Activity Detection

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

### Installation
```bash
pip install -r requirements.txt
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
