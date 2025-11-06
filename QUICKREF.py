"""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║              MEMBER 3 - QUICK REFERENCE CARD                     ║
║                   Anomaly Detection System                       ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
📦 INSTALLATION
═══════════════════════════════════════════════════════════════════

pip install -r requirements.txt
python setup.py


═══════════════════════════════════════════════════════════════════
🎓 TRAINING
═══════════════════════════════════════════════════════════════════

Train Autoencoder:
  python training/train_autoencoder.py --data_dir data/videos

Train CNN-LSTM:
  python training/train_cnn_lstm.py --data_dir data/videos


═══════════════════════════════════════════════════════════════════
🚀 INFERENCE
═══════════════════════════════════════════════════════════════════

Basic Usage:
─────────────
from inference.detector import AnomalyDetector

detector = AnomalyDetector(
    autoencoder_weights='weights/autoencoder_best.pth',
    cnn_lstm_weights='weights/cnn_lstm_best.pth',
    threshold=0.7
)

# Process video
results = detector.detect_from_video('video.mp4')

# Or stream
detector.detect_from_stream('rtsp://camera_url', callback=my_callback)


With Backend Integration:
──────────────────────────
from inference.detector import AnomalyDetector
from inference.integration import BackendIntegration, StreamCallback

detector = AnomalyDetector(threshold=0.7)
backend = BackendIntegration(
    api_url='http://localhost:8000/api/anomaly'
)
callback = StreamCallback(backend)

detector.detect_from_stream('rtsp://camera', callback=callback)


═══════════════════════════════════════════════════════════════════
⚙️ CONFIGURATION (config.py)
═══════════════════════════════════════════════════════════════════

Video:
  FRAME_WIDTH = 224
  FRAME_HEIGHT = 224
  SEQUENCE_LENGTH = 16

Autoencoder:
  AUTOENCODER_LATENT_DIM = 128
  AUTOENCODER_LEARNING_RATE = 0.001
  AUTOENCODER_BATCH_SIZE = 32
  AUTOENCODER_EPOCHS = 50

CNN-LSTM:
  CNN_LSTM_HIDDEN_DIM = 256
  CNN_LSTM_NUM_LAYERS = 2
  CNN_LSTM_LEARNING_RATE = 0.0001
  CNN_LSTM_BATCH_SIZE = 16
  CNN_LSTM_EPOCHS = 30

Detection:
  ANOMALY_THRESHOLD = 0.7          # Adjust sensitivity


═══════════════════════════════════════════════════════════════════
🎯 ANOMALY TYPES (7 classes)
═══════════════════════════════════════════════════════════════════

0: normal
1: suspicious_loitering
2: aggressive_behavior
3: crowd_panic
4: unauthorized_access
5: weapon_detected
6: unusual_movement


═══════════════════════════════════════════════════════════════════
📊 OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════

{
    'ready': True,
    'is_anomalous': True/False,
    'anomaly_score': 0.0-1.0,
    'ae_score': 0.0-1.0,              # Autoencoder score
    'lstm_score': 0.0-1.0,            # LSTM score
    'predicted_class': 'string',
    'predicted_class_idx': 0-6,
    'confidence': 0.0-1.0,
    'threshold': 0.7,
    'frame_number': int
}


═══════════════════════════════════════════════════════════════════
🔗 INTEGRATION WITH TEAM
═══════════════════════════════════════════════════════════════════

Member 1 (YOLO):
  Input  → Bounding boxes of detected objects
  Output → Focused anomaly detection

Member 2 (Tracking):
  Input  → Tracked IDs and trajectories
  Output → Movement pattern analysis

Member 4 (Backend):
  API    → POST /api/anomaly
  WS     → ws://localhost:8000/ws/anomaly
  Data   → JSON detection results

Member 5 (Dashboard):
  Via Member 4 → Visualization and alerts


═══════════════════════════════════════════════════════════════════
📁 PROJECT STRUCTURE
═══════════════════════════════════════════════════════════════════

anomaly_detection/
├── models/              Neural networks
├── data/                Data processing
├── training/            Training scripts
├── inference/           Real-time detection
├── utils/               Metrics & viz
├── weights/             Model checkpoints
├── logs/                Training logs
├── data/videos/         Input videos
│   ├── normal/         Normal footage
│   └── anomalous/      Anomaly footage
├── config.py           Settings
├── demo.py             Demo script
└── requirements.txt    Dependencies


═══════════════════════════════════════════════════════════════════
🎯 TARGET PERFORMANCE
═══════════════════════════════════════════════════════════════════

✓ Accuracy: >85%
✓ False Positive Rate: <10%
✓ Processing Speed: <200ms/frame
✓ ROC-AUC: >0.90


═══════════════════════════════════════════════════════════════════
🛠️ COMMON COMMANDS
═══════════════════════════════════════════════════════════════════

Setup:
  python setup.py

Demo:
  python demo.py

Train:
  python training/train_autoencoder.py
  python training/train_cnn_lstm.py

Test Model:
  python models/autoencoder.py
  python models/cnn_lstm.py

Test Detector:
  python inference/detector.py


═══════════════════════════════════════════════════════════════════
🐛 TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════

Import errors:
  pip install -r requirements.txt

CUDA out of memory:
  Reduce batch sizes in config.py

Low accuracy:
  • More training data
  • More epochs
  • Adjust threshold

Too many false positives:
  Increase ANOMALY_THRESHOLD (e.g., 0.8)

Missing anomalies:
  Decrease ANOMALY_THRESHOLD (e.g., 0.6)


═══════════════════════════════════════════════════════════════════
📚 DOCUMENTATION
═══════════════════════════════════════════════════════════════════

README.md    - Project overview
GUIDE.md     - Comprehensive usage guide
SUMMARY.md   - Implementation summary
demo.py      - Interactive demo
This file   - Quick reference


═══════════════════════════════════════════════════════════════════
🎓 KEY CONCEPTS
═══════════════════════════════════════════════════════════════════

Autoencoder:
  • Learns "normal" patterns
  • High reconstruction error = anomaly
  • Frame-level detection

CNN-LSTM:
  • CNN extracts spatial features
  • LSTM analyzes temporal patterns
  • Sequence-level detection

Combined Model:
  • Fusion of both approaches
  • Weighted scoring
  • Robust detection


═══════════════════════════════════════════════════════════════════
⏱️ 10-DAY ROADMAP
═══════════════════════════════════════════════════════════════════

Day 3-4: Autoencoder ✓ DONE
Day 5-6: CNN-LSTM ✓ DONE
Day 7:   Integration ✓ DONE
Day 8:   Delivery ← YOU ARE HERE


═══════════════════════════════════════════════════════════════════
✅ DELIVERABLES
═══════════════════════════════════════════════════════════════════

[x] Model architectures
[x] Training scripts
[x] Inference pipeline
[x] Backend integration
[x] Evaluation metrics
[x] Visualization
[x] Documentation
[ ] Trained weights (need data)
[ ] Demo video


═══════════════════════════════════════════════════════════════════
🎉 YOU'RE READY!
═══════════════════════════════════════════════════════════════════

All code is complete and documented.
Just add data, train, integrate, and demo!

Good luck with the hackathon! 🚀

═══════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(__doc__)
