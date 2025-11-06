"""
Configuration file for Anomaly Detection Module
Member 3 - CU Hackathon
"""

import os

class Config:
    """Configuration parameters for anomaly detection"""
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'videos')
    WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    
    # Video Processing
    FRAME_WIDTH = 224
    FRAME_HEIGHT = 224
    FRAME_CHANNELS = 3
    SEQUENCE_LENGTH = 16  # Number of frames in a sequence
    FPS = 30  # Frames per second
    
    # Autoencoder Configuration
    AUTOENCODER_LATENT_DIM = 128
    AUTOENCODER_LEARNING_RATE = 0.001
    AUTOENCODER_BATCH_SIZE = 32
    AUTOENCODER_EPOCHS = 50
    RECONSTRUCTION_THRESHOLD = 0.05  # Threshold for anomaly detection
    
    # CNN-LSTM Configuration
    CNN_LSTM_HIDDEN_DIM = 256
    CNN_LSTM_NUM_LAYERS = 2
    CNN_LSTM_DROPOUT = 0.3
    CNN_LSTM_LEARNING_RATE = 0.0001
    CNN_LSTM_BATCH_SIZE = 16
    CNN_LSTM_EPOCHS = 30
    
    # Combined Model
    ANOMALY_SCORE_WEIGHT_AE = 0.4  # Weight for autoencoder score
    ANOMALY_SCORE_WEIGHT_LSTM = 0.6  # Weight for CNN-LSTM score
    ANOMALY_THRESHOLD = 0.7  # Final threshold for flagging anomaly
    
    # Training
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10
    LEARNING_RATE_DECAY = 0.1
    LEARNING_RATE_DECAY_STEP = 20
    
    # Device
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    NUM_WORKERS = 4
    
    # Integration with Backend (Member 4)
    BACKEND_API_URL = 'http://localhost:8000/api/anomaly'
    WEBSOCKET_URL = 'ws://localhost:8000/ws/anomaly'
    
    # Anomaly Categories
    ANOMALY_TYPES = [
        'normal',
        'suspicious_loitering',
        'aggressive_behavior',
        'crowd_panic',
        'unauthorized_access',
        'weapon_detected',
        'unusual_movement'
    ]
    
    # Logging
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_INTERVAL = 5  # Save checkpoint every N epochs
    
    @staticmethod
    def create_directories():
        """Create necessary directories if they don't exist"""
        os.makedirs(Config.WEIGHTS_DIR, exist_ok=True)
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        os.makedirs(Config.DATA_DIR, exist_ok=True)

# Create directories on import
Config.create_directories()
