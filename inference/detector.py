"""
Real-time anomaly detector for inference
Integrates with backend (Member 4) for live anomaly detection
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import os
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CombinedAnomalyDetector
from config import Config
from data.preprocessing import VideoPreprocessor


class AnomalyDetector:
    """
    Real-time anomaly detector for video streams
    Processes frames and detects anomalies
    """
    
    def __init__(self,
                 autoencoder_weights: Optional[str] = None,
                 cnn_lstm_weights: Optional[str] = None,
                 combined_weights: Optional[str] = None,
                 device: str = 'cuda',
                 threshold: float = 0.7):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.sequence_length = Config.SEQUENCE_LENGTH
        
        # Initialize preprocessor
        self.preprocessor = VideoPreprocessor(
            frame_size=(Config.FRAME_WIDTH, Config.FRAME_HEIGHT),
            sequence_length=self.sequence_length,
            normalize=True
        )
        
        # Frame buffer for sequence creation
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Initialize model
        self.model = CombinedAnomalyDetector(
            autoencoder_latent_dim=Config.AUTOENCODER_LATENT_DIM,
            cnn_lstm_feature_dim=512,
            cnn_lstm_hidden_dim=Config.CNN_LSTM_HIDDEN_DIM,
            cnn_lstm_num_layers=Config.CNN_LSTM_NUM_LAYERS,
            num_classes=len(Config.ANOMALY_TYPES),
            ae_weight=Config.ANOMALY_SCORE_WEIGHT_AE,
            lstm_weight=Config.ANOMALY_SCORE_WEIGHT_LSTM
        )
        
        # Load weights
        if combined_weights:
            self._load_combined_weights(combined_weights)
        elif autoencoder_weights and cnn_lstm_weights:
            self.model.load_pretrained_models(autoencoder_weights, cnn_lstm_weights)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Anomaly Detector initialized on {self.device}")
    
    def _load_combined_weights(self, weights_path: str):
        """Load combined model weights"""
        checkpoint = torch.load(weights_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Loaded weights from {weights_path}")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a single frame"""
        # Resize
        frame = cv2.resize(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
        
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def add_frame(self, frame: np.ndarray):
        """Add a frame to the buffer"""
        preprocessed = self.preprocess_frame(frame)
        self.frame_buffer.append(preprocessed)
    
    def detect(self, frames: Optional[np.ndarray] = None) -> Dict:
        """
        Detect anomalies in frames
        
        Args:
            frames: numpy array of frames (seq_len, h, w, c) or None to use buffer
        
        Returns:
            Dictionary containing detection results
        """
        if frames is not None:
            # Process provided frames
            if len(frames) < self.sequence_length:
                raise ValueError(f"Need at least {self.sequence_length} frames")
            
            # Take last sequence_length frames
            frames = frames[-self.sequence_length:]
            sequence = np.array([self.preprocess_frame(f) for f in frames])
        else:
            # Use buffer
            if len(self.frame_buffer) < self.sequence_length:
                return {
                    'ready': False,
                    'message': f'Buffering frames ({len(self.frame_buffer)}/{self.sequence_length})'
                }
            
            sequence = np.array(list(self.frame_buffer))
        
        # Convert to tensor: (1, seq_len, c, h, w)
        sequence_tensor = torch.from_numpy(sequence).unsqueeze(0)
        sequence_tensor = sequence_tensor.permute(0, 1, 4, 2, 3).float()
        sequence_tensor = sequence_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(sequence_tensor)
        
        # Extract results
        anomaly_score = output['anomaly_score'].item()
        ae_score = output['ae_score'].item()
        lstm_score = output['lstm_score'].item()
        predicted_class_idx = torch.argmax(output['class_logits'], dim=1).item()
        predicted_class = Config.ANOMALY_TYPES[predicted_class_idx]
        
        is_anomalous = anomaly_score > self.threshold
        
        return {
            'ready': True,
            'is_anomalous': is_anomalous,
            'anomaly_score': float(anomaly_score),
            'ae_score': float(ae_score),
            'lstm_score': float(lstm_score),
            'predicted_class': predicted_class,
            'predicted_class_idx': int(predicted_class_idx),
            'confidence': float(anomaly_score),
            'threshold': self.threshold,
            'timestamp': None  # Can be set by caller
        }
    
    def detect_from_video(self, video_path: str) -> List[Dict]:
        """
        Process entire video and detect anomalies
        
        Returns:
            List of detection results for each sequence
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        results = []
        frame_count = 0
        
        print(f"Processing video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add frame to buffer
            self.add_frame(frame)
            
            # Detect if buffer is full
            if len(self.frame_buffer) == self.sequence_length:
                result = self.detect()
                result['frame_number'] = frame_count
                results.append(result)
            
            frame_count += 1
        
        cap.release()
        
        print(f"Processed {frame_count} frames, generated {len(results)} detections")
        
        return results
    
    def detect_from_stream(self, stream_url: str, callback=None):
        """
        Real-time detection from video stream (for integration with Member 4)
        
        Args:
            stream_url: URL or device ID for video stream
            callback: Function to call with detection results
        """
        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open stream: {stream_url}")
        
        print(f"Started streaming from: {stream_url}")
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add frame
                self.add_frame(frame)
                
                # Detect
                result = self.detect()
                
                if result['ready']:
                    result['frame_number'] = frame_count
                    
                    # Call callback if provided
                    if callback:
                        callback(result, frame)
                    
                    # Print anomaly alerts
                    if result['is_anomalous']:
                        print(f"\n⚠️  ANOMALY DETECTED at frame {frame_count}")
                        print(f"Score: {result['anomaly_score']:.3f}")
                        print(f"Type: {result['predicted_class']}")
                
                frame_count += 1
                
                # Break on 'q' key (if display window exists)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nStopping stream...")
        finally:
            cap.release()
    
    def reset_buffer(self):
        """Clear the frame buffer"""
        self.frame_buffer.clear()


def demo_detector():
    """Demo the anomaly detector"""
    print("Anomaly Detector Demo")
    print("=" * 50)
    
    # Initialize detector
    detector = AnomalyDetector(
        combined_weights=None,  # Will use untrained model for demo
        threshold=Config.ANOMALY_THRESHOLD
    )
    
    print("\nDetector ready!")
    print(f"Sequence length: {detector.sequence_length}")
    print(f"Threshold: {detector.threshold}")
    print(f"Device: {detector.device}")
    
    print("\nTo use the detector:")
    print("1. Train models using training scripts")
    print("2. Load weights: detector = AnomalyDetector(combined_weights='path/to/weights.pth')")
    print("3. Process video: results = detector.detect_from_video('video.mp4')")
    print("4. Or stream: detector.detect_from_stream('rtsp://stream_url', callback=my_callback)")
    

if __name__ == "__main__":
    demo_detector()
