"""
Video preprocessing utilities
Extract and prepare frames for model training
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class VideoPreprocessor:
    """Preprocess video files for anomaly detection"""
    
    def __init__(self, 
                 frame_size=(224, 224),
                 sequence_length=16,
                 fps=None,
                 normalize=True):
        self.frame_size = frame_size
        self.sequence_length = sequence_length
        self.fps = fps
        self.normalize = normalize
        
    def extract_frames(self, video_path: str, max_frames: Optional[int] = None) -> np.ndarray:
        """
        Extract frames from video file
        Returns: numpy array of shape (num_frames, height, width, channels)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling rate
        if self.fps and self.fps < video_fps:
            frame_interval = int(video_fps / self.fps)
        else:
            frame_interval = 1
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames based on interval
            if frame_count % frame_interval == 0:
                # Resize frame
                frame = cv2.resize(frame, self.frame_size)
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frames.append(frame)
                
                if max_frames and len(frames) >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        
        frames = np.array(frames)
        return frames
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame
        - Normalize to [0, 1]
        - Optional: Apply augmentations
        """
        frame = frame.astype(np.float32)
        
        if self.normalize:
            frame = frame / 255.0
        
        return frame
    
    def create_sequences(self, frames: np.ndarray, overlap: float = 0.5) -> List[np.ndarray]:
        """
        Create overlapping sequences from frames
        
        Args:
            frames: (num_frames, height, width, channels)
            overlap: overlap ratio between sequences (0-1)
        
        Returns:
            List of sequences, each of shape (sequence_length, height, width, channels)
        """
        num_frames = len(frames)
        step = max(1, int(self.sequence_length * (1 - overlap)))
        
        sequences = []
        for i in range(0, num_frames - self.sequence_length + 1, step):
            sequence = frames[i:i + self.sequence_length]
            sequences.append(sequence)
        
        return sequences
    
    def process_video(self, video_path: str, overlap: float = 0.5) -> Tuple[torch.Tensor, dict]:
        """
        Complete preprocessing pipeline for a video
        
        Returns:
            sequences: torch.Tensor of shape (num_sequences, seq_len, channels, height, width)
            metadata: dict with video information
        """
        # Extract frames
        frames = self.extract_frames(video_path)
        
        # Preprocess frames
        frames = np.array([self.preprocess_frame(f) for f in frames])
        
        # Create sequences
        sequences = self.create_sequences(frames, overlap=overlap)
        
        if len(sequences) == 0:
            raise ValueError(f"No sequences created from video: {video_path}")
        
        # Convert to torch tensor and rearrange dimensions
        # From (num_seq, seq_len, h, w, c) to (num_seq, seq_len, c, h, w)
        sequences = np.array(sequences)
        sequences = torch.from_numpy(sequences).permute(0, 1, 4, 2, 3).float()
        
        metadata = {
            'video_path': video_path,
            'num_frames': len(frames),
            'num_sequences': len(sequences),
            'sequence_length': self.sequence_length,
            'frame_size': self.frame_size
        }
        
        return sequences, metadata
    
    def process_video_directory(self, video_dir: str, pattern: str = "*.mp4") -> List[Tuple[torch.Tensor, dict]]:
        """
        Process all videos in a directory
        
        Returns:
            List of (sequences, metadata) tuples
        """
        video_dir = Path(video_dir)
        video_files = list(video_dir.glob(pattern))
        
        if not video_files:
            print(f"Warning: No videos found in {video_dir} with pattern {pattern}")
            return []
        
        print(f"Found {len(video_files)} videos in {video_dir}")
        
        results = []
        for video_path in video_files:
            try:
                sequences, metadata = self.process_video(str(video_path))
                results.append((sequences, metadata))
                print(f"Processed: {video_path.name} -> {len(sequences)} sequences")
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
        
        return results


class DataAugmentation:
    """Data augmentation techniques for video frames"""
    
    @staticmethod
    def random_flip(frames: np.ndarray, p: float = 0.5) -> np.ndarray:
        """Randomly flip frames horizontally"""
        if np.random.random() < p:
            return np.flip(frames, axis=2).copy()
        return frames
    
    @staticmethod
    def random_brightness(frames: np.ndarray, factor: float = 0.2) -> np.ndarray:
        """Randomly adjust brightness"""
        brightness = 1.0 + np.random.uniform(-factor, factor)
        frames = frames * brightness
        return np.clip(frames, 0, 1)
    
    @staticmethod
    def random_rotation(frames: np.ndarray, max_angle: float = 10) -> np.ndarray:
        """Randomly rotate frames"""
        angle = np.random.uniform(-max_angle, max_angle)
        h, w = frames.shape[1:3]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated = []
        for frame in frames:
            rotated_frame = cv2.warpAffine(frame, M, (w, h))
            rotated.append(rotated_frame)
        
        return np.array(rotated)
    
    @staticmethod
    def add_noise(frames: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_level, frames.shape)
        noisy_frames = frames + noise
        return np.clip(noisy_frames, 0, 1)


def test_preprocessing():
    """Test video preprocessing"""
    print("Testing Video Preprocessor...")
    
    preprocessor = VideoPreprocessor(
        frame_size=(224, 224),
        sequence_length=16,
        fps=10
    )
    
    # Create a dummy video for testing
    print("\nTo test with real video:")
    print("1. Place video files in the data/videos directory")
    print("2. Run: python data/preprocessing.py")
    print("\nPreprocessor initialized successfully! ✓")
    print(f"Frame size: {preprocessor.frame_size}")
    print(f"Sequence length: {preprocessor.sequence_length}")
    

if __name__ == "__main__":
    test_preprocessing()
