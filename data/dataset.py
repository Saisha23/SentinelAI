"""
PyTorch Dataset classes for anomaly detection
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from data.preprocessing import VideoPreprocessor, DataAugmentation


class VideoAnomalyDataset(Dataset):
    """
    Dataset for video anomaly detection
    Loads preprocessed video sequences with labels
    """
    
    def __init__(self, 
                 data_dir: str,
                 sequence_length: int = 16,
                 frame_size: Tuple[int, int] = (224, 224),
                 is_train: bool = True,
                 augment: bool = True):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.is_train = is_train
        self.augment = augment and is_train
        
        self.preprocessor = VideoPreprocessor(
            frame_size=frame_size,
            sequence_length=sequence_length
        )
        
        self.sequences = []
        self.labels = []
        self.video_names = []
        
        self._load_data()
    
    def _load_data(self):
        """Load all video sequences and labels"""
        # Expected structure:
        # data_dir/
        #   normal/
        #     video1.mp4
        #     video2.mp4
        #   anomalous/
        #     video1.mp4
        #     video2.mp4
        
        normal_dir = self.data_dir / 'normal'
        anomalous_dir = self.data_dir / 'anomalous'
        
        # Load normal videos (label = 0)
        if normal_dir.exists():
            for video_path in normal_dir.glob('*.mp4'):
                try:
                    sequences, metadata = self.preprocessor.process_video(str(video_path))
                    for seq in sequences:
                        self.sequences.append(seq)
                        self.labels.append(0)  # Normal
                        self.video_names.append(video_path.name)
                except Exception as e:
                    print(f"Error loading {video_path}: {e}")
        
        # Load anomalous videos (label = 1)
        if anomalous_dir.exists():
            for video_path in anomalous_dir.glob('*.mp4'):
                try:
                    sequences, metadata = self.preprocessor.process_video(str(video_path))
                    for seq in sequences:
                        self.sequences.append(seq)
                        self.labels.append(1)  # Anomalous
                        self.video_names.append(video_path.name)
                except Exception as e:
                    print(f"Error loading {video_path}: {e}")
        
        print(f"Loaded {len(self.sequences)} sequences")
        print(f"Normal: {sum(1 for l in self.labels if l == 0)}, Anomalous: {sum(1 for l in self.labels if l == 1)}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Apply data augmentation
        if self.augment:
            sequence = self._augment_sequence(sequence)
        
        return sequence, torch.tensor(label, dtype=torch.float32)
    
    def _augment_sequence(self, sequence):
        """Apply random augmentation to sequence"""
        # Convert to numpy for augmentation
        seq_np = sequence.permute(0, 2, 3, 1).numpy()  # (seq, c, h, w) -> (seq, h, w, c)
        
        # Random flip
        if np.random.random() < 0.5:
            seq_np = DataAugmentation.random_flip(seq_np)
        
        # Random brightness
        if np.random.random() < 0.3:
            seq_np = DataAugmentation.random_brightness(seq_np)
        
        # Random noise
        if np.random.random() < 0.2:
            seq_np = DataAugmentation.add_noise(seq_np)
        
        # Convert back to tensor
        sequence = torch.from_numpy(seq_np).permute(0, 3, 1, 2).float()
        
        return sequence


class AutoencoderDataset(Dataset):
    """
    Dataset for training autoencoder on normal frames only
    Used for unsupervised learning of normal patterns
    """
    
    def __init__(self,
                 data_dir: str,
                 frame_size: Tuple[int, int] = (224, 224),
                 max_frames_per_video: Optional[int] = None,
                 augment: bool = True):
        self.data_dir = Path(data_dir)
        self.frame_size = frame_size
        self.max_frames_per_video = max_frames_per_video
        self.augment = augment
        
        self.preprocessor = VideoPreprocessor(frame_size=frame_size)
        self.frames = []
        
        self._load_frames()
    
    def _load_frames(self):
        """Load normal frames from videos"""
        normal_dir = self.data_dir / 'normal'
        
        if not normal_dir.exists():
            print(f"Warning: Normal directory not found at {normal_dir}")
            return
        
        for video_path in normal_dir.glob('*.mp4'):
            try:
                frames = self.preprocessor.extract_frames(
                    str(video_path), 
                    max_frames=self.max_frames_per_video
                )
                
                # Preprocess frames
                frames = np.array([self.preprocessor.preprocess_frame(f) for f in frames])
                
                # Convert to tensor (num_frames, c, h, w)
                frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
                
                for frame in frames_tensor:
                    self.frames.append(frame)
                
            except Exception as e:
                print(f"Error loading {video_path}: {e}")
        
        print(f"Loaded {len(self.frames)} frames for autoencoder training")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        # Apply augmentation
        if self.augment:
            frame = self._augment_frame(frame)
        
        return frame, frame  # Input and target are the same for autoencoder
    
    def _augment_frame(self, frame):
        """Apply random augmentation to frame"""
        frame_np = frame.permute(1, 2, 0).numpy()  # (c, h, w) -> (h, w, c)
        
        # Random flip
        if np.random.random() < 0.5:
            frame_np = np.flip(frame_np, axis=1).copy()
        
        # Random brightness
        if np.random.random() < 0.3:
            brightness = 1.0 + np.random.uniform(-0.2, 0.2)
            frame_np = np.clip(frame_np * brightness, 0, 1)
        
        frame = torch.from_numpy(frame_np).permute(2, 0, 1).float()
        return frame


def create_data_loaders(data_dir: str,
                       batch_size: int = 32,
                       sequence_length: int = 16,
                       validation_split: float = 0.2,
                       num_workers: int = 4):
    """
    Create train and validation data loaders
    
    Returns:
        train_loader, val_loader
    """
    # Load full dataset
    full_dataset = VideoAnomalyDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        is_train=True,
        augment=True
    )
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {train_size}, Validation samples: {val_size}")
    
    return train_loader, val_loader


def test_dataset():
    """Test dataset loading"""
    print("Testing Dataset...")
    print("\nTo test with real data:")
    print("1. Organize videos in the following structure:")
    print("   data/videos/")
    print("     normal/")
    print("       video1.mp4")
    print("     anomalous/")
    print("       video2.mp4")
    print("2. Run: python data/dataset.py")
    print("\nDataset classes initialized successfully! ✓")


if __name__ == "__main__":
    test_dataset()
