"""
Training script for Autoencoder
Train on normal video frames to learn reconstruction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Autoencoder
from data import AutoencoderDataset
from config import Config
from utils.metrics import AverageMeter
from utils.visualization import plot_training_curves, visualize_reconstructions


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (frames, targets) in enumerate(pbar):
        frames = frames.to(device)
        targets = targets.to(device)
        
        # Forward pass
        reconstructions, _ = model(frames)
        loss = criterion(reconstructions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        losses.update(loss.item(), frames.size(0))
        pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
    
    return losses.avg


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    losses = AverageMeter()
    
    with torch.no_grad():
        for frames, targets in tqdm(dataloader, desc='Validation'):
            frames = frames.to(device)
            targets = targets.to(device)
            
            reconstructions, _ = model(frames)
            loss = criterion(reconstructions, targets)
            
            losses.update(loss.item(), frames.size(0))
    
    return losses.avg


def train_autoencoder(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print("Loading dataset...")
    full_dataset = AutoencoderDataset(
        data_dir=args.data_dir,
        frame_size=(Config.FRAME_WIDTH, Config.FRAME_HEIGHT),
        max_frames_per_video=args.max_frames,
        augment=True
    )
    
    # Split dataset
    val_size = int(len(full_dataset) * Config.VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.AUTOENCODER_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.AUTOENCODER_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Train samples: {train_size}, Validation samples: {val_size}")
    
    # Create model
    model = Autoencoder(latent_dim=Config.AUTOENCODER_LATENT_DIM)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.AUTOENCODER_LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=Config.LEARNING_RATE_DECAY_STEP,
        gamma=Config.LEARNING_RATE_DECAY
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0
    
    for epoch in range(1, Config.AUTOENCODER_EPOCHS + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f"Epoch {epoch}/{Config.AUTOENCODER_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = Path(Config.WEIGHTS_DIR) / 'autoencoder_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs")
            break
        
        # Save checkpoint every N epochs
        if epoch % Config.SAVE_INTERVAL == 0:
            checkpoint_path = Path(Config.WEIGHTS_DIR) / f'autoencoder_epoch_{epoch}.pth'
            torch.save(model.state_dict(), checkpoint_path)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, 
                        save_path=Path(Config.LOGS_DIR) / 'autoencoder_training.png')
    
    # Visualize reconstructions
    model.eval()
    sample_frames, _ = next(iter(val_loader))
    sample_frames = sample_frames[:8].to(device)
    with torch.no_grad():
        reconstructions, _ = model(sample_frames)
    
    visualize_reconstructions(
        sample_frames.cpu(), 
        reconstructions.cpu(),
        save_path=Path(Config.LOGS_DIR) / 'autoencoder_reconstructions.png'
    )
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to {Config.WEIGHTS_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Autoencoder for Anomaly Detection')
    parser.add_argument('--data_dir', type=str, default=Config.DATA_DIR,
                       help='Path to data directory')
    parser.add_argument('--max_frames', type=int, default=1000,
                       help='Maximum frames per video')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_autoencoder(args)
