"""
Training script for CNN-LSTM
Train on labeled video sequences for temporal anomaly detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CNNLSTM
from data import create_data_loaders
from config import Config
from utils.metrics import AverageMeter, evaluate_model
from utils.visualization import plot_training_curves, plot_confusion_matrix


def train_epoch(model, dataloader, criterion_class, criterion_anomaly, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    class_losses = AverageMeter()
    anomaly_losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for sequences, labels in pbar:
        sequences = sequences.to(device)
        labels = labels.to(device).long()
        
        # Forward pass
        class_logits, anomaly_score = model(sequences)
        
        # Compute losses
        loss_class = criterion_class(class_logits, labels)
        loss_anomaly = criterion_anomaly(anomaly_score.squeeze(), labels.float())
        
        # Combined loss
        loss = loss_class + 0.5 * loss_anomaly
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        losses.update(loss.item(), sequences.size(0))
        class_losses.update(loss_class.item(), sequences.size(0))
        anomaly_losses.update(loss_anomaly.item(), sequences.size(0))
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'class': f'{class_losses.avg:.4f}',
            'anomaly': f'{anomaly_losses.avg:.4f}'
        })
    
    return losses.avg


def validate(model, dataloader, criterion_class, criterion_anomaly, device):
    """Validate the model"""
    model.eval()
    losses = AverageMeter()
    
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc='Validation'):
            sequences = sequences.to(device)
            labels = labels.to(device).long()
            
            class_logits, anomaly_score = model(sequences)
            
            loss_class = criterion_class(class_logits, labels)
            loss_anomaly = criterion_anomaly(anomaly_score.squeeze(), labels.float())
            loss = loss_class + 0.5 * loss_anomaly
            
            losses.update(loss.item(), sequences.size(0))
    
    return losses.avg


def train_cnn_lstm(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=Config.CNN_LSTM_BATCH_SIZE,
        sequence_length=Config.SEQUENCE_LENGTH,
        validation_split=Config.VALIDATION_SPLIT,
        num_workers=Config.NUM_WORKERS
    )
    
    # Create model
    model = CNNLSTM(
        feature_dim=512,
        hidden_dim=Config.CNN_LSTM_HIDDEN_DIM,
        num_layers=Config.CNN_LSTM_NUM_LAYERS,
        num_classes=len(Config.ANOMALY_TYPES),
        dropout=Config.CNN_LSTM_DROPOUT
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion_class = nn.CrossEntropyLoss()
    criterion_anomaly = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.CNN_LSTM_LEARNING_RATE)
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
    
    for epoch in range(1, Config.CNN_LSTM_EPOCHS + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion_class, criterion_anomaly, 
            optimizer, device, epoch
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion_class, criterion_anomaly, device)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f"Epoch {epoch}/{Config.CNN_LSTM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint_path = Path(Config.WEIGHTS_DIR) / 'cnn_lstm_best.pth'
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
        
        # Save checkpoint
        if epoch % Config.SAVE_INTERVAL == 0:
            checkpoint_path = Path(Config.WEIGHTS_DIR) / f'cnn_lstm_epoch_{epoch}.pth'
            torch.save(model.state_dict(), checkpoint_path)
    
    # Plot training curves
    plot_training_curves(
        train_losses, val_losses,
        save_path=Path(Config.LOGS_DIR) / 'cnn_lstm_training.png'
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, val_loader, device, threshold=Config.ANOMALY_THRESHOLD)
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN-LSTM for Anomaly Detection')
    parser.add_argument('--data_dir', type=str, default=Config.DATA_DIR,
                       help='Path to data directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_cnn_lstm(args)
