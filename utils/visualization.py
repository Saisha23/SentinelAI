"""
Visualization utilities for training and results
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from pathlib import Path


def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_reconstructions(original, reconstructed, num_samples=8, save_path=None):
    """Visualize original frames and their reconstructions"""
    num_samples = min(num_samples, original.size(0))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    
    for i in range(num_samples):
        # Original
        orig_img = original[i].permute(1, 2, 0).numpy()
        orig_img = np.clip(orig_img, 0, 1)
        axes[0, i].imshow(orig_img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # Reconstructed
        recon_img = reconstructed[i].permute(1, 2, 0).numpy()
        recon_img = np.clip(recon_img, 0, 1)
        axes[1, i].imshow(recon_img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved reconstructions to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_anomaly_scores(scores, labels, threshold=0.5, save_path=None):
    """Plot distribution of anomaly scores"""
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Histogram
    plt.subplot(1, 2, 1)
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='green')
    plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomalous', color='red')
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
    
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Anomaly Scores', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Box plot
    plt.subplot(1, 2, 2)
    data_to_plot = [normal_scores, anomaly_scores]
    plt.boxplot(data_to_plot, labels=['Normal', 'Anomalous'])
    plt.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold')
    plt.ylabel('Anomaly Score', fontsize=12)
    plt.title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved anomaly score plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(y_true, y_scores, save_path=None):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_precision_recall_curve(y_true, y_scores, save_path=None):
    """Plot Precision-Recall curve"""
    from sklearn.metrics import precision_recall_curve, auc
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PR curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=class_names or ['Normal', 'Anomalous'],
                yticklabels=class_names or ['Normal', 'Anomalous'])
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_attention_weights(attention_weights, frame_indices=None, save_path=None):
    """Visualize attention weights across sequence"""
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
    
    # attention_weights shape: (batch_size, seq_len, 1)
    attention_weights = attention_weights.squeeze(-1)  # (batch_size, seq_len)
    
    plt.figure(figsize=(12, 6))
    
    for i, weights in enumerate(attention_weights[:4]):  # Plot first 4 samples
        plt.subplot(2, 2, i + 1)
        plt.bar(range(len(weights)), weights, color='steelblue')
        plt.xlabel('Frame Index', fontsize=10)
        plt.ylabel('Attention Weight', fontsize=10)
        plt.title(f'Sample {i+1}', fontsize=12)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention weights to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_temporal_anomaly_scores(scores, labels, save_path=None):
    """Plot anomaly scores over time"""
    plt.figure(figsize=(14, 6))
    
    time_steps = np.arange(len(scores))
    
    # Plot scores
    plt.plot(time_steps, scores, 'b-', alpha=0.7, linewidth=1, label='Anomaly Score')
    
    # Highlight anomalous regions
    anomalous_indices = np.where(labels == 1)[0]
    for idx in anomalous_indices:
        plt.axvspan(idx - 0.5, idx + 0.5, alpha=0.3, color='red')
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Anomaly Score', fontsize=12)
    plt.title('Temporal Anomaly Scores', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved temporal anomaly scores to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_evaluation_report(metrics, scores, labels, save_dir):
    """Create comprehensive evaluation report with all visualizations"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Binary predictions
    threshold = metrics.get('optimal_threshold', 0.5)
    y_pred = (scores > threshold).astype(int)
    
    # Plot all visualizations
    plot_anomaly_scores(scores, labels, threshold, 
                       save_path=save_dir / 'anomaly_scores.png')
    
    plot_roc_curve(labels, scores, 
                  save_path=save_dir / 'roc_curve.png')
    
    plot_precision_recall_curve(labels, scores,
                                save_path=save_dir / 'pr_curve.png')
    
    plot_confusion_matrix(labels, y_pred,
                         save_path=save_dir / 'confusion_matrix.png')
    
    print(f"\nEvaluation report saved to {save_dir}")


if __name__ == "__main__":
    print("Testing visualization utilities...")
    
    # Dummy data
    train_losses = np.random.rand(20) * 0.5 + 0.1
    val_losses = np.random.rand(20) * 0.5 + 0.15
    
    plot_training_curves(train_losses, val_losses)
    
    print("Visualization test passed! ✓")
