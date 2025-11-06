"""
Evaluation metrics for anomaly detection
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report, f1_score
)


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_reconstruction_error(model, dataloader, device):
    """
    Compute reconstruction errors for anomaly detection
    
    Returns:
        errors: list of reconstruction errors
        labels: list of ground truth labels
    """
    model.eval()
    errors = []
    labels = []
    
    with torch.no_grad():
        for frames, targets in dataloader:
            frames = frames.to(device)
            
            # Compute reconstruction error
            error = model.compute_reconstruction_error(frames)
            errors.extend(error.cpu().numpy())
            labels.extend(targets.numpy())
    
    return np.array(errors), np.array(labels)


def compute_anomaly_scores(model, dataloader, device):
    """
    Compute anomaly scores from combined model
    
    Returns:
        scores: anomaly scores
        labels: ground truth labels
        predictions: predicted classes
    """
    model.eval()
    scores = []
    labels = []
    predictions = []
    
    with torch.no_grad():
        for sequences, targets in dataloader:
            sequences = sequences.to(device)
            
            output = model(sequences)
            score = output['anomaly_score'].squeeze()
            pred = torch.argmax(output['class_logits'], dim=1)
            
            scores.extend(score.cpu().numpy())
            labels.extend(targets.numpy())
            predictions.extend(pred.cpu().numpy())
    
    return np.array(scores), np.array(labels), np.array(predictions)


def calculate_metrics(y_true, y_pred, y_scores=None):
    """
    Calculate various evaluation metrics
    
    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
        y_scores: Anomaly scores (optional, for ROC-AUC)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Basic metrics
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['f1_score'] = f1_score(y_true, y_pred)
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # ROC-AUC (requires scores)
    if y_scores is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        metrics['pr_auc'] = auc(recall, precision)
    
    # Confusion matrix values
    metrics['true_positives'] = tp
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    
    return metrics


def find_optimal_threshold(y_true, y_scores):
    """
    Find optimal threshold for binary classification
    Uses F1 score as the optimization metric
    
    Returns:
        optimal_threshold, best_f1_score
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Find threshold with best F1 score
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1_scores[best_idx]
    
    return optimal_threshold, best_f1


def print_metrics(metrics, title="Metrics"):
    """Pretty print metrics"""
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1 Score:          {metrics['f1_score']:.4f}")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
    print(f"False Negative Rate: {metrics['false_negative_rate']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    if 'pr_auc' in metrics:
        print(f"PR-AUC:            {metrics['pr_auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics['true_negatives']:4d}  |  FP: {metrics['false_positives']:4d}")
    print(f"  FN: {metrics['false_negatives']:4d}  |  TP: {metrics['true_positives']:4d}")
    print(f"{'='*50}\n")


def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Complete evaluation of anomaly detection model
    
    Returns:
        Dictionary of metrics and predictions
    """
    # Get predictions
    scores, y_true, y_pred_classes = compute_anomaly_scores(model, dataloader, device)
    
    # Binary predictions based on threshold
    y_pred_binary = (scores > threshold).astype(int)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred_binary, scores)
    
    # Find optimal threshold
    optimal_threshold, best_f1 = find_optimal_threshold(y_true, scores)
    metrics['optimal_threshold'] = optimal_threshold
    metrics['optimal_f1'] = best_f1
    
    # Print results
    print_metrics(metrics, title="Anomaly Detection Metrics")
    print(f"Optimal Threshold: {optimal_threshold:.4f} (F1: {best_f1:.4f})")
    
    return {
        'metrics': metrics,
        'scores': scores,
        'predictions': y_pred_binary,
        'true_labels': y_true,
        'predicted_classes': y_pred_classes
    }


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    # Dummy data
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 0])
    y_scores = np.array([0.1, 0.2, 0.9, 0.4, 0.3, 0.8, 0.6, 0.85, 0.95, 0.15])
    
    metrics = calculate_metrics(y_true, y_pred, y_scores)
    print_metrics(metrics)
    
    optimal_threshold, best_f1 = find_optimal_threshold(y_true, y_scores)
    print(f"Optimal threshold: {optimal_threshold:.4f} (F1: {best_f1:.4f})")
    
    print("\nMetrics test passed! ✓")
