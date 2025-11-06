"""
Combined Anomaly Detection Model
Integrates Autoencoder and CNN-LSTM for robust anomaly detection
"""

import torch
import torch.nn as nn
from .autoencoder import Autoencoder
from .cnn_lstm import CNNLSTM


class CombinedAnomalyDetector(nn.Module):
    """
    Combined model that uses both Autoencoder and CNN-LSTM
    - Autoencoder: Detects frame-level anomalies (reconstruction error)
    - CNN-LSTM: Detects temporal/sequence-level anomalies
    """
    
    def __init__(self, 
                 autoencoder_latent_dim=128,
                 cnn_lstm_feature_dim=512,
                 cnn_lstm_hidden_dim=256,
                 cnn_lstm_num_layers=2,
                 num_classes=7,
                 ae_weight=0.4,
                 lstm_weight=0.6):
        super(CombinedAnomalyDetector, self).__init__()
        
        self.autoencoder = Autoencoder(latent_dim=autoencoder_latent_dim)
        self.cnn_lstm = CNNLSTM(
            feature_dim=cnn_lstm_feature_dim,
            hidden_dim=cnn_lstm_hidden_dim,
            num_layers=cnn_lstm_num_layers,
            num_classes=num_classes
        )
        
        self.ae_weight = ae_weight
        self.lstm_weight = lstm_weight
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, frame_sequence):
        """
        frame_sequence: (batch_size, seq_len, channels, height, width)
        Returns: combined anomaly score and classification
        """
        batch_size, seq_len, c, h, w = frame_sequence.size()
        
        # 1. Autoencoder: Process each frame individually
        frames = frame_sequence.view(batch_size * seq_len, c, h, w)
        reconstruction_errors = self.autoencoder.compute_reconstruction_error(frames)
        reconstruction_errors = reconstruction_errors.view(batch_size, seq_len)
        
        # Average reconstruction error across sequence
        ae_anomaly_score = reconstruction_errors.mean(dim=1, keepdim=True)  # (batch_size, 1)
        
        # 2. CNN-LSTM: Process temporal sequence
        class_logits, lstm_anomaly_score = self.cnn_lstm(frame_sequence)
        
        # 3. Combine scores
        combined_scores = torch.cat([ae_anomaly_score, lstm_anomaly_score], dim=1)
        final_anomaly_score = self.fusion(combined_scores)
        
        # Alternative: Weighted average (simpler approach)
        # final_anomaly_score = (self.ae_weight * ae_anomaly_score + 
        #                        self.lstm_weight * lstm_anomaly_score)
        
        return {
            'anomaly_score': final_anomaly_score,
            'ae_score': ae_anomaly_score,
            'lstm_score': lstm_anomaly_score,
            'class_logits': class_logits,
            'reconstruction_errors': reconstruction_errors
        }
    
    def predict(self, frame_sequence, threshold=0.7):
        """
        Predict anomaly with threshold
        Returns: is_anomalous, anomaly_score, predicted_class
        """
        with torch.no_grad():
            output = self.forward(frame_sequence)
            
            anomaly_score = output['anomaly_score'].squeeze()
            is_anomalous = anomaly_score > threshold
            predicted_class = torch.argmax(output['class_logits'], dim=1)
            
            return is_anomalous, anomaly_score, predicted_class
    
    def load_pretrained_models(self, ae_path, lstm_path):
        """Load pre-trained autoencoder and CNN-LSTM weights"""
        if ae_path:
            ae_state = torch.load(ae_path)
            self.autoencoder.load_state_dict(ae_state)
            print(f"Loaded autoencoder from {ae_path}")
        
        if lstm_path:
            lstm_state = torch.load(lstm_path)
            self.cnn_lstm.load_state_dict(lstm_state)
            print(f"Loaded CNN-LSTM from {lstm_path}")


class EnsembleAnomalyDetector(nn.Module):
    """
    Ensemble of multiple models for more robust detection
    Useful when you have multiple trained models
    """
    
    def __init__(self, models):
        super(EnsembleAnomalyDetector, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
    def forward(self, x):
        """Average predictions from all models"""
        scores = []
        class_logits_list = []
        
        for model in self.models:
            output = model(x)
            scores.append(output['anomaly_score'])
            class_logits_list.append(output['class_logits'])
        
        # Average scores
        avg_score = torch.stack(scores).mean(dim=0)
        avg_class_logits = torch.stack(class_logits_list).mean(dim=0)
        
        return {
            'anomaly_score': avg_score,
            'class_logits': avg_class_logits,
            'individual_scores': scores
        }


def test_combined_model():
    """Test combined model with dummy data"""
    print("Testing Combined Anomaly Detector...")
    
    # Create model
    model = CombinedAnomalyDetector(
        autoencoder_latent_dim=128,
        cnn_lstm_feature_dim=512,
        cnn_lstm_hidden_dim=256,
        num_classes=7
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # Dummy input (batch_size=2, seq_len=16, channels=3, height=224, width=224)
    x = torch.randn(2, 16, 3, 224, 224)
    
    # Forward pass
    output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Final anomaly score: {output['anomaly_score'].squeeze()}")
    print(f"Autoencoder score: {output['ae_score'].squeeze()}")
    print(f"LSTM score: {output['lstm_score'].squeeze()}")
    print(f"Class logits shape: {output['class_logits'].shape}")
    
    # Test prediction
    is_anomalous, score, pred_class = model.predict(x, threshold=0.7)
    print(f"\nPrediction:")
    print(f"Is anomalous: {is_anomalous}")
    print(f"Anomaly score: {score}")
    print(f"Predicted class: {pred_class}")
    
    print("\nCombined model test passed! ✓")


if __name__ == "__main__":
    test_combined_model()
