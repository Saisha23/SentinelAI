"""
CNN-LSTM Model for Temporal Anomaly Detection
Analyzes sequences of video frames to detect temporal patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):
    """CNN to extract spatial features from each frame"""
    
    def __init__(self, feature_dim=512):
        super(CNNFeatureExtractor, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)  # 224 -> 112
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 112 -> 56
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 56 -> 28
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 28 -> 14
        self.bn4 = nn.BatchNorm2d(512)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected
        self.fc = nn.Linear(512, feature_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        x: (batch_size, channels, height, width)
        returns: (batch_size, feature_dim)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(F.relu(self.fc(x)))
        return x


class CNNLSTM(nn.Module):
    """
    CNN-LSTM for temporal sequence analysis
    CNN extracts features from each frame, LSTM analyzes temporal patterns
    """
    
    def __init__(self, feature_dim=512, hidden_dim=256, num_layers=2, 
                 num_classes=7, dropout=0.3):
        super(CNNLSTM, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # CNN feature extractor
        self.cnn = CNNFeatureExtractor(feature_dim)
        
        # LSTM for temporal analysis
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Classification head
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Anomaly score head
        self.anomaly_fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x, return_attention=False):
        """
        x: (batch_size, sequence_length, channels, height, width)
        returns: class logits and anomaly score
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Extract CNN features for each frame
        # Reshape to (batch_size * seq_len, channels, height, width)
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn(x)  # (batch_size * seq_len, feature_dim)
        
        # Reshape back to (batch_size, seq_len, feature_dim)
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(features)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)  # (batch_size, seq_len, 1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_dim * 2)
        
        # Classification
        x = F.relu(self.fc1(attended))
        x = self.dropout(x)
        class_logits = self.fc2(x)
        
        # Anomaly score (0-1 range)
        anomaly_score = torch.sigmoid(self.anomaly_fc(attended))
        
        if return_attention:
            return class_logits, anomaly_score, attention_weights
        
        return class_logits, anomaly_score
    
    def predict_anomaly(self, x):
        """
        Predict if sequence is anomalous
        Returns anomaly score and predicted class
        """
        with torch.no_grad():
            class_logits, anomaly_score = self.forward(x)
            predicted_class = torch.argmax(class_logits, dim=1)
            return anomaly_score.squeeze(), predicted_class


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell - Alternative implementation
    Maintains spatial structure while processing temporal information
    """
    
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class ConvLSTM(nn.Module):
    """
    Convolutional LSTM for spatiotemporal processing
    Better for preserving spatial information across frames
    """
    
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            cell_list.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size))
        
        self.cell_list = nn.ModuleList(cell_list)
        
    def forward(self, input_tensor, hidden_state=None):
        """
        input_tensor: (batch, seq_len, channels, height, width)
        """
        batch_size, seq_len, _, height, width = input_tensor.size()
        
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, height, width, input_tensor.device)
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], (h, c))
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
        
        return layer_output_list[-1], last_state_list
    
    def _init_hidden(self, batch_size, height, width, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append((
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
            ))
        return init_states


def test_cnn_lstm():
    """Test CNN-LSTM with dummy data"""
    print("Testing CNN-LSTM...")
    
    # Create model
    model = CNNLSTM(feature_dim=512, hidden_dim=256, num_layers=2, num_classes=7)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Dummy input (batch_size=2, seq_len=16, channels=3, height=224, width=224)
    x = torch.randn(2, 16, 3, 224, 224)
    
    # Forward pass
    class_logits, anomaly_score, attention = model(x, return_attention=True)
    print(f"Input shape: {x.shape}")
    print(f"Class logits shape: {class_logits.shape}")
    print(f"Anomaly score shape: {anomaly_score.shape}")
    print(f"Attention weights shape: {attention.shape}")
    print(f"Anomaly scores: {anomaly_score.squeeze()}")
    
    print("\nCNN-LSTM test passed! ✓")


if __name__ == "__main__":
    test_cnn_lstm()
