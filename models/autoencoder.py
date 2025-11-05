"""
Autoencoder Model for Anomaly Detection
Learns to reconstruct normal video frames - high reconstruction error indicates anomaly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder network to compress input frames into latent representation"""
    
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)  # 224 -> 112
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 112 -> 56
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 56 -> 28
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 28 -> 14
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # 14 -> 7
        self.bn5 = nn.BatchNorm2d(512)
        
        # Flatten and fully connected
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 7 * 7, latent_dim)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    """Decoder network to reconstruct frames from latent representation"""
    
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        
        # Fully connected
        self.fc = nn.Linear(latent_dim, 512 * 7 * 7)
        
        # Transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 7 -> 14
        self.bn1 = nn.BatchNorm2d(256)
        
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 14 -> 28
        self.bn2 = nn.BatchNorm2d(128)
        
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 28 -> 56
        self.bn3 = nn.BatchNorm2d(64)
        
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 56 -> 112
        self.bn4 = nn.BatchNorm2d(32)
        
        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)  # 112 -> 224
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 7, 7)
        
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = torch.sigmoid(self.deconv5(x))  # Output in [0, 1]
        
        return x


class Autoencoder(nn.Module):
    """Complete Autoencoder for video frame anomaly detection"""
    
    def __init__(self, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim
        
    def forward(self, x):
        """Forward pass through autoencoder"""
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent
    
    def encode(self, x):
        """Get latent representation only"""
        return self.encoder(x)
    
    def decode(self, latent):
        """Reconstruct from latent representation"""
        return self.decoder(latent)
    
    def compute_reconstruction_error(self, x, reduction='mean'):
        """
        Compute reconstruction error (anomaly score)
        Higher error = more likely to be anomalous
        """
        reconstruction, _ = self.forward(x)
        mse = F.mse_loss(reconstruction, x, reduction='none')
        
        if reduction == 'mean':
            return mse.mean(dim=[1, 2, 3])  # Per sample
        elif reduction == 'none':
            return mse
        else:
            return mse.mean()


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) - Alternative implementation
    Uses probabilistic latent space for better generalization
    """
    
    def __init__(self, latent_dim=128):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dim * 2)  # Output mean and logvar
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """Forward pass through VAE"""
        # Encode
        encoded = self.encoder(x)
        mu, logvar = encoded[:, :self.latent_dim], encoded[:, self.latent_dim:]
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstruction = self.decoder(z)
        
        return reconstruction, mu, logvar
    
    def compute_loss(self, x, beta=1.0):
        """
        Compute VAE loss: reconstruction + KL divergence
        beta: weight for KL divergence term
        """
        reconstruction, mu, logvar = self.forward(x)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + beta * kl_loss, recon_loss, kl_loss


def test_autoencoder():
    """Test autoencoder with dummy data"""
    print("Testing Autoencoder...")
    
    # Create model
    model = Autoencoder(latent_dim=128)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Dummy input (batch_size=4, channels=3, height=224, width=224)
    x = torch.randn(4, 3, 224, 224)
    
    # Forward pass
    reconstruction, latent = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    # Compute reconstruction error
    error = model.compute_reconstruction_error(x)
    print(f"Reconstruction error per sample: {error}")
    
    print("\nAutoencoder test passed! ✓")


if __name__ == "__main__":
    test_autoencoder()
