import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128, hidden_dims=None, dropout=0.2):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 512]  # default fallback

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Build encoder dynamically
        encoder_layers = []
        in_features = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_features, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            in_features = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def reparameterize(self, mu, logvar):
        """Improved reparameterization with numerical stability"""
        std = torch.exp(0.5 * logvar) + 1e-6
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input with dim {self.input_dim}, but got {x.shape[1]}")

        h = self.encoder(x)
        mu = self.mu(h)
        logvar = F.softplus(self.logvar(h))  # ensure logvar > 0
        logvar = torch.clamp(logvar, min=1e-4, max=10.0)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def kl_divergence(self, mu, logvar):
        """KL divergence between q(z|x) and N(0, I)"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def __repr__(self):
        return f"VAEEncoder(input_dim={self.input_dim}, latent_dim={self.latent_dim}, hidden_dims={self.hidden_dims})"
