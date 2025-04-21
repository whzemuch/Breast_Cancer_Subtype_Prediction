import torch
import torch.nn as nn


class MultimodalFusion(nn.Module):
    """Transformer-based fusion for 3 modalities"""
    def __init__(self, latent_dim=64, num_heads=4):
        super().__init__()
        self.position_embed = nn.Embedding(3, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=4*latent_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.attn_weights = None

    def forward(self, features):
        # features: list of [B, D] tensors (methy, mirna, rna_exp)
        stacked = torch.stack(features, dim=1)  # [B, 3, D]
        positions = self.position_embed(torch.arange(3, device=stacked.device))
        encoded = self.transformer(stacked + positions)
        return self.pool(encoded.transpose(1,2)).squeeze(-1)

 
