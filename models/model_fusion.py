# fusion_module.py

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerEncoderLayerWithAttention(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights = None

    def forward(self, src, src_mask=None, is_causal=False, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        self.attn_weights = attn_weights
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)


class MultimodalFusion(nn.Module):
    def __init__(self, latent_dim=64, num_heads=4):
        super().__init__()
        self.position_embed = nn.Embedding(3, latent_dim)
        encoder_layer = TransformerEncoderLayerWithAttention(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=4 * latent_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.attn_weights = None

    def forward(self, features):
        stacked = torch.stack(features, dim=1)  # [B, 3, D]
        positions = self.position_embed(torch.arange(3, device=stacked.device))
        encoded = self.transformer(stacked + positions)  # [B, 3, D]
        self.attn_weights = self.transformer.layers[0].attn_weights
        return self.pool(encoded.transpose(1, 2)).squeeze(-1)


 
