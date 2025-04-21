import torch
from torch import nn

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)  # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)  # → (B, H, W, C) for LayerNorm

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = x.permute(0, 3, 1, 2)  # → back to (B, C, H, W)
        return x + residual  # residual connection

class ChannelLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        # x: (B, C, H, W) → (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # → (B, H, W, C) → (B, C, H, W)
        return x.permute(0, 3, 1, 2)


class MiniConvNeXtMethylation(nn.Module):
    def __init__(self, latent_dim=128, input_size=(50, 100)):
        super().__init__()
        self.input_size = input_size
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=4),
            ChannelLayerNorm(32)
        )

        self.blocks = nn.Sequential(
            ConvNeXtBlock(32),
            ConvNeXtBlock(32)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(32),
            nn.Linear(32, latent_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        if x.shape[-2:] != self.input_size:
            raise ValueError(f"Expected input shape {self.input_size}, got {x.shape[-2:]}")

        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

