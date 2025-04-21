# losses/focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('weight', torch.tensor(weight) if weight is not None else None)
        self.label_smoothing = label_smoothing
        self.epsilon = 1e-8  # For numerical stability

    def forward(self, inputs, targets):
        # Input validation
        if inputs.dim() != 2:
            raise ValueError(f"Inputs must be 2D (batch_size, num_classes). Got {inputs.dim()}D")

        if targets.dim() != 1:
            raise ValueError(f"Targets must be 1D class indices. Got {targets.dim()}D")

        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )

        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return torch.mean(focal_loss + self.epsilon)  # Prevent zero loss