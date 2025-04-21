# losses/multi_omics_loss.py
import torch
from torch import nn


class MultiOmicsLoss(nn.Module):
    def __init__(self, beta=0.1, class_weights=None, annealing_steps=10000,
                 use_focal=False, label_smoothing=0.0, kl_epsilon=1e-8):
        super().__init__()
        self.register_buffer('current_step', torch.tensor(0))
        self.target_beta = beta
        self.annealing_steps = annealing_steps
        self.kl_epsilon = kl_epsilon

        # Handle class weights device compatibility
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights))
        else:
            self.class_weights = None

        # Initialize classification loss
        self.use_focal = use_focal
        self.label_smoothing = label_smoothing

        if use_focal:
            from .focal_loss import FocalLoss  # Local import
            self.ce_loss = FocalLoss(
                weight=self.class_weights,
                gamma=2.0,
                label_smoothing=label_smoothing
            )
        else:
            self.ce_loss = nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=label_smoothing
            )

    @property
    def beta(self):
        """Annealed beta value with progress clamping"""
        ratio = torch.clamp(self.current_step.float() / max(self.annealing_steps, 1), 0, 1)
        return self.target_beta * ratio

    def kl_divergence(self, mu, logvar):
        """Numerically stable KL calculation"""
        kl_terms = -0.5 * (1 + logvar - mu.pow(2) - (logvar.exp() + self.kl_epsilon))
        return torch.sum(kl_terms, dim=1).mean()  # Sum over latent dims, mean over batch

    def reset_annealing(self):
        """For training resumption support"""
        self.current_step.zero_()

    def forward(self, outputs, targets):
        # Validate inputs
        if outputs['logits'].shape[0] != targets.shape[0]:
            raise ValueError(
                f"Batch size mismatch: logits {outputs['logits'].shape[0]} vs "
                f"targets {targets.shape[0]}"
            )

        # Calculate components
        ce = self.ce_loss(outputs['logits'], targets)
        kl_mirna = self.kl_divergence(outputs['mu_mirna'], outputs['logvar_mirna'])
        kl_rna = self.kl_divergence(outputs['mu_rna'], outputs['logvar_rna'])
        print(
            f"[DEBUG] CE: {ce.item():.4f}, KL_mirna: {kl_mirna.item():.4f}, KL_rna: {kl_rna.item():.4f}")
        # Assemble loss dict
        loss_dict = {
            'total': ce + self.beta * (kl_mirna + kl_rna),
            'ce': ce.detach(),
            'kl_total': (kl_mirna + kl_rna).detach(),
            'kl_mirna': kl_mirna.detach(),
            'kl_rna': kl_rna.detach(),
            'beta': torch.tensor(self.beta, device=self.current_step.device)
        }

        # Update step counter (only during training)
        if self.training:
            self.current_step += 1

        return loss_dict