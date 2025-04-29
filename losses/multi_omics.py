# losses/multi_omics_loss.py
import torch
from torch import nn
from .focal import FocalLoss

class MultiOmicsLoss(nn.Module):
    def __init__(self, beta=0.1, class_weights=None, annealing_steps=10000,
                 use_focal=False, focal_gamma=2.0, label_smoothing=0.0, kl_epsilon=1e-8):
        super().__init__()

        # Validate focal parameters
        if use_focal and focal_gamma is None:
            raise ValueError("focal_gamma must be specified when use_focal=True")
        if not use_focal and focal_gamma is not None:
            raise ValueError("focal_gamma should be None when use_focal=False")

        self.register_buffer('current_step', torch.tensor(0))
        self.target_beta = beta
        self.annealing_steps = annealing_steps
        self.kl_epsilon = kl_epsilon

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights.clone().detach())

        else:
            self.class_weights = None

        self.use_focal = use_focal
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma if use_focal else None


        if use_focal:
            self.ce_loss = FocalLoss(
                weight=self.class_weights,
                gamma=focal_gamma,
                label_smoothing=label_smoothing
            )
        else:
            self.ce_loss = nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=label_smoothing
            )

    @property
    def beta(self):
        # Sigmoid warm-up curve
        center = self.annealing_steps // 2
        steepness = 5 / max(self.annealing_steps, 1)
        ratio = 1 / (1 + torch.exp(-steepness * (self.current_step.float() - center)))
        return self.target_beta * ratio
        
    # @property
    # def beta(self):
    #     ratio = torch.clamp(self.current_step.float() / max(self.annealing_steps, 1), 0, 1)
    #     return self.target_beta * ratio

    def kl_divergence(self, mu, logvar):
        kl_terms = -0.5 * (1 + logvar - mu.pow(2) - (logvar.exp() + self.kl_epsilon))
        return torch.sum(kl_terms, dim=1).mean()

    def reset_annealing(self):
        self.current_step.zero_()

    def forward(self, outputs, targets):
        if outputs['logits'].shape[0] != targets.shape[0]:
            raise ValueError(
                f"Batch size mismatch: logits {outputs['logits'].shape[0]} vs "
                f"targets {targets.shape[0]}"
            )

        ce = self.ce_loss(outputs['logits'], targets)
        kl_mirna = self.kl_divergence(outputs['mu_mirna'], outputs['logvar_mirna'])
        kl_rna = self.kl_divergence(outputs['mu_rna'], outputs['logvar_rna'])

        loss_dict = {
            'total': ce + self.beta * (kl_mirna + kl_rna),
            'ce': ce.detach(),
            'kl_total': (kl_mirna + kl_rna).detach(),
            'kl_mirna': kl_mirna.detach(),
            'kl_rna': kl_rna.detach(),
            'beta': self.beta.clone().detach().to(self.current_step.device)
        }

        if self.training:
            self.current_step += 1

        return loss_dict