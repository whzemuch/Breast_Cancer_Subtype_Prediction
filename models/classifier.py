from torch import nn
from .convnext import MiniConvNeXtMethylation
from .vae import VAEEncoder
from .model_fusion import MultimodalFusion

class MultiOmicsClassifier(nn.Module):
    def __init__(self,
                 mirna_dim=2000,
                 rna_exp_dim=20000,
                 methy_shape=(50, 100),
                 latent_dim=64,
                 num_classes=4):
        super().__init__()

        # Modality encoders
        self.methy_encoder = MiniConvNeXtMethylation(latent_dim)
        self.mirna_vae = VAEEncoder(mirna_dim, latent_dim)
        self.rna_vae = VAEEncoder(rna_exp_dim, latent_dim)

        # LayerNorm for each modality
        self.norm_methyl = nn.LayerNorm(latent_dim)
        self.norm_mirna = nn.LayerNorm(latent_dim)
        self.norm_rna = nn.LayerNorm(latent_dim)

        # Fusion + normalization before classification
        self.fusion = MultimodalFusion(latent_dim)
        self.norm_fused = nn.LayerNorm(latent_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, methyl, mirna, rna):
        # Encode and normalize each modality
        z_methyl = self.norm_methyl(self.methy_encoder(methyl))
        z_mirna, mu_mirna, logvar_mirna = self.mirna_vae(mirna)
        z_mirna = self.norm_mirna(z_mirna)
        z_rna, mu_rna, logvar_rna = self.rna_vae(rna)
        z_rna = self.norm_rna(z_rna)

        # Fuse and normalize

        fused = self.fusion([z_methyl, z_mirna, z_rna])
        # print(f"Fusion module attn_weights: {self.fusion.attn_weights.shape}")
        fused = self.norm_fused(fused)

        # Classify
        logits = self.classifier(fused)

        return {
            'logits': logits,
            'mu_mirna': mu_mirna,
            'logvar_mirna': logvar_mirna,
            'mu_rna': mu_rna,
            'logvar_rna': logvar_rna
        }
