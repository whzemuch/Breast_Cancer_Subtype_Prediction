import torch
from models.vae import VAEEncoder
from models.convnext import MiniConvNeXtMethylation
from models.model_fusion import MultimodalFusion
from models.classifier import MultiOmicsClassifier


def test_vae_forward():
    vae = VAEEncoder(input_dim=1000, latent_dim=64)
    x = torch.randn(4, 1000)
    z, mu, logvar = vae(x)
    assert z.shape == (4, 64)
    assert mu.shape == (4, 64)
    assert logvar.shape == (4, 64)


def test_mirna_vae_forward():
    vae = VAEEncoder(input_dim=50, latent_dim=32)
    x = torch.randn(4, 50)
    z, mu, logvar = vae(x)
    assert z.shape == (4, 32)


def test_mini_convnext():
    model = MiniConvNeXtMethylation()
    x = torch.randn(4, 1, 50, 100)
    out = model(x)
    assert out.shape == (4, 128)
    print("✅ Passed:", out.shape)




def test_transformer_fusion():
    transformer = MultimodalFusion( latent_dim=128)

    z_mrna = torch.randn(4, 128)
    z_protein = torch.randn(4, 128)
    z_methyl = torch.randn(4, 128)
    fused = transformer([z_mrna, z_protein, z_methyl])
    assert fused.shape == (4, 128)


def test_fusion_classifier():
    batch_size = 4
    mirna_dim = 2000
    rna_exp_dim = 20000
    methy_shape = (50, 100)
    latent_dim = 64
    num_classes = 4

    # Instantiate full model
    classifier = MultiOmicsClassifier(
        mirna_dim=mirna_dim,
        rna_exp_dim=rna_exp_dim,
        methy_shape=methy_shape,
        latent_dim=latent_dim,
        num_classes=num_classes
    )
    classifier.eval()

    # Create dummy inputs
    x_methy = torch.randn(batch_size, 1, *methy_shape)
    x_mirna = torch.randn(batch_size, mirna_dim)
    x_rna   = torch.randn(batch_size, rna_exp_dim)

    # Forward pass
    with torch.no_grad():
        outputs = classifier(x_methy, x_mirna, x_rna)

    # Assertions
    logits = outputs['logits']
    assert logits.shape == (batch_size, num_classes), f"Expected ({batch_size}, {num_classes}), got {logits.shape}"

    print("✅ test_classifier_head passed — logits shape:", logits.shape)
