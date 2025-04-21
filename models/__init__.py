# models/__init__.py

# Import core components
from .classifier import MultiOmicsClassifier
from .convnext import MiniConvNeXtMethylation
from .vae import VAEEncoder
from .model_fusion import MultimodalFusion

# Optional: Define public API
__all__ = [
    'MultiOmicsClassifier',
    'MiniConvNeXtMethylation',
    'VAEEncoder',
    'MultimodalFusion'
]