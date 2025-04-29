
"""
Configuration management for multi-omics training experiments with optional parameters.
"""

from dataclasses import dataclass
from pathlib import Path
import torch
from typing import Optional, Union, Tuple


@dataclass
class TrainingConfig:
    """
    Centralized configuration for multi-omics training experiments with optional parameters.

    Attributes:
        beta: KL divergence weight (None to disable KL loss term)
        use_focal: Whether to use focal loss
        focal_gamma: Gamma parameter for focal loss (None when use_focal=False)
        label_smoothing: Label smoothing factor (None to disable)
        class_weights: Per-class weights tensor (None for equal weights): torch.Tensor = torch.tensor([1.0, 1.0, 1.0, 1.5, 2.5], device="cuda")
        mirna_dim: Dimension of miRNA features
        rna_exp_dim: Dimension of RNA expression features
        methy_shape: Shape of methylation data as (rows, cols)
        latent_dim: Dimension of latent space
        num_classes: Number of output classes
        base_log_dir: Root directory for logs (str or Path)
        experiment_name: Optional subdirectory name
    """
    # Loss parameters (all optional except use_focal)
    beta: Optional[float] = 1.0
    use_focal: bool = False
    focal_gamma: Optional[float] = None
    label_smoothing: Optional[float] = None
    class_weights: Optional[torch.Tensor] = None

    # Model architecture (required parameters)
    mirna_dim: int = 1046
    rna_exp_dim: int = 13054
    methy_shape: Tuple[int, int] = (50, 100)
    latent_dim: int = 32
    num_classes: int = 5

    # Logging directories
    base_log_dir: Union[str, Path] = Path("logs")
    experiment_name: Optional[str] = None

    # Auto-generated paths (do not set manually)
    loss_history_path: Optional[Path] = None
    accuracy_history_path: Optional[Path] = None
    tsne_results_path: Optional[Path] = None
    attention_weights_path: Optional[Path] = None

    def __post_init__(self):
        """Initialize paths and validate configuration."""
        self._convert_paths()
        self._update_paths()
        self.validate()

    def _convert_paths(self):
        """Ensure all paths are Path objects."""
        if isinstance(self.base_log_dir, str):
            self.base_log_dir = Path(self.base_log_dir)

    def _update_paths(self):
        """Update all file paths based on current configuration."""
        log_dir = self.base_log_dir
        if self.experiment_name:
            log_dir = log_dir / self.experiment_name

        log_dir.mkdir(parents=True, exist_ok=True)

        self.loss_history_path = log_dir / "loss_history.json"
        self.accuracy_history_path = log_dir / "accuracy_history.json"
        self.tsne_results_path = log_dir / "tsne_results.pkl"
        self.attention_weights_path = log_dir / "attention_weights.pkl"

    def validate(self):
        """
        Validate configuration consistency.

        Raises:
            ValueError: If invalid parameter combinations are detected.
        """
        # Focal loss validation
        if self.use_focal and self.focal_gamma is None:
            raise ValueError("focal_gamma must be specified when use_focal=True")
        if not self.use_focal and self.focal_gamma is not None:
            raise ValueError("focal_gamma should be None when use_focal=False")

        # Class weights validation
        if self.class_weights is not None:
            if len(self.class_weights) != self.num_classes:
                raise ValueError(
                    f"class_weights length ({len(self.class_weights)}) "
                    f"must match num_classes ({self.num_classes})"
                )

    def update(self, **kwargs):
        """
        Safely update configuration parameters with validation.

        Args:
            **kwargs: Parameter names and values to update

        Returns:
            self: For method chaining

        Example:
            >>> config.update(beta=None, experiment_name="no_kl_experiment")
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Re-validate and update paths if needed
        self.validate()
        if any(k in kwargs for k in ('base_log_dir', 'experiment_name')):
            self._update_paths()

        return self

    def create_experiment_copy(self, experiment_name: str, **kwargs):
        """
        Create a new configuration with experiment-specific settings.

        Args:
            experiment_name: Name for the new experiment
            **kwargs: Additional parameters to override

        Returns:
            TrainingConfig: New validated configuration instance
        """
        new_config = TrainingConfig(
            **{f.name: getattr(self, f.name)
               for f in self.__dataclass_fields__.values()
               if not f.name.endswith('_path')}
        )
        return new_config.update(experiment_name=experiment_name, **kwargs)

    def __str__(self):
        """User-friendly string representation."""
        params = []
        for field in self.__dataclass_fields__.values():
            if not field.name.endswith('_path'):
                value = getattr(self, field.name)
                params.append(f"{field.name}: {value}")
        return "TrainingConfig:\n" + "\n".join(params)