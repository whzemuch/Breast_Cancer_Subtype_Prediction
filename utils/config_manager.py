"""Centralized management for multi-omics experiment configurations.

This module provides a ConfigManager class that serves as a registry for TrainingConfig
objects, enabling consistent model instantiation across experiments. The manager maintains
named configurations and provides utilities for model creation and configuration inspection.

Typical usage example:

  # Initialize manager and register configurations
  manager = ConfigManager()
  manager.add_config("default", TrainingConfig(), set_default=True)
  manager.add_config("large", TrainingConfig(latent_dim=128))

  # Retrieve models
  model = manager.get_model()  # Uses default config
  large_model = manager.get_model("large")
"""

from typing import Dict, Optional
from pathlib import Path
from

class ConfigManager:
    """Registry for managing and accessing experiment configurations.

    The ConfigManager provides centralized storage of TrainingConfig objects with
    named references, enabling:
    - Consistent configuration reuse across experiments
    - Default configuration fallback
    - Direct model instantiation
    - Runtime configuration inspection

    Attributes:
        _configs: Dictionary mapping configuration names to TrainingConfig instances
        _default_config: Name of the default configuration (optional)
    """

    def __init__(self):
        """Initializes an empty configuration registry."""
        self._configs: Dict[str, TrainingConfig] = {}
        self._default_config: Optional[str] = None

    def add_config(
        self,
        name: str,
        config: TrainingConfig,
        set_default: bool = False
    ) -> None:
        """Registers a new configuration with the manager.

        Args:
            name: Unique identifier for the configuration
            config: TrainingConfig instance to store
            set_default: Whether to set this as the default configuration

        Raises:
            ValueError: If configuration name already exists
        """
        if name in self._configs:
            raise ValueError(f"Configuration '{name}' already exists")
        self._configs[name] = config
        if set_default:
            self._default_config = name

    def get_config(self, name: Optional[str] = None) -> TrainingConfig:
        """Retrieves a configuration by name.

        Args:
            name: Name of the configuration to retrieve. If None, returns the default.

        Returns:
            Requested TrainingConfig instance

        Raises:
            KeyError: If requested configuration doesn't exist
            ValueError: If no default is set and no name provided
        """
        config_name = name or self._default_config
        if config_name is None:
            raise ValueError("No default configuration set and no name provided")
        if config_name not in self._configs:
            available = ", ".join(self._configs.keys())
            raise KeyError(
                f"Config '{config_name}' not found. Available: {available}"
            )
        return self._configs[config_name]

    def get_model(self, config_name: Optional[str] = None) -> MultiOmicsClassifier:
        """Instantiates a MultiOmicsClassifier from a named configuration.

        Args:
            config_name: Name of the configuration to use. Uses default if None.

        Returns:
            New MultiOmicsClassifier instance configured with the specified parameters

        Example:
            >>> manager.get_model("large")
            MultiOmicsClassifier(mirna_dim=1046, rna_exp_dim=13054, ...)
        """
        config = self.get_config(config_name)
        return MultiOmicsClassifier(
            mirna_dim=config.mirna_dim,
            rna_exp_dim=config.rna_exp_dim,
            methy_shape=config.methy_shape,
            latent_dim=config.latent_dim,
            num_classes=config.num_classes
        )

    @property
    def available_configs(self) -> Dict[str, TrainingConfig]:
        """Returns all registered configurations.

        Returns:
            Dictionary mapping configuration names to TrainingConfig instances
        """
        return self._configs.copy()

    @property
    def default_config_name(self) -> Optional[str]:
        """Returns the name of the default configuration.

        Returns:
            Name of the default config if set, else None
        """
        return self._default_config