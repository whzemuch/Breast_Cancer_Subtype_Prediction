from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any


class MultiOmicsDataset(Dataset):
    """Multi-omics dataset with configuration-driven loading"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config (dict): Configuration dictionary containing:
                - data_dir: Base directory path
                - omics: Dictionary of omics data specifications
                - labels: Label loading configuration
        """
        self.config = config
        self.data_dir = Path(config["data_dir"])
        self.data = {}
        self.labels = None
        self._load_data()
        self._validate_data_shapes()

    def _load_data(self) -> None:
        """Load all omics data and labels according to configuration"""

        # Load each omics modality
        for modality, specs in self.config["omics"].items():
            file_path = self.data_dir / specs["file"]
            df = pd.read_csv(file_path)
            values = df.values.astype(specs["dtype"])

            if "reshape" in specs:
                values = values.reshape(values.shape[0], *specs["reshape"])

            self.data[modality] = values

        # Load labels
        labels_spec = self.config["labels"]
        labels_df = pd.read_csv(self.data_dir / labels_spec["file"])
        self.labels = labels_df.values.astype(labels_spec["dtype"])

        if labels_spec.get("squeeze", False):
            self.labels = np.squeeze(self.labels)

    def _validate_data_shapes(self) -> None:
        """Verify consistent sample counts across modalities"""
        sample_counts = {mod: len(data) for mod, data in self.data.items()}
        sample_counts["labels"] = len(self.labels)

        if len(set(sample_counts.values())) > 1:
            raise ValueError(
                f"Inconsistent sample counts: {sample_counts}\n"
                f"All modalities must have the same number of samples"
            )

    # def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    #     """Return a multi-omics sample with proper tensor conversion"""
    #     sample = {
    #         modality: torch.tensor(data[idx], dtype=torch.float32)
    #         for modality, data in self.data.items()
    #     }

    #     # Special handling for methylation data
    #     if "methyl" in sample:
    #         sample["methyl"] = sample["methyl"].unsqueeze(0)  # Add channel dim

    #     label = torch.tensor(self.labels[idx], dtype=torch.long)

    #     return sample, label

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Return a tuple of (multi-omics data dict, label tensor)"""
        # Create sample dictionary
        sample = {
            modality: torch.tensor(data[idx], dtype=torch.float32)
            for modality, data in self.data.items()
        }
    
        # Add channel dimension for methylation data if needed
        if "methyl" in sample:
            sample["methyl"] = sample["methyl"].unsqueeze(0)
    
        # Create label tensor
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # my lable does not start as 0
        
    
        return sample, label  # Explicit tuple return

    def __len__(self) -> int:
        return len(self.labels)

    def get_modality_shapes(self) -> Dict[str, tuple]:
        """Return the shape of each omics modality (excluding batch dim)"""
        return {
            mod: data.shape[1:]  # Remove batch dimension
            for mod, data in self.data.items()
        }