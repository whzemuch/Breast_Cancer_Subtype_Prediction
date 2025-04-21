# tests/mocks/mock.py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict


class MockMultiOmicsDataset(Dataset):
    """Generates synthetic data_loader matching data_loader/simulation structure"""

    def __init__(self, num_samples=100):
        """
        Args:
            num_samples: Number of synthetic samples to generate (default: 100)
        """
        self.num_samples = num_samples
        self._generate_data()

    def _generate_data(self):
        """Create synthetic data_loader matching real dataset specs"""
        # Methylation data_loader: 100 samples, 5000 features -> reshaped to 50x100
        self.methyl = np.random.randn(self.num_samples, 50, 100).astype(np.float32)

        # miRNA expression: 100 samples, 5000 features
        self.mirna = np.random.randn(self.num_samples, 5000).astype(np.float32)

        # RNA expression: 100 samples, 5000 features
        self.rna_exp = np.random.randn(self.num_samples, 5000).astype(np.float32)

        # Binary labels (squeezed to 1D array)
        self.labels = np.random.randint(0, 2, size=self.num_samples).astype(np.int64)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "methyl": torch.tensor(self.methyl[idx]).unsqueeze(0),  # [1, 50, 100]
            "mirna": torch.tensor(self.mirna[idx]),
            "rna_exp": torch.tensor(self.rna_exp[idx])
        }, torch.tensor(self.labels[idx], dtype=torch.long)


def get_mock_config():
    """Returns config matching data_loader/simulation structure"""
    return {
        "data_dir": "/fake/path",
        "omics": {
            "methyl": {
                "file": "methylation.csv",
                "dtype": "float32",
                "reshape": [50, 100]
            },
            "mirna": {
                "file": "mirna.csv",
                "dtype": "float32"
            },
            "rna_exp": {
                "file": "rna_exp.csv",
                "dtype": "float32"
            }
        },
        "labels": {
            "file": "labels.csv",
            "dtype": "int64",
            "squeeze": True
        }
    }