from pathlib import Path
from typing import Dict, Any
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from .dataset import MultiOmicsDataset
from utils import set_seed


def create_stratified_splits(dataset: MultiOmicsDataset, config: dict) -> Dict[str, np.ndarray]:
    """
    Create stratified splits using parameters from config dictionary
    """
    # Access parameters through dictionary access
    splits = config['loader']['splits']
    seed = config['loader']['seed']

    set_seed(seed)
    labels = dataset.labels.squeeze()

    # First split: separate test set
    test_size = splits['test']
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    remaining_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))

    # Second split: separate val set from remaining
    remaining_labels = labels[remaining_idx]
    val_relative_size = splits['val'] / (1 - test_size)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(remaining_idx)), remaining_labels))

    return {
        'train': remaining_idx[train_idx],
        'val': remaining_idx[val_idx],
        'test': test_idx
    }


def create_dataloaders(dataset: MultiOmicsDataset, config: dict) -> Dict[str, DataLoader]:
    """
    Create dataloaders using parameters from config dictionary
    """
    # Access parameters through dictionary access
    batch_size = config['loader']['batch_size']
    num_workers = config['loader']['num_workers']
    seed = config['loader']['seed']

    set_seed(seed)
    split_indices = create_stratified_splits(dataset, config)

    subsets = {
        split: Subset(dataset, indices)
        for split, indices in split_indices.items()
    }

    return {
        'train': DataLoader(
            subsets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            subsets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            subsets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }