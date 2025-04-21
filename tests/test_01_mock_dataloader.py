import pytest
import torch
from torch.utils.data import DataLoader
from .mock import MockMultiOmicsDataset

et
from data_loader import create_dataloaders



def test_mock_dataloader_shuffling():
    dataset = MockMultiOmicsDataset(100)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Check order changes between epochs
    first_pass = next(iter(loader))[0]["methyl"][0]
    second_pass = next(iter(loader))[0]["methyl"][0]

    assert not torch.allclose(first_pass, second_pass), "Shuffling failed!"


@pytest.mark.parametrize("val_split, test_split, expected_splits", [
    (0.2, 0.1, (70, 20, 10)),  # 3-way split
    (0.3, 0.0, (70, 30, 0)),  # Train/Val only
    (0.0, 0.2, (80, 0, 20)),  # Train/Test only
    (0.0, 0.0, (100, 0, 0)),  # No split
])
def test_dataloader_splits(dataset_and_config, val_split, test_split, expected_splits):
    """Test different split configurations"""
    dataset, base_config = dataset_and_config
    total_samples = len(dataset)

    # Update config with current splits
    config = {
        **base_config,
        "val_split": val_split,
        "test_split": test_split
    }

    # Create loaders
    train_loader, val_loader, test_loader = create_dataloader(dataset, config)

    # Validate split sizes
    assert len(train_loader.dataset) == expected_splits[0]
    assert (val_loader is None) if (val_split == 0) else (len(val_loader.dataset) == expected_splits[1])
    assert (test_loader is None) if (test_split == 0) else (len(test_loader.dataset) == expected_splits[2])

    # Verify no data_loader loss
    total_split = sum([
        len(train_loader.dataset),
        len(val_loader.dataset) if val_loader else 0,
        len(test_loader.dataset) if test_loader else 0
    ])
    assert total_split == total_samples