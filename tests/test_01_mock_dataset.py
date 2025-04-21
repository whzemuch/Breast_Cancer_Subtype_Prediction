import pytest
from torch.utils.data import DataLoader
from tests.mock import MockMultiOmicsDataset
from data_loader import create_dataloaders


def test_mock_shapes():
    dataset = MockMultiOmicsDataset()
    sample, label = dataset[0]

    assert sample["methyl"].shape == (1, 50, 100)  # ConvNeXt-ready
    assert sample["mirna"].shape == (5000,)
    assert sample["rna_exp"].shape == (5000,)
    assert label.shape == ()


def test_mock_dataloader():
    dataset = MockMultiOmicsDataset()
    loader = DataLoader(dataset, batch_size=8)

    batch = next(iter(loader))
    inputs, labels = batch

    assert inputs["methyl"].shape == (8, 1, 50, 100)
    assert inputs["mirna"].shape == (8, 5000)
    assert labels.shape == (8,)


def test_mock_splits():

    dataset = MockMultiOmicsDataset(100)
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, {"val_split": 0.2, "test_split": 0.1}
    )

    assert len(train_loader.dataset) == 70
    assert len(val_loader.dataset) == 20
    assert len(test_loader.dataset) == 10