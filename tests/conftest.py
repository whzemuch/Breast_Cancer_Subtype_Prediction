import pytest
from tests.mock import MockMultiOmicsDataset

@pytest.fixture
def dataset_and_config():
    """Fixture providing mock dataset and base config"""
    # Create mock dataset
    dataset = MockMultiOmicsDataset(num_samples=100)

    # Base config (modify splits in tests)
    config = {
        "batch_size": 32,
        "val_split": 0.2,  # Default, can be overridden
        "test_split": 0.1,  # Default, can be overridden
        "seed": 42
    }

    return dataset, config