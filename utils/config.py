import yaml
import pytest
from pathlib import Path
from typing import Dict, Any, Union, List
from collections import UserDict  # Better for inheritance than dict



def pytest_report_header(config):
    return f"platform {sys.platform} -- Python {sys.version.split()[0]}, pytest-{pytest.__version__}, pluggy-{pluggy.__version__}"

def pytest_configure(config):
    """Disable all path reporting"""
    # These are the magic configs that actually remove the paths
    config.option.cache_dir = ""
    config._metadata = {}  # Clears all metadata

class Config(UserDict):  # Changed from Dict to UserDict
    """Configuration class that loads from YAML and supports nested dict access"""

    def __init__(self, config_dict: Dict[str, Any] = None):
        """Initialize with optional config dictionary"""
        super().__init__()  # Properly initialize UserDict
        if config_dict is not None:
            for key, value in config_dict.items():
                self[key] = self._convert_value(value)

    def _convert_value(self, value: Any) -> Union['Config', List, Any]:
        """Recursively convert nested dictionaries to Config objects"""
        if isinstance(value, dict):
            return Config(value)
        elif isinstance(value, list):
            return [self._convert_value(v) if isinstance(v, dict) else v for v in value]
        return value

    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file {path} not found")

        with open(path) as f:
            config_dict = yaml.safe_load(f) or {}  # Handle empty files

        return cls(config_dict)

    def __repr__(self) -> str:
        return f"Config({super().__repr__()})"