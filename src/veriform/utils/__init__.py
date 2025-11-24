"""
Utility functions for Veriform.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict

from veriform.config import BenchmarkConfig


def load_config(config_path: str) -> BenchmarkConfig:
    """
    Load configuration from a YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        BenchmarkConfig instance
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif path.suffix == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    return BenchmarkConfig(**config_dict)


def save_config(config: BenchmarkConfig, output_path: str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: BenchmarkConfig to save
        output_path: Path to save the configuration
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = config.dict()

    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(config_dict, f, default_flow_style=False)
        elif path.suffix == '.json':
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {path.suffix}")

    print(f"Saved config to {output_path}")


def load_results(results_path: str) -> Dict[str, Any]:
    """
    Load benchmark results from a JSON file.

    Args:
        results_path: Path to results file

    Returns:
        Results dictionary
    """
    with open(results_path, 'r') as f:
        return json.load(f)


__all__ = [
    "load_config",
    "save_config",
    "load_results",
]
