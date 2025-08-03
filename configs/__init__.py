"""
Configuration module for MedEyes
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    import copy
    merged = copy.deepcopy(base_config)

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def get_dataset_config(dataset_name: str, config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get dataset-specific configuration

    Args:
        dataset_name: Name of dataset
        config_dir: Directory containing configs

    Returns:
        Dataset configuration
    """
    if config_dir is None:
        config_dir = Path(__file__).parent

    # Load base config
    base_config_path = config_dir / "default.yaml"
    base_config = load_config(base_config_path)

    # Load dataset config
    dataset_config_path = config_dir / "datasets" / f"{dataset_name}.yaml"
    if dataset_config_path.exists():
        dataset_config = load_config(dataset_config_path)
        return merge_configs(base_config, dataset_config)
    else:
        return base_config


__all__ = ['load_config', 'merge_configs', 'get_dataset_config']