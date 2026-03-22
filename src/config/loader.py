"""
Handles loading and merging of YAML configuration files,
supporting base config inheritance via the `_base_` key.
"""
import os
import yaml
from typing import Any, Dict
from src.config.schema import MainConfig, DataConfig, ModelConfig, TrainConfig


def _load_yaml(path: str) -> Dict[str, Any]:
    """Loads a single YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Recursively merges override into base.
    Override values take precedence over base values.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _config_to_dict(cfg: MainConfig) -> Dict[str, Any]:
    """
    Converts a MainConfig dataclass into a plain nested dictionary,
    ready for YAML serialisation.
    """
    return {
        "data": cfg.data.__dict__.copy(),
        "model": cfg.model.__dict__.copy(),
        "train": cfg.train.__dict__.copy(),
    }


def load_config(config_path: str) -> MainConfig:
    """
    Loads a YAML config file, resolving `_base_` inheritance if present.

    Args:
        config_path (str): Path to the experiment YAML file.

    Returns:
        MainConfig: A fully populated config dataclass.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = _load_yaml(config_path)

    # Resolve _base_ inheritance
    if "_base_" in raw:
        base_path = os.path.join(os.path.dirname(config_path), raw.pop("_base_"))
        base_raw = _load_yaml(base_path)
        raw = _deep_merge(base_raw, raw)

    # Populate dataclasses
    cfg = MainConfig()
    for group, params in raw.items():
        if hasattr(cfg, group) and isinstance(params, dict):
            config_group = getattr(cfg, group)
            for key, value in params.items():
                if hasattr(config_group, key):
                    setattr(config_group, key, value)
    return cfg


def save_flattened_config(cfg: MainConfig, run_dir: str) -> None:
    """
    Saves the fully resolved (flattened) config as a single self-contained
    YAML file in the run directory.
    No `_base_` reference — the file alone is sufficient to reproduce the run.

    Args:
        cfg (MainConfig): The fully populated config dataclass.
        run_dir (str): The run directory to save the config into.
    """
    output_path = os.path.join(run_dir, "config.yaml")
    with open(output_path, "w") as f:
        yaml.dump(_config_to_dict(cfg), f, default_flow_style=False, sort_keys=False)