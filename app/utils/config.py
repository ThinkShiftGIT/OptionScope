"""
Configuration management utilities for OptionScope.

This module provides functions to load and access the application configuration
from the config.yml file.
"""

import os
from typing import Any, Dict, Optional

import yaml


_config_cache: Optional[Dict[str, Any]] = None


def get_config() -> Dict[str, Any]:
    """
    Get the application configuration from config.yml.
    
    Returns:
        Dict containing the configuration
    """
    global _config_cache
    
    # Return cached config if available
    if _config_cache is not None:
        return _config_cache
    
    # Determine the project root directory
    # If the file is in app/utils/config.py, project root is 2 levels up
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    
    # Load the config file
    config_path = os.path.join(project_root, "config.yml")
    
    try:
        with open(config_path, "r") as f:
            _config_cache = yaml.safe_load(f)
    except FileNotFoundError:
        raise RuntimeError(
            f"Configuration file not found at {config_path}. "
            "Make sure you have created a config.yml file in the project root."
        )
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error parsing config.yml: {e}")
    
    return _config_cache


def get_config_value(path: str, default: Any = None) -> Any:
    """
    Get a specific configuration value using a dot-notation path.
    
    Args:
        path: Dot-notation path to the config value (e.g., 'data_providers.rates.risk_free_rate')
        default: Default value to return if the path is not found
        
    Returns:
        Configuration value or default if not found
    """
    config = get_config()
    keys = path.split(".")
    
    try:
        result = config
        for key in keys:
            result = result[key]
        return result
    except (KeyError, TypeError):
        return default
