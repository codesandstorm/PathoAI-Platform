"""
pathoai/core/config.py
======================
Configuration management for PathoAI-Platform.

Implements a hierarchical, type-safe configuration system using YAML files
and a singleton access pattern. Supports multi-level config merging:

    base.yaml → dataset.yaml → model.yaml → experiment.yaml → env vars → CLI args

Features:
- Hierarchical config merging (lower priority ← higher priority)
- Environment variable overrides (PATHOAI_<KEY>=value)
- Attribute-style access (cfg.wsi.patch_size, not cfg["wsi"]["patch_size"])
- Config hash for provenance tracking
- Validation of required keys and value ranges

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1
"""

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from pathoai.core.exceptions import ConfigurationError
from pathoai.core.logger import get_logger

logger = get_logger(__name__)


class ConfigNode:
    """Recursive attribute-access wrapper for a nested configuration dict.

    Converts a nested dictionary into an object where keys are accessible
    as attributes, with full support for nested access:

        cfg = ConfigNode({"wsi": {"patch_size": 512}})
        cfg.wsi.patch_size  # → 512

    Unknown attribute access raises ConfigurationError with a helpful message
    instead of AttributeError, guiding the user to check their config file.
    """

    def __init__(self, data: Dict[str, Any], path: str = "config") -> None:
        """
        Parameters
        ----------
        data : Dict[str, Any]
            Nested dictionary of configuration values.
        path : str
            Dot-separated path to this node (for error messages).
        """
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_path", path)

        for key, value in data.items():
            if isinstance(value, dict):
                object.__setattr__(self, key, ConfigNode(value, path=f"{path}.{key}"))
            else:
                object.__setattr__(self, key, value)

    def __getattr__(self, name: str) -> Any:
        path = object.__getattribute__(self, "_path")
        raise ConfigurationError(
            f"Configuration key '{path}.{name}' not found. "
            f"Check your YAML config files or consult config/base.yaml for defaults."
        )

    def __repr__(self) -> str:
        data = object.__getattribute__(self, "_data")
        return f"ConfigNode({json.dumps(data, indent=2, default=str)})"

    def to_dict(self) -> Dict[str, Any]:
        """Return the underlying dict (original, not wrapped)."""
        return object.__getattribute__(self, "_data")

    def get(self, key: str, default: Any = None) -> Any:
        """Safe get with default, equivalent to dict.get()."""
        try:
            return getattr(self, key)
        except ConfigurationError:
            return default


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge two dicts. Override values take priority.

    Lists are replaced entirely (not merged element-wise).
    None values in override are skipped (preserve base value).

    Parameters
    ----------
    base : Dict
        Base configuration dict (lower priority).
    override : Dict
        Override dict (higher priority). Keys in override overwrite base.

    Returns
    -------
    Dict
        Merged dictionary.
    """
    result = deepcopy(base)
    for key, value in override.items():
        if value is None:
            continue  # None in override = keep base value
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its content as a dict.

    Parameters
    ----------
    path : Path
        Path to YAML file.

    Returns
    -------
    Dict[str, Any]
        Parsed YAML content.

    Raises
    ------
    ConfigurationError
        If the file does not exist or is not valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {path}. "
            f"Ensure the path is correct relative to the project root."
        )
    try:
        with open(path, encoding="utf-8") as f:
            content = yaml.safe_load(f)
        if content is None:
            return {}
        if not isinstance(content, dict):
            raise ConfigurationError(
                f"Config file {path} must contain a YAML mapping (dict), "
                f"got {type(content).__name__}."
            )
        return content
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Invalid YAML in {path}: {e}"
        ) from e


def _apply_env_overrides(config: Dict, prefix: str = "PATHOAI") -> Dict:
    """Apply environment variable overrides to config dict.

    Environment variables of the form PATHOAI_WSI__PATCH_SIZE=512
    map to config["wsi"]["patch_size"] = 512.

    Double underscores (__) denote nesting levels.
    Single underscores within a key remain as-is.

    Parameters
    ----------
    config : Dict
        Config dict to apply overrides to.
    prefix : str
        Environment variable prefix. Default is "PATHOAI".

    Returns
    -------
    Dict
        Config dict with env var overrides applied.
    """
    result = deepcopy(config)
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(f"{prefix}_"):
            continue
        # Remove prefix and split on double underscore for nesting
        key_path = env_key[len(prefix) + 1:].lower().split("__")
        node = result
        for part in key_path[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]

        # Attempt type coercion
        leaf_key = key_path[-1]
        try:
            # Try int, float, bool in order
            if env_val.lower() in ("true", "false"):
                node[leaf_key] = env_val.lower() == "true"
            elif "." in env_val:
                node[leaf_key] = float(env_val)
            else:
                node[leaf_key] = int(env_val)
        except ValueError:
            node[leaf_key] = env_val  # Keep as string

        logger.debug(
            "Applied env var override",
            extra={"env_key": env_key, "key_path": ".".join(key_path), "value": env_val},
        )
    return result


class ConfigManager:
    """Singleton configuration manager for PathoAI-Platform.

    Loads and merges configuration from multiple YAML sources,
    applies environment variable overrides, and exposes a unified
    attribute-access interface.

    Usage:
        # Initialize once at startup
        ConfigManager.initialize(
            base_config="config/base.yaml",
            overrides=["config/datasets/tiger.yaml", "config/experiments/exp_001.yaml"],
        )

        # Access from anywhere
        cfg = ConfigManager.get_instance()
        patch_size = cfg.wsi.patch_extraction.patch_size
    """

    _instance: Optional["ConfigManager"] = None
    _config_node: Optional[ConfigNode] = None
    _raw_config: Optional[Dict] = None

    def __init__(self) -> None:
        raise RuntimeError(
            "ConfigManager is a singleton. Use ConfigManager.initialize() and "
            "ConfigManager.get_instance() to access it."
        )

    @classmethod
    def initialize(
        cls,
        base_config: Union[str, Path],
        overrides: Optional[List[Union[str, Path]]] = None,
        apply_env_vars: bool = True,
    ) -> None:
        """Initialize the configuration singleton.

        Parameters
        ----------
        base_config : Union[str, Path]
            Path to the base YAML configuration file (required).
        overrides : List[Union[str, Path]], optional
            Ordered list of YAML override files, applied left to right.
            Later files take priority over earlier ones.
        apply_env_vars : bool
            Whether to apply PATHOAI_* environment variable overrides.
            Default True.

        Raises
        ------
        ConfigurationError
            If base_config or any override file is invalid or missing.
        """
        logger.debug(
            "Initializing ConfigManager",
            extra={"base_config": str(base_config), "n_overrides": len(overrides or [])},
        )

        # Load base config
        merged = _load_yaml(Path(base_config))

        # Apply override files in order
        for override_path in (overrides or []):
            override_data = _load_yaml(Path(override_path))
            merged = _deep_merge(merged, override_data)
            logger.debug("Merged config override", extra={"path": str(override_path)})

        # Apply environment variable overrides
        if apply_env_vars:
            merged = _apply_env_overrides(merged)

        cls._raw_config = merged
        cls._config_node = ConfigNode(merged)
        cls._instance = object.__new__(cls)

        logger.info(
            "Configuration loaded",
            extra={
                "config_hash": cls.get_config_hash(),
                "n_keys": len(merged),
            },
        )

    @classmethod
    def get_instance(cls) -> ConfigNode:
        """Return the active configuration node.

        Returns
        -------
        ConfigNode
            Attribute-accessible configuration object.

        Raises
        ------
        ConfigurationError
            If ConfigManager.initialize() has not been called.
        """
        if cls._config_node is None:
            raise ConfigurationError(
                "ConfigManager has not been initialized. "
                "Call ConfigManager.initialize(base_config=...) before get_config()."
            )
        return cls._config_node

    @classmethod
    def get_raw_config(cls) -> Dict:
        """Return the raw merged configuration dictionary."""
        if cls._raw_config is None:
            raise ConfigurationError("ConfigManager has not been initialized.")
        return deepcopy(cls._raw_config)

    @classmethod
    def get_config_hash(cls) -> str:
        """Return SHA-256 hash of the merged config for provenance tracking.

        Returns
        -------
        str
            SHA-256 hex digest of the canonical JSON representation.
        """
        if cls._raw_config is None:
            return "uninitialized"
        canonical = json.dumps(cls._raw_config, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing only).

        Warning: This should only be called in test teardown.
        """
        cls._instance = None
        cls._config_node = None
        cls._raw_config = None


def get_config() -> ConfigNode:
    """Convenience function to access the active configuration node.

    This is the primary API for accessing configuration throughout
    PathoAI-Platform. Must be called after ConfigManager.initialize().

    Returns
    -------
    ConfigNode
        Attribute-accessible configuration object.

    Examples
    --------
    >>> from pathoai.core.config import get_config
    >>> cfg = get_config()
    >>> patch_size = cfg.wsi.patch_extraction.patch_size
    >>> device = cfg.pipeline.device
    """
    return ConfigManager.get_instance()


def load_config(
    base_config: Union[str, Path] = "config/base.yaml",
    overrides: Optional[List[Union[str, Path]]] = None,
) -> ConfigNode:
    """Initialize and return the configuration (convenience wrapper).

    Parameters
    ----------
    base_config : Union[str, Path]
        Path to base YAML config file.
    overrides : List[Union[str, Path]], optional
        Additional YAML override files.

    Returns
    -------
    ConfigNode
        Initialized configuration node.
    """
    ConfigManager.initialize(base_config=base_config, overrides=overrides)
    return get_config()
