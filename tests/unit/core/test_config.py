"""
tests/unit/core/test_config.py
================================
Unit tests for pathoai.core.config.

Tests cover:
- ConfigNode: attribute access, nested access, unknown key error, get(), to_dict()
- _deep_merge: scalar override, nested override, None skip, list replacement
- _load_yaml: valid file, missing file, invalid YAML, empty YAML
- _apply_env_overrides: single key, nested key (double underscore), type coercion
- ConfigManager: initialize, get_instance, get_raw_config, get_config_hash, reset
- load_config / get_config: convenience wrappers
- Singleton behaviour: second initialize replaces first

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1.2
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import pytest

from pathoai.core.config import (
    ConfigManager,
    ConfigNode,
    _apply_env_overrides,
    _deep_merge,
    _load_yaml,
    get_config,
    load_config,
)
from pathoai.core.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_CONFIG: Path = Path(__file__).parents[2] / "fixtures" / "configs" / "test_base.yaml"


@pytest.fixture(autouse=True)
def reset_config_manager():
    """Ensure ConfigManager is clean before and after every test."""
    ConfigManager.reset()
    yield
    ConfigManager.reset()


@pytest.fixture()
def minimal_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid YAML config and return its path."""
    cfg = tmp_path / "minimal.yaml"
    cfg.write_text("pipeline:\n  seed: 99\n  device: cpu\n", encoding="utf-8")
    return cfg


@pytest.fixture()
def override_yaml(tmp_path: Path) -> Path:
    """Write a YAML override file and return its path."""
    cfg = tmp_path / "override.yaml"
    cfg.write_text("pipeline:\n  seed: 777\n", encoding="utf-8")
    return cfg


# ---------------------------------------------------------------------------
# ConfigNode
# ---------------------------------------------------------------------------

class TestConfigNode:
    """Tests for the ConfigNode attribute-access wrapper."""

    def test_scalar_attribute_access(self):
        """Top-level scalar values must be accessible as attributes."""
        node = ConfigNode({"name": "pathoai", "version": "0.1.0"})
        assert node.name == "pathoai"
        assert node.version == "0.1.0"

    def test_nested_attribute_access(self):
        """Nested dicts must be wrapped recursively for dot-access."""
        node = ConfigNode({"wsi": {"patch_size": 512, "stride": 256}})
        assert node.wsi.patch_size == 512
        assert node.wsi.stride == 256

    def test_deeply_nested_access(self):
        """Three-level nesting must work correctly."""
        node = ConfigNode({"a": {"b": {"c": 42}}})
        assert node.a.b.c == 42

    def test_unknown_key_raises_configuration_error(self):
        """Accessing a missing key must raise ConfigurationError."""
        node = ConfigNode({"x": 1})
        with pytest.raises(ConfigurationError, match="not found"):
            _ = node.missing_key

    def test_nested_unknown_key_raises_configuration_error(self):
        """Missing key in nested node must raise ConfigurationError."""
        node = ConfigNode({"wsi": {"patch_size": 512}})
        with pytest.raises(ConfigurationError):
            _ = node.wsi.unknown_param

    def test_get_returns_value_when_present(self):
        """node.get(key) must return the value when key exists."""
        node = ConfigNode({"seed": 42})
        assert node.get("seed") == 42

    def test_get_returns_default_when_absent(self):
        """node.get(key, default) must return default when key is missing."""
        node = ConfigNode({"seed": 42})
        assert node.get("missing", "fallback") == "fallback"
        assert node.get("missing") is None

    def test_to_dict_returns_original_data(self):
        """to_dict() must return the original nested dict."""
        data: Dict[str, Any] = {"pipeline": {"seed": 42, "device": "cpu"}}
        node = ConfigNode(data)
        result = node.to_dict()
        assert result == data
        assert isinstance(result, dict)

    def test_list_value_accessible_as_attribute(self):
        """List values must be returned directly (not wrapped)."""
        node = ConfigNode({"formats": [".svs", ".ndpi"]})
        assert node.formats == [".svs", ".ndpi"]

    def test_none_value_accessible(self):
        """None values must be accessible as None (not raise)."""
        node = ConfigNode({"checkpoint": None})
        assert node.checkpoint is None

    def test_bool_value_preserved(self):
        """Boolean values must be preserved as bool, not converted."""
        node = ConfigNode({"mixed_precision": False})
        assert node.mixed_precision is False
        assert isinstance(node.mixed_precision, bool)

    def test_repr_contains_data(self):
        """__repr__ must include config data for debugging."""
        node = ConfigNode({"seed": 42})
        assert "42" in repr(node)


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------

class TestDeepMerge:
    """Tests for the _deep_merge utility function."""

    def test_override_takes_priority_over_base(self):
        """Override values must replace base values."""
        base = {"seed": 42, "device": "cuda"}
        override = {"seed": 99}
        result = _deep_merge(base, override)
        assert result["seed"] == 99
        assert result["device"] == "cuda"  # preserved from base

    def test_nested_merge_preserves_unoverridden_keys(self):
        """Nested override must not wipe sibling keys not in override."""
        base = {"pipeline": {"seed": 42, "device": "cuda", "num_workers": 4}}
        override = {"pipeline": {"seed": 99}}
        result = _deep_merge(base, override)
        assert result["pipeline"]["seed"] == 99
        assert result["pipeline"]["device"] == "cuda"
        assert result["pipeline"]["num_workers"] == 4

    def test_none_in_override_skips_key(self):
        """None override values must leave the base value unchanged."""
        base = {"checkpoint": "/path/to/model.pth"}
        override = {"checkpoint": None}
        result = _deep_merge(base, override)
        assert result["checkpoint"] == "/path/to/model.pth"

    def test_list_replaced_not_merged(self):
        """Lists in override replace the base list entirely."""
        base = {"formats": [".svs", ".ndpi"]}
        override = {"formats": [".tif"]}
        result = _deep_merge(base, override)
        assert result["formats"] == [".tif"]

    def test_new_key_in_override_is_added(self):
        """Keys present only in override must be added to the result."""
        base = {"seed": 42}
        override = {"new_param": "hello"}
        result = _deep_merge(base, override)
        assert result["new_param"] == "hello"

    def test_base_is_not_mutated(self):
        """_deep_merge must not mutate the base dict."""
        base = {"seed": 42}
        original_base = base.copy()
        _deep_merge(base, {"seed": 99})
        assert base == original_base

    def test_override_is_not_mutated(self):
        """_deep_merge must not mutate the override dict."""
        override = {"seed": 99}
        original_override = override.copy()
        _deep_merge({"seed": 42}, override)
        assert override == original_override

    def test_deeply_nested_merge(self):
        """Three-level nesting must be merged correctly."""
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99}}}
        result = _deep_merge(base, override)
        assert result["a"]["b"]["c"] == 99
        assert result["a"]["b"]["d"] == 2


# ---------------------------------------------------------------------------
# _load_yaml
# ---------------------------------------------------------------------------

class TestLoadYaml:
    """Tests for the internal _load_yaml loader."""

    def test_loads_valid_yaml(self, tmp_path: Path):
        """Must parse valid YAML and return a dict."""
        f = tmp_path / "cfg.yaml"
        f.write_text("seed: 42\ndevice: cpu\n", encoding="utf-8")
        result = _load_yaml(f)
        assert result == {"seed": 42, "device": "cpu"}

    def test_empty_yaml_returns_empty_dict(self, tmp_path: Path):
        """Empty YAML file must return an empty dict, not None."""
        f = tmp_path / "empty.yaml"
        f.write_text("", encoding="utf-8")
        result = _load_yaml(f)
        assert result == {}

    def test_raises_on_missing_file(self, tmp_path: Path):
        """Must raise ConfigurationError for a non-existent file."""
        with pytest.raises(ConfigurationError, match="not found"):
            _load_yaml(tmp_path / "ghost.yaml")

    def test_raises_on_invalid_yaml(self, tmp_path: Path):
        """Must raise ConfigurationError for syntactically invalid YAML."""
        f = tmp_path / "bad.yaml"
        f.write_text("key: :\n  bad:\n  indent:", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            _load_yaml(f)

    def test_raises_when_yaml_is_not_a_dict(self, tmp_path: Path):
        """Must raise ConfigurationError when YAML root is a list, not a dict."""
        f = tmp_path / "list.yaml"
        f.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="must contain a YAML mapping"):
            _load_yaml(f)

    def test_nested_yaml_parsed_correctly(self, tmp_path: Path):
        """Nested YAML must be parsed into nested dicts."""
        f = tmp_path / "nested.yaml"
        f.write_text("pipeline:\n  seed: 42\n  device: cpu\n", encoding="utf-8")
        result = _load_yaml(f)
        assert result["pipeline"]["seed"] == 42


# ---------------------------------------------------------------------------
# _apply_env_overrides
# ---------------------------------------------------------------------------

class TestApplyEnvOverrides:
    """Tests for environment variable override application."""

    def test_applies_simple_override(self, monkeypatch):
        """A single PATHOAI_KEY=val must set config['key'] = val."""
        monkeypatch.setenv("PATHOAI_DEVICE", "cpu")
        config = {"device": "cuda"}
        result = _apply_env_overrides(config)
        assert result["device"] == "cpu"

    def test_applies_nested_override_with_double_underscore(self, monkeypatch):
        """PATHOAI_PIPELINE__SEED=99 must set config['pipeline']['seed'] = 99."""
        monkeypatch.setenv("PATHOAI_PIPELINE__SEED", "99")
        config = {"pipeline": {"seed": 42, "device": "cuda"}}
        result = _apply_env_overrides(config)
        assert result["pipeline"]["seed"] == 99
        assert result["pipeline"]["device"] == "cuda"  # untouched

    def test_coerces_bool_true(self, monkeypatch):
        """'true' must be coerced to Python True."""
        monkeypatch.setenv("PATHOAI_MIXED_PRECISION", "true")
        config = {"mixed_precision": False}
        result = _apply_env_overrides(config)
        assert result["mixed_precision"] is True

    def test_coerces_bool_false(self, monkeypatch):
        """'false' must be coerced to Python False."""
        monkeypatch.setenv("PATHOAI_MIXED_PRECISION", "false")
        config = {"mixed_precision": True}
        result = _apply_env_overrides(config)
        assert result["mixed_precision"] is False

    def test_coerces_int(self, monkeypatch):
        """An integer string must be coerced to int."""
        monkeypatch.setenv("PATHOAI_SEED", "123")
        config = {"seed": 42}
        result = _apply_env_overrides(config)
        assert result["seed"] == 123
        assert isinstance(result["seed"], int)

    def test_coerces_float(self, monkeypatch):
        """A float string must be coerced to float."""
        monkeypatch.setenv("PATHOAI_LR", "0.001")
        config = {"lr": 0.0001}
        result = _apply_env_overrides(config)
        assert abs(result["lr"] - 0.001) < 1e-9

    def test_string_value_preserved_when_no_coercion(self, monkeypatch):
        """Non-numeric strings must remain as strings."""
        monkeypatch.setenv("PATHOAI_MODEL_NAME", "deeplabv3plus")
        config = {"model_name": "resnet"}
        result = _apply_env_overrides(config)
        assert result["model_name"] == "deeplabv3plus"

    def test_ignores_non_prefixed_env_vars(self, monkeypatch):
        """Environment variables without PATHOAI_ prefix must be ignored."""
        monkeypatch.setenv("SOME_OTHER_VAR", "value")
        config = {"seed": 42}
        result = _apply_env_overrides(config)
        assert result == {"seed": 42}

    def test_does_not_mutate_input_config(self, monkeypatch):
        """_apply_env_overrides must not mutate the input dict."""
        monkeypatch.setenv("PATHOAI_SEED", "99")
        config = {"seed": 42}
        original = config.copy()
        _apply_env_overrides(config)
        assert config == original


# ---------------------------------------------------------------------------
# ConfigManager
# ---------------------------------------------------------------------------

class TestConfigManager:
    """Tests for the ConfigManager singleton."""

    def test_initialize_loads_config(self):
        """ConfigManager.initialize must succeed with a valid config file."""
        ConfigManager.initialize(base_config=TEST_CONFIG, apply_env_vars=False)
        cfg = ConfigManager.get_instance()
        assert cfg is not None

    def test_get_instance_raises_before_initialize(self):
        """get_instance must raise ConfigurationError if not yet initialized."""
        with pytest.raises(ConfigurationError, match="not been initialized"):
            ConfigManager.get_instance()

    def test_config_values_are_accessible(self):
        """Config values from the YAML file must be accessible."""
        ConfigManager.initialize(base_config=TEST_CONFIG, apply_env_vars=False)
        cfg = ConfigManager.get_instance()
        assert cfg.pipeline.seed == 42
        assert cfg.pipeline.device == "cpu"

    def test_nested_config_values_accessible(self):
        """Nested config values must be accessible via dot notation."""
        ConfigManager.initialize(base_config=TEST_CONFIG, apply_env_vars=False)
        cfg = ConfigManager.get_instance()
        assert cfg.wsi.patch_extraction.patch_size == 64

    def test_get_raw_config_returns_dict(self):
        """get_raw_config must return a plain dict."""
        ConfigManager.initialize(base_config=TEST_CONFIG, apply_env_vars=False)
        raw = ConfigManager.get_raw_config()
        assert isinstance(raw, dict)
        assert "pipeline" in raw

    def test_get_raw_config_returns_copy(self):
        """get_raw_config must return a copy, not a reference."""
        ConfigManager.initialize(base_config=TEST_CONFIG, apply_env_vars=False)
        raw1 = ConfigManager.get_raw_config()
        raw1["pipeline"]["seed"] = 9999
        raw2 = ConfigManager.get_raw_config()
        assert raw2["pipeline"]["seed"] == 42  # unchanged

    def test_get_config_hash_is_hex_string(self):
        """Config hash must be a non-empty hex string of SHA-256 length."""
        ConfigManager.initialize(base_config=TEST_CONFIG, apply_env_vars=False)
        h = ConfigManager.get_config_hash()
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 = 32 bytes = 64 hex chars
        assert all(c in "0123456789abcdef" for c in h)

    def test_config_hash_is_deterministic(self):
        """Same config file must always produce the same hash."""
        ConfigManager.initialize(base_config=TEST_CONFIG, apply_env_vars=False)
        h1 = ConfigManager.get_config_hash()
        ConfigManager.reset()
        ConfigManager.initialize(base_config=TEST_CONFIG, apply_env_vars=False)
        h2 = ConfigManager.get_config_hash()
        assert h1 == h2

    def test_config_hash_before_initialize_is_uninitialized(self):
        """get_config_hash must return 'uninitialized' before initialization."""
        assert ConfigManager.get_config_hash() == "uninitialized"

    def test_reset_clears_singleton(self):
        """After reset, get_instance must raise ConfigurationError."""
        ConfigManager.initialize(base_config=TEST_CONFIG, apply_env_vars=False)
        ConfigManager.reset()
        with pytest.raises(ConfigurationError):
            ConfigManager.get_instance()

    def test_second_initialize_replaces_first(self, tmp_path: Path):
        """Calling initialize twice must replace the singleton with new config."""
        f = tmp_path / "second.yaml"
        f.write_text("pipeline:\n  seed: 777\n  device: cpu\n", encoding="utf-8")

        ConfigManager.initialize(base_config=TEST_CONFIG, apply_env_vars=False)
        cfg1_seed = ConfigManager.get_instance().pipeline.seed

        ConfigManager.initialize(base_config=f, apply_env_vars=False)
        cfg2_seed = ConfigManager.get_instance().pipeline.seed

        assert cfg1_seed == 42
        assert cfg2_seed == 777

    def test_override_file_takes_priority(self, tmp_path: Path):
        """An override YAML must take priority over the base config."""
        override = tmp_path / "override.yaml"
        override.write_text("pipeline:\n  seed: 1234\n", encoding="utf-8")

        ConfigManager.initialize(
            base_config=TEST_CONFIG,
            overrides=[override],
            apply_env_vars=False,
        )
        cfg = ConfigManager.get_instance()
        assert cfg.pipeline.seed == 1234

    def test_multiple_overrides_applied_in_order(self, tmp_path: Path):
        """Later overrides must take priority over earlier ones."""
        o1 = tmp_path / "o1.yaml"
        o2 = tmp_path / "o2.yaml"
        o1.write_text("pipeline:\n  seed: 100\n", encoding="utf-8")
        o2.write_text("pipeline:\n  seed: 200\n", encoding="utf-8")

        ConfigManager.initialize(
            base_config=TEST_CONFIG,
            overrides=[o1, o2],
            apply_env_vars=False,
        )
        assert ConfigManager.get_instance().pipeline.seed == 200

    def test_get_raw_config_raises_before_initialize(self):
        """get_raw_config must raise ConfigurationError before initialization."""
        with pytest.raises(ConfigurationError):
            ConfigManager.get_raw_config()


# ---------------------------------------------------------------------------
# Public API: load_config / get_config
# ---------------------------------------------------------------------------

class TestPublicAPI:
    """Tests for the module-level convenience functions."""

    def test_load_config_initializes_and_returns_node(self):
        """load_config must initialize and return a ConfigNode."""
        cfg = load_config(base_config=TEST_CONFIG)
        assert cfg is not None
        assert cfg.pipeline.seed == 42

    def test_get_config_returns_same_instance_as_initialize(self):
        """get_config must return the same node as ConfigManager.get_instance."""
        ConfigManager.initialize(base_config=TEST_CONFIG, apply_env_vars=False)
        cfg_via_manager = ConfigManager.get_instance()
        cfg_via_get = get_config()
        assert cfg_via_manager is cfg_via_get

    def test_get_config_raises_without_initialize(self):
        """get_config must raise ConfigurationError if not yet initialized."""
        with pytest.raises(ConfigurationError):
            get_config()
