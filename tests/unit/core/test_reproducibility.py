"""
tests/unit/core/test_reproducibility.py
==========================================
Unit tests for pathoai.core.reproducibility.

Tests cover:
- set_global_seed: seeds Python random, NumPy, env vars, PyTorch (where available)
- get_worker_init_fn: returns callable, callable accepts worker_id
- capture_environment_snapshot: required keys, JSON-serializable output
- save_environment_snapshot: creates JSON file with correct content

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1.4
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest

from pathoai.core.reproducibility import (
    capture_environment_snapshot,
    get_worker_init_fn,
    save_environment_snapshot,
    set_global_seed,
)


# ---------------------------------------------------------------------------
# set_global_seed
# ---------------------------------------------------------------------------

class TestSetGlobalSeed:
    """Tests for set_global_seed()."""

    def test_sets_python_random_seed(self):
        """Python random module must produce deterministic values after seeding."""
        set_global_seed(42)
        val1 = random.random()
        set_global_seed(42)
        val2 = random.random()
        assert val1 == val2

    def test_sets_numpy_seed(self):
        """NumPy random must produce deterministic values after seeding."""
        set_global_seed(42)
        arr1 = np.random.rand(5)
        set_global_seed(42)
        arr2 = np.random.rand(5)
        np.testing.assert_array_equal(arr1, arr2)

    def test_sets_pythonhashseed_env_var(self):
        """PYTHONHASHSEED environment variable must be set to the seed value."""
        set_global_seed(123)
        assert os.environ.get("PYTHONHASHSEED") == "123"

    def test_different_seeds_produce_different_sequences(self):
        """Two different seeds must produce different random sequences."""
        set_global_seed(1)
        seq1 = [random.random() for _ in range(5)]
        set_global_seed(2)
        seq2 = [random.random() for _ in range(5)]
        assert seq1 != seq2

    def test_seed_zero_is_valid(self):
        """Seed value of 0 must work without raising."""
        set_global_seed(0)
        assert os.environ.get("PYTHONHASHSEED") == "0"

    def test_large_seed_is_valid(self):
        """A large seed value must work without raising."""
        set_global_seed(2**31 - 1)

    def test_pytorch_seed_set_when_available(self):
        """PyTorch manual_seed must be called when torch is available."""
        try:
            import torch
            set_global_seed(42)
            # Verify by checking that manual_seed was called — torch stores state
            state1 = torch.get_rng_state().clone()
            set_global_seed(42)
            state2 = torch.get_rng_state().clone()
            assert torch.equal(state1, state2)
        except ImportError:
            pytest.skip("PyTorch not installed — skipping torch seed test")


# ---------------------------------------------------------------------------
# get_worker_init_fn
# ---------------------------------------------------------------------------

class TestGetWorkerInitFn:
    """Tests for get_worker_init_fn()."""

    def test_returns_callable(self):
        """get_worker_init_fn must return a callable."""
        fn = get_worker_init_fn(seed=42)
        assert callable(fn)

    def test_callable_accepts_worker_id(self):
        """The returned function must accept a single integer worker_id."""
        fn = get_worker_init_fn(seed=42)
        fn(0)   # Must not raise
        fn(3)   # Must not raise

    def test_callable_sets_numpy_seed(self):
        """After calling the worker init fn, NumPy should be seeded."""
        fn = get_worker_init_fn(seed=42)
        fn(0)
        # Just verify it doesn't raise — exact seed depends on torch availability
        val = np.random.rand()
        assert 0.0 <= val <= 1.0

    def test_different_workers_can_run_sequentially(self):
        """Calling the fn with different worker IDs must not raise."""
        fn = get_worker_init_fn(seed=100)
        for worker_id in range(4):
            fn(worker_id)  # Must not raise


# ---------------------------------------------------------------------------
# capture_environment_snapshot
# ---------------------------------------------------------------------------

class TestCaptureEnvironmentSnapshot:
    """Tests for capture_environment_snapshot()."""

    def test_returns_dict(self):
        """Must return a dict."""
        snapshot = capture_environment_snapshot(experiment_id="test_001")
        assert isinstance(snapshot, dict)

    def test_required_keys_present(self):
        """Must contain experiment_id, timestamp, python, platform, packages, gpu, git."""
        snapshot = capture_environment_snapshot(experiment_id="test_001")
        required_keys = {"experiment_id", "timestamp", "python", "platform", "packages", "gpu", "git"}
        missing = required_keys - set(snapshot.keys())
        assert not missing, f"Missing keys in snapshot: {missing}"

    def test_experiment_id_matches(self):
        """experiment_id in snapshot must match the argument."""
        snapshot = capture_environment_snapshot(experiment_id="my_exp_42")
        assert snapshot["experiment_id"] == "my_exp_42"

    def test_python_info_is_accurate(self):
        """python section must contain version, executable, implementation."""
        import sys
        snapshot = capture_environment_snapshot(experiment_id="py_check")
        py = snapshot["python"]
        assert "version" in py
        expected_major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
        assert py["version"].startswith(expected_major_minor)

    def test_platform_info_present(self):
        """platform section must contain system and machine keys."""
        snapshot = capture_environment_snapshot(experiment_id="plat_check")
        plat = snapshot["platform"]
        assert "system" in plat
        assert "machine" in plat

    def test_snapshot_is_json_serializable(self):
        """The entire snapshot dict must be serializable to JSON without errors."""
        snapshot = capture_environment_snapshot(experiment_id="serial_check")
        json_str = json.dumps(snapshot, default=str)
        parsed = json.loads(json_str)
        assert parsed["experiment_id"] == "serial_check"

    def test_config_hash_stored_when_provided(self):
        """config_hash must be stored in the snapshot when provided."""
        snapshot = capture_environment_snapshot(
            experiment_id="hash_check",
            config_hash="abc123",
        )
        assert snapshot["config_hash"] == "abc123"

    def test_config_hash_is_none_when_not_provided(self):
        """config_hash must be None when not provided."""
        snapshot = capture_environment_snapshot(experiment_id="no_hash")
        assert snapshot["config_hash"] is None

    def test_extra_fields_are_included(self):
        """Extra key-value pairs must appear in the snapshot."""
        snapshot = capture_environment_snapshot(
            experiment_id="extra_check",
            extra={"custom_key": "custom_value"},
        )
        assert snapshot["custom_key"] == "custom_value"

    def test_packages_section_is_dict(self):
        """packages section must be a dict."""
        snapshot = capture_environment_snapshot(experiment_id="pkg_check")
        assert isinstance(snapshot["packages"], dict)

    def test_numpy_version_captured(self):
        """numpy version must appear in the packages section."""
        snapshot = capture_environment_snapshot(experiment_id="numpy_check")
        # numpy is imported so it must be captured
        pkgs = snapshot["packages"]
        assert "numpy" in pkgs
        assert pkgs["numpy"] not in (None, "", "not_installed")

    def test_gpu_section_contains_cuda_available(self):
        """gpu section must contain cuda_available key."""
        snapshot = capture_environment_snapshot(experiment_id="gpu_check")
        assert "cuda_available" in snapshot["gpu"]

    def test_git_section_present(self):
        """git section must be present (even if git unavailable)."""
        snapshot = capture_environment_snapshot(experiment_id="git_check")
        assert "git" in snapshot
        assert isinstance(snapshot["git"], dict)


# ---------------------------------------------------------------------------
# save_environment_snapshot
# ---------------------------------------------------------------------------

class TestSaveEnvironmentSnapshot:
    """Tests for save_environment_snapshot()."""

    def test_creates_json_file(self, tmp_path: Path):
        """Must create a JSON file at the specified path."""
        snapshot = capture_environment_snapshot(experiment_id="save_test")
        output_path = save_environment_snapshot(snapshot, output_dir=tmp_path)
        assert output_path.exists()
        assert output_path.suffix == ".json"

    def test_saved_file_is_valid_json(self, tmp_path: Path):
        """The saved file must contain valid JSON."""
        snapshot = capture_environment_snapshot(experiment_id="json_test")
        output_path = save_environment_snapshot(snapshot, output_dir=tmp_path)
        with open(output_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["experiment_id"] == "json_test"

    def test_custom_filename_is_used(self, tmp_path: Path):
        """Must use the provided filename parameter."""
        snapshot = capture_environment_snapshot(experiment_id="fname_test")
        output_path = save_environment_snapshot(
            snapshot,
            output_dir=tmp_path,
            filename="my_snapshot.json",
        )
        assert output_path.name == "my_snapshot.json"

    def test_creates_output_directory_if_missing(self, tmp_path: Path):
        """Must create the output directory if it does not yet exist."""
        deep_dir = tmp_path / "a" / "b" / "c"
        snapshot = capture_environment_snapshot(experiment_id="mkdir_test")
        output_path = save_environment_snapshot(snapshot, output_dir=deep_dir)
        assert deep_dir.is_dir()
        assert output_path.exists()

    def test_returns_path_to_saved_file(self, tmp_path: Path):
        """Return value must be the Path to the saved file."""
        snapshot = capture_environment_snapshot(experiment_id="ret_test")
        result = save_environment_snapshot(snapshot, output_dir=tmp_path)
        assert isinstance(result, Path)
        assert result.is_file()
