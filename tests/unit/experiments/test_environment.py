"""
tests/unit/experiments/test_environment.py
============================================
Unit tests for EnvironmentAuditor.

Author: PathoAI Research Team
Created: 2026-07-20
"""

from pathoai.experiments.environment import EnvironmentAuditor


class TestEnvironmentAuditor:
    """Test EnvironmentAuditor."""

    def test_capture_environment(self):
        """Test environment audit keys."""
        auditor = EnvironmentAuditor()
        env = auditor.capture_environment()

        assert "git_commit" in env
        assert "python_version" in env
        assert "pytorch_version" in env
        assert "cuda_version" in env
        assert "gpu_name" in env
