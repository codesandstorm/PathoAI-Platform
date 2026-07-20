"""
pathoai/experiments/environment.py
===================================
Environment Auditor for Reproducibility.

Captures hardware and software environment parameters: Python version, PyTorch,
CUDA, OS, GPU model, and git commit hash.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.5.1
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any, Dict

import torch


class EnvironmentAuditor:
    """Audits runtime environment hardware and software parameters."""

    def capture_environment(self) -> Dict[str, Any]:
        """Captures runtime environment dictionary.

        Returns
        -------
        Dict[str, Any]
            Environment key-values.
        """
        git_hash = "unknown"
        try:
            cmd = ["git", "rev-parse", "--short", "HEAD"]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            git_hash = out.decode("utf-8").strip()
        except Exception:
            pass

        gpu_name = "N/A (CPU)"
        cuda_ver = "N/A"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda or "available"

        return {
            "git_commit": git_hash,
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "cuda_version": cuda_ver,
            "gpu_name": gpu_name,
            "os_platform": platform.platform(),
            "cpu_architecture": platform.machine(),
        }
