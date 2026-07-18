"""
pathoai/core/reproducibility.py
================================
Reproducibility utilities for PathoAI-Platform.

Ensures deterministic behavior across all stages of training and inference.
Must be called at the start of every training run and inference run.

Reference:
    PyTorch Reproducibility Guide:
    https://pytorch.org/docs/stable/notes/randomness.html

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1
"""

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from pathoai.core.logger import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility.

    Sets seeds for Python's random, NumPy, PyTorch (CPU + CUDA),
    and configures PyTorch for deterministic operations.

    Must be called at the start of EVERY training and inference run
    before any data loading or model operations.

    Parameters
    ----------
    seed : int
        Random seed value. Use the seed from config.pipeline.seed.
        Recommended: 42 (conventional research default).

    Notes
    -----
    - torch.backends.cudnn.deterministic = True ensures deterministic
      convolution algorithms at the cost of some performance.
    - torch.backends.cudnn.benchmark = False disables the auto-tuner
      that selects the fastest algorithm (which may vary between runs).
    - These settings are critical for research reproducibility but may
      reduce GPU training throughput by 5-15%.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        logger.warning("NumPy not available — NumPy seed not set")

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Disable for reproducibility
        # For PyTorch >= 1.8: enable fully deterministic operations
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass  # Older PyTorch version — skip
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Required for CUDA determinism
    except ImportError:
        logger.warning("PyTorch not available — PyTorch seeds not set")

    logger.info("Global random seed set", extra={"seed": seed})


def get_worker_init_fn(seed: int):
    """Return a DataLoader worker_init_fn for reproducible multi-process loading.

    Parameters
    ----------
    seed : int
        Base random seed (same as passed to set_global_seed).

    Returns
    -------
    Callable
        Function suitable for DataLoader(worker_init_fn=...).

    Example
    -------
    >>> loader = DataLoader(
    ...     dataset,
    ...     num_workers=4,
    ...     worker_init_fn=get_worker_init_fn(seed=42),
    ...     generator=torch.Generator().manual_seed(42),
    ... )
    """
    def worker_init_fn(worker_id: int) -> None:
        try:
            import torch
            worker_seed = torch.initial_seed() % 2**32
        except ImportError:
            worker_seed = seed + worker_id

        random.seed(worker_seed)
        os.environ["PYTHONHASHSEED"] = str(worker_seed)

        try:
            import numpy as np
            np.random.seed(worker_seed)
        except ImportError:
            pass

    return worker_init_fn


def capture_environment_snapshot(
    experiment_id: str,
    config_hash: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    """Capture a complete snapshot of the runtime environment.

    Records all information needed to reproduce the experimental conditions:
    package versions, Python version, OS, GPU, git commit, config hash.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for this experiment run.
    config_hash : str, optional
        SHA-256 hash of the experiment configuration.
    extra : Dict, optional
        Additional key-value pairs to include in the snapshot.

    Returns
    -------
    Dict
        Complete environment snapshot, JSON-serializable.
    """
    snapshot: Dict = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "config_hash": config_hash,
        "python": {
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {},
        "gpu": {},
        "git": {},
    }

    # Package versions
    packages_to_capture = [
        "torch", "torchvision", "numpy", "pandas", "cv2",
        "PIL", "skimage", "albumentations", "timm",
        "segmentation_models_pytorch", "openslide", "yaml",
    ]
    for pkg_name in packages_to_capture:
        try:
            mod = __import__(pkg_name)
            snapshot["packages"][pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            snapshot["packages"][pkg_name] = "not_installed"

    # GPU info
    try:
        import torch
        snapshot["gpu"]["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            snapshot["gpu"]["device_count"] = torch.cuda.device_count()
            snapshot["gpu"]["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            snapshot["gpu"]["vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
        snapshot["gpu"]["torch_version"] = torch.__version__
    except ImportError:
        snapshot["gpu"]["torch_available"] = False

    # Git info
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        snapshot["git"]["commit"] = commit
        snapshot["git"]["branch"] = branch
    except (subprocess.CalledProcessError, FileNotFoundError):
        snapshot["git"]["error"] = "git not available or not a git repository"

    # Extra fields
    if extra:
        snapshot.update(extra)

    logger.info(
        "Environment snapshot captured",
        extra={"experiment_id": experiment_id},
    )

    return snapshot


def save_environment_snapshot(
    snapshot: Dict,
    output_dir: Path,
    filename: str = "environment_snapshot.json",
) -> Path:
    """Save environment snapshot to a JSON file.

    Parameters
    ----------
    snapshot : Dict
        Snapshot returned by capture_environment_snapshot().
    output_dir : Path
        Directory to save the snapshot file.
    filename : str
        Output filename. Default: "environment_snapshot.json".

    Returns
    -------
    Path
        Path to the saved JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)

    logger.info("Snapshot saved", extra={"path": str(output_path)})
    return output_path
