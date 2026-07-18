"""
pathoai/core/environment.py
===========================
Environment validation and audit for PathoAI-Platform.

Performs comprehensive checks of the runtime environment at startup:
- Python version compatibility
- Required package availability and versions
- GPU/CUDA availability
- OpenSlide binary installation
- Available disk space and RAM
- File system write permissions

Designed to be run:
1. At project setup (scripts/setup_environment.py)
2. At pipeline start (preflight validation)
3. In CI (automated environment report)

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1
"""

import importlib
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pathoai.core.constants import MIN_FREE_DISK_GB, MIN_RAM_GB, MIN_PYTHON_VERSION
from pathoai.core.exceptions import EnvironmentValidationError
from pathoai.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# RESULT DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class PackageStatus:
    """Status of a single Python package in the environment."""
    name: str
    required: bool
    installed: bool
    version: Optional[str] = None
    min_version: Optional[str] = None
    status_label: str = "UNKNOWN"  # READY | MISSING | OUTDATED | OPTIONAL_MISSING

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "required": self.required,
            "installed": self.installed,
            "version": self.version,
            "min_version": self.min_version,
            "status": self.status_label,
        }


@dataclass
class EnvironmentReport:
    """Complete environment audit report.

    Attributes
    ----------
    platform_info : Dict
        OS, CPU, RAM, Python version details.
    gpu_info : Dict
        GPU name, VRAM, driver version, CUDA version.
    packages : List[PackageStatus]
        Status of all required and optional packages.
    openslide_info : Dict
        OpenSlide binary and Python binding status.
    disk_info : Dict
        Available disk space on relevant drives.
    errors : List[str]
        Critical errors that will prevent the pipeline from running.
    warnings : List[str]
        Non-critical issues that may affect performance or quality.
    readiness_score : int
        Overall readiness score 0–100.
    is_ready : bool
        True if all critical requirements are met.
    """

    platform_info: Dict = field(default_factory=dict)
    gpu_info: Dict = field(default_factory=dict)
    packages: List[PackageStatus] = field(default_factory=list)
    openslide_info: Dict = field(default_factory=dict)
    disk_info: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    readiness_score: int = 0
    is_ready: bool = False

    def to_dict(self) -> Dict:
        return {
            "platform": self.platform_info,
            "gpu": self.gpu_info,
            "packages": [p.to_dict() for p in self.packages],
            "openslide": self.openslide_info,
            "disk": self.disk_info,
            "errors": self.errors,
            "warnings": self.warnings,
            "readiness_score": self.readiness_score,
            "is_ready": self.is_ready,
        }


# ---------------------------------------------------------------------------
# REQUIRED PACKAGES MANIFEST
# ---------------------------------------------------------------------------

# Format: (import_name, pip_name, min_version, is_required)
REQUIRED_PACKAGES: List[Tuple[str, str, str, bool]] = [
    # Core scientific computing
    ("numpy",                   "numpy",                      "1.24.0",  True),
    ("scipy",                   "scipy",                      "1.10.0",  True),

    # Image processing
    ("PIL",                     "Pillow",                     "9.5.0",   True),
    ("cv2",                     "opencv-python",              "4.8.0",   True),
    ("skimage",                 "scikit-image",               "0.21.0",  True),
    ("albumentations",          "albumentations",             "1.3.0",   True),

    # Deep learning
    ("torch",                   "torch",                      "2.0.0",   True),
    ("torchvision",             "torchvision",                "0.15.0",  True),
    ("timm",                    "timm",                       "0.9.0",   True),
    ("segmentation_models_pytorch", "segmentation-models-pytorch", "0.3.3", True),

    # Data science
    ("pandas",                  "pandas",                     "2.0.0",   True),
    ("sklearn",                 "scikit-learn",               "1.3.0",   True),

    # Configuration
    ("yaml",                    "PyYAML",                     "6.0.1",   True),

    # Experiment tracking
    ("tensorboard",             "tensorboard",                "2.13.0",  True),

    # Visualization
    ("matplotlib",              "matplotlib",                 "3.7.0",   True),

    # Testing
    ("pytest",                  "pytest",                     "7.4.0",   True),

    # Progress display
    ("tqdm",                    "tqdm",                       "4.66.0",  True),

    # Notebooks
    ("jupyterlab",              "jupyterlab",                 "4.0.0",   False),  # Optional

    # WSI reading — special case (binary dependency)
    ("openslide",               "openslide-python",           "1.3.0",   True),
]


# ---------------------------------------------------------------------------
# AUDIT FUNCTIONS
# ---------------------------------------------------------------------------

def audit_platform() -> Dict:
    """Collect platform information.

    Returns
    -------
    Dict
        OS, CPU, RAM, Python version, architecture.
    """
    import psutil  # optional; fall back gracefully
    try:
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        ram_gb = -1.0

    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "ram_gb": round(ram_gb, 2),
        "architecture": platform.architecture()[0],
    }


def audit_gpu() -> Dict:
    """Audit GPU availability, CUDA, and cuDNN.

    Returns
    -------
    Dict
        GPU name, VRAM, driver, CUDA version, cuDNN availability.
    """
    result: Dict = {
        "nvidia_gpu_detected": False,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_count": 0,
        "vram_gb": None,
        "driver_version": None,
        "cuda_version": None,
        "cudnn_available": False,
        "cudnn_version": None,
        "torch_cuda_available": False,
    }

    # Check nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            proc = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                lines = proc.stdout.strip().split("\n")
                result["nvidia_gpu_detected"] = True
                result["gpu_count"] = len(lines)
                first = lines[0].split(", ")
                if len(first) >= 3:
                    result["gpu_name"] = first[0].strip()
                    try:
                        vram_mib = int(first[1].strip().replace(" MiB", ""))
                        result["vram_gb"] = round(vram_mib / 1024, 2)
                    except ValueError:
                        pass
                    result["driver_version"] = first[2].strip()

            # Get CUDA version from nvidia-smi header
            proc2 = subprocess.run(
                [nvidia_smi], capture_output=True, text=True, timeout=10,
            )
            if "CUDA Version:" in proc2.stdout:
                for line in proc2.stdout.split("\n"):
                    if "CUDA Version:" in line:
                        result["cuda_version"] = line.split("CUDA Version:")[-1].strip().rstrip("|").strip()
                        break
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    # Check PyTorch CUDA availability
    try:
        import torch
        result["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["cuda_available"] = True
            result["gpu_count"] = result["gpu_count"] or torch.cuda.device_count()
            if result["gpu_count"] > 0 and result["gpu_name"] is None:
                result["gpu_name"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                result["vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
        try:
            import torch.backends.cudnn as cudnn
            result["cudnn_available"] = cudnn.is_available()
            if cudnn.is_available():
                result["cudnn_version"] = str(cudnn.version())
        except Exception:  # noqa: BLE001
            pass
    except ImportError:
        pass

    return result


def audit_package(import_name: str, pip_name: str, min_version: str, required: bool) -> PackageStatus:
    """Check if a single package is installed and meets version requirement.

    Parameters
    ----------
    import_name : str
        Python module name for import.
    pip_name : str
        PyPI package name (for user instructions).
    min_version : str
        Minimum acceptable version string (e.g., "2.0.0").
    required : bool
        Whether this package is mandatory.

    Returns
    -------
    PackageStatus
        Status object for this package.
    """
    status = PackageStatus(
        name=pip_name,
        required=required,
        installed=False,
    )
    status.min_version = min_version

    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        status.installed = True
        status.version = version

        # Check minimum version
        if version != "unknown" and min_version:
            try:
                from packaging.version import Version
                if Version(version) >= Version(min_version):
                    status.status_label = "READY"
                else:
                    status.status_label = "OUTDATED"
            except Exception:  # noqa: BLE001
                status.status_label = "READY"  # Cannot compare, assume OK
        else:
            status.status_label = "READY"

    except ImportError:
        status.installed = False
        status.status_label = "MISSING" if required else "OPTIONAL_MISSING"

    return status


def audit_openslide() -> Dict:
    """Check OpenSlide binary and Python binding availability.

    Returns
    -------
    Dict
        Status of OpenSlide C library and Python bindings.
    """
    result: Dict = {
        "python_binding_installed": False,
        "python_binding_version": None,
        "c_library_found": False,
        "c_library_version": None,
        "dll_path": None,
        "status": "MISSING",
        "install_instructions": (
            "1. Download OpenSlide Windows binaries from: "
            "https://github.com/openslide/openslide-winbuild/releases\n"
            "2. Extract and add the 'bin' directory to your system PATH\n"
            "3. Install Python bindings: pip install openslide-python"
        ),
    }

    # Check Python bindings
    try:
        import openslide
        result["python_binding_installed"] = True
        result["python_binding_version"] = getattr(openslide, "__version__", "unknown")

        # Try to get C library version
        try:
            result["c_library_version"] = openslide.OPENSLIDE_VERSION_STRING
            result["c_library_found"] = True
            result["status"] = "READY"
        except AttributeError:
            result["status"] = "PARTIAL"  # Python binding OK, C lib unclear
    except ImportError:
        result["status"] = "MISSING"
    except Exception as e:  # noqa: BLE001
        result["status"] = "ERROR"
        result["error"] = str(e)

    # Check for DLL on Windows PATH
    if platform.system() == "Windows":
        dll = shutil.which("libopenslide-0.dll") or shutil.which("openslide-0.dll")
        if dll:
            result["dll_path"] = dll
            result["c_library_found"] = True

    return result


def audit_disk_space(paths: Optional[List[Path]] = None) -> Dict:
    """Check available disk space on relevant paths.

    Parameters
    ----------
    paths : List[Path], optional
        Paths to check. Defaults to current drive and D:\\ (project data drive).

    Returns
    -------
    Dict
        Available GB per path.
    """
    if paths is None:
        paths = [Path.cwd(), Path("D:\\")]

    result = {}
    for path in paths:
        try:
            stat = shutil.disk_usage(path)
            result[str(path)] = {
                "total_gb": round(stat.total / (1024 ** 3), 2),
                "used_gb": round(stat.used / (1024 ** 3), 2),
                "free_gb": round(stat.free / (1024 ** 3), 2),
                "sufficient": stat.free / (1024 ** 3) >= MIN_FREE_DISK_GB,
            }
        except OSError:
            result[str(path)] = {"error": "Path not accessible"}

    return result


def audit_external_tools() -> Dict:
    """Check external tools (git, AWS CLI, VS Code).

    Returns
    -------
    Dict
        Version strings for available tools.
    """
    tools = {}
    for tool, args in [
        ("git", ["git", "--version"]),
        ("aws", ["aws", "--version"]),
        ("code", ["code", "--version"]),
    ]:
        found = shutil.which(tool)
        if found:
            try:
                proc = subprocess.run(args, capture_output=True, text=True, timeout=5)
                output = (proc.stdout + proc.stderr).strip().split("\n")[0]
                tools[tool] = {"found": True, "path": found, "version": output}
            except (subprocess.TimeoutExpired, OSError):
                tools[tool] = {"found": True, "path": found, "version": "unknown"}
        else:
            tools[tool] = {"found": False}

    return tools


# ---------------------------------------------------------------------------
# MAIN AUDIT ORCHESTRATOR
# ---------------------------------------------------------------------------

def run_full_audit(
    output_path: Optional[Path] = None,
) -> EnvironmentReport:
    """Run a complete environment audit and return a structured report.

    Checks all system components: platform, GPU, Python packages, OpenSlide,
    disk space, and external tools.

    Parameters
    ----------
    output_path : Path, optional
        If provided, saves the JSON report to this path.

    Returns
    -------
    EnvironmentReport
        Complete audit results.
    """
    report = EnvironmentReport()

    logger.info("Starting full environment audit...")

    # 1. Platform audit
    logger.info("Phase 1: Platform audit")
    report.platform_info = audit_platform()

    # Python version check
    py_ver = sys.version_info[:2]
    if py_ver < MIN_PYTHON_VERSION:
        report.errors.append(
            f"Python {py_ver[0]}.{py_ver[1]} detected. "
            f"Minimum required: {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}"
        )
    elif py_ver < (3, 11):
        report.warnings.append(
            f"Python {py_ver[0]}.{py_ver[1]} detected. "
            f"Recommended: 3.11 for full compatibility."
        )

    # 2. GPU audit
    logger.info("Phase 2: GPU audit")
    report.gpu_info = audit_gpu()

    if not report.gpu_info.get("nvidia_gpu_detected"):
        report.warnings.append(
            "No NVIDIA GPU detected. Pipeline will run on CPU — significantly slower. "
            "Expected full-slide processing time: ~30-45 minutes (vs ~5 min on GPU)."
        )
    if not report.gpu_info.get("cuda_available"):
        report.warnings.append(
            "CUDA not available. Using CPU PyTorch. This is expected for development "
            "on machines without NVIDIA GPU."
        )

    # 3. Package audit
    logger.info("Phase 3: Python package audit")
    for import_name, pip_name, min_ver, required in REQUIRED_PACKAGES:
        pkg_status = audit_package(import_name, pip_name, min_ver, required)
        report.packages.append(pkg_status)

        if not pkg_status.installed and required:
            report.errors.append(f"Required package '{pip_name}' is not installed. Run: pip install {pip_name}")
        elif pkg_status.status_label == "OUTDATED":
            report.warnings.append(
                f"Package '{pip_name}' version {pkg_status.version} < minimum {min_ver}. "
                f"Run: pip install --upgrade {pip_name}"
            )

    # 4. OpenSlide audit
    logger.info("Phase 4: OpenSlide audit")
    report.openslide_info = audit_openslide()
    if report.openslide_info["status"] != "READY":
        report.errors.append(
            "OpenSlide is not properly installed. WSI reading will not work.\n"
            f"{report.openslide_info.get('install_instructions', '')}"
        )

    # 5. Disk space audit
    logger.info("Phase 5: Disk space audit")
    report.disk_info = audit_disk_space()
    for path, info in report.disk_info.items():
        if "error" not in info and not info.get("sufficient", True):
            report.warnings.append(
                f"Low free disk space on {path}: {info['free_gb']} GB "
                f"(minimum recommended: {MIN_FREE_DISK_GB} GB)"
            )

    # 6. Compute readiness score
    n_packages_ok = sum(1 for p in report.packages if p.status_label in {"READY", "OPTIONAL_MISSING"})
    n_packages_total = len(report.packages)
    pkg_score = int(100 * n_packages_ok / max(n_packages_total, 1))

    openslide_score = 10 if report.openslide_info["status"] == "READY" else 0
    error_penalty = min(len(report.errors) * 5, 30)

    report.readiness_score = max(0, min(100, pkg_score - error_penalty + (openslide_score // 2)))
    report.is_ready = len(report.errors) == 0

    # 7. Save report if path provided
    if output_path is not None:
        import json
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info("Audit report saved", extra={"path": str(output_path)})

    logger.info(
        "Environment audit complete",
        extra={
            "readiness_score": report.readiness_score,
            "is_ready": report.is_ready,
            "n_errors": len(report.errors),
            "n_warnings": len(report.warnings),
        },
    )

    return report


def validate_environment(raise_on_error: bool = True) -> EnvironmentReport:
    """Quick validation — raises if critical requirements not met.

    Parameters
    ----------
    raise_on_error : bool
        If True, raises EnvironmentValidationError on critical failures.

    Returns
    -------
    EnvironmentReport
        Audit results.

    Raises
    ------
    EnvironmentValidationError
        If critical requirements are not met and raise_on_error=True.
    """
    report = run_full_audit()
    if not report.is_ready and raise_on_error:
        error_msg = "\n".join(report.errors)
        raise EnvironmentValidationError(
            f"Environment validation failed:\n{error_msg}\n\n"
            f"Run scripts/setup_environment.py to resolve these issues."
        )
    return report
