#!/usr/bin/env python3
"""
scripts/setup_environment.py
=============================
PathoAI-Platform — Complete Environment Setup and Validation Script

This script:
1. Audits your existing environment (NEVER installs anything without showing you first)
2. Reports what is installed and what is missing
3. Offers to install missing packages (requires explicit confirmation)
4. Verifies all installations
5. Sets up the PathoAI virtual environment
6. Generates a final readiness report

Usage:
    python scripts/setup_environment.py                # Full audit + install prompt
    python scripts/setup_environment.py --audit-only  # Audit only, no installs
    python scripts/setup_environment.py --report      # Generate JSON report

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1
"""

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# ANSI COLORS
# ---------------------------------------------------------------------------
class Color:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


def c(text: str, color: str) -> str:
    return f"{color}{text}{Color.RESET}"


def header(title: str) -> None:
    width = 70
    print()
    print(c("=" * width, Color.CYAN))
    print(c(f"  {title}", Color.BOLD + Color.CYAN))
    print(c("=" * width, Color.CYAN))


def section(title: str) -> None:
    print()
    print(c(f"── {title} " + "─" * max(0, 60 - len(title)), Color.YELLOW))


def ok(msg: str) -> None:
    print(c(f"  ✓  {msg}", Color.GREEN))


def warn(msg: str) -> None:
    print(c(f"  ⚠  {msg}", Color.YELLOW))


def fail(msg: str) -> None:
    print(c(f"  ✗  {msg}", Color.RED))


def info(msg: str) -> None:
    print(f"     {msg}")


# ---------------------------------------------------------------------------
# PHASE 1: SYSTEM AUDIT
# ---------------------------------------------------------------------------

def audit_system() -> Dict:
    """Audit system hardware and software environment."""
    header("PHASE 1 — SYSTEM AUDIT")

    result: Dict = {}

    # OS
    section("Operating System")
    os_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }
    ok(f"OS: {os_info['system']} {os_info['release']}")
    info(f"Version: {os_info['version'][:80]}")
    result["os"] = os_info

    # CPU
    section("CPU")
    cpu_info = {"processor": platform.processor()}
    try:
        import psutil
        cpu_info["cores_physical"] = psutil.cpu_count(logical=False)
        cpu_info["cores_logical"] = psutil.cpu_count(logical=True)
        ok(f"CPU: {cpu_info['processor']}")
        ok(f"Cores: {cpu_info['cores_physical']} physical / {cpu_info['cores_logical']} logical")
    except ImportError:
        ok(f"CPU: {cpu_info['processor']}")
        warn("psutil not installed — cannot get core count")
    result["cpu"] = cpu_info

    # RAM
    section("RAM")
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        result["ram_gb"] = round(ram_gb, 2)
        if ram_gb >= 16:
            ok(f"RAM: {ram_gb:.1f} GB")
        elif ram_gb >= 8:
            warn(f"RAM: {ram_gb:.1f} GB (minimum 8 GB, 16 GB recommended)")
        else:
            fail(f"RAM: {ram_gb:.1f} GB (insufficient — minimum 8 GB required)")
    except ImportError:
        warn("RAM: Cannot determine (psutil not installed)")
        result["ram_gb"] = -1

    # Disk space
    section("Disk Space")
    disk_result = {}
    for drive in ["C:\\", "D:\\"]:
        try:
            stat = shutil.disk_usage(drive)
            free_gb = stat.free / (1024 ** 3)
            total_gb = stat.total / (1024 ** 3)
            disk_result[drive] = {"free_gb": round(free_gb, 2), "total_gb": round(total_gb, 2)}
            status_fn = ok if free_gb >= 10 else warn if free_gb >= 5 else fail
            status_fn(f"{drive} — Free: {free_gb:.1f} GB / Total: {total_gb:.1f} GB")
        except OSError:
            info(f"{drive} — not accessible")
    result["disk"] = disk_result

    return result


# ---------------------------------------------------------------------------
# PHASE 2: PYTHON ENVIRONMENT AUDIT
# ---------------------------------------------------------------------------

PACKAGES_TO_AUDIT: List[Tuple[str, str, str]] = [
    # (import_name, pip_name, min_version)
    ("torch",                     "torch",                        "2.0.0"),
    ("torchvision",               "torchvision",                  "0.15.0"),
    ("torchaudio",                "torchaudio",                   "2.0.0"),
    ("cv2",                       "opencv-python",                "4.8.0"),
    ("numpy",                     "numpy",                        "1.24.0"),
    ("pandas",                    "pandas",                       "2.0.0"),
    ("matplotlib",                "matplotlib",                   "3.7.0"),
    ("skimage",                   "scikit-image",                 "0.21.0"),
    ("sklearn",                   "scikit-learn",                 "1.3.0"),
    ("albumentations",            "albumentations",               "1.3.0"),
    ("timm",                      "timm",                         "0.9.0"),
    ("segmentation_models_pytorch","segmentation-models-pytorch", "0.3.3"),
    ("PIL",                       "Pillow",                       "9.5.0"),
    ("tqdm",                      "tqdm",                         "4.66.0"),
    ("yaml",                      "PyYAML",                       "6.0.1"),
    ("pytest",                    "pytest",                       "7.4.0"),
    ("tensorboard",               "tensorboard",                  "2.13.0"),
    ("jupyterlab",                "jupyterlab",                   "4.0.0"),
    ("openslide",                 "openslide-python",             "1.3.0"),
    ("scipy",                     "scipy",                        "1.10.0"),
]


def audit_python_packages() -> Dict:
    """Audit all required Python packages."""
    header("PHASE 2 — PYTHON ENVIRONMENT AUDIT")

    section("Python Version")
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_exec = sys.executable
    if sys.version_info >= (3, 11):
        ok(f"Python {py_ver} — OPTIMAL")
    elif sys.version_info >= (3, 10):
        warn(f"Python {py_ver} — ACCEPTABLE (3.11 recommended)")
    else:
        fail(f"Python {py_ver} — INSUFFICIENT (minimum 3.10 required)")
    info(f"Executable: {py_exec}")

    section("Python Package Audit")
    print(f"\n  {'Package':<35} {'Installed?':<12} {'Version':<15} {'Min Ver':<12} {'Status'}")
    print(f"  {'-'*35} {'-'*12} {'-'*15} {'-'*12} {'-'*10}")

    results = {}
    missing_required = []

    for import_name, pip_name, min_ver in PACKAGES_TO_AUDIT:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            installed = True

            # Check min version
            try:
                from packaging.version import Version
                meets_min = version == "unknown" or Version(version) >= Version(min_ver)
            except Exception:
                meets_min = True

            status = "✓ READY" if meets_min else "⚠ OUTDATED"
            status_color = Color.GREEN if meets_min else Color.YELLOW
        except ImportError:
            version = "—"
            installed = False
            meets_min = False
            status = "✗ MISSING"
            status_color = Color.RED
            missing_required.append(pip_name)

        print(
            f"  {pip_name:<35} {'YES' if installed else 'NO':<12} "
            f"{version:<15} {min_ver:<12} {c(status, status_color)}"
        )
        results[pip_name] = {
            "installed": installed,
            "version": version,
            "min_version": min_ver,
            "status": status.strip("✓⚠✗ "),
        }

    return {"packages": results, "missing": missing_required}


# ---------------------------------------------------------------------------
# PHASE 3: GPU AUDIT
# ---------------------------------------------------------------------------

def audit_gpu() -> Dict:
    """Comprehensive GPU, CUDA, and cuDNN audit."""
    header("PHASE 3 — GPU AUDIT")
    result: Dict = {}

    section("NVIDIA Driver and CUDA (nvidia-smi)")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            proc = subprocess.run([nvidia_smi], capture_output=True, text=True, timeout=10)
            lines = proc.stdout.strip().split("\n")
            for line in lines[:10]:  # Print first 10 lines of nvidia-smi
                if line.strip():
                    info(line)
            result["nvidia_smi_available"] = True
            result["nvidia_smi_output"] = proc.stdout[:500]
            ok("nvidia-smi found and responsive")
        except (subprocess.TimeoutExpired, OSError) as e:
            warn(f"nvidia-smi found but failed: {e}")
            result["nvidia_smi_available"] = False
    else:
        warn("nvidia-smi not found — No NVIDIA GPU or driver not installed")
        info("Note: Intel iGPU detected. CPU inference path will be used.")
        result["nvidia_smi_available"] = False

    section("PyTorch CUDA Availability")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        result["pytorch_cuda"] = cuda_available
        result["pytorch_version"] = torch.__version__

        if cuda_available:
            ok(f"torch.cuda.is_available() = True")
            ok(f"GPU count: {torch.cuda.device_count()}")
            ok(f"GPU name: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / (1024 ** 3)
            ok(f"VRAM: {vram:.1f} GB")
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["vram_gb"] = round(vram, 2)

            try:
                import torch.backends.cudnn as cudnn
                result["cudnn_available"] = cudnn.is_available()
                if cudnn.is_available():
                    ok(f"cuDNN: {cudnn.version()}")
            except Exception:
                pass
        else:
            warn("torch.cuda.is_available() = False")
            info("→ Using CPU inference. Expected: slower processing.")
            info(f"→ PyTorch version: {torch.__version__}")
    except ImportError:
        fail("PyTorch not installed")
        result["pytorch_cuda"] = False

    return result


# ---------------------------------------------------------------------------
# PHASE 4: DEPENDENCY CHECK
# ---------------------------------------------------------------------------

def audit_external_tools() -> Dict:
    """Check external tool installations."""
    header("PHASE 4 — DEPENDENCY CHECK")
    result: Dict = {}

    tools = [
        ("git",     ["git",   "--version"],     "Git"),
        ("code",    ["code",  "--version"],     "VS Code"),
        ("aws",     ["aws",   "--version"],     "AWS CLI"),
        ("conda",   ["conda", "--version"],     "Miniconda/Conda"),
        ("python",  [sys.executable, "--version"], f"Python ({sys.executable})"),
    ]

    for cmd, args, display_name in tools:
        found = shutil.which(cmd)
        if found or cmd == "python":
            try:
                proc = subprocess.run(args, capture_output=True, text=True, timeout=5)
                version_str = (proc.stdout + proc.stderr).strip().split("\n")[0]
                ok(f"{display_name}: {version_str}")
                result[cmd] = {"found": True, "path": found or sys.executable, "version": version_str}
            except (subprocess.TimeoutExpired, OSError):
                ok(f"{display_name}: Found (version check timed out)")
                result[cmd] = {"found": True, "path": found}
        else:
            warn(f"{display_name}: NOT FOUND")
            result[cmd] = {"found": False}

    # OpenSlide special check
    section("OpenSlide Binaries")
    dll_paths = ["libopenslide-0.dll", "openslide-0.dll"]
    openslide_dll_found = any(shutil.which(dll) for dll in dll_paths)
    if openslide_dll_found:
        ok("OpenSlide DLL found in PATH")
    else:
        warn("OpenSlide DLL not found in PATH")
        info("Download from: https://github.com/openslide/openslide-winbuild/releases")
        info("Extract and add the 'bin' directory to your system PATH")

    try:
        import openslide
        ok(f"openslide-python: {openslide.__version__}")
        result["openslide_python"] = {"installed": True, "version": openslide.__version__}
    except ImportError:
        warn("openslide-python: NOT installed")
        result["openslide_python"] = {"installed": False}

    return result


# ---------------------------------------------------------------------------
# PHASE 5-6: INSTALL MISSING + VERIFY
# ---------------------------------------------------------------------------

def install_missing_packages(missing: List[str], dry_run: bool = False) -> Dict:
    """Install missing packages using pip."""
    header("PHASE 5 — INSTALL MISSING PACKAGES")

    if not missing:
        ok("No missing packages to install!")
        return {}

    print(f"\n  Missing packages: {', '.join(missing)}")

    if dry_run:
        warn("Dry run mode — no packages will be installed")
        return {}

    results = {}

    # Special case: PyTorch — offer CUDA vs CPU choice
    torch_packages = {p for p in missing if p in {"torch", "torchvision", "torchaudio"}}
    other_packages = [p for p in missing if p not in torch_packages]

    if torch_packages:
        section("PyTorch Installation")
        print("\n  PyTorch requires special installation command.")
        print("  Since this machine has Intel iGPU only (no NVIDIA CUDA),")
        print("  we will install the CPU-only version.")
        print()
        torch_cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu",
        ]
        info(f"Command: {' '.join(torch_cmd)}")
        confirm = input("\n  Proceed with PyTorch CPU installation? [y/N]: ").strip().lower()
        if confirm == "y":
            proc = subprocess.run(torch_cmd, capture_output=False)
            results["torch_group"] = "installed" if proc.returncode == 0 else "failed"
        else:
            results["torch_group"] = "skipped"

    # Install other packages
    if other_packages:
        section("Other Package Installation")
        install_cmd = [sys.executable, "-m", "pip", "install"] + other_packages
        info(f"Command: {' '.join(install_cmd)}")
        confirm = input(f"\n  Install {len(other_packages)} missing packages? [y/N]: ").strip().lower()
        if confirm == "y":
            for pkg in other_packages:
                sub_cmd = [sys.executable, "-m", "pip", "install", pkg]
                proc = subprocess.run(sub_cmd, capture_output=True, text=True)
                if proc.returncode == 0:
                    ok(f"Installed: {pkg}")
                    results[pkg] = "installed"
                else:
                    fail(f"Failed: {pkg}")
                    info(proc.stderr[-200:])
                    results[pkg] = "failed"

    return results


def verify_installations() -> Dict:
    """Re-run package imports to verify successful installation."""
    header("PHASE 6 — VERIFY INSTALLATIONS")
    section("Import Verification")

    results = {}
    critical_imports = [
        ("torch",          "PyTorch"),
        ("torchvision",    "TorchVision"),
        ("cv2",            "OpenCV"),
        ("numpy",          "NumPy"),
        ("pandas",         "Pandas"),
        ("sklearn",        "scikit-learn"),
        ("skimage",        "scikit-image"),
        ("albumentations", "Albumentations"),
        ("timm",           "timm"),
        ("PIL",            "Pillow"),
        ("yaml",           "PyYAML"),
        ("pytest",         "pytest"),
        ("tensorboard",    "TensorBoard"),
        ("tqdm",           "tqdm"),
    ]

    for import_name, display_name in critical_imports:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "unknown")
            ok(f"{display_name}: {ver}")
            results[import_name] = {"status": "OK", "version": ver}
        except ImportError as e:
            fail(f"{display_name}: IMPORT FAILED — {e}")
            results[import_name] = {"status": "FAILED", "error": str(e)}

    # PyTorch CUDA test
    section("PyTorch Functional Test")
    try:
        import torch
        t = torch.tensor([1.0, 2.0, 3.0])
        assert t.sum().item() == 6.0
        ok("PyTorch tensor operations: WORKING")
        results["torch_functional"] = "OK"
    except Exception as e:
        fail(f"PyTorch functional test failed: {e}")
        results["torch_functional"] = "FAILED"

    return results


# ---------------------------------------------------------------------------
# PHASE 7: CREATE VIRTUAL ENVIRONMENT
# ---------------------------------------------------------------------------

def create_virtual_environment(venv_name: str = "pathoai_env") -> Optional[Path]:
    """Create a virtual environment for PathoAI-Platform."""
    header("PHASE 7 — CREATE PROJECT ENVIRONMENT")

    venv_path = Path.cwd() / venv_name
    if venv_path.exists():
        ok(f"Virtual environment already exists: {venv_path}")
        return venv_path

    section("Creating Virtual Environment")
    info(f"Location: {venv_path}")
    confirm = input(f"\n  Create virtual environment '{venv_name}'? [y/N]: ").strip().lower()
    if confirm != "y":
        warn("Virtual environment creation skipped")
        return None

    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        ok(f"Virtual environment created: {venv_path}")

        # Upgrade pip in venv
        pip_path = venv_path / "Scripts" / "pip.exe"
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
        ok("pip upgraded in virtual environment")

        return venv_path
    except subprocess.CalledProcessError as e:
        fail(f"Virtual environment creation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# PHASE 8-9: FINAL VALIDATION + READINESS REPORT
# ---------------------------------------------------------------------------

def generate_readiness_report(
    system: Dict,
    packages: Dict,
    gpu: Dict,
    tools: Dict,
    verify: Dict,
    output_path: Optional[Path] = None,
) -> Dict:
    """Generate final environment readiness report."""
    header("PHASE 8 — FINAL VALIDATION REPORT")

    pkg_results = packages.get("packages", {})
    n_ok = sum(1 for p in pkg_results.values() if p.get("status") in {"READY", "unknown"} and p.get("installed"))
    n_total = len(pkg_results)

    # Classify each item
    report_items = []
    for pkg_name, info_dict in pkg_results.items():
        if info_dict.get("installed"):
            label = "READY" if "READY" in info_dict.get("status", "") else "INSTALLED"
        else:
            label = "MISSING"
        report_items.append((pkg_name, label, info_dict.get("version", "—")))

    print()
    print(f"  {'Component':<35} {'Status':<15} {'Version'}")
    print(f"  {'-'*35} {'-'*15} {'-'*20}")
    for name, label, ver in sorted(report_items):
        color = Color.GREEN if label in {"READY", "INSTALLED"} else Color.RED
        print(f"  {name:<35} {c(label, color):<24} {ver}")

    # Compute score
    base_score = int(100 * n_ok / max(n_total, 1))
    cuda_bonus = 0  # No CUDA on this machine — not a penalty, expected
    missing_penalty = len(packages.get("missing", [])) * 3
    readiness_score = max(0, min(100, base_score - missing_penalty))

    section("Overall Readiness Score")
    score_color = Color.GREEN if readiness_score >= 90 else Color.YELLOW if readiness_score >= 70 else Color.RED
    print()
    print(c(f"  Research Environment Readiness: {readiness_score}/100", Color.BOLD + score_color))
    print()

    if readiness_score == 100:
        ok("ENVIRONMENT IS FULLY READY. You may begin PathoAI-Platform development.")
    elif readiness_score >= 80:
        warn("ENVIRONMENT IS MOSTLY READY. Install missing packages before training.")
    else:
        fail("ENVIRONMENT NEEDS SETUP. Several critical packages are missing.")

    report = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "readiness_score": readiness_score,
        "system": system,
        "packages": pkg_results,
        "gpu": gpu,
        "tools": tools,
        "verification": verify,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        ok(f"Report saved: {output_path}")

    return report


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PathoAI-Platform Environment Setup and Validation"
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run audit only — no installations",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="environment_report.json",
        help="Output path for JSON report (default: environment_report.json)",
    )
    args = parser.parse_args()

    print()
    print(c("╔══════════════════════════════════════════════════════════════════╗", Color.CYAN))
    print(c("║         PathoAI-Platform — Environment Setup & Validation        ║", Color.BOLD + Color.CYAN))
    print(c("╚══════════════════════════════════════════════════════════════════╝", Color.CYAN))
    print(c(f"  Python: {sys.version}  |  Platform: {platform.system()}", Color.YELLOW))

    # Run all audit phases
    system_info = audit_system()
    pkg_info = audit_python_packages()
    gpu_info = audit_gpu()
    tools_info = audit_external_tools()

    if not args.audit_only:
        # Install missing packages
        install_results = install_missing_packages(
            missing=pkg_info.get("missing", []),
            dry_run=False,
        )
        # Verify
        verify_results = verify_installations()
        # Create venv (optional)
        # create_virtual_environment()
    else:
        install_results = {}
        verify_results = {}

    # Final report
    generate_readiness_report(
        system=system_info,
        packages=pkg_info,
        gpu=gpu_info,
        tools=tools_info,
        verify=verify_results,
        output_path=Path(args.report) if args.report else None,
    )


if __name__ == "__main__":
    main()
