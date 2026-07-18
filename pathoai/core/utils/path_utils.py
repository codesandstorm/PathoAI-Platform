"""
pathoai/core/utils/path_utils.py
=================================
Directory and filesystem utility functions for PathoAI-Platform.

Provides deterministic, logged helpers for:
- Project directory tree creation and validation
- Safe path resolution with platform normalization
- Atomic file-existence checks with informative error messages
- .gitkeep placeholder management for version-controlled empty directories

All functions are stateless, idempotent, and produce structured log output.

Design principles:
    - Never silently fail — raise domain exceptions with actionable messages.
    - Use pathlib.Path throughout; never manipulate path strings directly.
    - All directory-creation operations are idempotent (exist_ok=True).
    - Preserve directory structure as declared in Folder_Structure.md.

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1.1
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pathoai.core.exceptions import PathoAIException
from pathoai.core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Exceptions specific to path operations
# ---------------------------------------------------------------------------

class PathError(PathoAIException):
    """Raised when a required path does not exist or cannot be created.

    Examples:
        - A required input directory is missing and auto-creation is disabled.
        - A file path resolves outside the expected project root.
        - Insufficient filesystem permissions to create a directory.
    """


# ---------------------------------------------------------------------------
# Constants: canonical project directory names
# ---------------------------------------------------------------------------

#: Directories that must exist at runtime but whose contents are git-ignored.
#: Each contains a single .gitkeep placeholder file.
_GITKEEP_PLACEHOLDER: str = ".gitkeep"

#: Canonical first-level project directories relative to the project root.
#: Keeping this list here ensures path_utils is the single source of truth
#: for directory topology — do not scatter `mkdir` calls across the codebase.
PROJECT_DIRECTORIES: Dict[str, str] = {
    "pathoai": "Source package — always tracked by Git",
    "config": "YAML configuration files — always tracked by Git",
    "tests": "Unit and integration tests — always tracked by Git",
    "docs": "Project documentation — always tracked by Git",
    "scripts": "Utility and setup scripts — always tracked by Git",
    "notebooks": "Jupyter notebooks (clean, no output cells) — tracked by Git",
    "data": "Raw and processed datasets — Git-ignored, .gitkeep only",
    "models": "Pretrained and trained model weights — Git-ignored, .gitkeep only",
    "logs": "Experiment and run logs — Git-ignored, .gitkeep only",
    "outputs": "Pipeline outputs and reports — Git-ignored, .gitkeep only",
    "results": "Evaluation results — Git-ignored, .gitkeep only",
    "checkpoints": "Training checkpoints — Git-ignored, .gitkeep only",
    "splits": "Dataset split definitions — Git-ignored, .gitkeep only",
}

#: Subdirectories that are always git-ignored (must contain .gitkeep).
_GITIGNORED_DIRS: frozenset[str] = frozenset({
    "data", "models", "logs", "outputs", "results", "checkpoints", "splits",
})


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def resolve_path(path: str | Path) -> Path:
    """Resolve a path to an absolute, normalized ``pathlib.Path``.

    Expands user home (``~``) and environment variables, then resolves
    symlinks and ``..`` components. The resulting path is always absolute.

    Args:
        path: A relative or absolute path string or ``Path`` object.

    Returns:
        Absolute, resolved ``Path``.

    Raises:
        PathError: If ``path`` is empty or cannot be represented as a
            filesystem path.

    Example:
        >>> p = resolve_path("~/research/PathoAI-Platform")
        >>> p.is_absolute()
        True
    """
    if not path and not isinstance(path, Path):
        raise PathError("path must be a non-empty string or Path object.")
    try:
        return Path(os.path.expandvars(str(path))).expanduser().resolve()
    except (TypeError, ValueError) as exc:
        raise PathError(f"Cannot resolve path {path!r}: {exc}") from exc


def ensure_directory(
    path: str | Path,
    *,
    parents: bool = True,
    exist_ok: bool = True,
) -> Path:
    """Create a directory (and any missing parents) if it does not exist.

    This function is idempotent — calling it on an existing directory
    is always safe and produces no side effects.

    Args:
        path: Target directory path.
        parents: If ``True``, create missing parent directories. Defaults
            to ``True``.
        exist_ok: If ``True``, do not raise an error if the directory
            already exists. Defaults to ``True``.

    Returns:
        Resolved absolute ``Path`` to the created (or existing) directory.

    Raises:
        PathError: If the directory cannot be created (e.g., a file with
            the same name already exists, or permission is denied).

    Example:
        >>> p = ensure_directory("/tmp/pathoai_test/subdir")
        >>> p.is_dir()
        True
    """
    resolved = resolve_path(path)
    try:
        resolved.mkdir(parents=parents, exist_ok=exist_ok)
    except FileExistsError as exc:
        raise PathError(
            f"A file (not a directory) already exists at {resolved}: {exc}"
        ) from exc
    except PermissionError as exc:
        raise PathError(
            f"Permission denied when creating directory {resolved}: {exc}"
        ) from exc
    except OSError as exc:
        raise PathError(
            f"Cannot create directory {resolved}: {exc}"
        ) from exc
    logger.debug("Directory ensured: %s", resolved)
    return resolved


def ensure_file_exists(path: str | Path) -> Path:
    """Assert that a file exists and is a regular file (not a directory).

    Args:
        path: Path to the file.

    Returns:
        Resolved absolute ``Path`` to the file.

    Raises:
        PathError: If the path does not exist or is not a regular file.

    Example:
        >>> p = ensure_file_exists("/etc/hosts")
        >>> p.is_file()
        True
    """
    resolved = resolve_path(path)
    if not resolved.exists():
        raise PathError(f"Required file does not exist: {resolved}")
    if not resolved.is_file():
        raise PathError(
            f"Path exists but is not a regular file: {resolved} "
            f"(type: {'directory' if resolved.is_dir() else 'other'})"
        )
    return resolved


def ensure_parent_exists(path: str | Path) -> Path:
    """Ensure the parent directory of a file path exists, creating it if needed.

    Useful before writing a new file — guarantees the parent directory
    is present without having to call ``ensure_directory`` separately.

    Args:
        path: Path to the file whose parent directory should exist.

    Returns:
        Resolved absolute ``Path`` to the input file path (not the parent).

    Raises:
        PathError: If the parent directory cannot be created.

    Example:
        >>> p = ensure_parent_exists("/tmp/pathoai/new_file.json")
        >>> p.parent.is_dir()
        True
    """
    resolved = resolve_path(path)
    ensure_directory(resolved.parent)
    return resolved


# ---------------------------------------------------------------------------
# Project structure management
# ---------------------------------------------------------------------------

def create_project_structure(
    project_root: str | Path,
    directories: Optional[Sequence[str]] = None,
    *,
    create_gitkeep: bool = True,
) -> Dict[str, Path]:
    """Create the standard PathoAI-Platform directory structure.

    Creates all required top-level directories under ``project_root``.
    Git-ignored directories (data/, models/, logs/, etc.) receive a
    ``.gitkeep`` placeholder file so they are trackable by Git without
    committing their contents.

    Args:
        project_root: Root directory of the project.
        directories: Optional list of directory names to create. If
            ``None``, uses the canonical ``PROJECT_DIRECTORIES`` list.
        create_gitkeep: If ``True``, write a ``.gitkeep`` file into
            git-ignored directories. Defaults to ``True``.

    Returns:
        Dictionary mapping directory name → resolved absolute ``Path``.

    Raises:
        PathError: If any directory cannot be created.

    Example:
        >>> paths = create_project_structure("/tmp/my_project")
        >>> all(p.is_dir() for p in paths.values())
        True
    """
    root = resolve_path(project_root)
    dir_names: Sequence[str] = (
        directories if directories is not None else list(PROJECT_DIRECTORIES.keys())
    )

    created: Dict[str, Path] = {}
    for name in dir_names:
        target = ensure_directory(root / name)
        created[name] = target

        # Write .gitkeep for git-ignored directories
        if create_gitkeep and name in _GITIGNORED_DIRS:
            _write_gitkeep(target)

    logger.info(
        "Project directory structure ensured at %s (%d directories)",
        root,
        len(created),
    )
    return created


def _write_gitkeep(directory: Path) -> None:
    """Write (or refresh) a ``.gitkeep`` placeholder in a git-ignored directory.

    The file contains a human-readable comment explaining its purpose.
    If the file already exists, it is left unchanged to avoid noisy Git diffs.

    Args:
        directory: The directory in which to place the ``.gitkeep`` file.
    """
    gitkeep_path = directory / _GITKEEP_PLACEHOLDER
    if not gitkeep_path.exists():
        content = (
            "# Contents of this directory are excluded from version control.\n"
            "# See .gitignore and CONTRIBUTING.md §5 for the directory policy.\n"
            "# This placeholder ensures the directory is tracked by Git.\n"
        )
        gitkeep_path.write_text(content, encoding="utf-8")
        logger.debug("Created .gitkeep at %s", gitkeep_path)


def validate_project_structure(
    project_root: str | Path,
    required_dirs: Optional[Sequence[str]] = None,
) -> List[str]:
    """Validate that all required project directories exist.

    Args:
        project_root: Root directory of the project.
        required_dirs: List of directory names that must exist. If
            ``None``, checks all directories in ``PROJECT_DIRECTORIES``.

    Returns:
        List of missing directory names. Empty list means all are present.

    Example:
        >>> missing = validate_project_structure("/path/to/project")
        >>> if missing:
        ...     print(f"Missing directories: {missing}")
    """
    root = resolve_path(project_root)
    check_dirs: Sequence[str] = (
        required_dirs if required_dirs is not None else list(PROJECT_DIRECTORIES.keys())
    )

    missing = [name for name in check_dirs if not (root / name).is_dir()]

    if missing:
        logger.warning(
            "Missing project directories under %s: %s",
            root,
            ", ".join(missing),
        )
    else:
        logger.debug(
            "All %d project directories present under %s",
            len(check_dirs),
            root,
        )

    return missing


# ---------------------------------------------------------------------------
# Path listing and discovery
# ---------------------------------------------------------------------------

def list_files_with_extension(
    directory: str | Path,
    extension: str,
    *,
    recursive: bool = True,
) -> List[Path]:
    """List all files with a given extension under a directory.

    Args:
        directory: Root directory to search.
        extension: File extension to filter by, including the leading dot
            (e.g., ``".svs"``, ``".yaml"``).
        recursive: If ``True``, searches all subdirectories recursively.
            Defaults to ``True``.

    Returns:
        Sorted list of absolute ``Path`` objects matching the extension.
        Returns an empty list if the directory does not exist.

    Raises:
        PathError: If ``extension`` does not start with a dot.

    Example:
        >>> paths = list_files_with_extension("/data/slides", ".svs")
        >>> len(paths)
        42
    """
    if not extension.startswith("."):
        raise PathError(
            f"extension must start with a dot (e.g., '.svs'), got {extension!r}"
        )

    resolved = resolve_path(directory)
    if not resolved.is_dir():
        logger.warning("Directory does not exist, returning empty list: %s", resolved)
        return []

    ext_lower = extension.lower()
    pattern = f"**/*{ext_lower}" if recursive else f"*{ext_lower}"
    files = sorted(resolved.glob(pattern))

    logger.debug(
        "Found %d '%s' files in %s (recursive=%s)",
        len(files),
        extension,
        resolved,
        recursive,
    )
    return files


def safe_copy_file(
    src: str | Path,
    dst: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Copy a file, optionally creating the destination parent directory.

    Args:
        src: Source file path. Must exist and be a regular file.
        dst: Destination file path.
        overwrite: If ``False`` (default), raises ``PathError`` if the
            destination file already exists. Set to ``True`` to allow
            overwriting.

    Returns:
        Resolved absolute ``Path`` to the destination file after copying.

    Raises:
        PathError: If the source does not exist, is not a file, or the
            destination exists and ``overwrite=False``.

    Example:
        >>> dst = safe_copy_file("/src/model.pth", "/backup/model.pth")
        >>> dst.is_file()
        True
    """
    src_path = ensure_file_exists(src)
    dst_path = resolve_path(dst)

    if dst_path.exists() and not overwrite:
        raise PathError(
            f"Destination already exists and overwrite=False: {dst_path}"
        )

    ensure_directory(dst_path.parent)
    shutil.copy2(str(src_path), str(dst_path))
    logger.debug("Copied %s -> %s", src_path, dst_path)
    return dst_path


def get_file_size_bytes(path: str | Path) -> int:
    """Return the size of a file in bytes.

    Args:
        path: Path to the file.

    Returns:
        File size in bytes.

    Raises:
        PathError: If the path does not exist or is not a regular file.

    Example:
        >>> size = get_file_size_bytes("/path/to/slide.svs")
        >>> size > 0
        True
    """
    resolved = ensure_file_exists(path)
    return resolved.stat().st_size


def get_free_disk_space_gb(path: str | Path = ".") -> float:
    """Return the available free disk space at the given path in gigabytes.

    Args:
        path: A filesystem path. The free space is measured for the
            partition/drive that contains this path. Defaults to the
            current working directory.

    Returns:
        Free disk space in gigabytes (GB), rounded to 2 decimal places.

    Raises:
        PathError: If the path does not exist.

    Example:
        >>> gb = get_free_disk_space_gb("/data")
        >>> gb > 0.0
        True
    """
    resolved = resolve_path(path)
    if not resolved.exists():
        raise PathError(f"Path does not exist, cannot check disk space: {resolved}")
    try:
        usage = shutil.disk_usage(str(resolved))
    except OSError as exc:
        raise PathError(f"Cannot query disk usage at {resolved}: {exc}") from exc
    free_gb = usage.free / (1024 ** 3)
    logger.debug("Free disk space at %s: %.2f GB", resolved, free_gb)
    return round(free_gb, 2)
