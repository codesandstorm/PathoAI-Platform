"""
tests/unit/core/utils/test_path_utils.py
==========================================
Unit tests for pathoai.core.utils.path_utils.

Tests cover:
- resolve_path: normalization, symlink resolution, user home expansion
- ensure_directory: creation, idempotency, error cases
- ensure_file_exists: happy path, missing file, directory-not-file
- ensure_parent_exists: creates parent, returns file path
- create_project_structure: creates all dirs, gitkeep for ignored dirs
- validate_project_structure: detects missing dirs, all-present case
- list_files_with_extension: recursive and non-recursive, extension filter
- safe_copy_file: copies correctly, overwrite semantics
- get_file_size_bytes: correct size, error on missing file
- get_free_disk_space_gb: returns positive float, error on bad path
- PathError: is-a PathoAIException

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1.1
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pathoai.core.exceptions import PathoAIException
from pathoai.core.utils.path_utils import (
    PathError,
    create_project_structure,
    ensure_directory,
    ensure_file_exists,
    ensure_parent_exists,
    get_file_size_bytes,
    get_free_disk_space_gb,
    list_files_with_extension,
    resolve_path,
    safe_copy_file,
    validate_project_structure,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_project(tmp_path: Path) -> Path:
    """Return a temporary directory that acts as a project root."""
    return tmp_path


@pytest.fixture()
def populated_dir(tmp_path: Path) -> Path:
    """Create a nested directory with test files of various extensions."""
    root = tmp_path / "slides"
    root.mkdir()

    # Top-level files
    (root / "slide_a.svs").write_text("dummy")
    (root / "slide_b.svs").write_text("dummy")
    (root / "report.pdf").write_text("dummy")

    # Nested subdir with more files
    sub = root / "subdir"
    sub.mkdir()
    (sub / "slide_c.svs").write_text("dummy")
    (sub / "meta.json").write_text("{}")

    return root


# ---------------------------------------------------------------------------
# PathError
# ---------------------------------------------------------------------------

class TestPathError:
    """PathError must be a subclass of PathoAIException."""

    def test_is_pathoai_exception(self):
        """PathError must inherit from PathoAIException."""
        err = PathError("something went wrong")
        assert isinstance(err, PathoAIException)

    def test_can_be_raised_and_caught_as_pathoai_exception(self):
        """PathError must be catchable as PathoAIException."""
        with pytest.raises(PathoAIException):
            raise PathError("test error")

    def test_message_is_preserved(self):
        """Error message must be accessible as a string."""
        msg = "disk is full"
        err = PathError(msg)
        assert msg in str(err)


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------

class TestResolvePath:
    """Tests for resolve_path()."""

    def test_resolves_relative_path(self, tmp_path: Path):
        """A relative path should resolve to an absolute path."""
        os.chdir(tmp_path)
        result = resolve_path("subdir")
        assert result.is_absolute()

    def test_resolves_home_tilde(self):
        """Tilde ~ should expand to the user's home directory."""
        result = resolve_path("~")
        assert result == Path.home().resolve()

    def test_resolves_absolute_path(self, tmp_path: Path):
        """An already-absolute path should be returned as resolved."""
        result = resolve_path(str(tmp_path))
        assert result == tmp_path.resolve()

    def test_accepts_path_object(self, tmp_path: Path):
        """Should accept pathlib.Path objects in addition to strings."""
        result = resolve_path(tmp_path)
        assert isinstance(result, Path)
        assert result == tmp_path.resolve()

    def test_normalizes_double_slashes(self, tmp_path: Path):
        """Double slashes and .. components should be normalized."""
        messy = str(tmp_path) + "/foo/../bar"
        result = resolve_path(messy)
        assert ".." not in str(result)
        assert "//" not in str(result)

    def test_raises_on_empty_string(self):
        """Empty string should raise PathError."""
        with pytest.raises(PathError, match="non-empty"):
            resolve_path("")


# ---------------------------------------------------------------------------
# ensure_directory
# ---------------------------------------------------------------------------

class TestEnsureDirectory:
    """Tests for ensure_directory()."""

    def test_creates_new_directory(self, tmp_path: Path):
        """Must create a directory that does not yet exist."""
        target = tmp_path / "new_dir"
        result = ensure_directory(target)
        assert result.is_dir()

    def test_creates_nested_directories(self, tmp_path: Path):
        """Must create all missing parent directories when parents=True."""
        target = tmp_path / "a" / "b" / "c"
        result = ensure_directory(target)
        assert result.is_dir()

    def test_idempotent_on_existing_directory(self, tmp_path: Path):
        """Calling ensure_directory on an existing directory must not raise."""
        target = tmp_path / "already_exists"
        target.mkdir()
        result = ensure_directory(target)  # should not raise
        assert result.is_dir()

    def test_returns_absolute_path(self, tmp_path: Path):
        """Return value must always be an absolute Path."""
        result = ensure_directory(tmp_path / "x")
        assert result.is_absolute()

    def test_raises_when_file_exists_at_path(self, tmp_path: Path):
        """Must raise PathError if a regular file occupies the path."""
        conflict = tmp_path / "file_not_dir"
        conflict.write_text("I am a file")
        with pytest.raises(PathError):
            ensure_directory(conflict, exist_ok=False)

    def test_raises_on_permission_error(self, tmp_path: Path, monkeypatch):
        """Must raise PathError on PermissionError from mkdir."""
        def mock_mkdir(*args, **kwargs):
            raise PermissionError("access denied")

        monkeypatch.setattr(Path, "mkdir", mock_mkdir)
        with pytest.raises(PathError, match="Permission denied"):
            ensure_directory(tmp_path / "blocked")


# ---------------------------------------------------------------------------
# ensure_file_exists
# ---------------------------------------------------------------------------

class TestEnsureFileExists:
    """Tests for ensure_file_exists()."""

    def test_returns_path_for_existing_file(self, tmp_path: Path):
        """Must return a resolved Path for an existing regular file."""
        f = tmp_path / "test.txt"
        f.write_text("content")
        result = ensure_file_exists(f)
        assert result.is_file()
        assert result.is_absolute()

    def test_raises_on_missing_file(self, tmp_path: Path):
        """Must raise PathError when file does not exist."""
        with pytest.raises(PathError, match="does not exist"):
            ensure_file_exists(tmp_path / "ghost.txt")

    def test_raises_when_path_is_directory(self, tmp_path: Path):
        """Must raise PathError when the path points to a directory."""
        with pytest.raises(PathError, match="not a regular file"):
            ensure_file_exists(tmp_path)


# ---------------------------------------------------------------------------
# ensure_parent_exists
# ---------------------------------------------------------------------------

class TestEnsureParentExists:
    """Tests for ensure_parent_exists()."""

    def test_creates_parent_of_nonexistent_file(self, tmp_path: Path):
        """Must create all parent directories of a not-yet-existing file."""
        target = tmp_path / "deep" / "nested" / "output.json"
        result = ensure_parent_exists(target)
        assert result.parent.is_dir()

    def test_returns_file_path_not_parent(self, tmp_path: Path):
        """Return value must be the file path, not the parent directory."""
        target = tmp_path / "subdir" / "file.json"
        result = ensure_parent_exists(target)
        assert result == target.resolve()

    def test_idempotent_when_parent_exists(self, tmp_path: Path):
        """Must not raise if parent already exists."""
        target = tmp_path / "output.json"
        ensure_parent_exists(target)  # parent is tmp_path — always exists


# ---------------------------------------------------------------------------
# create_project_structure
# ---------------------------------------------------------------------------

class TestCreateProjectStructure:
    """Tests for create_project_structure()."""

    def test_creates_all_standard_directories(self, tmp_project: Path):
        """All directories in PROJECT_DIRECTORIES must be created."""
        from pathoai.core.utils.path_utils import PROJECT_DIRECTORIES
        paths = create_project_structure(tmp_project)
        for name in PROJECT_DIRECTORIES:
            assert (tmp_project / name).is_dir(), f"Missing: {name}"

    def test_returns_dict_of_absolute_paths(self, tmp_project: Path):
        """Return value must be a dict mapping name → absolute Path."""
        paths = create_project_structure(tmp_project)
        assert isinstance(paths, dict)
        for name, p in paths.items():
            assert isinstance(p, Path), f"Value for {name} is not a Path"
            assert p.is_absolute(), f"Path for {name} is not absolute"

    def test_creates_gitkeep_in_ignored_dirs(self, tmp_project: Path):
        """Git-ignored directories must receive a .gitkeep placeholder."""
        from pathoai.core.utils.path_utils import _GITIGNORED_DIRS
        create_project_structure(tmp_project, create_gitkeep=True)
        for name in _GITIGNORED_DIRS:
            gitkeep = tmp_project / name / ".gitkeep"
            assert gitkeep.is_file(), f"Missing .gitkeep in {name}/"

    def test_no_gitkeep_when_disabled(self, tmp_project: Path):
        """When create_gitkeep=False, no .gitkeep files should be created."""
        from pathoai.core.utils.path_utils import _GITIGNORED_DIRS
        create_project_structure(tmp_project, create_gitkeep=False)
        for name in _GITIGNORED_DIRS:
            gitkeep = tmp_project / name / ".gitkeep"
            assert not gitkeep.exists(), f"Unexpected .gitkeep in {name}/"

    def test_idempotent_on_existing_structure(self, tmp_project: Path):
        """Calling twice must not raise and must not corrupt existing structure."""
        create_project_structure(tmp_project)
        create_project_structure(tmp_project)  # second call — must not raise

    def test_custom_directory_list(self, tmp_project: Path):
        """Must create only the directories explicitly provided."""
        paths = create_project_structure(tmp_project, directories=["data", "logs"])
        assert set(paths.keys()) == {"data", "logs"}
        assert (tmp_project / "data").is_dir()
        assert (tmp_project / "logs").is_dir()


# ---------------------------------------------------------------------------
# validate_project_structure
# ---------------------------------------------------------------------------

class TestValidateProjectStructure:
    """Tests for validate_project_structure()."""

    def test_returns_empty_list_when_all_dirs_present(self, tmp_project: Path):
        """Must return an empty list when all required directories exist."""
        create_project_structure(tmp_project)
        missing = validate_project_structure(tmp_project)
        assert missing == []

    def test_returns_names_of_missing_dirs(self, tmp_project: Path):
        """Must list directories that are absent from the project root."""
        # Create only a subset
        (tmp_project / "pathoai").mkdir()
        (tmp_project / "tests").mkdir()
        missing = validate_project_structure(
            tmp_project, required_dirs=["pathoai", "tests", "data", "logs"]
        )
        assert set(missing) == {"data", "logs"}

    def test_returns_all_dirs_missing_on_empty_root(self, tmp_project: Path):
        """Empty root → all directories in PROJECT_DIRECTORIES are missing."""
        from pathoai.core.utils.path_utils import PROJECT_DIRECTORIES
        missing = validate_project_structure(tmp_project)
        assert set(missing) == set(PROJECT_DIRECTORIES.keys())

    def test_custom_required_dirs(self, tmp_project: Path):
        """Must respect a custom list of required directory names."""
        (tmp_project / "pathoai").mkdir()
        missing = validate_project_structure(tmp_project, required_dirs=["pathoai", "models"])
        assert missing == ["models"]


# ---------------------------------------------------------------------------
# list_files_with_extension
# ---------------------------------------------------------------------------

class TestListFilesWithExtension:
    """Tests for list_files_with_extension()."""

    def test_finds_files_recursively(self, populated_dir: Path):
        """Should find files in root and all subdirectories by default."""
        files = list_files_with_extension(populated_dir, ".svs")
        assert len(files) == 3  # slide_a, slide_b, slide_c

    def test_non_recursive_finds_only_top_level(self, populated_dir: Path):
        """With recursive=False, must not descend into subdirectories."""
        files = list_files_with_extension(populated_dir, ".svs", recursive=False)
        assert len(files) == 2  # slide_a, slide_b

    def test_returns_sorted_list(self, populated_dir: Path):
        """Results must be sorted for deterministic ordering."""
        files = list_files_with_extension(populated_dir, ".svs")
        assert files == sorted(files)

    def test_returns_empty_list_for_missing_extension(self, populated_dir: Path):
        """Must return an empty list if no files match the extension."""
        files = list_files_with_extension(populated_dir, ".ndpi")
        assert files == []

    def test_returns_empty_list_for_missing_directory(self, tmp_path: Path):
        """Must return an empty list (not raise) if the directory is absent."""
        files = list_files_with_extension(tmp_path / "nonexistent", ".svs")
        assert files == []

    def test_raises_on_extension_without_dot(self, populated_dir: Path):
        """Must raise PathError if extension does not start with a dot."""
        with pytest.raises(PathError, match="must start with a dot"):
            list_files_with_extension(populated_dir, "svs")

    def test_all_returned_paths_are_absolute(self, populated_dir: Path):
        """Every returned Path must be absolute."""
        files = list_files_with_extension(populated_dir, ".svs")
        assert all(f.is_absolute() for f in files)


# ---------------------------------------------------------------------------
# safe_copy_file
# ---------------------------------------------------------------------------

class TestSafeCopyFile:
    """Tests for safe_copy_file()."""

    def test_copies_file_to_new_location(self, tmp_path: Path):
        """Must copy source file to the destination."""
        src = tmp_path / "source.txt"
        src.write_text("hello pathoai")
        dst = tmp_path / "dest" / "copy.txt"
        result = safe_copy_file(src, dst)
        assert result.is_file()
        assert result.read_text() == "hello pathoai"

    def test_creates_destination_parent(self, tmp_path: Path):
        """Must create destination parent directory if it does not exist."""
        src = tmp_path / "a.txt"
        src.write_text("data")
        dst = tmp_path / "new" / "subdir" / "b.txt"
        safe_copy_file(src, dst)
        assert dst.parent.is_dir()

    def test_raises_if_destination_exists_without_overwrite(self, tmp_path: Path):
        """Must raise PathError if destination exists and overwrite=False."""
        src = tmp_path / "s.txt"
        src.write_text("original")
        dst = tmp_path / "d.txt"
        dst.write_text("existing")
        with pytest.raises(PathError, match="overwrite=False"):
            safe_copy_file(src, dst, overwrite=False)

    def test_overwrites_when_flag_is_true(self, tmp_path: Path):
        """Must overwrite destination when overwrite=True."""
        src = tmp_path / "src.txt"
        src.write_text("new content")
        dst = tmp_path / "dst.txt"
        dst.write_text("old content")
        safe_copy_file(src, dst, overwrite=True)
        assert dst.read_text() == "new content"

    def test_raises_on_missing_source(self, tmp_path: Path):
        """Must raise PathError if the source file does not exist."""
        with pytest.raises(PathError):
            safe_copy_file(tmp_path / "ghost.txt", tmp_path / "copy.txt")

    def test_returns_absolute_destination_path(self, tmp_path: Path):
        """Return value must be the absolute resolved destination path."""
        src = tmp_path / "src.txt"
        src.write_text("x")
        dst = tmp_path / "dst.txt"
        result = safe_copy_file(src, dst)
        assert result.is_absolute()


# ---------------------------------------------------------------------------
# get_file_size_bytes
# ---------------------------------------------------------------------------

class TestGetFileSizeBytes:
    """Tests for get_file_size_bytes()."""

    def test_returns_correct_size(self, tmp_path: Path):
        """Must return the actual file size in bytes."""
        content = b"PathoAI test content"
        f = tmp_path / "test.bin"
        f.write_bytes(content)
        size = get_file_size_bytes(f)
        assert size == len(content)

    def test_returns_zero_for_empty_file(self, tmp_path: Path):
        """Must return 0 for an empty file."""
        f = tmp_path / "empty.txt"
        f.write_text("")
        assert get_file_size_bytes(f) == 0

    def test_raises_on_missing_file(self, tmp_path: Path):
        """Must raise PathError for a non-existent file."""
        with pytest.raises(PathError):
            get_file_size_bytes(tmp_path / "no_such_file.txt")

    def test_returns_int(self, tmp_path: Path):
        """Return type must be int."""
        f = tmp_path / "int_test.txt"
        f.write_text("abc")
        assert isinstance(get_file_size_bytes(f), int)


# ---------------------------------------------------------------------------
# get_free_disk_space_gb
# ---------------------------------------------------------------------------

class TestGetFreeDiskSpaceGb:
    """Tests for get_free_disk_space_gb()."""

    def test_returns_positive_float(self, tmp_path: Path):
        """Must return a positive float for a valid path."""
        gb = get_free_disk_space_gb(tmp_path)
        assert isinstance(gb, float)
        assert gb > 0.0

    def test_default_path_works(self):
        """Must work with the default path (current working directory)."""
        gb = get_free_disk_space_gb()
        assert gb > 0.0

    def test_raises_on_nonexistent_path(self, tmp_path: Path):
        """Must raise PathError if the path does not exist."""
        with pytest.raises(PathError, match="does not exist"):
            get_free_disk_space_gb(tmp_path / "nonexistent")

    def test_result_is_rounded_to_two_decimal_places(self, tmp_path: Path):
        """Result must be rounded to 2 decimal places."""
        gb = get_free_disk_space_gb(tmp_path)
        assert gb == round(gb, 2)
