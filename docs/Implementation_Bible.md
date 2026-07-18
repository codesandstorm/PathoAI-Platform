# PathoAI-Platform: Implementation Bible

> **Document Version**: 1.0.0
> **Date**: 2026-07-18
> **Authority**: This document supersedes all other coding conventions within PathoAI-Platform.
> **Scope**: Every Python file, config, test, and script in this repository.

---

## Table of Contents

1. [Python Standards](#1-python-standards)
2. [Type Annotation Requirements](#2-type-annotation-requirements)
3. [Docstring Standards](#3-docstring-standards)
4. [Naming Conventions](#4-naming-conventions)
5. [Module and Package Organization](#5-module-and-package-organization)
6. [Configuration Access Rules](#6-configuration-access-rules)
7. [Logging Rules](#7-logging-rules)
8. [Error Handling Rules](#8-error-handling-rules)
9. [Testing Standards](#9-testing-standards)
10. [NumPy and PyTorch Conventions](#10-numpy-and-pytorch-conventions)
11. [Pathology Domain Conventions](#11-pathology-domain-conventions)
12. [Git Workflow](#12-git-workflow)
13. [Commit Message Format](#13-commit-message-format)
14. [Code Review Checklist](#14-code-review-checklist)
15. [Performance Rules](#15-performance-rules)
16. [Reproducibility Requirements](#16-reproducibility-requirements)
17. [Forbidden Patterns](#17-forbidden-patterns)
18. [Approved Patterns](#18-approved-patterns)
19. [Dependency Management](#19-dependency-management)
20. [Architecture Decision Records (ADR)](#20-architecture-decision-records)

---

## 1. Python Standards

### Python Version
- **Required**: Python 3.11.x
- **Minimum**: Python 3.10 (for structural pattern matching)
- **Never**: Python < 3.10

### Style
- PEP 8 strictly enforced via `ruff` linter
- Line length: **100 characters** maximum
- Quotes: **double quotes** for strings (configured in ruff)
- Trailing commas in multi-line structures: **required**

### Formatting
```bash
# All code must pass:
ruff check pathoai/ tests/
ruff format --check pathoai/ tests/
```

### Import Order (enforced by ruff/isort):
```python
# 1. Standard library
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 2. Third-party packages
import numpy as np
import torch
import torch.nn as nn

# 3. Internal pathoai imports
from pathoai.core.config import get_config
from pathoai.core.logger import get_logger
from pathoai.core.types import WSIMetadata
```

---

## 2. Type Annotation Requirements

**Rule**: Every function signature MUST have complete type annotations. No bare `Any` unless absolutely unavoidable with a comment explaining why.

```python
# CORRECT
def extract_patches(
    slide: openslide.OpenSlide,
    coordinates: List[Tuple[int, int]],
    patch_size: int,
    level: int,
) -> torch.Tensor:
    ...

# WRONG — missing return type, missing parameter types
def extract_patches(slide, coordinates, patch_size, level):
    ...
```

### Complex Types
Use `TypeAlias` (Python 3.10+) for complex repeated types:

```python
from typing import TypeAlias

PatchCoordinate: TypeAlias = Tuple[int, int]      # (x, y) at level 0
ClassProbMap: TypeAlias = np.ndarray               # Shape: (H, W, n_classes)
BoundingBoxes: TypeAlias = np.ndarray              # Shape: (N, 4) [x1,y1,x2,y2]
```

---

## 3. Docstring Standards

**Format**: NumPy docstring style — required for all public functions, classes, and modules.

```python
def compute_stil_score(
    n_lymphocytes: int,
    stroma_area_um2: float,
    min_stroma_area: float = 0.5e6,
) -> float:
    """Compute the stromal TIL (sTIL) score for a slide or patch.

    Implements the simplified area-normalized sTIL computation following
    Salgado et al. (2015) TILs Working Group recommendations. The score
    represents the density of lymphocytes within the stromal compartment,
    normalized to unit area.

    Parameters
    ----------
    n_lymphocytes : int
        Count of lymphocytes whose centroid lies within the stromal mask.
        Must be non-negative.
    stroma_area_um2 : float
        Total area of stromal compartment in square micrometers (μm²).
        Must be positive.
    min_stroma_area : float, optional
        Minimum stroma area in μm² below which scoring is unreliable.
        Default is 0.5 mm² = 500,000 μm².

    Returns
    -------
    float
        sTIL score in the range [0.0, 100.0], representing percentage
        of stromal area occupied by lymphocytes (density-based approximation).

    Raises
    ------
    FusionError
        If stroma_area_um2 <= 0 (division by zero would result).
    DataError
        If n_lymphocytes < 0.

    Notes
    -----
    This is a density-based approximation of the TILs Working Group visual
    scoring method. The visual method estimates % area occupied; this method
    uses cell count / area and requires calibration against pathologist scores.

    References
    ----------
    Salgado, R. et al. (2015). The evaluation of tumor-infiltrating lymphocytes
    (TILs) in breast cancer. Annals of Oncology, 26(2), 259–271.
    https://doi.org/10.1093/annonc/mdu450

    Examples
    --------
    >>> score = compute_stil_score(n_lymphocytes=500, stroma_area_um2=2_000_000)
    >>> print(f"sTIL score: {score:.1f}%")
    sTIL score: 25.0%
    """
```

### Module-Level Docstrings
Every Python file must begin with a module docstring:

```python
"""
pathoai/wsi/reader.py
=====================
WSI file reader wrapping OpenSlide.

Provides a unified interface for opening Whole Slide Images in any
OpenSlide-supported format (.svs, .ndpi, .tif, .scn, .mrxs).

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1
"""
```

---

## 4. Naming Conventions

### Variables and Functions
```python
# snake_case for everything
patch_size = 512
tissue_mask = np.zeros((100, 100), dtype=bool)

def extract_tissue_patches(slide, mask, stride):
    ...
```

### Classes
```python
# PascalCase
class WSIReader:
    ...

class DeepLabV3PlusEffNetB3(BaseSegmentationModel):
    ...
```

### Constants
```python
# SCREAMING_SNAKE_CASE in constants.py ONLY
STROMA_CLASS_ID = 2
LYMPHOCYTE_CLASS_ID = 3
DEFAULT_PATCH_SIZE = 512
MIN_TISSUE_RATIO = 0.05
SEGMENTATION_TARGET_MPP = 0.50
```

### Configuration Keys
```python
# snake_case in YAML
wsi:
  patch_extraction:
    patch_size: 512
    target_mpp: 0.50
```

### File Names
```
# Python files: snake_case
wsi_reader.py
patch_extractor.py
stil_computer.py

# Test files: test_<module_name>.py
test_wsi_reader.py
test_patch_extractor.py

# Config files: snake_case.yaml
base.yaml
tiger_dataset.yaml

# Documentation: PascalCase.md
Architecture.md
Pipeline.md
```

---

## 5. Module and Package Organization

### Package Structure Rule
Every directory that is a Python package must have an `__init__.py` that:
1. Declares `__version__`, `__author__` at package root level
2. Explicitly exports public API via `__all__`
3. Does NOT perform heavy computation (no model loading, no file I/O at import time)

```python
# pathoai/wsi/__init__.py
"""pathoai.wsi — Whole Slide Image processing engine."""

from pathoai.wsi.reader import WSIReader
from pathoai.wsi.extractor import PatchExtractor
from pathoai.wsi.tissue import TissueDetector
from pathoai.wsi.coordinate_map import PatchCoordinateMap

__all__ = [
    "WSIReader",
    "PatchExtractor",
    "TissueDetector",
    "PatchCoordinateMap",
]
```

### Circular Import Prevention
- Never import from a sibling or parent engine in `__init__.py` initialization code
- Use TYPE_CHECKING guards for type-only imports:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathoai.core.types import WSIMetadata  # Only imported for type checking
```

---

## 6. Configuration Access Rules

### Rule: Never parse YAML directly in engine code

```python
# CORRECT
from pathoai.core.config import get_config

def my_function():
    cfg = get_config()
    patch_size = cfg.wsi.patch_extraction.patch_size

# WRONG — never do this
import yaml
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
patch_size = cfg["wsi"]["patch_extraction"]["patch_size"]
```

### Rule: Never use magic numbers

```python
# CORRECT
from pathoai.core.constants import STROMA_CLASS_ID

in_stroma = mask[cy, cx] == STROMA_CLASS_ID

# WRONG
in_stroma = mask[cy, cx] == 2  # What is 2? Reader shouldn't need to know.
```

### Rule: Never hardcode file paths

```python
# CORRECT
from pathoai.core.config import get_config
cfg = get_config()
output_dir = Path(cfg.output.base_dir) / slide_id

# WRONG
output_dir = Path("results") / slide_id
```

---

## 7. Logging Rules

### Rule: Always use module-level logger

```python
# At top of each module:
from pathoai.core.logger import get_logger
logger = get_logger(__name__)

# In functions:
logger.info("Extracting patches", extra={"slide_id": slide_id, "n_coords": len(coords)})
logger.warning("Low tissue ratio", extra={"ratio": tissue_ratio, "threshold": min_ratio})
logger.error("Failed to read patch", extra={"coord": (x, y), "error": str(e)})
```

### Rule: Log at appropriate levels

| Action | Level |
|--------|-------|
| Function entry with inputs | DEBUG |
| Stage start/completion | INFO |
| Performance metrics (timing) | INFO |
| Quality warnings (non-fatal) | WARNING |
| Recoverable errors (slide skipped) | ERROR |
| Pipeline-breaking errors | CRITICAL |

### Rule: Never print() in production code

```python
# WRONG
print(f"Processing slide {slide_id}")

# CORRECT
logger.info(f"Processing slide", extra={"slide_id": slide_id})
```

---

## 8. Error Handling Rules

### Rule: Raise specific exceptions, always with context

```python
# CORRECT
from pathoai.core.exceptions import WSIReadError

try:
    slide = openslide.OpenSlide(str(slide_path))
except openslide.OpenSlideError as e:
    raise WSIReadError(
        f"Cannot open WSI file: {slide_path}. "
        f"Ensure the file is not corrupted and OpenSlide supports the format."
    ) from e

# WRONG — loses original error context and is too generic
try:
    slide = openslide.OpenSlide(str(slide_path))
except Exception as e:
    print(f"Error: {e}")
    return None
```

### Rule: Use context managers for resources

```python
# CORRECT — slide is always closed, even on exception
with WSIReader(slide_path) as reader:
    metadata = reader.get_metadata()
    patches = reader.extract_patches(coords)

# WRONG — reader may leak if exception occurs
reader = WSIReader(slide_path)
metadata = reader.get_metadata()
reader.close()  # May never be called
```

---

## 9. Testing Standards

### Rule: Every public function has a test

### Rule: Tests are independent — no shared mutable state

### Rule: Tests use fixtures, not hardcoded paths

```python
# tests/unit/wsi/test_tissue_detector.py

import numpy as np
import pytest
from pathoai.wsi.tissue import TissueDetector
from tests.fixtures.synthetic_wsi import make_synthetic_thumbnail


class TestTissueDetector:
    """Unit tests for TissueDetector.

    All tests use synthetic thumbnails — no real WSI data required.
    Tests are independent and can run in any order.
    """

    @pytest.fixture
    def detector(self, base_config):
        """Create TissueDetector with test config."""
        return TissueDetector(config=base_config.wsi.tissue_detection)

    def test_detects_tissue_in_synthetic_image(self, detector):
        """Tissue detector correctly identifies tissue region in synthetic image."""
        thumbnail = make_synthetic_thumbnail(
            size=(256, 256),
            tissue_fraction=0.6,
            tissue_color=(180, 140, 180),  # Purple-pink H&E tissue
            background_color=(245, 245, 245),  # White background
        )
        mask, ratio = detector.detect(thumbnail)
        assert ratio >= 0.5, f"Expected ratio >= 0.5, got {ratio:.3f}"
        assert mask.dtype == bool
        assert mask.shape == (256, 256)

    def test_rejects_all_background_slide(self, detector):
        """Tissue detector returns near-zero ratio for blank slide."""
        blank = np.full((256, 256, 3), 250, dtype=np.uint8)
        mask, ratio = detector.detect(blank)
        assert ratio < 0.05, f"Expected ratio < 0.05, got {ratio:.3f}"

    def test_mask_is_boolean(self, detector):
        """Output mask must be boolean dtype for downstream operations."""
        thumbnail = make_synthetic_thumbnail(size=(128, 128), tissue_fraction=0.5)
        mask, _ = detector.detect(thumbnail)
        assert mask.dtype == bool, f"Expected bool dtype, got {mask.dtype}"
```

### Test Naming Convention
```
test_<what_it_does>_<expected_outcome>

Examples:
test_detects_tissue_in_synthetic_image
test_raises_wsiread_error_for_missing_file
test_returns_zero_score_for_empty_stroma
test_patch_coordinates_are_within_slide_bounds
```

### Coverage Target
- Unit tests: > 80% line coverage
- Critical path functions (sTIL equation, tissue detection): 100% branch coverage

---

## 10. NumPy and PyTorch Conventions

### Array Shape Documentation
Always document array shapes in comments when shapes are non-obvious:

```python
probs: np.ndarray  # Shape: (H, W, n_classes), float32, softmax probabilities
boxes: np.ndarray  # Shape: (N, 4), float32, [x1, y1, x2, y2] in pixel coords
```

### Device Management

```python
# CORRECT — always specify device explicitly
tensor = torch.tensor(data, device=device)
model = model.to(device)

# WRONG — implicit CPU assumption
tensor = torch.tensor(data)  # Assumed CPU — breaks on GPU
```

### Tensor vs. Array Boundaries
- **Inside engine**: Use PyTorch tensors
- **At engine output**: Convert to NumPy arrays (CPU) for serialization and inter-engine transfer
- **Never**: Pass GPU tensors between engines

```python
# At engine output boundary:
result_array = tensor.cpu().numpy()  # Explicit CPU + numpy conversion
```

### Batch Dimension Convention
```
Batch dimension is always FIRST: (B, C, H, W) for images
No implicit squeezing/unsqueezing — always be explicit:
    patch.unsqueeze(0)     # (C, H, W) → (1, C, H, W)
    output.squeeze(0)      # (1, ...) → (...)
```

---

## 11. Pathology Domain Conventions

### Coordinate Systems
PathoAI uses **two** coordinate systems that must never be mixed:

| System | Description | Units |
|--------|-------------|-------|
| **Slide coordinates** | Level 0 pixel coordinates (OpenSlide convention) | pixels at level 0 |
| **Physical coordinates** | Real-world tissue coordinates | micrometers (μm) |

```python
# Variable naming disambiguates:
x_slide: int    # pixel x at level 0
x_um: float     # x in micrometers = x_slide * mpp_x
```

### MPP (Microns Per Pixel)
- MPP is the fundamental spatial calibration constant
- All area computations must use MPP, not pixel counts alone
- MPP is stored in `WSIMetadata.mpp_x` and `WSIMetadata.mpp_y`

```python
# Area computation:
pixel_area = mask.sum()
area_um2 = pixel_area * (mpp_x * mpp_y)      # μm²
area_mm2 = area_um2 / 1e6                     # mm²
```

### Magnification vs. MPP
- Use **MPP** for all computations (objective magnification alone is insufficient)
- Magnification is only used for display labels and user-facing messages
- Different scanners at "40×" may have different MPPs (0.225 vs. 0.252)

---

## 12. Git Workflow

### Branch Strategy

```
main           ← Production-ready, tagged releases only
  └── develop  ← Integration branch
        └── milestone/m1-infrastructure  ← Per-milestone feature branch
              └── feat/wsi-reader        ← Per-feature branches
              └── fix/tissue-detection-threshold
              └── test/patch-extractor-unit-tests
```

### Branch Naming

```
feat/<short-description>       ← New features
fix/<short-description>        ← Bug fixes
test/<short-description>       ← Tests only
docs/<short-description>       ← Documentation only
refactor/<short-description>   ← Refactoring (no behavior change)
milestone/m<N>-<name>          ← Milestone integration branch
```

---

## 13. Commit Message Format

### Format: Conventional Commits

```
<type>(<scope>): <short description>

<body — optional, wrap at 72 chars>

<footer — optional: refs, breaking changes>
```

### Types:
```
feat      — New feature
fix       — Bug fix
docs      — Documentation change
test      — Test additions/changes
refactor  — Code restructure (no behavior change)
perf      — Performance improvement
config    — Configuration change
chore     — Maintenance (deps, CI, tooling)
```

### Scopes (PathoAI-specific):
```
core, wsi, segmentation, detection, fusion, validation, viz, report, dashboard
config, docs, tests, ci, deps
```

### Examples:

```
feat(wsi): implement tissue detection via Otsu HSV thresholding

Add TissueDetector class in pathoai/wsi/tissue.py implementing
the Otsu-based tissue/background separation in HSV color space
following computational pathology best practices.

- Converts thumbnail from RGB to HSV
- Applies Otsu threshold to saturation channel
- Applies morphological cleanup (open + close)
- Removes small connected components below MIN_SIZE
- Returns binary mask and tissue ratio

Refs: Architecture.md#stage-4-tissue-detection

---

fix(wsi): correct MPP extraction for Leica SCN format

SCN files store MPP in different OpenSlide property keys than SVS.
Add fallback property lookup chain for missing MPP values.

Closes #12

---

docs(core): add Implementation_Bible.md

Complete coding standards and conventions document for all
PathoAI-Platform contributors.

Milestone: M1
```

---

## 14. Code Review Checklist

Before merging any PR, verify:

### Architecture
- [ ] No engine imports another engine's internals (only types from core)
- [ ] No circular imports
- [ ] New classes implement the appropriate base class / protocol

### Code Quality
- [ ] All public functions have complete type annotations
- [ ] All public functions have NumPy-format docstrings
- [ ] No magic numbers (use constants.py)
- [ ] No hardcoded file paths (use config)
- [ ] No print() statements (use logger)
- [ ] No bare except: clauses

### Testing
- [ ] All new public functions have unit tests
- [ ] Tests use fixtures, not real WSI data
- [ ] Test names follow naming convention
- [ ] All tests pass locally

### Git
- [ ] Commits follow Conventional Commits format
- [ ] No large binary files committed (> 1 MB)
- [ ] requirements.txt updated if new dependency added

---

## 15. Performance Rules

### Rule: Profile before optimizing

```python
# Use built-in timers for logging, not for optimization decisions
import time
start = time.perf_counter()
result = expensive_operation()
elapsed = time.perf_counter() - start
logger.info("Operation complete", extra={"elapsed_s": elapsed})
```

### Rule: Use vectorized operations (NumPy/PyTorch), never Python loops on arrays

```python
# CORRECT — vectorized
centroids = (boxes[:, :2] + boxes[:, 2:]) / 2.0

# WRONG — Python loop
centroids = []
for box in boxes:
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    centroids.append((cx, cy))
```

### Rule: Read patches lazily (on demand), never preload entire WSI

```python
# CORRECT — lazy loading in Dataset.__getitem__
class PatchDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        x, y = self.coords[idx]
        region = self.slide.read_region((x, y), self.level, (self.size, self.size))
        return np.array(region)[:, :, :3]  # Loaded only when needed
```

### Rule: Close all OpenSlide handles when done

```python
# Always use context manager or try/finally
with WSIReader(slide_path) as reader:
    ...  # reader.slide is closed automatically on exit
```

---

## 16. Reproducibility Requirements

### Rule: Set all random seeds at experiment start

```python
# pathoai/core/reproducibility.py
import random
import numpy as np
import torch

def set_global_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility.

    Must be called at the start of every training and inference run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable for reproducibility
```

### Rule: Record environment at experiment start

```python
# Automatically captured in experiment manifest:
{
  "python_version": "3.11.3",
  "torch_version": "2.x.x+cpu",
  "numpy_version": "1.x.x",
  "platform": "Windows-11-26200",
  "cuda_available": false,
  "config_hash": "sha256:...",
  "git_commit": "abc123...",
  "timestamp": "2026-07-18T17:09:45+05:30"
}
```

### Rule: Use deterministic DataLoader

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

loader = DataLoader(
    dataset,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=torch.Generator().manual_seed(cfg.pipeline.seed),
)
```

---

## 17. Forbidden Patterns

The following are **strictly prohibited** in all PathoAI code:

```python
# ❌ FORBIDDEN: Bare except
try:
    ...
except:  # Catches everything including KeyboardInterrupt
    pass

# ❌ FORBIDDEN: Mutable default arguments
def process(patches, results=[]):  # Bug: list persists across calls

# ❌ FORBIDDEN: Global mutable state
GLOBAL_SLIDE = None  # Race condition risk; untestable

# ❌ FORBIDDEN: Hardcoded paths
mask_path = "C:\\Users\\ASUS\\data\\masks\\slide001.npy"

# ❌ FORBIDDEN: Magic numbers
if ratio > 0.05:  # What is 0.05? Use MIN_TISSUE_RATIO

# ❌ FORBIDDEN: Silent failure
def load_slide(path):
    try:
        return openslide.OpenSlide(path)
    except:
        return None  # Caller gets None with no explanation

# ❌ FORBIDDEN: Modifying input arrays in-place without documentation
def normalize(patch):
    patch /= 255.0  # Mutates caller's array — use patch.copy() or be explicit

# ❌ FORBIDDEN: Mixed coordinate systems without conversion
x_um = x_slide  # Missing * mpp_x conversion — produces wrong area computation

# ❌ FORBIDDEN: GPU tensors as function return values across engine boundaries
def get_mask(patch) -> torch.Tensor:
    return model(patch)  # Returns GPU tensor — should be .cpu().numpy()
```

---

## 18. Approved Patterns

### Context Manager for Resources

```python
class WSIReader:
    def __enter__(self):
        self._slide = openslide.OpenSlide(str(self._path))
        return self

    def __exit__(self, *args):
        if self._slide:
            self._slide.close()
            self._slide = None
```

### Dataclass for Data Contracts

```python
from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)  # frozen=True for immutable contracts
class WSIMetadata:
    slide_id: str
    mpp_x: float
    mpp_y: float
    width: int
    height: int
    # ... etc.
```

### Factory Function for Complex Objects

```python
def create_patch_dataset(
    slide_path: Path,
    config: PatchExtractionConfig,
    device: torch.device,
) -> PatchDataset:
    """Factory for PatchDataset — handles all validation and setup."""
    ...
```

### Strategy Pattern for Pluggable Algorithms

```python
class BaseStainNormalizer(ABC):
    @abstractmethod
    def fit(self, reference: np.ndarray) -> None: ...

    @abstractmethod
    def transform(self, patch: np.ndarray) -> np.ndarray: ...

class MacenkoNormalizer(BaseStainNormalizer):
    def fit(self, reference: np.ndarray) -> None: ...
    def transform(self, patch: np.ndarray) -> np.ndarray: ...
```

---

## 19. Dependency Management

### requirements.txt format

```
# PathoAI-Platform requirements
# Generated: 2026-07-18
# Python: 3.11
# Platform: Windows 11 x64

# Core scientific computing
numpy>=1.24.0,<2.0.0
scipy>=1.10.0

# Image processing
Pillow>=9.5.0
opencv-python>=4.8.0
scikit-image>=0.21.0
albumentations>=1.3.0

# Deep learning
torch>=2.0.0          # CPU version or CUDA version — see INSTALL.md
torchvision>=0.15.0
timm>=0.9.0

# WSI reading
openslide-python>=1.3.0    # Requires OpenSlide binaries — see INSTALL.md

# Segmentation models (reference/baseline)
segmentation-models-pytorch>=0.3.3

# Data science
pandas>=2.0.0
scikit-learn>=1.3.0

# Configuration
PyYAML>=6.0.1

# Experiment tracking
tensorboard>=2.13.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Notebooks
jupyterlab>=4.0.0

# Progress display
tqdm>=4.66.0

# Utilities
python-dateutil>=2.8.2
```

### Adding a New Dependency
1. Verify it is necessary (no existing package provides this)
2. Check license compatibility (must be permissive or LGPL minimum)
3. Add to `requirements.txt` with version bounds
4. Document why it was added in the PR description
5. Run `pip-audit` to check for known vulnerabilities

### Pinning Strategy
- Development: Use `>=` with upper bounds for major version compatibility
- Production/reproduction: Use `pip freeze > requirements.lock` and commit the lock file

---

## 20. Architecture Decision Records (ADR)

All significant architecture decisions must be documented as ADRs in `docs/adr/`.

### ADR Format

```markdown
# ADR-001: Use OpenSlide as Primary WSI Reader

**Date**: 2026-07-18
**Status**: Accepted
**Deciders**: PathoAI Research Team

## Context
We need a library to read Whole Slide Images in multiple formats from different scanner vendors.

## Decision
Use the OpenSlide library (via `openslide-python`) as the primary WSI reader.

## Rationale
1. Supports all major scanner formats: SVS (Aperio), NDPI (Hamamatsu), SCN (Leica), MRXS (3DHISTECH), generic TIFF
2. Well-maintained, battle-tested in computational pathology community
3. Python bindings available via openslide-python
4. Free and open source (LGPL 2.1)
5. Used by leading computational pathology frameworks (HistoQC, CLAM, etc.)

## Rejected Alternatives
- **tifffile**: Only supports TIFF format — insufficient scanner coverage
- **pyvips**: High performance but less community adoption in pathology
- **QuPath Java integration**: Excessive complexity for a Python-first pipeline

## Consequences
- Requires OpenSlide C binaries installed on OS (not just pip install)
- Windows installation requires manual DLL management (documented in INSTALL.md)
- If OpenSlide becomes unmaintained, migration path exists to tifffile for TIFF-only slides
```

### Current ADRs
- `docs/adr/ADR-001-openslide-reader.md`
- `docs/adr/ADR-002-deeplabv3plus-segmentation.md`
- `docs/adr/ADR-003-faster-rcnn-detection.md`
- `docs/adr/ADR-004-tiger-dataset.md`
- `docs/adr/ADR-005-cpu-first-design.md`
- `docs/adr/ADR-006-pytorch-over-tensorflow.md`

---

*End of Implementation Bible v1.0.0*
*This document is the law of this repository. When in doubt, consult this document.*
