# Contributing to PathoAI-Platform

> **This document is non-negotiable.** Every contributor — human or AI — must follow
> these rules without exception. They exist to keep the repository lean, reproducible,
> and suitable for open-source publication in a research context.

---

## Table of Contents

1. [Repository Philosophy](#1-repository-philosophy)
2. [Development Workflow](#2-development-workflow)
3. [Git Rules](#3-git-rules)
4. [Code Quality Standards](#4-code-quality-standards)
5. [Directory Policy](#5-directory-policy)
6. [Pre-Commit Checklist](#6-pre-commit-checklist)
7. [Commit Message Format](#7-commit-message-format)
8. [AI Assistant Guidelines](#8-ai-assistant-guidelines)

---

## 1. Repository Philosophy

PathoAI-Platform is a **research-grade, publication-ready** computational pathology
platform. The repository must always reflect that standard.

### What belongs in this repository

| Category | Examples |
|---|---|
| Source code | `pathoai/**/*.py` |
| Configuration | `configs/*.yaml` |
| Tests | `tests/**/*.py` |
| Scripts | `scripts/*.py` |
| Documentation | `docs/**/*.md` |
| Requirements | `requirements.txt`, `pyproject.toml` |
| CI/CD | `.github/workflows/*.yml` |

### What never belongs in this repository

| Category | Examples |
|---|---|
| Raw WSIs | `.svs`, `.tiff`, `.ndpi`, `.mrxs`, `.scn` |
| Processed images | `.png`, `.jpg` patches and masks at scale |
| Datasets | Any directory under `data/` beyond `.gitkeep` |
| Trained models | `.pth`, `.pt`, `.ckpt`, `.onnx`, `.h5` |
| Experiment outputs | `outputs/`, `results/`, `reports/` |
| TensorBoard logs | `runs/`, `tb_logs/`, `tensorboard_logs/` |
| Checkpoints | `checkpoints/` |
| Large archives | `.zip`, `.tar`, `.gz`, `.tar.gz` |
| Cached data | `__pycache__/`, `.cache/`, `.pytest_cache/` |

> **A healthy source-only repository** should remain small (< 50 MB). If your
> repository grows unexpectedly large, stop and audit it before your next push.

---

## 2. Development Workflow

Every sub-milestone follows this exact cycle:

```
Implement
    |
    v
Local Testing  (pytest, linting)
    |
    v
Codex Review   (code quality)
    |
    v
Claude Review  (research correctness)
    |
    v
Architecture Review  (manual)
    |
    v
Fix Issues
    |
    v
Run Unit Tests  (must pass 100%)
    |
    v
git commit
    |
    v
git push        <- after every sub-milestone, not after every milestone
```

### Frequency

Push after **every completed sub-milestone**, not at the end of a larger milestone.

```
# Correct cadence
Milestone 1.1 -> commit -> push
Milestone 1.2 -> commit -> push
Milestone 1.3 -> commit -> push

# Never do this
Milestone 1.1
Milestone 1.2
Milestone 1.3
-> single push
```

---

## 3. Git Rules

### Rule 1 — Push Frequently

See [Development Workflow](#2-development-workflow). Push after every sub-milestone.

### Rule 2 — Never Commit Large Files

The following file types must **never** be tracked by Git:

```
# Whole Slide Images
*.svs  *.ndpi  *.scn  *.mrxs  *.tif  *.tiff  *.vms  *.vmu  *.bif

# Model Artifacts
*.pth  *.pt  *.ckpt  *.onnx  *.h5  *.pkl  *.safetensors

# Data Archives
*.zip  *.tar  *.gz  *.tar.gz  *.7z  *.bz2

# Large Numerical Arrays
*.npy  *.npz  *.hdf5
```

### Rule 3 — Keep the Repository Source-Code-Only

```
PathoAI/
|
|-- src/          [Git - always]
|-- configs/      [Git - always]
|-- docs/         [Git - always]
|-- tests/        [Git - always]
|-- scripts/      [Git - always]
|
|-- data/         [Only .gitkeep tracked]
|-- models/       [Only .gitkeep tracked]
|-- outputs/      [Only .gitkeep tracked]
|-- logs/         [Only .gitkeep tracked]
|-- checkpoints/  [Only .gitkeep tracked]
```

### Rule 4 — Verify Before Every Push

Run these three commands before every `git push`:

```bash
# 1. Check working tree status
git status

# 2. Review what changed
git diff --stat

# 3. Verify tracked files do not include large or sensitive data
git ls-files | grep -E "\.(svs|tiff|ndpi|pth|pt|ckpt|zip|tar|gz)$"
# The above command must return ZERO results.
```

If you see any datasets, model weights, slide images, or binary blobs in the tracked
files list — **stop immediately**, remove them with `git rm --cached`, and update
`.gitignore` before proceeding.

### Rule 5 — Repository Size Budget

| Threshold | Action |
|---|---|
| < 50 MB | Healthy |
| 50-200 MB | Audit — find and remove unintended large files |
| > 200 MB | Stop — mandatory investigation before any further pushes |

Check repository size at any time:

```bash
git count-objects -vH
```

### Rule 6 — Stage Interactively

Never use `git add .` — always stage interactively to review every change:

```bash
git add -p   # review every hunk before staging
```

### Rule 7 — Keep Git History Clean

Never rewrite shared history with `--force-push` on the main branch. If you need
to fix a mistake, use `git revert` to create a corrective commit.

---

## 4. Code Quality Standards

Every module must meet all of the following standards before being committed.

### 4.1 Type Hints

All public functions, methods, and module-level variables must have complete type
annotations using Python's `typing` module or built-in generics (Python >= 3.10).

```python
# Correct
def compute_stil_score(
    lymphocyte_area: float,
    stroma_area: float,
    confidence_threshold: float = 0.50,
) -> float:
    ...

# Incorrect — missing type hints
def compute_stil_score(lymphocyte_area, stroma_area, confidence_threshold=0.50):
    ...
```

### 4.2 Google-Style Docstrings

Every public class, method, and function requires a Google-style docstring.

```python
def compute_stil_score(
    lymphocyte_area: float,
    stroma_area: float,
    confidence_threshold: float = 0.50,
) -> float:
    """Compute the stromal TIL (sTIL) score per Salgado et al. (2015).

    The sTIL score is defined as the ratio of lymphocyte-occupied stroma area
    to total stroma area, expressed as a percentage.

    Args:
        lymphocyte_area: Area occupied by lymphocytes in the stromal
            compartment, in micrometers squared.
        stroma_area: Total stromal compartment area, in micrometers squared.
        confidence_threshold: Minimum score below which the result is
            flagged as LOW_CONFIDENCE. Defaults to 0.50.

    Returns:
        sTIL score in the range [0.0, 100.0].

    Raises:
        ValueError: If stroma_area <= 0 or either area is negative.
        PathoAIValidationError: If inputs fail scientific plausibility checks.

    Example:
        >>> score = compute_stil_score(lymphocyte_area=1000.0, stroma_area=5000.0)
        >>> assert 0.0 <= score <= 100.0
    """
```

### 4.3 Logging

Use the PathoAI structured logger — never `print()`.

```python
# Correct
from pathoai.core.logger import get_logger
logger = get_logger(__name__)
logger.info("Processing slide", extra={"slide_id": slide_id})

# Never use print
print(f"Processing slide {slide_id}")
```

### 4.4 Input Validation

Validate all public function inputs at entry. Raise domain-specific exceptions from
`pathoai.core.exceptions` — never raw `ValueError` or `TypeError` in production code.

```python
from pathoai.core.exceptions import PathoAIValidationError

def process_patch(patch: np.ndarray) -> np.ndarray:
    if patch.ndim != 3:
        raise PathoAIValidationError(
            f"patch must be a 3-D array (H, W, C), got shape {patch.shape}"
        )
```

### 4.5 No Magic Numbers

All constants must be defined in `pathoai/core/constants.py`. Direct use of numeric
literals in business logic is forbidden.

```python
# Correct
from pathoai.core.constants import DEFAULT_DETECTION_CONFIDENCE
if score < DEFAULT_DETECTION_CONFIDENCE:
    ...

# Forbidden
if score < 0.5:
    ...
```

### 4.6 No Placeholders or TODOs

Do not commit code containing:

- `TODO`, `FIXME`, `HACK`, `XXX` comments (unless explicitly planned for future milestone)
- `pass` in non-abstract methods
- `raise NotImplementedError` in non-abstract methods
- Stub functions that silently return incorrect values

### 4.7 Linting

All code must pass `ruff` with zero errors before commit:

```bash
ruff check pathoai/ tests/
ruff format --check pathoai/ tests/
```

---

## 5. Directory Policy

### 5.1 Placeholder Files

Directories that must exist at runtime but whose contents are excluded from Git
(data, models, logs, outputs, checkpoints) must contain a single `.gitkeep` file.

```bash
echo "# Contents excluded from version control — see .gitignore" > data/.gitkeep
```

### 5.2 Protected Directories

| Directory | Policy |
|---|---|
| `data/` | `.gitkeep` only — no data files |
| `models/` | `.gitkeep` only — no model weights |
| `logs/` | `.gitkeep` only — no log files |
| `outputs/` | `.gitkeep` only — no generated outputs |
| `checkpoints/` | `.gitkeep` only — no checkpoint files |
| `results/` | `.gitkeep` only — no result artifacts |

---

## 6. Pre-Commit Checklist

Before every `git commit`, verify the following:

```
[ ] All new code has type hints
[ ] All public functions have Google-style docstrings
[ ] All inputs are validated; domain exceptions are raised
[ ] No magic numbers — constants are in constants.py
[ ] No print() statements — structured logger is used
[ ] No TODO / FIXME / HACK comments
[ ] No placeholder or stub code
[ ] ruff check passes with zero errors
[ ] All unit tests pass: pytest tests/ -v
[ ] git status shows no large files (datasets, models, WSIs)
[ ] git ls-files contains no binary artifacts
[ ] Repository size is within budget (< 50 MB)
```

---

## 7. Commit Message Format

Use the [Conventional Commits](https://www.conventionalcommits.org/) format.

### Structure

```
<type>(<scope>): <short summary>

[Optional body — explain WHY, not WHAT]

[Optional footer — breaking changes, issue references]
```

### Types

| Type | Use for |
|---|---|
| `feat` | New feature or module |
| `fix` | Bug fix |
| `test` | Adding or updating tests |
| `docs` | Documentation changes |
| `refactor` | Code restructuring without behavior change |
| `perf` | Performance improvement |
| `chore` | Tooling, CI, dependency updates |
| `style` | Formatting, linting (no logic change) |

### Scope

Use the milestone or module name:

```
feat(core/constants): add normalization constants for Macenko staining
test(core/config): add unit tests for YAML config loader
docs(milestone-1.1): add architecture decision record for exception hierarchy
```

### Examples

```bash
# Sub-milestone commit
git commit -m "feat(core): implement Milestone 1.1 — project foundation

- Add versioned __init__.py with Python version enforcement
- Add global constants module (constants.py)
- Add exception hierarchy (exceptions.py)
- Add type definitions (types.py)
- Add directory utility functions (utils/path_utils.py)

Milestone: 1.1
Tests: 47 passed, 0 failed
Coverage: 94%"

# Bug fix commit
git commit -m "fix(core/config): resolve YAML merge key handling in nested configs

Fixes edge case where anchors in default.yaml were not correctly
resolved when user config overrides a partial nested mapping."
```

---

## 8. AI Assistant Guidelines

> **This section is specifically for AI coding assistants working on this codebase.**
> (Antigravity, Codex, Claude, Gemini, or any other AI tool)

### Mandatory Rules for Every Code Generation Task

1. **Never add datasets, checkpoints, logs, outputs, raw images, or generated artifacts
   to Git.** Ensure `.gitignore` prevents accidental tracking. The repository must
   remain source-code-only and suitable for frequent incremental pushes.

2. **Every function must have**: type hints, a Google-style docstring, input
   validation, and structured logging via `pathoai.core.logger`.

3. **Never use magic numbers.** All constants belong in `pathoai/core/constants.py`.

4. **Never use `print()`.** Use `get_logger(__name__)`.

5. **Never leave placeholders.** If a feature is not being implemented in the current
   sub-milestone, do not create a stub function. Either implement it fully or omit it
   and plan it explicitly in the next sub-milestone.

6. **Every module must include unit tests** in `tests/unit/` with >= 85% code coverage.

7. **Follow the existing code style** in `pathoai/core/` for consistency.
   Read existing modules before writing new ones.

8. **Check `pathoai/core/constants.py` before defining any numeric literal.**
   If a constant is missing, add it there — do not inline it.

9. **Use domain exceptions** from `pathoai/core/exceptions.py` — never raw
   `ValueError` or `TypeError` in production paths.

10. **Respect the sub-milestone boundary.** Implement exactly what is scoped for the
    current sub-milestone. Do not reach ahead into future milestones.

---

## Quick Reference

```bash
# Run linter
ruff check pathoai/ tests/ && ruff format --check pathoai/ tests/

# Run all tests
pytest tests/ -v --tb=short

# Run tests with coverage
pytest tests/ -v --cov=pathoai --cov-report=term-missing

# Check repository size
git count-objects -vH

# Verify no large files are tracked
git ls-files | grep -E "\.(svs|tiff|ndpi|pth|pt|ckpt|zip|tar|gz|npy|npz)$"

# Standard sub-milestone push sequence
git status
git diff --stat
git add -p          # stage interactively — review every hunk
git commit -m "feat(scope): Milestone X.Y — short description"
git push origin main
```

---

*PathoAI-Platform — Built for Research, Designed for Reproducibility.*
