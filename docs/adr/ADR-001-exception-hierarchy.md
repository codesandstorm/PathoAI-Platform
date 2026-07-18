# ADR-001: Exception Hierarchy Design

**Status:** Accepted
**Date:** 2026-07-18
**Milestone:** 1
**Author:** PathoAI Research Team

---

## Context

PathoAI-Platform is a multi-stage research pipeline with distinct failure modes:
configuration errors, WSI read failures, model inference errors, and statistical
validation failures. Without a structured exception hierarchy, callers cannot
distinguish between these categories, making robust error handling and logging
impossible.

Common anti-patterns we wanted to avoid:
- Raising bare `Exception` with string messages
- Using the same exception class for structurally different errors
- `except Exception: pass` silently swallowing failures
- No domain-specific exceptions, making log analysis difficult

---

## Decision

We adopt a three-level exception hierarchy rooted at `PathoAIException`:

```
PathoAIException                   ← Catch-all for any PathoAI error
├── ConfigurationError             ← Config file invalid or missing
├── EnvironmentValidationError     ← Runtime environment inadequate
├── DataError                      ← All data-layer failures
│   ├── WSIReadError               ← OpenSlide cannot open/read WSI
│   ├── MetadataExtractionError    ← MPP/objective power unavailable
│   ├── PatchExtractionError       ← Region read fails
│   ├── DatasetValidationError     ← Structural dataset integrity
│   └── StainNormalizationError    ← Stain normalization failure
├── ModelError                     ← All model-layer failures
│   ├── CheckpointLoadError        ← Checkpoint missing/corrupt/mismatch
│   ├── InferenceError             ← CUDA OOM, NaN outputs
│   └── ModelArchitectureError     ← Registry miss, invalid hyperparams
├── FusionError                    ← Spatial fusion stage
│   └── sTILComputationError       ← sTIL score computation
├── ValidationError                ← Statistical validation
├── ReportGenerationError          ← Report file generation
└── PipelineError                  ← Orchestration errors
```

### Rules enforced by the hierarchy

1. **Every `raise` statement** in production code uses a domain-specific class.
2. **`except` clauses** catch the narrowest applicable class.
3. **Root `PathoAIException`** is caught only at the outermost pipeline boundary.
4. **Preserve cause chain**: always use `raise X from original`.
5. **Actionable messages**: every exception message tells the user what to do.

---

## Rationale

**Why a custom base class?**
Research pipelines are often run unattended overnight. A single catch clause
(`except PathoAIException as e:`) at the pipeline level can log the full error
and gracefully terminate without crashing other slides.

**Why separate DataError from ModelError?**
Data failures (missing WSI, corrupt mask) require different remediation than
model failures (checkpoint mismatch). Separating them enables targeted retry
logic and better structured log analysis.

**Why sub-classes within DataError?**
`WSIReadError` vs `PatchExtractionError` have distinct causes and remediation.
Logging these as separate categories allows filtering: "show me all WSI open
failures" vs "show me all coordinate errors".

---

## Consequences

- **Positive**: Callers can write targeted `except WSIReadError` handlers.
- **Positive**: Log aggregation can group errors by class name.
- **Positive**: Tests can verify exact exception types, not just messages.
- **Positive**: Adding new exception types is non-breaking (they inherit from existing parent).
- **Negative**: Slightly more code in the `exceptions.py` module (~230 lines).
- **Negative**: Developers must learn the hierarchy to use the correct class.

---

## Compliance

All code in `pathoai/` must:
- Import exceptions from `pathoai.core.exceptions`
- Not catch `Exception` or `BaseException` directly (use specific types)
- Not use `raise Exception(...)` (use domain-specific types)

CI enforces this via ruff rules `B017` (blind exception) and custom linting.

---

## References

- [exceptions.py](../../pathoai/core/exceptions.py)
- [test_constants.py](../../tests/unit/core/test_constants.py)
- Salgado et al. (2015) — TILs Working Group guidelines
- Martin Fowler, "Replacing Error Codes with Exceptions"
