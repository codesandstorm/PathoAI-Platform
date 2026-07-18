# ADR-002: Configuration System Design

**Status:** Accepted
**Date:** 2026-07-18
**Milestone:** 1
**Author:** PathoAI Research Team

---

## Context

Research pipelines require flexible configuration: the same codebase runs on a
laptop (CPU, small batches), a workstation (single GPU), and HPC clusters (multi-GPU).
Configuration must be:

- **Hierarchical**: common defaults → dataset-specific → experiment-specific → environment
- **Reproducible**: same config hash must produce the same results
- **Auditable**: config must be logged alongside results for provenance
- **Type-safe**: accessing a missing key should produce a clear error, not `None`

Considered approaches:
1. Python `argparse` (flat, no nesting, no defaults hierarchy)
2. `hydra-core` (powerful but adds significant complexity, decorator coupling)
3. Plain `dict` with JSON/YAML (no type safety, no attribute access)
4. **Custom YAML + singleton with attribute access** ← chosen

---

## Decision

We implement a custom YAML-based configuration system with:

### Layer 1: YAML Files (Hierarchical)
```
config/base.yaml              ← Canonical defaults (all params documented)
config/datasets/<name>.yaml   ← Dataset-specific overrides (e.g., tiger.yaml)
config/models/<name>.yaml     ← Model-specific overrides
config/experiments/<id>.yaml  ← Experiment-specific overrides
```

Override priority (lowest → highest):
```
base.yaml → dataset.yaml → model.yaml → experiment.yaml → env vars → CLI
```

### Layer 2: ConfigNode (Attribute Access)
```python
cfg = ConfigManager.get_instance()
patch_size = cfg.wsi.patch_extraction.patch_size  # ← dict access: no
```

Unknown keys raise `ConfigurationError` with the full dotted path:
```
ConfigurationError: Configuration key 'wsi.patch_size' not found.
Check your YAML config files or consult config/base.yaml for defaults.
```

### Layer 3: Environment Variable Overrides
```
PATHOAI_PIPELINE__DEVICE=cpu       → config['pipeline']['device'] = 'cpu'
PATHOAI_WSI__PATCH_SIZE=256        → config['wsi']['patch_size'] = 256
```
Double underscore (`__`) separates nesting levels. Values auto-coerce to
`bool`, `int`, `float`, or `str` as appropriate.

### Layer 4: ConfigManager Singleton
- Initialized once at startup: `ConfigManager.initialize(base_config=...)`
- Accessed from anywhere: `cfg = get_config()`
- Hash computed after merging: `ConfigManager.get_config_hash()` → SHA-256

---

## Rationale

**Why not Hydra?**
Hydra is excellent but introduces decorator coupling (`@hydra.main`), which
makes unit testing harder and forces a particular script entry point structure.
Our simpler approach allows `ConfigManager.reset()` in tests for full isolation.

**Why a singleton?**
The configuration is global state by definition. A singleton with a `.reset()`
method for testing gives the benefits of global access with clean test isolation.

**Why SHA-256 hash?**
Every pipeline run logs the config hash. This allows exact reproduction:
given the hash, git commit, and dataset, the exact configuration is recoverable.
The hash is logged to `environment_snapshot.json` and every structured log record.

**Why `None` in override skips rather than overwrites?**
Dataset configs often need to leave most parameters as base defaults and only
override 1–2 keys. Using `None` as "use base default" avoids brittle full
repetition of the base config in every override file.

---

## Consequences

- **Positive**: All parameters have documented defaults in one place (`base.yaml`)
- **Positive**: Experiments are fully reproducible from config hash + git commit
- **Positive**: Unknown config access raises immediately with helpful message
- **Positive**: CI can validate config hash stability across runs
- **Negative**: Custom implementation requires maintenance vs. Hydra's ecosystem
- **Negative**: No tab-completion for config keys (unlike dataclass-based systems)
- **Negative**: Adding a new parameter requires updating `base.yaml` explicitly

---

## File Layout

```
config/
├── base.yaml                    ← ALL defaults (never import-time generated)
├── datasets/
│   ├── tiger.yaml               ← TIGER dataset paths and specific params
│   └── tcga.yaml                ← TCGA dataset (future)
├── models/
│   ├── deeplabv3plus.yaml       ← Segmentation model config
│   └── faster_rcnn.yaml         ← Detection model config
└── experiments/
    └── baseline_001.yaml        ← First experiment config
```

---

## Config Versioning

`base.yaml` carries a `version` field at the top level. On every `initialize()`,
the version is logged alongside the hash. When the YAML schema changes in a
backward-incompatible way, the version is bumped and release notes document
the migration.

---

## Testing Strategy

- All tests use `tests/fixtures/configs/test_base.yaml` (smallest valid config)
- `ConfigManager.reset()` is called in `autouse=True` fixtures to ensure isolation
- Each test that calls `initialize()` must call `ConfigManager.reset()` in teardown
- Config hash determinism is tested explicitly (same file = same hash)

---

## References

- [config.py](../../pathoai/core/config.py)
- [config/base.yaml](../../config/base.yaml)
- [test_config.py](../../tests/unit/core/test_config.py)
- [test_base.yaml](../../tests/fixtures/configs/test_base.yaml)
