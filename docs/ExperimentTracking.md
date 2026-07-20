# Experiment Tracking & Provenance Framework (Milestone 10.5)

The Experiment Tracking & Provenance Framework (`pathoai/experiments/`) logs run manifests, runtime hardware/software environments, seeds, and validation metrics for 100% scientific reproducibility.

---

## 🏗️ Architecture

```
ExperimentConfig + ValidationResult
                │
                ▼
        ExperimentTracker
                ├── EnvironmentAuditor (Git commit, Python, PyTorch, CUDA, GPU)
                ├── ManifestGenerator (ExperimentManifest DTO JSON/YAML export)
                ├── ReproducibilityManager (Seed determinism enforcement)
                ├── PublicationTableGenerator (Table 1, Table 2, Table 3 Markdown)
                ├── LaTeXExporter (Journal LaTeX \begin{table} exports)
                └── SupplementaryPackageGenerator (Research package bundling)
```
